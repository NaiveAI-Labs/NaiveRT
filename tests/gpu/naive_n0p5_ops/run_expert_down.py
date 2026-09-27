#!/usr/bin/env python3
"""Per-kernel check: expert_down_allreduce on 8 GPUs at one model's geometry, against a
torch reference.

usage: run_expert_down.py [--model naive_n0p5] [--seed 7] [--generations 3]

Every rank holds its 256-wide slice of each expert's down projection (fp8 block-128,
naivert.kernels.packing.pack_down) and the SiLU(gate)*up rows of the slots for that slice;
routing (indices, probs) is replicated. Slots below the model's shared-expert count address
packed expert `slot` with weight 1; routed slots address `index + shared` with their prob.
The kernel computes residual + sum over ranks and slots of weight[s] * W_down[e_s] @ up[s]
through the TP8 mailbox. Checks per generation: ranks within one bf16 ulp; max |out - ref|
below 2 % of max |ref - residual|. --diag prints the worst output tiles.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CUDA_HOME", "/opt/cuda132-min")

import torch  # noqa: E402

from _model import add_model_argument, close_bf16, dequant, select  # noqa: E402
from naivert.kernels.packing import (  # noqa: E402
    dequantize_mxfp8_block32, pack_down, pack_down_mx,
    quantize_fp8_block, quantize_mxfp8_block32)

MAILBOX_BYTES = 0x70000


def main() -> int:
    ap = argparse.ArgumentParser()
    add_model_argument(ap)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--diag", action="store_true")
    args = ap.parse_args()
    M = select(args.model, args.weights)
    t0 = time.time()
    ext = M.load_extension()
    print(f"{M.name} model_ops extension loaded in {time.time() - t0:.0f} s; hidden {M.hidden}, "
          f"{M.total_experts} experts ({M.shared_experts} shared), {M.slots} slots, tile {M.expert_down_tile}")
    ext.enable_tp8_peer_access()
    domain_size = int(ext.expert_down_domain_size())
    devices = [torch.device(f"cuda:{r}") for r in range(M.ranks)]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    inter = M.expert_inter

    indices_cpu = torch.randperm(M.routed_experts, generator=gen)[:M.selected_experts].to(torch.int32)
    sig = torch.rand(M.selected_experts, generator=gen) * 0.5 + 0.25
    probs_cpu = (sig / sig.sum() * M.route_scale).float()
    used = list(range(M.shared_experts)) + [int(i) + M.shared_experts for i in indices_cpu.tolist()]  # packed indices
    weights = [1.0] * M.shared_experts + probs_cpu.tolist()
    print(f"selected experts {indices_cpu.tolist()} -> packed slots {used}")

    per_rank = []
    for r, dev in enumerate(devices):
        deq = []
        if M.weights == "mx":
            q_all = torch.zeros((M.total_experts, M.hidden, inter), dtype=torch.float8_e4m3fn)
            s_all = torch.full((M.total_experts, M.hidden, inter // 32), 127, dtype=torch.uint8)
            for e in used:
                w = (torch.randn((M.hidden, inter), generator=gen) * 0.02).to(torch.bfloat16)
                q, s = quantize_mxfp8_block32(w)
                q_all[e], s_all[e] = q, s
                deq.append(dequantize_mxfp8_block32(q, s))
            packed = pack_down_mx(q_all, s_all, hidden=M.hidden, outputs_per_tile=M.expert_down_tile, inter=inter).view(torch.float8_e4m3fn)
            scale = torch.zeros(16, dtype=torch.bfloat16)
        else:
            q_all = torch.zeros((M.total_experts, M.hidden, inter), dtype=torch.float8_e4m3fn)
            s_all = torch.zeros((M.total_experts, M.hidden // 128, inter // 128), dtype=torch.bfloat16)
            for e in used:
                w = (torch.randn((M.hidden, inter), generator=gen) * 0.02).to(torch.bfloat16)
                q, s = quantize_fp8_block(w)
                q_all[e], s_all[e] = q, s.to(torch.bfloat16)
                deq.append(dequant(q, s))
            packed, scale = pack_down(q_all, s_all, hidden=M.hidden, outputs_per_tile=M.expert_down_tile, inter=inter)
        per_rank.append({
            "packed": packed.to(dev), "scale": scale.to(dev), "deq": torch.stack(deq),   # [slots, hidden, inter]
            "mailbox": torch.zeros(MAILBOX_BYTES, dtype=torch.uint8, device=dev),
            "domain": torch.zeros(domain_size, dtype=torch.uint8, device=dev),
            "indices": indices_cpu.to(dev), "probs": probs_cpu.to(dev),
        })
    mailboxes = [st["mailbox"] for st in per_rank]
    print(f"weights: packed {tuple(per_rank[0]['packed'].shape)}, scale {tuple(per_rank[0]['scale'].shape)}")

    ok = True
    for g in range(args.generations):
        # Eager launches carry no replay epoch, so the generation is the tag alone and its
        # low byte (the layer tag) must differ between consecutive exchanges on one mailbox.
        tag = 0x11 + g
        up = [(torch.randn((M.slots, inter), generator=gen) * 2.0).to(torch.bfloat16) for _ in range(M.ranks)]
        residual = (torch.randn(M.hidden, generator=gen) * 4.0).to(torch.bfloat16)
        ref = residual.float().clone()
        for r in range(M.ranks):
            partial = torch.zeros(M.hidden)
            for s in range(M.slots):
                partial += weights[s] * (per_rank[r]["deq"][s] @ up[r][s].float())
            ref += partial.to(torch.bfloat16).float()
        ref = ref.to(torch.bfloat16)
        for r, dev in enumerate(devices):
            st = per_rank[r]
            st["up"] = up[r].to(dev)
            st["residual"] = residual.to(dev)
            st["out"] = torch.full((M.hidden,), float("nan"), dtype=torch.bfloat16, device=dev)
            ext.prepare_expert_down_replay_domain(
                st["domain"], st["up"], st["packed"], st["scale"], st["probs"], st["indices"],
                st["out"], st["residual"], mailboxes, tag, 0, r)
        for r, dev in enumerate(devices):
            ext.launch_expert_down_rows_from_replay_domain(per_rank[r]["domain"], 0, r, 1)
        for dev in devices:
            torch.cuda.synchronize(dev)
        outs = [per_rank[r]["out"].cpu() for r in range(M.ranks)]
        finite = all(bool(torch.isfinite(o.float()).all()) for o in outs)
        cross = max(int((o.view(torch.int16).int() - outs[0].view(torch.int16).int()).abs().max()) for o in outs)
        scale = float((ref.float() - residual.float()).abs().max())
        diff = (outs[0].float() - ref.float()).abs()
        checks = [close_bf16(o, ref, scale) for o in outs]
        max_abs = max(c[1] for c in checks)
        g_ok = finite and cross <= 1 and all(c[0] for c in checks)
        ok = ok and g_ok
        print(f"  generation {g} (tag {tag:#x}): finite={finite}, cross-rank max ulp={cross}, "
              f"max|out-ref|={max_abs:.3e} / scale {scale:.3e} = {max_abs / scale:.4f}, "
              f"worst diff/threshold {max(c[2] for c in checks):.2f} -> {'ok' if g_ok else 'MISMATCH'}")
        if args.diag and not g_ok:
            tile = M.expert_down_tile
            tile_err = diff[: (M.hidden // tile) * tile].reshape(-1, tile).max(dim=1).values / scale
            bad = (tile_err > 0.02).nonzero().flatten()
            print(f"    tiles over 2 %: {int(bad.numel())} of {tile_err.numel()}; first {bad[:12].tolist()}")
            print(f"    out[:6]={outs[0][:6].float().tolist()}\n    ref[:6]={ref[:6].float().tolist()}")
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
