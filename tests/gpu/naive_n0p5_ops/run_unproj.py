#!/usr/bin/env python3
"""Per-kernel check: unproj_o_allreduce on 8 GPUs (attention output projection + TP8
all-reduce) at one model's geometry, against a torch reference.

usage: run_unproj.py [--model naive_n0p5] [--seed 7] [--generations 3]

Each rank holds its [hidden, K] slice of o_proj as fp8 block-128 (naivert.kernels.packing.
pack_unproj) and its own bf16 proj_o row; output = residual + sum over ranks of W_r @ proj_o_r
on every rank through the TP8 mailbox. Checks per generation: every rank within one bf16 ulp
of rank 0, and max |out - ref| below 2 % of max |ref - residual| (fp8 products exact; f32 sum
order and the bf16 exchange differ). --diag prints the worst 64-wide output tiles.
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
from naivert.kernels.packing import pack_unproj, quantize_fp8_block  # noqa: E402

MAILBOX_BYTES = 0x70000


def main() -> int:
    ap = argparse.ArgumentParser()
    add_model_argument(ap)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--diag", action="store_true")
    args = ap.parse_args()
    M = select(args.model)
    t0 = time.time()
    ext = M.load_extension()
    print(f"{M.name} model_ops extension loaded in {time.time() - t0:.0f} s; hidden {M.hidden}, K {M.unproj_k}, tile {M.unproj_tile}")
    ext.enable_tp8_peer_access()
    devices = [torch.device(f"cuda:{r}") for r in range(M.ranks)]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    per_rank = []
    for r, dev in enumerate(devices):
        w = (torch.randn((M.hidden, M.unproj_k), generator=gen) * 0.02).to(torch.bfloat16)
        q, s = quantize_fp8_block(w)
        packed, packed_scale = pack_unproj(q, s, hidden=M.hidden, outputs_per_tile=M.unproj_tile, input_slots=4)
        per_rank.append({
            "packed": packed.to(dev), "scale": packed_scale.to(dev), "deq": dequant(q, s),
            "mailbox": torch.zeros(MAILBOX_BYTES, dtype=torch.uint8, device=dev),
        })
    mailboxes = [st["mailbox"] for st in per_rank]
    print(f"weights: packed {tuple(per_rank[0]['packed'].shape)} {per_rank[0]['packed'].dtype}, scale {tuple(per_rank[0]['scale'].shape)}")
    ok = True
    for g in range(args.generations):
        # Eager launches carry no replay epoch, so the generation is the tag alone and its
        # low byte (the layer tag) must differ between consecutive exchanges on one mailbox.
        tag = 0x11 + g
        proj = [(torch.randn(M.unproj_k, generator=gen) * 2.0).to(torch.bfloat16) for _ in range(M.ranks)]
        residual = (torch.randn(M.hidden, generator=gen) * 4.0).to(torch.bfloat16)
        ref = residual.float().clone()
        for r in range(M.ranks):
            ref += (per_rank[r]["deq"] @ proj[r].float()).to(torch.bfloat16).float()
        ref = ref.to(torch.bfloat16)
        for r, dev in enumerate(devices):
            st = per_rank[r]
            st["proj"] = proj[r].to(dev)
            st["residual"] = residual.to(dev)
            st["out"] = torch.full((M.hidden,), float("nan"), dtype=torch.bfloat16, device=dev)
        for r, dev in enumerate(devices):
            st = per_rank[r]
            ext.launch_unproj_allreduce(st["proj"], st["packed"], st["scale"], st["residual"], st["out"], mailboxes, r, tag)
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
            tile_err = diff.reshape(-1, M.unproj_tile).max(dim=1).values / scale
            bad = (tile_err > 0.02).nonzero().flatten()
            print(f"    tiles over 2 %: {int(bad.numel())} of {tile_err.numel()}; first {bad[:12].tolist()}")
            print(f"    tile 0 out[:8]={outs[0][:8].float().tolist()}\n    tile 0 ref[:8]={ref[:8].float().tolist()}")
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
