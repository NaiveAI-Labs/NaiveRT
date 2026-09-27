#!/usr/bin/env python3
"""Per-kernel check: the dense MLP path, rmsnorm_up_gate (one GPU) and down_allreduce
(8 GPUs), at one model's geometry (dense intermediate per rank = n native 256-wide slots).

usage: run_dense_ffn.py [--model naive_n0p5] [--seed 7] [--generations 3]

rmsnorm_up_gate: x_norm = x * gamma * rsqrt(mean(x^2) + eps) as fp16, then SiLU(gate) * up
  from fp8 block-128 weights in the compact workspace ([tiles, pages, 0x8020] bytes:
  pack_gate_up's pages without their spare bytes, the layout the checkpoint loader compacts to).
  Check: max |y - y_ref| below 2 % of max |y_ref|.
down_allreduce: output = residual + sum over ranks of W_down_r @ up_gate_r (fp8 block-128,
  pack_down layout). Checks: ranks within one bf16 ulp, max |out - ref| below 2 %.
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
from naivert.kernels.packing import pack_down, pack_gate_up, permute_for_up_gate, quantize_fp8_block  # noqa: E402

SLOT = 256
PANE = 0x8020
MAILBOX_BYTES = 4 * 0x70000


def pages_of(hidden):
    return hidden // 1024


def compact_workspace(gate, up, hidden):
    """[inter, hidden] bf16 gate/up -> the up-gate kernel's compact page workspace, plus the
    fp8 parts of the NATURAL weights for the reference.

    The kernel reads pack_gate_up's pages in its own fragment order, so the weights are
    pre-permuted (packing.permute_for_up_gate) exactly as the Naive N dense converter does.
    """
    slots, pages = gate.shape[0] // SLOT, hidden // 1024

    def parts_of(g, u):
        gq, gs = zip(*(quantize_fp8_block(t) for t in g.reshape(slots, SLOT, hidden)))
        uq, us = zip(*(quantize_fp8_block(t) for t in u.reshape(slots, SLOT, hidden)))
        return tuple(torch.stack(t) for t in (gq, gs, uq, us))

    kernel_parts = parts_of(permute_for_up_gate(gate), permute_for_up_gate(up))
    native = pack_gate_up(*kernel_parts, hidden=hidden).view(torch.uint8).reshape(slots * 16, pages, 32 * 1024 + 128)
    return native[:, :, :PANE].contiguous(), parts_of(gate, up)


def main() -> int:
    ap = argparse.ArgumentParser()
    add_model_argument(ap)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--diag", action="store_true")
    args = ap.parse_args()
    M = select(args.model)
    hidden, inter, slots = M.hidden, M.dense_inter, M.dense_inter // SLOT
    t0 = time.time()
    ext = M.load_extension()
    print(f"{M.name} model_ops extension loaded in {time.time() - t0:.0f} s; hidden {hidden}, dense inter {inter} = {slots} slots, down tile {M.down_tile}")
    ext.enable_tp8_peer_access()
    devices = [torch.device(f"cuda:{r}") for r in range(M.ranks)]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    ok = True

    # --- rmsnorm_up_gate on rank 0 ---------------------------------------------------
    dev0 = devices[0]
    gate = (torch.randn((inter, hidden), generator=gen) * 0.02).to(torch.bfloat16)
    up = (torch.randn((inter, hidden), generator=gen) * 0.02).to(torch.bfloat16)
    workspace, (gate_q, gate_s, up_q, up_s) = compact_workspace(gate, up, hidden)
    gamma = 1.0 + 0.1 * torch.randn(hidden, generator=gen)
    print(f"up-gate workspace {tuple(workspace.shape)} bytes = {workspace.numel()}")
    dummy8 = torch.zeros(16, dtype=torch.float8_e4m3fn, device=dev0)
    dummy16 = torch.zeros(16, dtype=torch.bfloat16, device=dev0)
    ws = workspace.to(dev0)
    for trial in range(args.generations):
        x = (torch.randn((1, hidden), generator=gen) * (1.0 + trial)).to(torch.bfloat16).to(dev0)
        y = torch.full((1, inter), float("nan"), dtype=torch.bfloat16, device=dev0)
        ext.launch_rmsnorm_up_gate(x, gamma.to(dev0), dummy8, dummy16, dummy8, dummy16, ws, y, hidden, inter)
        torch.cuda.synchronize(dev0)
        xf = x.float().cpu()
        inv = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + M.eps)
        x_half = (xf * gamma * inv).to(torch.float16).float()[0]
        g_ref = torch.cat([dequant(gate_q[s], gate_s[s]) @ x_half for s in range(slots)])
        u_ref = torch.cat([dequant(up_q[s], up_s[s]) @ x_half for s in range(slots)])
        y_ref = torch.nn.functional.silu(g_ref) * u_ref
        scale = float(y_ref.abs().max())
        diff = (y[0].float().cpu() - y_ref).abs()
        max_abs = float(diff.max())
        t_ok = bool(torch.isfinite(y.float()).all()) and max_abs <= 0.02 * scale
        ok = ok and t_ok
        print(f"  rmsnorm_up_gate trial {trial}: max|y-ref|={max_abs:.3e} / scale {scale:.3e} = {max_abs / scale:.4f} -> {'ok' if t_ok else 'MISMATCH'}")
        if args.diag and not t_ok:
            tile_err = diff.reshape(-1, 16).max(dim=1).values / scale
            bad = (tile_err > 0.02).nonzero().flatten()
            print(f"    16-wide tiles over 2 %: {int(bad.numel())} of {tile_err.numel()}; first {bad[:12].tolist()}")
            print(f"    y[:6]={y[0][:6].float().tolist()}\n    ref[:6]={y_ref[:6].tolist()}")
            # Is the kernel's output order within a 16-wide tile a fixed permutation of ours?
            yt = y[0].float().cpu().reshape(-1, 16)
            rt = y_ref.reshape(-1, 16)
            perm = (yt[0][:, None] - rt[0][None, :]).abs().argmin(dim=1)
            agree = int(((yt - rt[:, perm]).abs().max(dim=1).values <= 0.02 * scale).sum())
            print(f"    tile-0 permutation y[i] = ref[perm[i]]: {perm.tolist()}; tiles matching that permutation: {agree} of {yt.shape[0]}")
            swapped = torch.nn.functional.silu(u_ref) * g_ref
            print(f"    if gate/up were swapped: max|y-ref|/scale = {float((y[0].float().cpu() - swapped).abs().max()) / scale:.4f}")
            if M.name == "naive_n0p5" and slots <= M.routed_experts:
                # Cross-check against expert_select on the very same pages: the slots are
                # experts 0..slots-1 of a routed table; force their selection with the scores.
                table = torch.zeros((M.routed_experts, 16, pages_of(hidden), 32 * 1024 + 128), dtype=torch.float8_e4m3fn, device=dev0)
                table[:slots] = pack_gate_up(gate_q, gate_s, up_q, up_s, hidden=hidden).to(dev0)
                scores = torch.full((1, M.routed_experts), -30.0, device=dev0)
                scores[0, :slots] = torch.arange(slots, 0, -1, device=dev0).float()   # experts 0..slots-1 win, in order
                bias = torch.zeros(M.routed_experts, device=dev0)
                x_norm = (xf * gamma * inv).to(torch.bfloat16).to(dev0)
                es_y = torch.empty((1, M.selected_experts, 16, 16), dtype=torch.bfloat16, device=dev0)
                es_p = torch.empty((1, M.selected_experts), dtype=torch.float32, device=dev0)
                es_i = torch.empty((1, M.selected_experts), dtype=torch.int32, device=dev0)
                ext.launch_expert_select_up_gate_silu_rows1(x_norm, scores, bias, table, es_y, es_p, es_i)
                torch.cuda.synchronize(dev0)
                order = es_i[0].tolist()
                es_rows = torch.cat([es_y[0][s].reshape(-1) for s in sorted(range(M.selected_experts), key=lambda s: order[s])]).float().cpu()
                n = min(es_rows.numel(), inter)
                print(f"    expert_select on the same pages (fp8 activation), experts {order}: "
                      f"max|y_upgate - y_expert_select| / scale = {float((y[0].float().cpu()[:n] - es_rows[:n]).abs().max()) / scale:.4f}, "
                      f"max|ref - y_expert_select| / scale = {float((y_ref[:n] - es_rows[:n]).abs().max()) / scale:.4f}")

    # --- down_allreduce on 8 GPUs ----------------------------------------------------
    per_rank = []
    for r, dev in enumerate(devices):
        w = (torch.randn((inter, hidden), generator=gen) * 0.02).to(torch.bfloat16)      # down^T: [inter, hidden]
        cols = w.T.reshape(hidden, slots, SLOT).transpose(0, 1)                          # [slots, hidden, 256]
        q, s = zip(*(quantize_fp8_block(c) for c in cols))
        q, s = torch.stack(q), torch.stack(s).to(torch.bfloat16)
        packed, packed_scale = pack_down(q, s, hidden=hidden, outputs_per_tile=M.down_tile, inter=SLOT)
        per_rank.append({
            "packed": packed.to(dev), "scale": packed_scale.to(dev), "deq": dequant(q, s),
            "mailbox": torch.zeros(MAILBOX_BYTES, dtype=torch.uint8, device=dev),
            "step_block": torch.zeros(16, dtype=torch.uint8, device=dev),
            "generation": torch.zeros(1, dtype=torch.int32, device=dev),
        })
    mailboxes = [st["mailbox"] for st in per_rank]
    print(f"down weights: packed {tuple(per_rank[0]['packed'].shape)}, scale {tuple(per_rank[0]['scale'].shape)}")
    for g in range(args.generations):
        up_rows = [(torch.randn(inter, generator=gen) * 2.0).to(torch.bfloat16) for _ in range(M.ranks)]
        residual = (torch.randn(hidden, generator=gen) * 4.0).to(torch.bfloat16)
        ref = residual.float().clone()
        for r in range(M.ranks):
            partial = torch.zeros(hidden)
            for s in range(slots):
                partial += per_rank[r]["deq"][s] @ up_rows[r][s * SLOT:(s + 1) * SLOT].float()
            ref += partial.to(torch.bfloat16).float()
        ref = ref.to(torch.bfloat16)
        for r, dev in enumerate(devices):
            st = per_rank[r]
            st["up"] = up_rows[r].to(dev).reshape(1, inter)
            st["residual"] = residual.to(dev)
            st["out"] = torch.full((hidden,), float("nan"), dtype=torch.bfloat16, device=dev)
            st["generation"].fill_(g + 1)
        for r, dev in enumerate(devices):
            st = per_rank[r]
            ext.launch_down_allreduce(st["up"], st["packed"], st["scale"], st["residual"], st["out"],
                                      st["step_block"], st["generation"], 3, mailboxes, r, hidden, inter)
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
        print(f"  down_allreduce generation {g}: finite={finite}, cross-rank max ulp={cross}, "
              f"max|out-ref|={max_abs:.3e} / scale {scale:.3e} = {max_abs / scale:.4f}, "
              f"worst diff/threshold {max(c[2] for c in checks):.2f} -> {'ok' if g_ok else 'MISMATCH'}")
        if args.diag and not g_ok:
            tile = M.down_tile
            tile_err = diff[: (hidden // tile) * tile].reshape(-1, tile).max(dim=1).values / scale
            bad = (tile_err > 0.02).nonzero().flatten()
            print(f"    tiles over 2 %: {int(bad.numel())} of {tile_err.numel()}; first {bad[:12].tolist()}")
            print(f"    out[:6]={outs[0][:6].float().tolist()}\n    ref[:6]={ref[:6].float().tolist()}")
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
