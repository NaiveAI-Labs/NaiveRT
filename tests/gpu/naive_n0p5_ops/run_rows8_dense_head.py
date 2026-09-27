#!/usr/bin/env python3
"""Naive N per-kernel check for the eight-row launches of the layer-0 dense MLP and the LM
head: rmsnorm_up_gate rows8 (one GPU), down_allreduce rows8 (8 GPUs) and rmsnorm_head_proj
rows8 (one GPU), checked against the
torch reference of run_dense_ffn.py / run_head.py.

usage: run_rows8_dense_head.py [--seed 7] [--trials 2] [--timing]
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

from _model import close_bf16, dequant, select  # noqa: E402
from naivert.kernels.packing import pack_down, pack_head, quantize_fp8_block  # noqa: E402
from run_dense_ffn import PANE, SLOT, compact_workspace  # noqa: E402,F401

ROWS = 8
MAILBOX_BYTES = ROWS * 4 * 0x70000


def timed(fn, dev, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize(dev)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(dev)
    return start.elapsed_time(end) / iters * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    M = select("naive_n0p5")
    hidden, inter, slots = M.hidden, M.dense_inter, M.dense_inter // SLOT
    t0 = time.time()
    ext = M.load_extension()
    print(f"{M.name} model_ops extension loaded in {time.time() - t0:.0f} s; hidden {hidden}, dense inter {inter}")
    ext.enable_tp8_peer_access()
    devices = [torch.device(f"cuda:{r}") for r in range(M.ranks)]
    dev0 = devices[0]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    ok = True

    # --- rmsnorm_up_gate rows8 vs torch, rank 0 ---------------------------------
    gate = (torch.randn((inter, hidden), generator=gen) * 0.02).to(torch.bfloat16)
    up = (torch.randn((inter, hidden), generator=gen) * 0.02).to(torch.bfloat16)
    workspace, (gate_q, gate_s, up_q, up_s) = compact_workspace(gate, up, hidden)
    gamma = 1.0 + 0.1 * torch.randn(hidden, generator=gen)
    dummy8 = torch.zeros(16, dtype=torch.float8_e4m3fn, device=dev0)
    dummy16 = torch.zeros(16, dtype=torch.bfloat16, device=dev0)
    ws, gamma0 = workspace.to(dev0), gamma.to(dev0)
    for trial in range(args.trials):
        x = (torch.randn((ROWS, hidden), generator=gen) * (1.0 + trial)).to(torch.bfloat16).to(dev0)
        y8 = torch.full((ROWS, inter), float("nan"), dtype=torch.bfloat16, device=dev0)
        ext.launch_rmsnorm_up_gate(x, gamma0, dummy8, dummy16, dummy8, dummy16, ws, y8, hidden, inter)
        torch.cuda.synchronize(dev0)
        xf = x.float().cpu()
        inv = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + M.eps)
        x_half = (xf * gamma * inv).to(torch.float16).float()
        worst = 0.0
        for row in range(ROWS):
            g_ref = torch.cat([dequant(gate_q[s], gate_s[s]) @ x_half[row] for s in range(slots)])
            u_ref = torch.cat([dequant(up_q[s], up_s[s]) @ x_half[row] for s in range(slots)])
            y_ref = torch.nn.functional.silu(g_ref) * u_ref
            worst = max(worst, float((y8[row].float().cpu() - y_ref).abs().max()) / float(y_ref.abs().max()))
        t_ok = bool(torch.isfinite(y8.float()).all()) and worst <= 0.02
        ok = ok and t_ok
        print(f"  rmsnorm_up_gate rows8 trial {trial}: worst row max|y-ref|/scale = {worst:.4f} -> {'ok' if t_ok else 'MISMATCH'}")
    if args.timing:
        def f8():
            ext.launch_rmsnorm_up_gate(x, gamma0, dummy8, dummy16, dummy8, dummy16, ws, y8, hidden, inter)

        print(f"  rmsnorm_up_gate L2-warm: rows8 {timed(f8, dev0):.2f} us")

    # --- down_allreduce rows8 vs torch, 8 GPUs ---------------------------------------------
    per_rank = []
    for r, dev in enumerate(devices):
        w = (torch.randn((inter, hidden), generator=gen) * 0.02).to(torch.bfloat16)
        cols = w.T.reshape(hidden, slots, SLOT).transpose(0, 1)
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
    generation = 0
    for trial in range(args.trials):
        up_rows = [(torch.randn((ROWS, inter), generator=gen) * 2.0).to(torch.bfloat16) for _ in range(M.ranks)]
        residual = (torch.randn((ROWS, hidden), generator=gen) * 4.0).to(torch.bfloat16)
        ref = residual.float().clone()
        for r in range(M.ranks):
            partial = torch.zeros((ROWS, hidden))
            for s in range(slots):
                partial += up_rows[r][:, s * SLOT:(s + 1) * SLOT].float() @ per_rank[r]["deq"][s].T
            ref += partial.to(torch.bfloat16).float()
        ref = ref.to(torch.bfloat16)
        for r, dev in enumerate(devices):
            st = per_rank[r]
            st["up"] = up_rows[r].to(dev)
            st["residual"] = residual.to(dev)
            st["out8"] = torch.full((ROWS, hidden), float("nan"), dtype=torch.bfloat16, device=dev)
        generation += 1
        for st in per_rank:
            st["generation"].fill_(generation)
        for r, dev in enumerate(devices):
            st = per_rank[r]
            ext.launch_down_allreduce(st["up"], st["packed"], st["scale"], st["residual"], st["out8"],
                                      st["step_block"], st["generation"], 3, mailboxes, r, hidden, inter)
        for dev in devices:
            torch.cuda.synchronize(dev)
        outs8 = [per_rank[r]["out8"].cpu() for r in range(M.ranks)]
        finite = all(bool(torch.isfinite(o.float()).all()) for o in outs8)
        cross = max(int((o.view(torch.int16).int() - outs8[0].view(torch.int16).int()).abs().max()) for o in outs8)
        worst = 0.0
        for row in range(ROWS):
            scale = float((ref[row].float() - residual[row].float()).abs().max())
            worst = max(worst, max(close_bf16(o[row], ref[row], scale)[2] for o in outs8))
        t_ok = finite and cross <= 1 and worst <= 1.0
        ok = ok and t_ok
        print(f"  down_allreduce rows8 trial {trial}: finite={finite}, cross-rank max ulp={cross}, "
              f"worst row diff/threshold {worst:.2f} -> {'ok' if t_ok else 'MISMATCH'}")

    # --- rmsnorm_head_proj rows8 vs torch, rank 0 ------------------------------------------
    VOCAB = M.vocab_shard if hasattr(M, "vocab_shard") else 19072
    w_head = (torch.randn((VOCAB, hidden), generator=gen) * 0.02).to(torch.bfloat16)
    lm_head = pack_head(w_head, hidden=hidden).to(dev0)
    gamma_h = (1.0 + 0.1 * torch.randn(hidden, generator=gen))
    ws8 = torch.zeros(ROWS * 64, dtype=torch.uint8, device=dev0)
    inboxes = [torch.zeros(4096, dtype=torch.uint8, device=dev0) for _ in range(8)]
    inbox_pointers = [t.data_ptr() for t in inboxes]
    for trial in range(args.trials):
        x = (torch.randn((ROWS, hidden), generator=gen) * (1.0 + trial)).to(torch.bfloat16)
        xd = x.to(dev0)
        logits8 = torch.full((ROWS, VOCAB), float("nan"), dtype=torch.float32, device=dev0)
        last8 = torch.zeros((ROWS, hidden), dtype=torch.bfloat16, device=dev0)
        ext.launch_rmsnorm_head_proj_rows1(xd, gamma_h.to(dev0), lm_head, logits8, last8, ws8, hidden, VOCAB, inbox_pointers, 0)
        torch.cuda.synchronize(dev0)
        xf = x.float()
        inv = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + M.eps)
        x_norm = (xf * gamma_h * inv).to(torch.bfloat16)
        logits_ref = x_norm.float() @ w_head.float().T
        ulps = int(((last8.cpu().view(torch.int16).int() - x_norm.view(torch.int16).int()).abs() > 1).sum())
        max_diff = float((logits8.cpu() - logits_ref).abs().max())
        argmax_same = int((logits8.cpu().argmax(dim=1) == logits_ref.argmax(dim=1)).sum())
        # Compare all logits and normalized hidden states with the independent reference.
        logit_scale = float(logits_ref.abs().max())
        t_ok = (bool(torch.isfinite(logits8).all()) and ulps == 0 and max_diff < 0.02 * logit_scale
                and argmax_same == ROWS)
        ok = ok and t_ok
        print(f"  rmsnorm_head_proj rows8 trial {trial}: last_hidden >1ulp={ulps}, "
              f"logits max|diff|={max_diff:.2e} (max|logit| {logit_scale:.2f}), argmax same {argmax_same}/{ROWS} -> {'ok' if t_ok else 'MISMATCH'}")
    if args.timing:
        def h8():
            ext.launch_rmsnorm_head_proj_rows1(xd, gamma_h.to(dev0), lm_head, logits8, last8, ws8, hidden, VOCAB, inbox_pointers, 0)

        print(f"  rmsnorm_head_proj L2-warm: rows8 {timed(h8, dev0, 50):.1f} us")
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
