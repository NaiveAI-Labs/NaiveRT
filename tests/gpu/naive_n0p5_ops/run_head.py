#!/usr/bin/env python3
"""Naive N per-kernel check: rmsnorm_head_proj rows1 on one GPU at Naive N shapes (hidden 4096,
vocab shard 19072 of 152576), built through naivert.kernels.naive_n0p5.model_ops.

usage: run_head.py [--seed 7] [--trials 3] [--timing]

x_norm = bf16(x * gamma * rsqrt(mean(x^2) + 1e-5)); logits = x_norm . W_shard (bf16 weights
packed with naivert.kernels.packing.pack_head, f32 accumulation). The sampler inboxes are
null (no top-1 handoff; that is the runtime's sampler contract, tested with the graph).
Checks: last_hidden_states within one bf16 ulp of x_norm, logits within 2e-4 of the f32
torch dot products, and the shard argmax identical.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_HOME", "/opt/cuda132-min")

import torch  # noqa: E402

from naivert.kernels.packing import pack_head  # noqa: E402
from naivert.models.naive_n0p5.model import NAIVE_N0P5_HYBRID, NAIVE_N0P5_GEOMETRY  # noqa: E402

HIDDEN = NAIVE_N0P5_GEOMETRY.hidden_size
VOCAB = NAIVE_N0P5_GEOMETRY.vocab_shard_size   # 19072
EPS = float(NAIVE_N0P5_HYBRID.norm_eps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    from naivert.kernels.naive_n0p5.model_ops import load_extension
    t0 = time.time()
    ext = load_extension()
    print(f"Naive N model_ops extension loaded in {time.time() - t0:.0f} s")
    device = torch.device("cuda:0")
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    w = (torch.randn((VOCAB, HIDDEN), generator=gen) * 0.02).to(torch.bfloat16)
    lm_head = pack_head(w, hidden=HIDDEN).to(device)
    gamma = (1.0 + 0.1 * torch.randn(HIDDEN, generator=gen))
    print(f"lm_head packed {tuple(lm_head.shape)} {lm_head.dtype}")
    workspace = torch.zeros(64, dtype=torch.uint8, device=device)
    # The kernel publishes its shard's top-1 candidate into every rank's sampler inbox
    # (128-byte SamplerPeerState); on one GPU all eight inboxes are local scratch.
    inboxes = [torch.zeros(4096, dtype=torch.uint8, device=device) for _ in range(8)]
    inbox_pointers = [t.data_ptr() for t in inboxes]
    ok = True
    for trial in range(args.trials):
        x = (torch.randn((1, HIDDEN), generator=gen) * (1.0 + trial)).to(torch.bfloat16)
        logits = torch.full((1, VOCAB), float("nan"), dtype=torch.float32, device=device)
        last_hidden = torch.zeros((1, HIDDEN), dtype=torch.bfloat16, device=device)
        ext.launch_rmsnorm_head_proj_rows1(x.to(device), gamma.to(device), lm_head, logits, last_hidden, workspace,
                                           HIDDEN, VOCAB, inbox_pointers, 0)
        torch.cuda.synchronize(device)
        xf = x.float()
        inv = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + EPS)
        x_norm = (xf * gamma * inv).to(torch.bfloat16)
        logits_ref = x_norm.float() @ w.float().T
        ulps = (last_hidden.cpu().view(torch.int16).int() - x_norm.view(torch.int16).int()).abs()
        max_diff = float((logits.cpu() - logits_ref).abs().max())
        same_argmax = int(logits.cpu().argmax()) == int(logits_ref.argmax())
        t_ok = bool(torch.isfinite(logits).all()) and int((ulps > 1).sum()) == 0 and max_diff < 2e-4 and same_argmax
        ok = ok and t_ok
        print(f"  trial {trial}: last_hidden >1ulp={int((ulps > 1).sum())}, logits max|diff|={max_diff:.2e} "
              f"(max|logit| {float(logits_ref.abs().max()):.2f}), argmax same={same_argmax} -> {'ok' if t_ok else 'MISMATCH'}")
    if args.timing:
        for _ in range(10):
            ext.launch_rmsnorm_head_proj_rows1(x.to(device), gamma.to(device), lm_head, logits, last_hidden, workspace, HIDDEN, VOCAB, inbox_pointers, 0)
        torch.cuda.synchronize(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(200):
            ext.launch_rmsnorm_head_proj_rows1(x.to(device), gamma.to(device), lm_head, logits, last_hidden, workspace, HIDDEN, VOCAB, inbox_pointers, 0)
        end.record()
        torch.cuda.synchronize(device)
        print(f"  rmsnorm_head_proj rows1: {start.elapsed_time(end) / 200 * 1000:.1f} us per launch")
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
