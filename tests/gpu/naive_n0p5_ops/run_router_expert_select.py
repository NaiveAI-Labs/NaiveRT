#!/usr/bin/env python3
"""Naive N per-kernel check: rmsnorm_expert_proj (router) + expert_select_up_gate_silu rows1
on one GPU at Naive N shapes (hidden 4096, 256 experts, top-8, no shared expert, inter 256
per rank), built through naivert.kernels.naive_n0p5.model_ops, against a torch reference.

usage: run_router_expert_select.py [--seed 7] [--trials 3] [--timing]

Stage 1 (router): x_norm = bf16(x * gamma * rsqrt(mean(x^2) + 1e-5)), scores = x_norm . W_r
  (bf16 router weights, f32 accumulation).  Checks: x_mlp within one bf16 ulp of the torch
  x_norm, scores within 1e-4 of the f32 torch dot products.
Stage 2 (expert_select, fed the kernel's own scores): sigmoid + bias top-8, probs =
  sigmoid / sum * route_scale (1.0), y[slot] = SiLU(gate) * up for the 8 selected experts
  from fp8 block-128 gate/up weights (packed with naivert.kernels.packing).  Checks:
  the top-8 set and the probs (1e-5), and max |y - y_ref| below 2 % of max |y_ref|
  (fp8 products are exact, the f32 sum order and the bf16 store differ).
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

from naivert.kernels.packing import (  # noqa: E402
    dequantize_mxfp8_block32, pack_gate_up, pack_gate_up_mx,
    quantize_fp8_block, quantize_mxfp8_block32)
from naivert.models.naive_n0p5.model import NAIVE_N0P5_HYBRID, NAIVE_N0P5_GEOMETRY  # noqa: E402

HIDDEN = NAIVE_N0P5_GEOMETRY.hidden_size            # 4096
EXPERTS = NAIVE_N0P5_GEOMETRY.routed_expert_count    # 256
TOP_K = NAIVE_N0P5_GEOMETRY.selected_expert_count    # 8
SLOTS = TOP_K + (NAIVE_N0P5_GEOMETRY.total_expert_count - EXPERTS)  # 8 (no shared expert)
INTER = 256                                    # per rank
EPS = float(NAIVE_N0P5_HYBRID.norm_eps)          # 1e-5
ROUTE_SCALE = float(NAIVE_N0P5_HYBRID.routing.route_scale)  # 1.0


def bf16_ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()


def stage_router(ext, x, gamma, router, device):
    scores = torch.empty((1, EXPERTS), dtype=torch.float32, device=device)
    x_mlp = torch.empty((1, HIDDEN), dtype=torch.bfloat16, device=device)
    ext.launch_rmsnorm_expert_proj(x, gamma, router, scores, x_mlp, HIDDEN, EXPERTS)
    torch.cuda.synchronize(device)
    xf = x.float()
    inv = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + EPS)
    x_norm_ref = (xf * gamma * inv).to(torch.bfloat16)
    scores_ref = x_norm_ref.float() @ router.float().T
    ulps = bf16_ulp_diff(x_mlp, x_norm_ref)
    ok = int((ulps > 1).sum()) == 0 and float((scores - scores_ref).abs().max()) < 1e-4
    print(f"  router: x_mlp >1ulp={int((ulps > 1).sum())} (max {int(ulps.max())}), "
          f"scores max|diff|={float((scores - scores_ref).abs().max()):.2e} -> {'ok' if ok else 'MISMATCH'}")
    return ok, scores, x_mlp


def stage_expert_select(ext, x_mlp, scores, bias, packed, gate_q, gate_s, up_q, up_s, device, dequant_fn=None):
    up_gate = torch.full((1, SLOTS, 16, 16), float("nan"), dtype=torch.bfloat16, device=device)
    probs = torch.full((1, TOP_K), float("nan"), dtype=torch.float32, device=device)
    indices = torch.full((1, TOP_K), -1, dtype=torch.int32, device=device)
    ext.launch_expert_select_up_gate_silu_rows1(x_mlp, scores, bias, packed, up_gate, probs, indices)
    torch.cuda.synchronize(device)

    sig = torch.sigmoid(scores[0])
    select = sig + bias
    ref_idx = torch.topk(select, TOP_K).indices
    ref_set = set(ref_idx.tolist())
    got = indices[0].tolist()
    set_ok = set(got) == ref_set and len(set(got)) == TOP_K
    ref_probs = sig[indices[0].long()] / sig[indices[0].long()].sum() * ROUTE_SCALE
    probs_ok = bool(torch.isfinite(probs).all()) and float((probs[0] - ref_probs).abs().max()) < 1e-5

    # y[slot] for the kernel's slot order: dequantised fp8 weights, fp8 activation.
    x8 = x_mlp[0].to(torch.float8_e4m3fn).float()
    sel = indices[0].long()
    def dequant(q, s):
        w = q[sel].float()                       # [8, 256, 4096]
        sc = s[sel].float()                      # [8, 2, 32]
        sc = sc.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
        return w * sc
    if dequant_fn is not None:                   # MX8: (codes, block scales) of the selected experts
        dequant = lambda q, s: dequant_fn(q[sel], s[sel])  # noqa: E731
    gate = dequant(gate_q, gate_s) @ x8          # [8, 256]
    up = dequant(up_q, up_s) @ x8
    y_ref = torch.nn.functional.silu(gate) * up
    y = up_gate[0].reshape(SLOTS, INTER).float()
    scale = float(y_ref.abs().max())
    max_abs = float((y - y_ref).abs().max())
    y_ok = bool(torch.isfinite(y).all()) and max_abs <= 0.02 * scale
    print(f"  expert_select: top8 set {'ok' if set_ok else 'MISMATCH'} (kernel order {got}), "
          f"probs max|diff|={float((probs[0] - ref_probs).abs().max()):.1e} {'ok' if probs_ok else 'MISMATCH'}, "
          f"y max|diff|={max_abs:.3e} / scale {scale:.3e} = {max_abs / scale:.4f} -> {'ok' if y_ok else 'MISMATCH'}")
    return set_ok and probs_ok and y_ok


def timing(ext, x, gamma, router, scores, x_mlp, bias, packed, device, iters=200):
    up_gate = torch.empty((1, SLOTS, 16, 16), dtype=torch.bfloat16, device=device)
    probs = torch.empty((1, TOP_K), dtype=torch.float32, device=device)
    indices = torch.empty((1, TOP_K), dtype=torch.int32, device=device)
    for name, fn in (
        ("rmsnorm_expert_proj", lambda: ext.launch_rmsnorm_expert_proj(x, gamma, router, scores, x_mlp, HIDDEN, EXPERTS)),
        ("expert_select_rows1", lambda: ext.launch_expert_select_up_gate_silu_rows1(x_mlp, scores, bias, packed, up_gate, probs, indices)),
    ):
        for _ in range(10):
            fn()
        torch.cuda.synchronize(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        print(f"  {name}: {start.elapsed_time(end) / iters * 1000:.1f} us per launch")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--weights", choices=("fp8", "mx"), default="fp8", help="expert weight format build")
    args = ap.parse_args()
    from naivert.kernels.naive_n0p5.model_ops import load_extension
    t0 = time.time()
    ext = load_extension(weights=args.weights)
    print(f"Naive N model_ops extension loaded in {time.time() - t0:.0f} s")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    gate = (torch.randn((EXPERTS, INTER, HIDDEN), device=device, generator=gen) * 0.02).to(torch.bfloat16)
    up = (torch.randn((EXPERTS, INTER, HIDDEN), device=device, generator=gen) * 0.02).to(torch.bfloat16)
    dequant_fn = None
    if args.weights == "mx":
        gate_q, gate_s = zip(*(quantize_mxfp8_block32(g) for g in gate))
        up_q, up_s = zip(*(quantize_mxfp8_block32(u) for u in up))
        gate_q, gate_s, up_q, up_s = (torch.stack(t).to(device) for t in (gate_q, gate_s, up_q, up_s))
        packed = pack_gate_up_mx(gate_q, gate_s, up_q, up_s, hidden=HIDDEN).view(torch.float8_e4m3fn)
        dequant_fn = dequantize_mxfp8_block32
    else:
        gate_q, gate_s = zip(*(quantize_fp8_block(g) for g in gate))
        up_q, up_s = zip(*(quantize_fp8_block(u) for u in up))
        gate_q, gate_s, up_q, up_s = (torch.stack(t).to(device) for t in (gate_q, gate_s, up_q, up_s))
        packed = pack_gate_up(gate_q, gate_s, up_q, up_s, hidden=HIDDEN)
    print(f"weights: packed {tuple(packed.shape)} {packed.dtype}, scales {tuple(gate_s.shape)}")
    router = (torch.randn((EXPERTS, HIDDEN), device=device, generator=gen) * 0.02).to(torch.bfloat16)
    bias = torch.randn((EXPERTS,), device=device, generator=gen) * 0.1
    gamma = 1.0 + 0.1 * torch.randn((HIDDEN,), device=device, generator=gen)
    ok = True
    for trial in range(args.trials):
        x = (torch.randn((1, HIDDEN), device=device, generator=gen) * (1.0 + trial)).to(torch.bfloat16)
        print(f"trial {trial}: |x| scale {1.0 + trial}")
        router_ok, scores, x_mlp = stage_router(ext, x, gamma, router, device)
        select_ok = stage_expert_select(ext, x_mlp, scores, bias, packed, gate_q, gate_s, up_q, up_s, device, dequant_fn)
        ok = ok and router_ok and select_ok
    if args.timing:
        timing(ext, x, gamma, router, scores, x_mlp, bias, packed, device)
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
