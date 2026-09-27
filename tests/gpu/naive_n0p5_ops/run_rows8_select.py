#!/usr/bin/env python3
"""Eight-row router and expert selector checks against PyTorch.
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

from _model import add_model_argument, dequant, select  # noqa: E402
from naivert.kernels.packing import (  # noqa: E402
    dequantize_mxfp8_block32, pack_gate_up, pack_gate_up_mx, quantize_fp8_block,
    quantize_mxfp8_block32)

ROWS = 8


def rmsnorm(x, gamma, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * gamma).to(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    add_model_argument(ap)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trials", type=int, default=3, help="independent random inputs (routing patterns)")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    M = select(args.model, args.weights)
    assert M.weights in ("fp8", "mx") and M.shared_experts == 0, "rows8 body: fp8 / mx pages, no shared expert"
    quantize = quantize_mxfp8_block32 if M.weights == "mx" else quantize_fp8_block
    deq = dequantize_mxfp8_block32 if M.weights == "mx" else dequant
    H, inter = M.hidden, M.expert_inter
    t0 = time.time()
    ext = M.load_extension()
    print(f"{M.name} model_ops loaded in {time.time() - t0:.0f} s; rows {ROWS}", flush=True)
    dev0 = torch.device("cuda:0")
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    gamma = 1.0 + 0.1 * torch.randn(H, generator=gen)
    router = (torch.randn((M.routed_experts, H), generator=gen) * 0.02).to(torch.bfloat16)
    bias = 0.1 * torch.randn(M.routed_experts, generator=gen)
    gate = (torch.randn((M.total_experts, inter, H), generator=gen) * 0.02).to(torch.bfloat16)
    up = (torch.randn((M.total_experts, inter, H), generator=gen) * 0.02).to(torch.bfloat16)
    gq, gs = zip(*(quantize(g) for g in gate))
    uq, us = zip(*(quantize(u) for u in up))
    gq, gs, uq, us = (torch.stack(t) for t in (gq, gs, uq, us))
    if M.weights == "mx":
        table = pack_gate_up_mx(gq, gs, uq, us, hidden=H).view(torch.float8_e4m3fn).to(dev0)
    else:
        table = pack_gate_up(gq, gs, uq, us, hidden=H).to(dev0)
    bias_d, gamma_d, router_d = bias.to(dev0), gamma.to(dev0), router.to(dev0)
    ok = True
    for trial in range(args.trials):
        x = (torch.randn((ROWS, H), generator=gen) * 1.5).to(torch.bfloat16).to(dev0)
        scores = torch.empty((ROWS, M.routed_experts), dtype=torch.float32, device=dev0)
        x_mlp = torch.empty((ROWS, H), dtype=torch.bfloat16, device=dev0)
        ext.launch_rmsnorm_expert_proj(x, gamma_d, router_d, scores, x_mlp, H, M.routed_experts)
        torch.cuda.synchronize(dev0)
        norm_ref = rmsnorm(x, gamma_d, M.eps)
        score_ref = norm_ref.float() @ router_d.float().T
        router_ok = (torch.allclose(x_mlp.float(), norm_ref.float(), atol=0.016, rtol=0.008)
                     and torch.allclose(scores, score_ref, atol=0.02, rtol=0.02))
        ok = ok and router_ok
        print(f"trial {trial}: router rows8 vs torch: {router_ok}", flush=True)
        y8 = torch.full((ROWS, M.slots, 16, 16), float("nan"), dtype=torch.bfloat16, device=dev0)
        p8 = torch.full((ROWS, M.selected_experts), float("nan"), dtype=torch.float32, device=dev0)
        i8 = torch.full((ROWS, M.selected_experts), -1, dtype=torch.int32, device=dev0)
        ext.launch_expert_select_up_gate_silu_rows8(x_mlp, scores, bias_d, table, y8, p8, i8)
        torch.cuda.synchronize(dev0)
        distinct = len(set(i8.flatten().tolist()))
        nan8 = int(torch.isnan(y8.float()).sum())
        ref_ok, worst = True, 0.0
        for r in range(ROWS):
            sig = torch.sigmoid(scores[r].cpu())
            route_score = sig + bias
            if M.route_kept_groups < 8:
                group_score = route_score.view(8, -1).topk(2, dim=-1).values.sum(-1)
                kept = torch.zeros(8, dtype=torch.bool)
                kept[group_score.topk(M.route_kept_groups).indices] = True
                route_score = torch.where(kept.repeat_interleave(M.routed_experts // 8), route_score,
                                          torch.full_like(route_score, -float("inf")))
            ref_idx = set(torch.topk(route_score, M.selected_experts).indices.tolist())
            got = i8[r].tolist()
            set_ok = set(got) == ref_idx and len(set(got)) == M.selected_experts
            ref_p = sig[i8[r].cpu().long()] / sig[i8[r].cpu().long()].sum() * M.route_scale
            p_ok = float((p8[r].cpu() - ref_p).abs().max()) < 1e-5
            x8 = x_mlp[r].cpu().to(torch.float8_e4m3fn).float()
            g = torch.stack([deq(gq[e], gs[e]) @ x8 for e in got])
            u = torch.stack([deq(uq[e], us[e]) @ x8 for e in got])
            y_ref = torch.nn.functional.silu(g) * u
            y = y8[r].reshape(M.slots, inter).float().cpu()
            rel = float((y - y_ref).abs().max()) / max(float(y_ref.abs().max()), 1e-6)
            worst = max(worst, rel)
            ref_ok = ref_ok and set_ok and p_ok and rel <= 0.02
        good = nan8 == 0 and ref_ok
        ok = ok and good
        print(f"trial {trial}: distinct experts {distinct}/64; nan {nan8}; "
              f"vs torch: {ref_ok}, worst rel {worst:.4f} -> {'ok' if good else 'MISMATCH'}", flush=True)
        if args.timing:
            def loop(fn):
                for _ in range(10):
                    fn()
                torch.cuda.synchronize(dev0)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.reps):
                    fn()
                end.record()
                torch.cuda.synchronize(dev0)
                return start.elapsed_time(end) * 1e3 / args.reps
            def rows8():
                ext.launch_expert_select_up_gate_silu_rows8(x_mlp, scores, bias_d, table, y8, p8, i8)
            def router8():
                ext.launch_rmsnorm_expert_proj(x, gamma_d, router_d, scores, x_mlp, H, M.routed_experts)
            print(f"  timing (us per layer, L2-warm): expert_select rows8 {loop(rows8):.2f}; "
                  f"router rows8 {loop(router8):.2f}", flush=True)
    print("ALL OK" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
