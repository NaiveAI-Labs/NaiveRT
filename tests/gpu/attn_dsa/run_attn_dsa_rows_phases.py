#!/usr/bin/env python3
"""attn_dsa with N query rows (the verify build) against the torch reference.

usage: run_attn_dsa_rows.py [--rows 8] [--positions 0,5,120,1000,2040,3000] [--seed 7]
                            [--timing] [--repeat N]

Builds the attn_dsa test extension with NAIVERT_ATTN_QUERY_ROWS=<rows> and, for every
base position T, feeds the rows at positions T..T+rows-1 in one launch: the caches for
positions < T come from the reference, the kernel appends its own rows, and each query
row must attend causally to everything up to its own position (dense below 2048 tokens,
top-k selection above). Reference helpers come from run_attn_dsa.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
CUDA_HOME = os.environ.get("NAIVERT_CUDA_HOME", "/opt/cuda132-min")

from naivert.kernels.toolchain import SM_COUNT  # noqa: E402
from naivert.models.naive_n0p5.kernel_geometry import geometry_cuda_cflags  # noqa: E402

GEOMETRY_FLAGS = list(geometry_cuda_cflags(sm_count=SM_COUNT))

spec = importlib.util.spec_from_file_location("run_attn_dsa", HERE / "run_attn_dsa.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def build(rows: int):
    from torch.utils.cpp_extension import load

    from naivert.kernels.toolchain import scoped_cuda_toolchain

    build_dir = ROOT / "tests/build/attn_dsa" / f"rows{rows}_timers"
    build_dir.mkdir(parents=True, exist_ok=True)
    with scoped_cuda_toolchain(CUDA_HOME, suppress_arch_flags=True):
        return load(
            name=f"naivert_attn_dsa_test_rows{rows}_timers",
            sources=[str(ROOT / "naivert/body/attn_dsa/attn_dsa_entry.cu"),
                     str(ROOT / "tests/gpu/attn_dsa/attn_dsa_bindings.cpp")],
            extra_include_paths=[str(ROOT)],
            # The bindings compute the workspace layout too: both compilers must see the row count.
            extra_cflags=["-O2", "-std=c++17", f"-DNAIVERT_ATTN_QUERY_ROWS={rows}", *GEOMETRY_FLAGS],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                               "-gencode=arch=compute_103a,code=sm_103a",
                               "-DNAIVERT_MLA_ROWS_ENABLE_PHASE_TIMERS=1",
                               f"-DNAIVERT_ATTN_QUERY_ROWS={rows}", *GEOMETRY_FLAGS],
            build_directory=str(build_dir),
            verbose=os.environ.get("ATTN_DSA_VERBOSE") == "1",
        )


def run_case(ext, layer, kw, T, rows, device, timing, max_seq_len, repeat=0):
    import torch
    torch.manual_seed(1000 + T)
    x_rows = (torch.randn(T + rows, base.DIM, device=device) * 1.5).to(torch.bfloat16)
    k_ref, v_ref, ki_ref = base.reference_caches(layer, x_rows)
    per_query = [base.reference_query(layer, x_rows[T + q], T + q) for q in range(rows)]
    k_cache = torch.zeros(max_seq_len, base.HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros(max_seq_len, base.VDIM, dtype=torch.bfloat16, device=device)
    ki_cache = torch.zeros(max_seq_len, base.IDIM, dtype=torch.bfloat16, device=device)
    if T > 0:
        k_cache[:T] = base.interleave_rope(k_ref[:T], base.HEAD_DIM)
        v_cache[:T] = v_ref[:T]
        ki_cache[:T] = base.interleave_rope(ki_ref[:T], base.IDIM)
    rope = base.rope_table(max_seq_len, device)
    proj_o = torch.zeros(rows, base.HEADS, base.VDIM, dtype=torch.bfloat16, device=device)
    workspace = torch.zeros(ext.workspace_bytes(), dtype=torch.uint8, device=device)
    ext.prepare_maps(workspace, kw["wq"], kw["wk"], kw["wv"], kw["wik"])
    cur_pos = torch.tensor([T], dtype=torch.int32, device=device)

    def launch():
        ext.run(x_rows[T:T + rows].contiguous(), layer.gamma, kw["wq"], kw["wk"], kw["wv"], kw["wiq"],
                kw["wik"], kw["ik_w"], kw["ik_b"], kw["wis"], rope, k_cache, v_cache, ki_cache,
                proj_o, workspace, cur_pos, max_seq_len, 0)

    launch()
    torch.cuda.synchronize()
    lay = ext.layout()
    off, nbytes = lay["fused"]
    fused = workspace[off:off + nbytes].view(torch.bfloat16).view(rows, -1)
    report = {"T": T, "rows": rows, "visible_last": T + rows}
    worst = {"fused>1ulp": 0, "k>1ulp": 0, "v>1ulp": 0, "ki>1ulp": 0, "out_vs_bf16probs_rel": 0.0, "out_vs_fp32_rel": 0.0,
             "fused_rel": 0.0, "k_rel": 0.0, "v_rel": 0.0, "ki_rel": 0.0}

    def rel(a, b):
        return float((a.float() - b.float()).abs().max()) / max(float(b.float().abs().max()), 1e-6)
    sparse_rows = 0
    selection_ok = True
    for q in range(rows):
        pos = T + q
        q_ref, iq_ref, w_ref, fused_ref, _ = per_query[q]
        visible = pos + 1
        compared = fused.shape[1] if visible > base.TOPK else lay["fused_bases"][4]
        fused_kernel_order = base.kernel_order_fused(fused_ref)[:compared]
        n, _ = base.bf16_ulp_mismatch(fused[q, :compared], fused_kernel_order)
        worst["fused>1ulp"] = max(worst["fused>1ulp"], n)
        worst["fused_rel"] = max(worst["fused_rel"], rel(fused[q, :compared], fused_kernel_order))
        k_row = base.interleave_rope(k_ref[pos:pos + 1], base.HEAD_DIM)[0]
        n, _ = base.bf16_ulp_mismatch(k_cache[pos], k_row)
        worst["k>1ulp"] = max(worst["k>1ulp"], n)
        worst["k_rel"] = max(worst["k_rel"], rel(k_cache[pos], k_row))
        n, _ = base.bf16_ulp_mismatch(v_cache[pos], v_ref[pos])
        worst["v>1ulp"] = max(worst["v>1ulp"], n)
        worst["v_rel"] = max(worst["v_rel"], rel(v_cache[pos], v_ref[pos]))
        ki_row = base.interleave_rope(ki_ref[pos:pos + 1], base.IDIM)[0]
        n, _ = base.bf16_ulp_mismatch(ki_cache[pos], ki_row)
        worst["ki>1ulp"] = max(worst["ki>1ulp"], n)
        worst["ki_rel"] = max(worst["ki_rel"], rel(ki_cache[pos], ki_row))
        sel_ref, logits_ref = base.reference_selection(iq_ref, w_ref, ki_ref, visible)
        if logits_ref is not None:
            sparse_rows += 1
            # kernel-side selection for this row: rows are laid out per query in the workspace
            soff, snbytes = lay["selections"]
            sel = workspace[soff:soff + snbytes].view(torch.int32).view(rows, -1)[q].long()
            selection_ok = selection_ok and bool((sel >= 0).all() and (sel <= pos).all())
            sel_ref = sel.sort().values  # judge the attention core on the kernel's own picks
        out_fp32, out_bf16p = base.reference_attention(q_ref, k_ref, v_ref, sel_ref)
        o = proj_o[q].float()
        scale = max(float(out_fp32.abs().max()), 1e-3)
        worst["out_vs_bf16probs_rel"] = max(worst["out_vs_bf16probs_rel"], float((o - out_bf16p).abs().max()) / scale)
        worst["out_vs_fp32_rel"] = max(worst["out_vs_fp32_rel"], float((o - out_fp32).abs().max()) / scale)
    report.update(worst)
    report["sparse_rows"] = sparse_rows
    report["selection_in_range"] = selection_ok
    report["out_nonzero_rows"] = int((proj_o.float().abs().sum((1, 2)) > 0).sum())
    if repeat:
        first = proj_o.clone()
        differing = 0
        for _ in range(repeat):
            proj_o.zero_()
            launch()
            torch.cuda.synchronize()
            differing += int(not torch.equal(proj_o, first))
        report["repeat_bitwise_mismatches"] = differing
    if timing:
        for _ in range(3):
            launch()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        n = 20
        start.record()
        for _ in range(n):
            launch()
        end.record()
        torch.cuda.synchronize()
        report["us_per_step"] = round(start.elapsed_time(end) / n * 1e3, 1)
        poff, pn = lay["phase_timers"]
        workspace[poff:poff + pn].zero_()
        launch()
        torch.cuda.synchronize()
        report["phases_us"] = base.phase_report(workspace, lay)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--positions", default="0,5,120,1000,2040,3000")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.build_only:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    base.torch = torch
    t0 = time.time()
    ext = build(args.rows)
    print(f"built rows{args.rows} extension in {time.time() - t0:.1f}s; workspace {ext.workspace_bytes() / 2**20:.1f} MiB", flush=True)
    if args.build_only:
        return 0
    device = torch.device("cuda")
    layer = base.Layer(device, args.seed)
    kw = layer.kernel_weights()
    ok = True
    for T in (int(p) for p in args.positions.split(",")):
        report = run_case(ext, layer, kw, T, args.rows, device, args.timing, args.max_seq_len, repeat=args.repeat)
        print(report, flush=True)
        # Projections / cache rows: bounded relative to their own scale (the >1 ulp counts
        # include sub-ulp differences of near-zero elements and are informational).
        bad = (max(report["fused_rel"], report["k_rel"], report["v_rel"], report["ki_rel"]) > 0.01 or
               report["out_nonzero_rows"] != args.rows or report["out_vs_bf16probs_rel"] > 0.01 or
               not report["selection_in_range"] or report.get("repeat_bitwise_mismatches", 0) > 0)
        ok = ok and not bad
        print("  ->", "ok" if not bad else "MISMATCH", flush=True)
    print("ALL OK" if ok else "FAILURES", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
