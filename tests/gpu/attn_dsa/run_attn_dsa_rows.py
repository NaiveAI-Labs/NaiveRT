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


def build(rows: int, phase_timers: bool = False, index_heads: int = 64):
    from torch.utils.cpp_extension import load

    from naivert.kernels.toolchain import scoped_cuda_toolchain

    flavour = f"rows{rows}_ih{index_heads}" + ("_timers" if phase_timers else "")
    head_flags = [f"-DNAIVERT_DSA_INDEX_HEADS={index_heads}"]
    build_dir = ROOT / "tests/build/attn_dsa" / flavour
    build_dir.mkdir(parents=True, exist_ok=True)
    with scoped_cuda_toolchain(CUDA_HOME, suppress_arch_flags=True):
        return load(
            name=f"naivert_attn_dsa_test_{flavour}",
            sources=[str(ROOT / "naivert/body/attn_dsa/attn_dsa_entry.cu"),
                     str(ROOT / "tests/gpu/attn_dsa/attn_dsa_bindings.cpp")],
            extra_include_paths=[str(ROOT), base.CUTLASS_INCLUDE],
            # The bindings compute the workspace layout too: both compilers must see the row count.
            extra_cflags=["-O2", "-std=c++17", f"-DNAIVERT_ATTN_QUERY_ROWS={rows}", *GEOMETRY_FLAGS, *head_flags],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                               "-gencode=arch=compute_103a,code=sm_103a",
                               f"-DNAIVERT_MLA_ROWS_ENABLE_PHASE_TIMERS={int(phase_timers)}",
                               f"-DNAIVERT_ATTN_QUERY_ROWS={rows}", *GEOMETRY_FLAGS, *head_flags],
            build_directory=str(build_dir),
            verbose=os.environ.get("ATTN_DSA_VERBOSE") == "1",
        )


def run_case(ext, layer, kw, T, rows, device, timing, max_seq_len, repeat=0, phases=False):
    import torch
    torch.manual_seed(1000 + T)
    x_rows = (torch.randn(T + rows, base.DIM, device=device) * 1.5).to(torch.bfloat16)
    k_ref, v_ref, ki_ref = base.reference_caches(layer, x_rows)
    per_query = [base.reference_query(layer, x_rows[T + q], T + q) for q in range(rows)]
    k_cache = torch.zeros(max_seq_len, base.HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros(max_seq_len, base.VDIM, dtype=torch.bfloat16, device=device)
    ki_cache = torch.zeros(max_seq_len, base.IDIM, dtype=torch.float8_e4m3fn, device=device)
    ki_scale = torch.zeros(max_seq_len, dtype=torch.float32, device=device)
    if T > 0:
        k_cache[:T] = base.interleave_rope(k_ref[:T], base.HEAD_DIM)
        v_cache[:T] = v_ref[:T]
        ki_codes_plain, ki_scale[:T] = base.quantize_e4m3_rows(base.interleave_rope(ki_ref[:T], base.IDIM),
                                                                    1e-4 if getattr(layer,"training_graph",False) else 1e-30)
        ki_cache[:T] = base.swizzle_e4m3_rows(ki_codes_plain)
    rope = base.rope_table(max_seq_len, device)
    proj_o = torch.zeros(rows, base.HEADS, base.VDIM, dtype=torch.bfloat16, device=device)
    workspace = torch.zeros(ext.workspace_bytes(), dtype=torch.uint8, device=device)
    ext.prepare_maps(workspace, kw["wq"], kw["wk"], kw["wv"], kw["wik"])
    cur_pos = torch.tensor([T], dtype=torch.int32, device=device)

    def launch():
        ext.run_scaled(x_rows[T:T + rows].contiguous(), layer.gamma, kw["wq"], kw["wk"], kw["wv"], kw["wiq"], kw["wiq_scale"],
                kw["wik"], kw["ik_w"], kw["ik_b"], kw["wis"], rope, k_cache, v_cache, ki_cache, ki_scale,
                proj_o, workspace, cur_pos, max_seq_len, 0, getattr(layer, "value_scale", 1.0))

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
    selection_vs_ref = []
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
        ki_row_q, _ = base.quantize_e4m3_rows(base.interleave_rope(ki_ref[pos:pos + 1], base.IDIM))
        ki_row_kernel = base.swizzle_e4m3_rows(ki_cache[pos:pos + 1], torch.tensor([pos]))[0]
        n = int(((ki_row_kernel.view(torch.uint8).int() - ki_row_q[0].view(torch.uint8).int()).abs() > 1).sum())
        worst["ki>1ulp"] = max(worst["ki>1ulp"], n)
        ki_row = base.interleave_rope(ki_ref[pos:pos + 1], base.IDIM)[0]
        ki_row_q, ki_row_s = base.quantize_e4m3_rows(ki_row[None])
        ki_deq = ki_row_kernel.float() * ki_scale[pos]
        ki_ref_deq = ki_row_q[0].float() * ki_row_s[0]
        far = (ki_row_kernel.view(torch.uint8).int() - ki_row_q[0].view(torch.uint8).int()).abs() > 1
        ki_far_rel = float((ki_deq - ki_ref_deq).abs()[far].max()) / max(float(ki_ref_deq.abs().max()), 1e-6) if bool(far.any()) else 0.0
        worst["ki_rel"] = max(worst["ki_rel"], ki_far_rel)  # codes > 1 step apart only; tie flips are counted in ki>1ulp
        worst["ki_quant_rel"] = max(worst.get("ki_quant_rel", 0.0), rel(ki_deq, ki_row))
        sel_ref, logits_ref = base.reference_selection(iq_ref, w_ref, ki_ref, visible)
        if logits_ref is not None:
            sparse_rows += 1
            # kernel-side selection for this row: rows are laid out per query in the workspace
            soff, snbytes = lay["selections"]
            sel = workspace[soff:soff + snbytes].view(torch.int32).view(rows, -1)[q].long()
            selection_ok = selection_ok and bool((sel >= 0).all() and (sel <= pos).all())
            try:
                selection_vs_ref.append(len(set(sel.tolist()) & set(sel_ref.tolist())))
            except Exception:
                selection_vs_ref.append(-1)
            sel_ref = sel.sort().values  # judge the attention core on the kernel's own picks
        out_fp32, out_bf16p = base.reference_attention(q_ref, k_ref, v_ref, sel_ref)
        o = proj_o[q].float()
        scale = max(float(out_fp32.abs().max()), 1e-3)
        worst["out_vs_bf16probs_rel"] = max(worst["out_vs_bf16probs_rel"], float((o - out_bf16p).abs().max()) / scale)
        worst["out_vs_fp32_rel"] = max(worst["out_vs_fp32_rel"], float((o - out_fp32).abs().max()) / scale)
    report.update(worst)
    report["sparse_rows"] = sparse_rows
    report["selection_in_range"] = selection_ok
    report["selection_vs_ref"] = selection_vs_ref
    report["out_nonzero_rows"] = int((proj_o.float().abs().sum((1, 2)) > 0).sum())
    # Check the sweep independently of projection rounding, using the exact
    # published FP8 operands. This catches mixing two 16-head query rows in
    # one 32-column TMEM load, even when attention on the wrong picks looks sane.
    report["sweep_rel"] = 0.0
    report["index_weights_rel"] = 0.0
    report["topk_valid"] = True
    if T + rows > base.TOPK:
        def view(name, dtype):
            off, size = lay[name]
            return workspace[off:off + size].view(dtype)
        iq = view("iq_packed", torch.float8_e4m3fn)[:rows * base.IHEADS * base.IDIM].reshape(-1, base.IDIM)
        iq = base.swizzle_e4m3_rows(iq).float()
        ki = base.swizzle_e4m3_rows(ki_cache[:T + rows]).float()
        ws = view("index_weights_scaled", torch.float32).reshape(rows, base.IHEADS)
        iw = view("index_weights", torch.float32).reshape(rows, base.IHEADS)
        report["index_weights_rel"] = rel(iw, torch.stack([query[2] for query in per_query]))
        logits = view("logits", torch.float32).reshape(rows, -1)
        selections = view("selections", torch.int32).reshape(rows, -1).long()
        for q in range(rows):
            visible = T + q + 1
            scores = ki[:visible] @ iq[q * base.IHEADS:(q + 1) * base.IHEADS].T
            expected = (torch.relu(scores * (base.IDIM * base.IHEADS) ** -0.5) * ws[q]).sum(-1) * ki_scale[:visible]
            report["sweep_rel"] = max(report["sweep_rel"], rel(logits[q, :visible], expected))
            if visible > base.TOPK:
                sel = selections[q]
                valid = bool((sel >= 0).all() and (sel < visible).all() and sel.unique().numel() == base.TOPK)
                if valid:
                    mask = torch.ones(visible, dtype=torch.bool, device=device)
                    mask[sel] = False
                    valid = bool(logits[q, sel].min() >= logits[q, :visible][mask].max())
                report["topk_valid"] = report["topk_valid"] and valid
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
        if os.environ.get("ATTN_DSA_COLD"):
            # Cold-L2 timing: a 256 MB write between launches evicts the weights (untimed).
            flush = torch.empty(256 << 20, dtype=torch.uint8, device=device)
            total = 0.0
            for i in range(n):
                flush.fill_(i & 0xff)
                torch.cuda.synchronize()
                s2, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s2.record(); launch(); e2.record(); torch.cuda.synchronize()
                total += s2.elapsed_time(e2)
            report["us_per_step_cold"] = round(total / n * 1e3, 1)
        if phases:
            off, nbytes = lay["phase_timers"]
            workspace[off:off + nbytes].zero_()
            launch()
            torch.cuda.synchronize()
            report["phases_us"] = base.phase_report(workspace, lay)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--index-heads", type=int, choices=(16, 64), default=64)
    parser.add_argument("--checkpoint", help="FP8 checkpoint directory; selects 16 index heads")
    parser.add_argument("--training-graph", action="store_true", help="BF16 checkpoint IQ projection and training amax floor")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0, help="TP weight shard to test on the visible GPU")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--positions", default="0,5,120,1000,2040,3000")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--phases", action="store_true", help="build with phase timers and report them")
    args = parser.parse_args()
    if args.build_only:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    base.torch = torch
    t0 = time.time()
    if args.checkpoint:
        args.index_heads = 16
    base.configure_index_heads(args.index_heads)
    ext = build(args.rows, args.phases, args.index_heads)
    print(f"built rows{args.rows} extension in {time.time() - t0:.1f}s; workspace {ext.workspace_bytes() / 2**20:.1f} MiB", flush=True)
    if args.build_only:
        return 0
    device = torch.device("cuda")
    if args.checkpoint:
        from naivert.models.naive_n0p5.checkpoint_fp8 import NaiveNCheckpoint
        checkpoint = NaiveNCheckpoint(args.checkpoint)
        layer = base.Layer.from_checkpoint(checkpoint, args.layer, args.rank, device, training_graph=args.training_graph)
    else:
        layer = base.Layer(device, args.seed)
    kw = layer.kernel_weights()
    ok = True
    reports = []
    for T in (int(p) for p in args.positions.split(",")):
        report = run_case(ext, layer, kw, T, args.rows, device, args.timing or args.phases, args.max_seq_len,
                          repeat=args.repeat, phases=args.phases)
        print(report, flush=True)
        # Projections / cache rows: bounded relative to their own scale (the >1 ulp counts
        # include sub-ulp differences of near-zero elements and are informational).
        bad = (max(report["fused_rel"], report["k_rel"], report["v_rel"], report["ki_rel"]) > 0.01 or
               report["out_nonzero_rows"] != args.rows or report["out_vs_bf16probs_rel"] > 0.01 or
               not report["selection_in_range"] or report.get("repeat_bitwise_mismatches", 0) > 0 or
               report["sweep_rel"] > 1e-5 or report["index_weights_rel"] > 0.01 or not report["topk_valid"])
        report["passed"] = not bad
        reports.append(report)
        ok = ok and not bad
        print("  ->", "ok" if not bad else "MISMATCH", flush=True)
    print("ALL OK" if ok else "FAILURES", flush=True)
    if args.out:
        import json
        args.out.write_text(json.dumps(dict(index_heads=args.index_heads, checkpoint=args.checkpoint,
                                            layer=args.layer, rank=args.rank, cases=reports, passed=ok), indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
