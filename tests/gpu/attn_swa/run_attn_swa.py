#!/usr/bin/env python3
"""Standalone check of the Naive N attn_swa body against a torch reference.

One TP8 rank of a Naive N0.5 sliding-window layer at real shapes (hidden 4096,
8 local heads x 192 with 64 rope dims at theta 1e4, kv head 192/128, window 128,
one sink logit per head) with random bf16 weights. For every requested base position T
the reference fills the ring caches for the positions the window can still see, then the
kernel processes the query rows T..T+rows-1 in one launch and we compare the fused
projections, the cache rows it appended, and each row's attention output (every row
attends causally up to its own position, including the rows appended before it).

usage: run_attn_swa.py [--rows 1] [--positions 0,1,5,127,128,129,300,2047,4095] [--seed 7]
                       [--build-only] [--timing] [--repeat N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
CUDA_HOME = os.environ.get("NAIVERT_CUDA_HOME", "/opt/cuda132-min")

from naivert.kernels.toolchain import SM_COUNT  # noqa: E402
from naivert.models.naive_n0p5.kernel_geometry import geometry_cuda_cflags  # noqa: E402

GEOMETRY_FLAGS = list(geometry_cuda_cflags(sm_count=SM_COUNT))

DIM, HEADS, HEAD_DIM, ROPE, VDIM, WINDOW, WINDOW_ROWS = 4096, 8, 192, 64, 128, 128, 136
THETA, EPS, VSCALE = 1.0e4, 1.0e-5, 0.707
Q_ROWS = HEADS * HEAD_DIM


def build(rows: int = 1):
    from torch.utils.cpp_extension import load

    from naivert.kernels.toolchain import scoped_cuda_toolchain
    from naivert.kernels.build import CUTLASS_INCLUDE_ROOT

    build_dir = ROOT / "tests/build/attn_swa" / f"rows{rows}"
    build_dir.mkdir(parents=True, exist_ok=True)
    with scoped_cuda_toolchain(CUDA_HOME, suppress_arch_flags=True):
        return load(
            name=f"naivert_attn_swa_test_rows{rows}",
            sources=[str(ROOT / "naivert/body/attn_swa/attn_swa_entry.cu"),
                     str(ROOT / "tests/gpu/attn_swa/attn_swa_bindings.cpp")],
            extra_include_paths=[str(ROOT), str(CUTLASS_INCLUDE_ROOT)],
            # The bindings compute the workspace layout too: both compilers must see the row count.
            extra_cflags=["-O2", "-std=c++17", f"-DNAIVERT_ATTN_QUERY_ROWS={rows}", *GEOMETRY_FLAGS],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                               "-gencode=arch=compute_103a,code=sm_103a",
                               "-DNAIVERT_MLA_ROWS_ENABLE_PHASE_TIMERS=0",
                               f"-DNAIVERT_ATTN_QUERY_ROWS={rows}", *GEOMETRY_FLAGS],
            build_directory=str(build_dir),
            verbose=os.environ.get("ATTN_SWA_VERBOSE") == "1",
        )


# ------------------------------------------------------------ rope helpers
def rope_permutation():
    """HF rotate_half order (i, i+32) -> interleaved pairs (2i, 2i+1)."""
    import torch
    perm = torch.empty(ROPE, dtype=torch.long)
    perm[0::2] = torch.arange(ROPE // 2)
    perm[1::2] = torch.arange(ROPE // 2) + ROPE // 2
    return perm


def interleave_rope(x, head_dim):
    shape = x.shape
    x = x.reshape(-1, head_dim).clone()
    x[:, :ROPE] = x[:, rope_permutation().to(x.device)]
    return x.reshape(shape)


def interleave_rope_rows(w, head_dim):
    return interleave_rope(w.transpose(0, 1), head_dim).transpose(0, 1).contiguous()


def rope_table(max_pos, device):
    import torch
    freqs = 1.0 / (THETA ** (torch.arange(0, ROPE, 2, dtype=torch.float32, device=device) / ROPE))
    angles = torch.outer(torch.arange(max_pos, dtype=torch.float32, device=device), freqs)
    return torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).reshape(max_pos, ROPE).contiguous()


def hf_rope(x, positions):
    """rotate_half rope on the first 64 dims (given as x [..., P, 64], fp32), positions [P]."""
    import torch
    inv = 1.0 / (THETA ** (torch.arange(0, ROPE, 2, dtype=torch.float32, device=x.device) / ROPE))
    ang = positions.float()[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)
    sin = torch.cat([ang.sin(), ang.sin()], -1)
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    x1, x2 = x[..., :ROPE // 2], x[..., ROPE // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


# ---------------------------------------------------------------- reference
class Layer:
    def __init__(self, device, seed):
        import torch
        g = torch.Generator(device="cpu").manual_seed(seed)

        def w(rows, scale=0.02):
            return (torch.randn(rows, DIM, generator=g) * scale).to(torch.bfloat16).to(device)

        self.wq, self.wk = w(Q_ROWS), w(HEAD_DIM)
        self.wv = (w(VDIM).float() * VSCALE).to(torch.bfloat16)  # value scale folded
        self.gamma = (1.0 + 0.1 * torch.randn(DIM, generator=g)).to(device)
        self.sink = (0.5 * torch.randn(HEADS, generator=g)).to(device)

    def kernel_weights(self):
        return dict(wq=interleave_rope_rows(self.wq, HEAD_DIM), wk=interleave_rope_rows(self.wk, HEAD_DIM),
                    wv=self.wv.contiguous())


def kernel_order_fused(fused_ref):
    q, k, v = fused_ref[:Q_ROWS], fused_ref[Q_ROWS:Q_ROWS + HEAD_DIM], fused_ref[Q_ROWS + HEAD_DIM:]
    return torch.cat([interleave_rope(q, HEAD_DIM), interleave_rope(k, HEAD_DIM), v])


def rmsnorm(x, gamma):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * gamma).to(torch.bfloat16)


def project(xn, w):
    return (xn.float() @ w.float().T).to(torch.bfloat16)


def reference_caches(layer, x_rows):
    """k [P,192] (rope applied, HF order) and v [P,128] for x_rows [P, 4096]."""
    P = x_rows.shape[0]
    pos = torch.arange(P, device=x_rows.device)
    xn = rmsnorm(x_rows, layer.gamma)
    k = project(xn, layer.wk).float()
    k[:, :ROPE] = hf_rope(k[:, :ROPE], pos)
    return k.to(torch.bfloat16), project(xn, layer.wv)


def reference_query(layer, x, T):
    xn = rmsnorm(x[None], layer.gamma)
    q_raw, k_raw, v_raw = project(xn, layer.wq)[0], project(xn, layer.wk)[0], project(xn, layer.wv)[0]
    q = q_raw.view(HEADS, HEAD_DIM).float()
    q[:, :ROPE] = hf_rope(q[:, :ROPE], torch.tensor([T], device=x.device))
    return q.to(torch.bfloat16), torch.cat([q_raw, k_raw, v_raw])


def reference_attention(q, k, v, sink, first, last):
    """Softmax attention of q [8,192] over rows first..last of k / v with the sink logit."""
    ks, vs = k[first:last + 1].float(), v[first:last + 1].float()
    scores = (q.float() @ ks.T) * HEAD_DIM ** -0.5                      # [8, n]
    logits = torch.cat([scores, sink[:, None]], -1)
    probs = torch.softmax(logits, -1)[:, :-1]
    out_fp32 = probs @ vs
    probs_bf16 = probs.to(torch.bfloat16).float()
    return out_fp32, probs_bf16 @ vs


def bf16_ulp_mismatch(a, b):
    af, bf = a.float(), b.float()
    ulp = torch.maximum(af.abs(), bf.abs()).clamp_min(1e-30)
    ulp = 2.0 ** (torch.floor(torch.log2(ulp)) - 7)
    return int(((af - bf).abs() > ulp * 1.001).sum()), float((af - bf).abs().max())


def run_case(ext, layer, kw, T, rows, device, timing, max_seq_len, repeat=0, dump=None):
    import torch
    torch.manual_seed(1000 + T)
    x_rows = (torch.randn(T + rows, DIM, device=device) * 1.5).to(torch.bfloat16)
    k_ref, v_ref = reference_caches(layer, x_rows)
    # Ring caches: every position the window of any row can still see, at its ring row.
    ring_rows = ext.window_rows()
    k_cache = torch.zeros(ring_rows, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros(ring_rows, VDIM, dtype=torch.bfloat16, device=device)
    for j in range(max(0, T - WINDOW), T):
        k_cache[j % WINDOW_ROWS] = interleave_rope(k_ref[j:j + 1], HEAD_DIM)[0]
        v_cache[j % WINDOW_ROWS] = v_ref[j]
    rope = rope_table(max_seq_len, device)
    proj_o = torch.zeros(rows, HEADS, VDIM, dtype=torch.bfloat16, device=device)
    workspace = torch.zeros(ext.workspace_bytes(), dtype=torch.uint8, device=device)
    ext.prepare_maps(workspace, kw["wq"], kw["wk"], kw["wv"])
    cur_pos = torch.tensor([T], dtype=torch.int32, device=device)

    def launch():
        ext.run(x_rows[T:T + rows].contiguous(), layer.gamma, kw["wq"], kw["wk"], kw["wv"], rope,
                layer.sink, k_cache, v_cache, proj_o, workspace, cur_pos, max_seq_len, 0)

    launch()
    torch.cuda.synchronize()
    lay = ext.layout()
    off, nbytes = lay["fused"]
    fused = workspace[off:off + nbytes].view(torch.bfloat16).view(rows, -1)
    report = {"T": T, "rows": rows, "fused>1ulp": 0, "k>1ulp": 0, "v>1ulp": 0, "proj_rel": 0.0,
              "out_vs_bf16probs_rel": 0.0, "out_vs_fp32_rel": 0.0}

    def rel(a, b):
        return float((a.float() - b.float()).abs().max()) / max(float(b.float().abs().max()), 1e-6)

    for q in range(rows):
        pos = T + q
        q_ref, fused_ref = reference_query(layer, x_rows[pos], pos)
        visible = min(pos + 1, WINDOW)
        out_fp32, out_bf16p = reference_attention(q_ref, k_ref, v_ref, layer.sink, pos + 1 - visible, pos)
        fused_ko = kernel_order_fused(fused_ref)
        k_row = interleave_rope(k_ref[pos:pos + 1], HEAD_DIM)[0]
        report["fused>1ulp"] = max(report["fused>1ulp"], bf16_ulp_mismatch(fused[q], fused_ko)[0])
        report["k>1ulp"] = max(report["k>1ulp"], bf16_ulp_mismatch(k_cache[pos % WINDOW_ROWS], k_row)[0])
        report["v>1ulp"] = max(report["v>1ulp"], bf16_ulp_mismatch(v_cache[pos % WINDOW_ROWS], v_ref[pos])[0])
        report["proj_rel"] = max(report["proj_rel"], rel(fused[q], fused_ko), rel(k_cache[pos % WINDOW_ROWS], k_row),
                                 rel(v_cache[pos % WINDOW_ROWS], v_ref[pos]))
        o = proj_o[q].float()
        scale = max(float(out_fp32.abs().max()), 1e-3)
        report["out_vs_bf16probs_rel"] = max(report["out_vs_bf16probs_rel"], float((o - out_bf16p).abs().max()) / scale)
        report["out_vs_fp32_rel"] = max(report["out_vs_fp32_rel"], float((o - out_fp32).abs().max()) / scale)
    report["out_nonzero_rows"] = int((proj_o.float().abs().sum((1, 2)) > 0).sum())
    if dump:
        torch.save({"fused": fused.cpu(), "proj_o": proj_o.cpu(), "k_cache": k_cache.cpu(), "v_cache": v_cache.cpu()},
                   f"{dump}/rows{rows}_T{T}.pt")
    if repeat:
        first_out = proj_o.clone()
        differing = 0
        for _ in range(repeat):
            proj_o.zero_()
            launch()
            torch.cuda.synchronize()
            differing += int(not torch.equal(proj_o, first_out))
        report["repeat_runs"], report["repeat_bitwise_mismatches"] = repeat, differing
    if timing:
        for _ in range(3):
            launch()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        n = 50
        start.record()
        for _ in range(n):
            launch()
        end.record()
        torch.cuda.synchronize()
        report["us_per_step"] = round(start.elapsed_time(end) / n * 1e3, 1)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--positions", default="0,1,5,127,128,129,300,2047,4095")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--dump", default="", help="directory: save fused / proj_o / caches per position (bitwise A/B across trees)")
    args = parser.parse_args()
    if args.build_only:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    global torch
    import torch
    t0 = time.time()
    ext = build(args.rows)
    print(f"built rows{args.rows} extension in {time.time() - t0:.1f}s; workspace {ext.workspace_bytes()} bytes, "
          f"window rows {ext.window_rows()}", flush=True)
    if args.build_only:
        return 0
    device = torch.device("cuda")
    layer = Layer(device, args.seed)
    kw = layer.kernel_weights()
    ok = True
    for T in (int(p) for p in args.positions.split(",")):
        report = run_case(ext, layer, kw, T, args.rows, device, args.timing, args.max_seq_len, repeat=args.repeat, dump=args.dump or None)
        print(report, flush=True)
        # Projections / cache rows: bounded relative to their own scale (the >1 ulp counts
        # include sub-ulp differences of near-zero elements and are informational).
        bad = (report["proj_rel"] > 0.01 or
               report["out_nonzero_rows"] != args.rows or report["out_vs_bf16probs_rel"] > 0.01 or
               report.get("repeat_bitwise_mismatches", 0) > 0)
        ok = ok and not bad
        print("  ->", "ok" if not bad else "MISMATCH", flush=True)
    print("ALL OK" if ok else "FAILURES", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
