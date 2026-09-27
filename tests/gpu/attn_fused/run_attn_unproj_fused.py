#!/usr/bin/env python3
"""Attention with the fused o-projection + TP8 all-reduce tail vs attention followed by
the standalone unproj kernel, on 8 GPUs, for both attention kinds and 1 / 8 query rows.

usage: run_attn_unproj_fused.py [--rows 1,8] [--kinds swa,dsa] [--reps 20] [--seed 7]

Per rank: its own head shards (wq/wk/wv, indexer for dsa), its o_proj shard (fp8 block-128,
pack_unproj), the replicated residual row(s). Reference = ext.run (proj_o) then
launch_unproj_allreduce_replay (one launch per row).
Fused = ext.run_fused with the same weights, mailboxes and epoch. Checks: fused output equals
the reference bit for bit (1 row: same summation recipe) or within one bf16 ulp (8 rows), and is
identical on every rank. Timing: per-rank graphs of N repeated launches (epoch advanced in
between) replayed in lockstep: attention + unproj vs fused.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_HOME", "/opt/cuda132-min")

import torch  # noqa: E402

from naivert.kernels.packing import pack_unproj, quantize_fp8_block  # noqa: E402
from naivert.models.naive_n0p5.kernel_geometry import ALLREDUCE_GEOMETRY  # noqa: E402
from naivert.models.naive_n0p5.model import NAIVE_N0P5_HYBRID, NAIVE_N0P5_GEOMETRY  # noqa: E402

RANKS, HIDDEN = NAIVE_N0P5_GEOMETRY.rank_count, NAIVE_N0P5_GEOMETRY.hidden_size
FULL = NAIVE_N0P5_HYBRID.attention_layouts["full_dsa"]
SWA = NAIVE_N0P5_HYBRID.attention_layouts["sliding_window"]
HEADS, HEAD_DIM, VDIM, ROPE = FULL.local_heads(RANKS), FULL.head_dim, FULL.v_head_dim, FULL.rope_dim
IHEADS, IDIM = NAIVE_N0P5_HYBRID.indexer.head_count, NAIVE_N0P5_HYBRID.indexer.head_dim
K_UNPROJ = HEADS * VDIM
SLAB = 0x70000
MAX_SEQ = 4096


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rope_table(max_pos, theta, device):
    freqs = 1.0 / (theta ** (torch.arange(0, ROPE, 2, dtype=torch.float32, device=device) / ROPE))
    angles = torch.outer(torch.arange(max_pos, dtype=torch.float32, device=device), freqs)
    return torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).reshape(max_pos, ROPE).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,8")
    ap.add_argument("--kinds", default="swa,dsa")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from naivert.kernels.naive_n0p5.model_ops import load_extension
    ops = load_extension()
    ops.enable_tp8_peer_access()
    swa_mod = load_module(ROOT / "tests/gpu/attn_swa/run_attn_swa.py")
    dsa_mod = load_module(ROOT / "tests/gpu/attn_dsa/run_attn_dsa.py")
    dsa_rows = load_module(ROOT / "tests/gpu/attn_dsa/run_attn_dsa_rows.py")
    devices = [torch.device(f"cuda:{r}") for r in range(RANKS)]
    streams = [torch.cuda.Stream(device=d) for d in devices]
    mailboxes = [torch.zeros(16 * SLAB, dtype=torch.uint8, device=d) for d in devices]
    results, ok = {}, True

    def sync_all():
        for d in devices:
            torch.cuda.synchronize(d)

    def lockstep(fns):
        for r, d in enumerate(devices):
            with torch.cuda.device(d), torch.cuda.stream(streams[r]):
                fns[r]()
        sync_all()

    for kind in args.kinds.split(","):
        for rows in (int(x) for x in args.rows.split(",")):
            ext = (swa_mod.build(rows) if kind == "swa" else (dsa_mod.build(False, IHEADS) if rows == 1 else dsa_rows.build(rows, index_heads=IHEADS)))
            gen = torch.Generator().manual_seed(args.seed)
            residual = (torch.randn((rows, HIDDEN), generator=gen) * 1.5).to(torch.bfloat16)
            gamma = 1.0 + 0.1 * torch.randn(HIDDEN, generator=gen)
            per = []
            for r, d in enumerate(devices):
                g = torch.Generator().manual_seed(args.seed * 1000 + r)
                st = {"residual": residual.to(d), "gamma": gamma.to(d),
                      "wq": (torch.randn((HEADS * HEAD_DIM, HIDDEN), generator=g) * 0.02).to(torch.bfloat16).to(d),
                      "wk": (torch.randn((HEAD_DIM, HIDDEN), generator=g) * 0.02).to(torch.bfloat16).to(d),
                      "wv": (torch.randn((VDIM, HIDDEN), generator=g) * 0.02 * float(FULL.value_scale)).to(torch.bfloat16).to(d),
                      "proj_o": torch.zeros((rows, HEADS, VDIM), dtype=torch.bfloat16, device=d),
                      "ws": torch.zeros(ext.workspace_bytes(), dtype=torch.uint8, device=d),
                      "pos": torch.full((1,), 64, dtype=torch.int32, device=d),
                      "epoch": torch.ones(1, dtype=torch.int64, device=d),
                      "hid_ref": torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=d),
                      "hid_fused": torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=d)}
                oq, os_ = quantize_fp8_block((torch.randn((HIDDEN, K_UNPROJ), generator=g) * 0.02).to(torch.bfloat16))
                st["o_packed"], st["o_scale"] = (t.to(d) for t in pack_unproj(oq, os_, hidden=HIDDEN, outputs_per_tile=ALLREDUCE_GEOMETRY.unproj_outputs_per_tile, input_slots=4))
                if kind == "swa":
                    ext.prepare_maps(st["ws"], st["wq"], st["wk"], st["wv"])
                    st["rope"] = rope_table(MAX_SEQ, SWA.rope_theta, d)
                    st["sink"] = (0.5 * torch.randn(HEADS, generator=g)).to(d)
                    st["k_cache"] = torch.randn((ext.window_rows(), HEAD_DIM), generator=g).to(torch.bfloat16).to(d)
                    st["v_cache"] = torch.randn((ext.window_rows(), VDIM), generator=g).to(torch.bfloat16).to(d)
                else:
                    st["rope"] = rope_table(MAX_SEQ, FULL.rope_theta, d)
                    st["wiq"] = (torch.randn((IHEADS * IDIM, HIDDEN), generator=g) * 0.02).to(torch.bfloat16).to(d)
                    st["wik"] = (torch.randn((IDIM, HIDDEN), generator=g) * 0.02).to(torch.bfloat16).to(d)
                    ext.prepare_maps(st["ws"], st["wq"], st["wk"], st["wv"], st["wik"])
                    st["ik_w"] = (1.0 + 0.1 * torch.randn(IDIM, generator=g)).to(d)
                    st["ik_b"] = (0.1 * torch.randn(IDIM, generator=g)).to(d)
                    st["wis"] = (torch.randn((IHEADS, HIDDEN), generator=g) * 0.05).to(torch.bfloat16).to(d)
                    st["k_cache"] = torch.randn((MAX_SEQ, HEAD_DIM), generator=g).to(torch.bfloat16).to(d)
                    st["v_cache"] = torch.randn((MAX_SEQ, VDIM), generator=g).to(torch.bfloat16).to(d)
                    st["ki_cache"] = torch.randn((MAX_SEQ, IDIM), generator=g).to(torch.bfloat16).to(d)
                per.append(st)

            def attention_args(st):
                if kind == "swa":
                    return (st["residual"], st["gamma"], st["wq"], st["wk"], st["wv"], st["rope"], st["sink"],
                            st["k_cache"], st["v_cache"], st["proj_o"], st["ws"], st["pos"], MAX_SEQ, 0)
                return (st["residual"], st["gamma"], st["wq"], st["wk"], st["wv"], st["wiq"], st["wik"], st["ik_w"],
                        st["ik_b"], st["wis"], st["rope"], st["k_cache"], st["v_cache"], st["ki_cache"], st["proj_o"],
                        st["ws"], st["pos"], MAX_SEQ, 0)

            def reference(r):
                st = per[r]
                ext.run(*attention_args(st))
                if rows == 1:
                    ops.launch_unproj_allreduce_replay(st["proj_o"].view(1, K_UNPROJ), st["o_packed"], st["o_scale"],
                                                       st["residual"], st["hid_ref"], mailboxes, r, 0x22,
                                                       st["epoch"].view(torch.uint8))
                else:
                    for row in range(rows):
                        s = slice(row, row + 1)
                        row_mailboxes = [mb[row * SLAB:(row + 1) * SLAB] for mb in mailboxes]
                        ops.launch_unproj_allreduce_replay(st["proj_o"][s].reshape(1, K_UNPROJ), st["o_packed"], st["o_scale"],
                                                           st["residual"][s], st["hid_ref"][s], row_mailboxes, r, 0x22,
                                                           st["epoch"].view(torch.uint8))

            def fused(r):
                st = per[r]
                ext.run_fused(*attention_args(st), st["o_packed"], st["o_scale"], st["hid_fused"], mailboxes,
                              st["epoch"], r, 0x33)

            def advance(r):
                per[r]["epoch"] += 1

            # Correctness (eager, lockstep; the epoch advances between the two paths).
            lockstep([lambda r=r: reference(r) for r in range(RANKS)])
            lockstep([lambda r=r: advance(r) for r in range(RANKS)])
            lockstep([lambda r=r: fused(r) for r in range(RANKS)])
            lockstep([lambda r=r: advance(r) for r in range(RANKS)])
            refs = [st["hid_ref"].cpu() for st in per]
            fus = [st["hid_fused"].cpu() for st in per]
            ulps = max(int((f.view(torch.int16).int() - rf.view(torch.int16).int()).abs().max()) for f, rf in zip(fus, refs))
            cross = max(int((f.view(torch.int16).int() - fus[0].view(torch.int16).int()).abs().max()) for f in fus)
            finite = all(bool(torch.isfinite(f.float()).all()) for f in fus)
            scale = float((refs[0].float() - residual.float()).abs().max())
            diff = max(float((f.float() - rf.float()).abs().max()) for f, rf in zip(fus, refs))
            good = finite and cross == 0 and (ulps == 0 if rows == 1 else diff <= max(0.02 * scale, 2 ** -7 * float(refs[0].abs().max())))
            ok = ok and good
            print(f"  {kind} rows{rows}: fused vs reference max ulp {ulps} (max|diff| {diff:.3e} on scale {scale:.3f}), "
                  f"cross-rank ulp {cross}, finite={finite} -> {'ok' if good else 'MISMATCH'}", flush=True)

            # Timing: repeated-launch graphs.
            def time_graphs(build):
                graphs = []
                for r, d in enumerate(devices):
                    with torch.cuda.device(d):
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, stream=streams[r]):
                            for fn in build(r):
                                fn()
                    graphs.append(g)
                for _ in range(2):
                    for r, d in enumerate(devices):
                        with torch.cuda.device(d), torch.cuda.stream(streams[r]):
                            graphs[r].replay()
                    sync_all()
                n = 10
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with torch.cuda.device(devices[0]):
                    s.record(streams[0])
                for _ in range(n):
                    for r, d in enumerate(devices):
                        with torch.cuda.device(d), torch.cuda.stream(streams[r]):
                            graphs[r].replay()
                with torch.cuda.device(devices[0]):
                    e.record(streams[0])
                sync_all()
                return s.elapsed_time(e) / n * 1e3 / args.reps

            adv = time_graphs(lambda r: [lambda r=r: advance(r)] * args.reps)
            sep = time_graphs(lambda r: [f for _ in range(args.reps) for f in (lambda r=r: reference(r), lambda r=r: advance(r))]) - adv
            fus_t = time_graphs(lambda r: [f for _ in range(args.reps) for f in (lambda r=r: fused(r), lambda r=r: advance(r))]) - adv
            print(f"  {kind} rows{rows}: attention + unproj {sep:.2f} us per launch pair, fused {fus_t:.2f} us -> "
                  f"{sep - fus_t:+.2f} us saved", flush=True)
            results[f"{kind}_rows{rows}"] = {"ok": good, "max_ulp": ulps, "separate_us": round(sep, 2), "fused_us": round(fus_t, 2)}
    print("RESULT", "PASS" if ok else "FAIL")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
