#!/usr/bin/env python3
"""Per-kernel time of the verify-8 and single-token phases on 8 ranks.

For every phase kind of one MoE layer (and the dense layer's phases, and the head) a per-rank
CUDA graph of N repetitions [phase, advance] is captured and replayed in lockstep; `advance`
(epoch / generation / position increments) keeps every all-reduce exchange on a fresh tag and is
timed alone and subtracted. This avoids the profiler (node-level graph tracing stalled on the
8-rank driver) and the host issue skew of eager per-phase timing.

usage: bench_verify_kernels.py [--reps 20] [--pdl] [--out file.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_HOME", "/opt/cuda132-min")
import torch  # noqa: E402


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decode = load_module(ROOT / "tests/gpu/naive_n0p5_e2e/run_naive_n0p5_decode.py")
verify = load_module(ROOT / "tests/gpu/naive_n0p5_e2e/run_naive_n0p5_verify.py")
RANKS = decode.RANKS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--layers", type=int, default=2, help="layer 0 dense (full_dsa), layer 1 MoE (sliding window)")
    ap.add_argument("--pdl", action="store_true")
    ap.add_argument("--select", choices=("rows8",), default="rows8")
    ap.add_argument("--router", choices=("rows8", "halves"), default="rows8")
    ap.add_argument("--dense", choices=("rows8", "halves"), default="rows8")
    ap.add_argument("--head", choices=("rows8", "halves"), default="rows8")
    ap.add_argument("--fused-unproj", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--pos", type=int, default=64, help="position of the timed step (caches random-filled up to it)")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--experts", choices=("fp8", "mx"), default="fp8", help="routed-expert weight format build")
    ap.add_argument("--expert-down", choices=("bf16", "w8a8"), default="bf16", help="The W8A8 production graph")
    args = ap.parse_args()
    from naivert.kernels.naive_n0p5.model_ops import load_extension
    dsa_mod = load_module(ROOT / "tests/gpu/attn_dsa/run_attn_dsa.py")
    dsa_rows = load_module(ROOT / "tests/gpu/attn_dsa/run_attn_dsa_rows.py")
    swa_mod = load_module(ROOT / "tests/gpu/attn_swa/run_attn_swa.py")
    ops = load_extension(weights=args.experts, expert_down=args.expert_down)
    ext_dsa, ext_swa = dsa_mod.build(False, decode.IHEADS), swa_mod.build(1)
    ext_dsa8, ext_swa8 = dsa_rows.build(8, index_heads=decode.IHEADS), swa_mod.build(8)
    ops.enable_tp8_peer_access()
    weights = [decode.RankWeights(r, args.layers, 7, args.max_seq_len, ext_dsa.workspace_bytes(), ext_swa.workspace_bytes(),
                                  experts=args.experts, expert_down=args.expert_down)
               for r in range(RANKS)]
    mailboxes = [torch.zeros(verify.mailbox_bytes(ops), dtype=torch.uint8, device=w.dev) for w in weights]   # Region + rows, not max()
    inboxes = [torch.zeros(4096, dtype=torch.uint8, device=w.dev) for w in weights]
    runtimes = [decode.RankRuntime(r, weights[r], ops, ext_dsa, ext_swa, args.max_seq_len, mailboxes, inboxes, pdl=args.pdl)
                for r in range(RANKS)]
    for rt in runtimes: rt.pdl_markers = decode.PDL_W8A8_MARKERS if args.expert_down == 'w8a8' else None
    verifies = [verify.VerifyRuntime(rt, ext_dsa8, ext_swa8, ext_dsa8.workspace_bytes(), ext_swa8.workspace_bytes(),
                                     pdl=args.pdl, select=args.select, router=args.router, fused_unproj=args.fused_unproj,
                                     dense=args.dense, head=args.head) for rt in runtimes]

    def sync_all():
        for rt in runtimes:
            torch.cuda.synchronize(rt.dev)

    def lockstep(phase_lists):
        for k in range(len(phase_lists[0])):
            for rt, phases in zip(runtimes, phase_lists):
                with torch.cuda.device(rt.dev), torch.cuda.stream(rt.stream):
                    phases[k][1]()
            sync_all()

    # Caches random-filled up to the timed position (as run_naive_n0p5_verify --random-fill-pos does).
    P = args.pos
    for w in weights:
        with torch.cuda.device(w.dev):
            for L in range(w.layers):
                lw = w.layer[L]
                if lw["kind"] == "full_dsa":
                    lw["k_cache"][:P].normal_(); lw["v_cache"][:P].normal_()
                    rows = torch.randn(P, lw["ki_cache"].shape[1], device=w.dev)
                    scale = rows.abs().amax(dim=1).clamp_min(1e-30) / 448.0
                    lw["ki_cache"][:P] = (rows / scale[:, None]).to(torch.float8_e4m3fn)
                    lw["ki_scale"][:P] = scale
                else:
                    lw["k_cache"].normal_(); lw["v_cache"].normal_()
    tgen = torch.Generator().manual_seed(1007)
    row_tokens = torch.randint(0, decode.VOCAB, (verify.VROWS,), generator=tgen)
    for rt, v in zip(runtimes, verifies):
        with torch.cuda.device(rt.dev), torch.cuda.stream(rt.stream):
            rt.pos.fill_(P)
            rt.token.fill_(int(row_tokens[0]))
            v.tokens.copy_(row_tokens.to(rt.dev))   # eight different tokens: diverse expert routing
    # Warm up both paths once in lockstep (first launches configure attributes and would deadlock
    # under rank-sequential eager issue).
    lockstep([rt.phases() for rt in runtimes])
    lockstep([v.phases() for v in verifies])

    def reset_pos():
        for rt in runtimes:
            with torch.cuda.device(rt.dev), torch.cuda.stream(rt.stream):
                rt.pos.fill_(P)
        sync_all()

    def time_graphs(build_fn):
        """build_fn(rank_index) -> list of callables; capture per rank, replay in lockstep, ms per replay."""
        graphs = []
        for i, rt in enumerate(runtimes):
            fns = build_fn(i)
            with torch.cuda.device(rt.dev):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=rt.stream):
                    for fn in fns:
                        fn()
            graphs.append(g)
        for _ in range(2):
            for rt, g in zip(runtimes, graphs):
                with torch.cuda.device(rt.dev), torch.cuda.stream(rt.stream):
                    g.replay()
            sync_all()
        reset_pos()
        n = 10
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(runtimes[0].dev):
            start.record(runtimes[0].stream)
        for _ in range(n):
            for rt, g in zip(runtimes, graphs):
                with torch.cuda.device(rt.dev), torch.cuda.stream(rt.stream):
                    g.replay()
        with torch.cuda.device(runtimes[0].dev):
            end.record(runtimes[0].stream)
        sync_all()
        reset_pos()
        return start.elapsed_time(end) / n

    results = {"reps": args.reps, "pdl": args.pdl, "pos": P, "max_seq_len": args.max_seq_len, "experts": args.experts,
               "seq1_us": {}, "verify8_us": {}}
    for label, holders, key in (("seq1", runtimes, "seq1_us"), ("verify8", verifies, "verify8_us")):
        phase_lists = [h.phases() for h in holders]
        names = [name for name, _ in phase_lists[0]]
        advance_name = names[-1]
        # advance moves pos by VROWS / 1; a fill after it holds the timed position (in the baseline too).
        holds = [(lambda rt=rt: rt.pos.fill_(P)) for rt in runtimes]
        adv = time_graphs(lambda i: [f for _ in range(args.reps) for f in (dict(phase_lists[i])[advance_name], holds[i])])
        results[key]["advance_only"] = round(adv / args.reps * 1e3, 2)
        for name in names:
            if name == advance_name:
                continue
            fns = {i: dict(phase_lists[i]) for i in range(RANKS)}
            total = time_graphs(lambda i, name=name: [f for _ in range(args.reps) for f in (fns[i][name], fns[i][advance_name], holds[i])])
            per = (total - adv) / args.reps * 1e3
            results[key][name] = round(per, 2)
            print(f"  {label:8s} {name:24s} {per:8.2f} us", flush=True)
    # Whole-step replay of each path for the same window.
    for label, holders, key in (("seq1", runtimes, "seq1_step_us"), ("verify8", verifies, "verify8_step_us")):
        phase_lists = [h.phases() for h in holders]
        step = time_graphs(lambda i: [fn for _, fn in phase_lists[i]])
        results[key] = round(step * 1e3, 1)
        print(f"  {label} whole step ({args.layers} layers): {step * 1e3:.1f} us", flush=True)
    print(json.dumps(results, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
