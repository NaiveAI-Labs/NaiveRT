"""Runtime sampling controls in the TP8 samplers: temperature and top-p, deterministic checks.

Logits are built so the outcome is forced: with A=3 and B=0 (everything else far below), 1/T = 10
must always pick A and 1/T = 0.1 must pick B within twenty seeds; with A > B > C by 0.1 logits,
top_p = 0.3 keeps only A, top_p = 0.65 keeps A and B (0.367 + 0.332 = 0.699) and must never pick C. The parameter block's
defaults must reproduce the stock sampler token for token.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo))
    spec = importlib.util.spec_from_file_location("run_naive_n0p5_verify", a.repo / "tests/gpu/naive_n0p5_e2e/run_naive_n0p5_verify.py")
    verify = importlib.util.module_from_spec(spec); spec.loader.exec_module(verify)
    verify.decode.enable_torch_tp8_peers()
    from naivert.kernels.naive_n0p5.dspark_kernel_set import prepare_draft
    from naivert.kernels.naive_n0p5.dspark_sampling import TargetSampler, SHARD
    from naivert.models.naive_n0p5.dspark import DraftBuffers, VOCAB, sampling_params_tensor
    torch.manual_seed(0)
    kernels = prepare_draft(range(8))
    buffers = [DraftBuffers(kernels, rank=r, capacity=64, sampling=True) for r in range(8)]
    logits1 = [torch.randn(1, SHARD, device=f"cuda:{r}") for r in range(8)]
    out1 = [torch.zeros(1, dtype=torch.int32, device=f"cuda:{r}") for r in range(8)]
    anchor = TargetSampler(logits1, buffers, out1, anchor=True, persistent=True)
    failures = 0

    def run(seed):
        for r, b in enumerate(buffers):
            with torch.cuda.device(r):
                b.sampling_state.copy_(torch.tensor([seed, 0], dtype=torch.int64))
        for r in range(8):
            with torch.cuda.device(r):
                anchor.run(r)
        for r in range(8):
            torch.cuda.synchronize(r)
        outs = [int(anchor.outputs[r][0]) for r in range(8)]
        assert len(set(outs)) == 1, f"ranks disagree {outs}"
        return outs[0]

    def set_params(**kw):
        t = sampling_params_tensor(**kw)
        for r, b in enumerate(buffers):
            with torch.cuda.device(r):
                b.sampling_params.copy_(t)

    def set_logits(values):
        """values: {global token: logit}; everything else -1e4."""
        for r in range(8):
            with torch.cuda.device(r):
                logits1[r].fill_(-1e4)
                for tok, v in values.items():
                    if tok // SHARD == r:
                        logits1[r][0, tok % SHARD] = v

    # 0. defaults reproduce the stock token on random logits
    for r in range(8):
        with torch.cuda.device(r):
            logits1[r].copy_(torch.randn(1, SHARD, generator=torch.Generator().manual_seed(7)).to(f"cuda:{r}"))
    set_params(); base = [run(s) for s in range(5)]
    set_params(inv_temperature=1.0, top_p=0.95); same = [run(s) for s in range(5)]
    ok = base == same
    print("explicit defaults == stock:", "ok" if ok else f"FAIL {base} {same}", flush=True); failures += not ok
    # 1. temperature: A=3, B=0
    A, B, C = 40000, 90000, 140000
    set_logits({A: 3.0, B: 0.0})
    set_params(inv_temperature=10.0); got = [run(s) for s in range(a.trials)]
    ok = all(g == A for g in got); print(f"1/T=10: {sum(g == A for g in got)}/{a.trials} picked A", "ok" if ok else f"FAIL {got[:5]}"); failures += not ok
    set_params(inv_temperature=0.1); got = [run(s) for s in range(a.trials)]
    ok = any(g == B for g in got) and all(g in (A, B) for g in got)
    print(f"1/T=0.1: A {sum(g == A for g in got)} B {sum(g == B for g in got)} of {a.trials}", "ok" if ok else f"FAIL {got[:5]}"); failures += not ok
    set_params(inv_temperature=1.0); got = [run(s) for s in range(a.trials)]
    print(f"1/T=1: A {sum(g == A for g in got)} B {sum(g == B for g in got)} (p(A)=0.953 expected)", flush=True)
    # 2. top-p: A > B > C by 0.1
    set_logits({A: 20.0, B: 19.9, C: 19.8})
    set_params(top_p=0.3); got = [run(s) for s in range(a.trials)]
    ok = all(g == A for g in got); print(f"top_p=0.3: {sum(g == A for g in got)}/{a.trials} picked A", "ok" if ok else f"FAIL {got[:5]}"); failures += not ok
    set_params(top_p=0.65); got = [run(s) for s in range(a.trials)]
    ok = all(g in (A, B) for g in got) and any(g == B for g in got)
    print(f"top_p=0.65: A {sum(g == A for g in got)} B {sum(g == B for g in got)} C {sum(g == C for g in got)}", "ok" if ok else f"FAIL {got[:5]}"); failures += not ok
    set_params(top_p=0.7); got = [run(s) for s in range(a.trials)]   # 0.699 < 0.7: C joins the nucleus
    ok = any(g == C for g in got)
    print(f"top_p=0.7: A {sum(g == A for g in got)} B {sum(g == B for g in got)} C {sum(g == C for g in got)} (C expected)", "ok" if ok else f"FAIL {got[:5]}"); failures += not ok
    set_params(top_p=1.0); got = [run(s) for s in range(a.trials)]
    print(f"top_p=1.0: A {sum(g == A for g in got)} B {sum(g == B for g in got)} C {sum(g == C for g in got)}", flush=True)
    # 3. seed: same seed same token, across seeds not constant at T=1 on the three-way tie
    set_params(); got_a = [run(s) for s in range(a.trials)]; got_b = [run(s) for s in range(a.trials)]
    ok = got_a == got_b and len(set(got_a)) > 1
    print("seed: repeatable", got_a == got_b, "and varies across seeds", len(set(got_a)) > 1, "ok" if ok else "FAIL"); failures += not ok
    print("SAMPLING_OK" if failures == 0 else f"SAMPLING_FAIL {failures}")


if __name__ == "__main__":
    main()
