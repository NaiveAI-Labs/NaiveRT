"""Repeat penalty in the TP8 target sampler: deterministic set-membership checks.

With presence beta = -1e30 every token with count > 0 is boosted far above the rest, so the sample
must come from the counted set; with count on a single token the sample must be that token.
Row d of a block also counts proposals[0..d-1]. penalty active = 0 must reproduce the unpenalized
tokens bitwise (same seed, same logits).
"""
import argparse
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=6)
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo))
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_naive_n0p5_verify", a.repo / "tests/gpu/naive_n0p5_e2e/run_naive_n0p5_verify.py")
    verify = importlib.util.module_from_spec(spec); spec.loader.exec_module(verify)
    verify.decode.enable_torch_tp8_peers()   # the ranks read each other's mailboxes and flags
    from naivert.kernels.naive_n0p5.dspark_kernel_set import prepare_draft
    from naivert.kernels.naive_n0p5.dspark_sampling import TargetSampler, SHARD
    from naivert.models.naive_n0p5.dspark import DraftBuffers, VOCAB, sampling_params_tensor
    torch.manual_seed(0)
    kernels = prepare_draft(range(8))
    buffers = [DraftBuffers(kernels, rank=r, capacity=64, sampling=True) for r in range(8)]
    logits8 = [torch.randn(8, SHARD, device=f"cuda:{r}") for r in range(8)]
    draft8 = [torch.randn(7, SHARD, device=f"cuda:{r}") for r in range(8)]
    logits1 = [torch.randn(1, SHARD, device=f"cuda:{r}") for r in range(8)]
    out8 = [torch.zeros(8, dtype=torch.int32, device=f"cuda:{r}") for r in range(8)]
    out1 = [torch.zeros(1, dtype=torch.int32, device=f"cuda:{r}") for r in range(8)]
    # The eight-row persistent sampler only runs inside the round graph with the tail's rendezvous;
    # standalone it is the sharded build (same penalty code). The anchor runs standalone in both.
    block = TargetSampler(logits8, buffers, out8, persistent=False)
    anchor = TargetSampler(logits1, buffers, out1, anchor=True, persistent=True)
    for r, b in enumerate(buffers):
        if hasattr(block, "draft_logits"):
            with torch.cuda.device(r):
                block.draft_logits[r].copy_(draft8[r])

    def run(s):
        for r, b in enumerate(buffers):
            with torch.cuda.device(r):
                b.sampling_state.copy_(torch.tensor([1234, 0], dtype=torch.int64))
        for r in range(8):
            with torch.cuda.device(r):
                s.run(r)
        for r in range(8):
            torch.cuda.synchronize(r)
        outs = [int(x) for x in (s.outputs[0] if s.anchor else s.outputs[0])]
        for r in range(8):
            assert [int(x) for x in s.outputs[r]] == outs, "ranks disagree"
        return outs

    def set_penalty(theta, alpha, beta, counts, proposals):
        for r, b in enumerate(buffers):
            with torch.cuda.device(r):
                b.sampling_params.copy_(sampling_params_tensor(repetition=theta, frequency=alpha, presence=beta))
                b.rep_count.copy_(counts.to(b.device, torch.int32))
                b.draft_tokens.copy_(proposals.to(b.device))

    failures = 0
    g = torch.Generator().manual_seed(1)
    zero = torch.zeros(VOCAB, dtype=torch.int32)
    props = torch.randint(0, VOCAB, (7,), generator=g)
    # 1. anchor: inactive == bitwise the unpenalized path (counts ignored)
    set_penalty(1.0, 0.0, 0.0, zero, props); base1 = run(anchor)
    set_penalty(1.0, 0.0, 0.0, torch.randint(0, 3, (VOCAB,), generator=g).to(torch.int32), props); same1 = run(anchor)
    ok = same1 == base1
    print("anchor inactive identity:", "ok" if ok else f"FAIL {base1} {same1}", flush=True); failures += not ok
    # 2. anchor: a single counted token with presence boost must be the sample
    for t in range(a.trials):
        tok = int(torch.randint(0, VOCAB, (1,), generator=g))
        counts = zero.clone(); counts[tok] = 1
        set_penalty(1.0, 0.0, -1e30, counts, props)
        got = run(anchor)[0]
        ok = got == tok
        print(f"anchor presence trial {t}: counted {tok} -> sampled {got}", "ok" if ok else "FAIL", flush=True); failures += not ok
    # 3. anchor: repetition theta on a dominant positive logit drops it below the rest
    for t in range(3):
        tok = int(torch.randint(0, SHARD, (1,), generator=g))
        for r in range(8):
            with torch.cuda.device(r):
                logits1[r].zero_(); logits1[r][0, tok] = 50.0
        counts = zero.clone()
        for r in range(8): counts[r * SHARD + tok] = 1
        set_penalty(1.0, 0.0, 0.0, zero, props); got_plain = run(anchor)[0]
        set_penalty(10.0, 0.0, 0.0, counts, props); got_pen = run(anchor)[0]
        ok = (got_plain % SHARD == tok) and (got_pen % SHARD != tok)
        print(f"anchor theta trial {t}: plain {got_plain} penalized {got_pen} (dominant shard index {tok})", "ok" if ok else "FAIL", flush=True); failures += not ok
    # 4. eight rows (sharded build): inactive identity, then row d samples from {counted} | proposals[:d]
    try:
        set_penalty(1.0, 0.0, 0.0, zero, props); base8 = run(block)
        set_penalty(1.0, 0.0, 0.0, torch.randint(0, 3, (VOCAB,), generator=g).to(torch.int32), props); same8 = run(block)
        ok = same8 == base8
        print("block inactive identity:", "ok" if ok else f"FAIL {base8} {same8}", flush=True); failures += not ok
        for t in range(a.trials):
            props = torch.randperm(VOCAB, generator=g)[:7]
            tok0 = int(torch.randint(0, VOCAB, (1,), generator=g))
            counts = zero.clone(); counts[tok0] = 1
            set_penalty(1.0, 0.0, -1e30, counts, props)
            got = run(block)
            allowed = [{tok0} | set(int(x) for x in props[:d]) for d in range(8)]
            ok = all(got[d] in allowed[d] for d in range(8))
            print(f"block trial {t}: rows -> {got}", "ok" if ok else f"FAIL allowed {[sorted(s) for s in allowed]}", flush=True); failures += not ok
    except Exception as exc:
        print("BLOCK_CRASH", type(exc).__name__, str(exc).splitlines()[0][:200], flush=True); failures += 1
    print("PENALTY_OK" if failures == 0 else f"PENALTY_FAIL {failures}")


if __name__ == "__main__":
    main()
