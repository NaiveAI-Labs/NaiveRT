"""The sampler's processed distribution (its p buffer after a run) against a torch nucleus reference.

For each top-p and logit shape, every rank's shard of p is gathered, summed (must be 1) and compared
with softmax -> descending cumulative mass -> keep through the first position reaching top-p (ties
kept) -> renormalize. Near-uniform logits put every shard over the candidate cap, which is the
overflow (fallback) path a full nucleus takes on real logits.
"""
import argparse, importlib.util, sys
from pathlib import Path
import torch


def reference(logits, top_p):
    p = torch.softmax(logits.double(), dim=0)
    order = torch.argsort(p, descending=True)
    cum = torch.cumsum(p[order], dim=0)
    k = int(torch.searchsorted(cum, torch.tensor(top_p, dtype=torch.float64)).item())
    k = min(k, p.numel() - 1)
    threshold = p[order[k]]
    keep = p >= threshold
    out = torch.where(keep, p, torch.zeros_like(p))
    return out / out.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo))
    spec = importlib.util.spec_from_file_location("run_naive_n0p5_verify", a.repo / "tests/gpu/naive_n0p5_e2e/run_naive_n0p5_verify.py")
    verify = importlib.util.module_from_spec(spec); spec.loader.exec_module(verify)
    verify.decode.enable_torch_tp8_peers()
    from naivert.kernels.naive_n0p5.dspark_kernel_set import prepare_draft
    from naivert.kernels.naive_n0p5.dspark_sampling import TargetSampler, SHARD
    from naivert.models.naive_n0p5.dspark import DraftBuffers, VOCAB, sampling_params_tensor, TOP_P_MAX
    kernels = prepare_draft(range(8))
    buffers = [DraftBuffers(kernels, rank=r, capacity=64, sampling=True) for r in range(8)]
    logits1 = [torch.zeros(1, SHARD, device=f"cuda:{r}") for r in range(8)]
    out1 = [torch.zeros(1, dtype=torch.int32, device=f"cuda:{r}") for r in range(8)]
    anchor = TargetSampler(logits1, buffers, out1, anchor=True, persistent=True)
    g = torch.Generator().manual_seed(3)
    shapes = {
        "peaky (randn*4)": torch.randn(VOCAB, generator=g) * 4,
        "moderate (randn*2)": torch.randn(VOCAB, generator=g) * 2,
        "near-uniform (randn*0.05)": torch.randn(VOCAB, generator=g) * 0.05,
    }
    failures = 0
    for name, full in shapes.items():
        for r in range(8):
            with torch.cuda.device(r):
                logits1[r].copy_(full[r * SHARD:(r + 1) * SHARD].view(1, SHARD))
        for top_p in (0.95, 0.999, 1.0):
            t = sampling_params_tensor(top_p=top_p)
            for r, b in enumerate(buffers):
                with torch.cuda.device(r):
                    b.sampling_params.copy_(t); b.sampling_state.copy_(torch.tensor([1, 0], dtype=torch.int64))
            for r in range(8):
                with torch.cuda.device(r):
                    anchor.run(r)
            for r in range(8):
                torch.cuda.synchronize(r)
            p = torch.cat([anchor.full[r][0].double().cpu() for r in range(8)])
            ref = reference(full, min(top_p, TOP_P_MAX))   # 1.0 maps to TOP_P_MAX in the parameter block
            tok = int(anchor.outputs[0][0])
            nucleus_ref = int((ref > 0).sum()); nucleus_dev = int((p > 0).sum())
            mass = float(p.sum()); maxdiff = float((p - ref).abs().max()); inref = ref[tok].item() > 0
            # The nucleus edge among tail tokens of ~1e-11 mass depends on float32 vs float64 summation order;
            # the distribution itself must agree to 1e-6 and the edge to one percent of the nucleus.
            ok = abs(mass - 1) < 1e-3 and maxdiff < 1e-6 and abs(nucleus_dev - nucleus_ref) <= max(2, nucleus_ref // 100) and inref
            print(f"{name:28s} top_p={top_p:<6} nucleus dev {nucleus_dev:6d} ref {nucleus_ref:6d}  mass {mass:.5f}  max|dp| {maxdiff:.2e}  sample in nucleus {inref}", "ok" if ok else "FAIL", flush=True)
            failures += not ok
    print("DIST_OK" if failures == 0 else f"DIST_FAIL {failures}")


if __name__ == "__main__":
    main()
