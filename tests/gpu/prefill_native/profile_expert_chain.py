"""Nsight Compute target: unchanged W8A8 expert chain with recorded routing counts.

Use ncu --profile-from-start off through gpu_profile_queue. Random weights and
activations, actual executor shapes/strides, and one representative real count
vector. No profile timing is an end-to-end latency claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--occupancy', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo))
    import torch
    from naivert.kernels.naive_n0p5 import prefill as P
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(20260920)
    torch.cuda.set_device(0)
    runtime = P.prepare_device(torch.device('cuda:0'))
    records = json.loads(args.occupancy.read_text())['records']
    real = [dict(layer=r['layer'], rank=rank, counts=counts, valid=r['valid'],
                 tokens_sha256=r['tokens_sha256'])
            for r in records if r['start'] == 0 and r['valid'] < 4000
            for rank, counts in enumerate(r['counts'])]
    real.sort(key=lambda r: sum(r['counts']))
    case = real[len(real)//2]
    e, cap, hidden, inter = 32, 8192, 4096, 2048
    dev = torch.device('cuda:0')
    counts = torch.tensor(case['counts'], device=dev, dtype=torch.int32)
    a = torch.randn((e, cap, hidden), device=dev, dtype=torch.bfloat16).to(P.FP8_DTYPE)
    sa = P.ue8m0_scale_buffer((e,), cap, hidden, device=dev); sa.fill_(0x7c7d7e7f)
    w13 = torch.randn((e, 2*inter, hidden), device=dev, dtype=torch.bfloat16).to(P.FP8_DTYPE)
    s13 = P.pack_expert_weight_scales(torch.ones((e, 2*inter//128, hidden//128), device=dev))
    w2 = torch.randn((e, hidden, inter), device=dev, dtype=torch.bfloat16).to(P.FP8_DTYPE)
    s2 = P.pack_expert_weight_scales(torch.ones((e, hidden//128, inter//128), device=dev))
    up = torch.empty((e, cap, 2*inter), device=dev, dtype=torch.bfloat16)
    aq = torch.empty((e, cap, inter), device=dev, dtype=P.FP8_DTYPE)
    aqs = P.ue8m0_scale_buffer((e,), cap, inter, device=dev)
    dn = torch.empty((e, cap, hidden), device=dev, dtype=torch.bfloat16)

    def run():
        P.grouped_gemm(a, sa, w13, s13, up, counts, P.expected_m(4096))
        P.act_quant_masked(up, aq, aqs, counts)
        P.grouped_gemm(aq, aqs, w2, s2, dn, counts, P.expected_m(4096))

    run(); run(); torch.cuda.synchronize()
    torch.cuda.profiler.start()
    run()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    args.out.write_text(json.dumps(dict(scope=__doc__, case=case,
        deep_gemm_sha256=runtime.sha256, success=True,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (Path(__file__), Path(P.__file__), args.occupancy)}), indent=2)+'\n')


if __name__ == '__main__':
    main()
