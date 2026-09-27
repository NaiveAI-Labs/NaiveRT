"""Measure masked SiLU/FP8 quantization with 4096 versus 8192 row capacity.

Uses recorded real expert counts and identical random BF16 inputs. Run through
gpu_profile_queue. A unique top-8 route can assign at most 4096 chunk rows to
one expert; the smaller capacity is a shape bound, not a fitted percentile.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
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
    helper_path = Path(__file__).with_name('bench_expert_tiles.py')
    if not helper_path.exists():
        helper_path = Path(__file__).with_name('bench_expert_configs.py')
    spec = importlib.util.spec_from_file_location('_expert_case_selection', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    cases = helper.select_cases(json.loads(args.occupancy.read_text())['records'])
    cases.append(dict(valid=4096, start=0, layer=None, rank=None,
                      counts=[4096] * 8 + [0] * 24, scope='maximum unique-routing concentration'))
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(20260920)
    versions = P.check_versions()
    tensors, graphs = {}, {}
    counts = torch.empty(32, device='cuda', dtype=torch.int32)
    base = torch.randn((32, 4096, 4096), device='cuda', dtype=torch.bfloat16)
    for cap in (8192, 4096):
        up = torch.zeros((32, cap, 4096), device='cuda', dtype=torch.bfloat16)
        up[:, :4096].copy_(base)
        q = torch.empty((32, cap, 2048), device='cuda', dtype=P.FP8_DTYPE)
        s = P.ue8m0_scale_buffer((32,), cap, 2048, device=torch.device('cuda:0'))
        counts.fill_(4096)
        P.act_quant_masked(up, q, s, counts)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            P.act_quant_masked(up, q, s, counts)
        tensors[cap] = up, q, s
        graphs[cap] = graph
    del base
    flush = torch.empty(256*1024*1024, device='cuda', dtype=torch.uint8)
    reports = []
    for case in cases:
        counts.copy_(torch.tensor(case['counts'], device='cuda', dtype=torch.int32))
        for graph in graphs.values():
            graph.replay()
        for expert, m in enumerate(case['counts']):
            if m:
                assert torch.equal(tensors[8192][1][expert, :m].view(torch.uint8),
                                   tensors[4096][1][expert, :m].view(torch.uint8))
                assert torch.equal(tensors[8192][2][expert, :m], tensors[4096][2][expert, :m])
        samples = {cap: [] for cap in graphs}
        for rep in range(34):
            for cap in ((8192, 4096) if rep % 2 == 0 else (4096, 8192)):
                flush.zero_()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); graphs[cap].replay(); b.record(); b.synchronize()
                if rep >= 4:
                    samples[cap].append(a.elapsed_time(b)*1000)
        row = dict(**case, exact_codes_and_scales=True,
                   median_us={cap: statistics.median(v) for cap, v in samples.items()}, samples_us=samples)
        reports.append(row)
        print('RESULT', case.get('layer'), case.get('rank'), sum(case['counts']), row['median_us'], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(scope=__doc__, versions=versions, success=True, cases=reports,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (Path(__file__), helper_path, args.occupancy, Path(P.__file__))}), indent=2)+'\n')


if __name__ == '__main__':
    main()
