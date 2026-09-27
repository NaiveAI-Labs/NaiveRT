"""Screen the existing quantization kernel on compact, expert-aligned rows.

Run through gpu_profile_queue. Kernel-only results assume a preceding GEMM
already produces the compact layout. A separate measured variant includes an
index-select copy, but excludes construction of its row indices and is only
a lower bound for converting the current masked layout during inference.
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
    helper_path = Path(__file__).with_name('bench_expert_configs.py')
    spec = importlib.util.spec_from_file_location('_expert_case_selection', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    cases = helper.select_cases(json.loads(args.occupancy.read_text())['records'])
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(20260920)
    torch.cuda.set_device(0)
    dev = torch.device('cuda:0')
    cap, hidden, inter = 8192, 4096, 2048
    up = torch.randn((32, cap, hidden), device=dev, dtype=torch.bfloat16)
    q = torch.empty((32, cap, inter), device=dev, dtype=P.FP8_DTYPE)
    s = P.ue8m0_scale_buffer((32,), cap, inter, device=dev)
    counts = torch.empty(32, device=dev, dtype=torch.int32)
    flush = torch.empty(256*1024*1024, device=dev, dtype=torch.uint8)
    results = dict(scope=__doc__, versions=P.check_versions(), cases=[], success=False,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (Path(__file__), helper_path, args.occupancy, Path(P.__file__))})
    for case in cases:
        ms = case['counts']
        aligned = [(m + 127)//128*128 for m in ms]
        total = sum(aligned)
        counts.copy_(torch.tensor(ms, device=dev, dtype=torch.int32))
        flat_counts = torch.tensor([total], device=dev, dtype=torch.int32)
        gather_ids = torch.cat([torch.arange(m, device=dev) + expert*cap
                                for expert, m in enumerate(aligned) if m])
        flat_up = torch.empty((1, total, hidden), device=dev, dtype=torch.bfloat16)
        flat_q = torch.empty((1, total, inter), device=dev, dtype=P.FP8_DTYPE)
        flat_s = P.ue8m0_scale_buffer((1,), total, inter, device=dev)

        def pack():
            torch.index_select(up.view(-1, hidden), 0, gather_ids, out=flat_up[0])

        def run(name):
            if name == 'masked':
                P.act_quant_masked(up, q, s, counts)
            else:
                if name == 'compact_with_copy':
                    pack()
                P.act_quant_masked(flat_up, flat_q, flat_s, flat_counts)

        def check():
            offset = 0
            for expert, m in enumerate(ms):
                if m:
                    assert torch.equal(q[expert, :m].view(torch.uint8), flat_q[0, offset:offset+m].view(torch.uint8))
                    assert torch.equal(s[expert, :m], flat_s[0, offset:offset+m])
                offset += aligned[expert]

        pack()
        names = ['masked', 'compact', 'compact_with_copy']
        graphs = {}
        for name in names:
            run(name)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(name)
            graphs[name] = graph
        check()
        times = {name: [] for name in names}
        for rep in range(34):
            for name in (names if rep % 2 == 0 else list(reversed(names))):
                flush.zero_()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); graphs[name].replay(); b.record(); b.synchronize()
                if rep >= 4:
                    times[name].append(a.elapsed_time(b)*1000)
        check()
        up.mul_(0.5)
        pack()
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        check()
        row = dict(**case, aligned_rows=total, exact_codes_and_scales=True,
                   changed_input_exact=True, median_us={n:statistics.median(v) for n,v in times.items()},
                   samples_us=times)
        results['cases'].append(row)
        print('RESULT', case['layer'], case['rank'], sum(ms), total, row['median_us'], flush=True)
        del graphs, graph
    results['success'] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
