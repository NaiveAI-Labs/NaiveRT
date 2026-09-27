"""Screen existing DeepGEMM launch hints with real routing counts and random W8A8 data.

Run through gpu_profile_queue. This changes neither the library nor runtime.
Single-GEMM graphs and cache eviction avoid claiming a repeated hot-weight win.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys


def select_cases(records):
    candidates = [dict(tokens_sha256=r['tokens_sha256'], start=r['start'], valid=r['valid'],
                       layer=r['layer'], rank=rank, counts=counts)
                  for r in records for rank, counts in enumerate(r['counts'])]
    candidates.sort(key=lambda r: sum(r['counts']))
    picked = [candidates[round((len(candidates) - 1) * f)] for f in (0, .25, .5, .75, 1)]
    picked.append(max(candidates, key=lambda r: sum((m + 127) // 128 * 128 for m in r['counts']) /
                       max(1, sum(r['counts']))))
    unique = {tuple(r['counts']): r for r in picked}
    return list(unique.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--occupancy', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--samples', type=int, default=30)
    ap.add_argument('--sweep', choices=('hints', 'tiles'), default='hints')
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo))
    import torch
    from naivert.kernels.naive_n0p5 import prefill as P
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(20260920)
    runtime = P.prepare_device(torch.device('cuda:0'))
    module = runtime.module
    original_sms = int(module.get_num_sms())
    original_alignment = int(module.get_mk_alignment_for_contiguous_layout())
    configs = [dict(expected=129, sms=original_sms, block_m=original_alignment)]
    if args.sweep == 'hints':
        configs += [dict(expected=m, sms=original_sms, block_m=original_alignment) for m in (32, 64, 128, 256, 512)]
        configs += [dict(expected=129, sms=s, block_m=original_alignment) for s in (64, 96, 112, 128) if s != original_sms]
    else:
        configs += [dict(expected=129, sms=original_sms, block_m=m)
                    for m in (32, 64, 96, 160, 192, 224) if m != original_alignment]
    data = json.loads(args.occupancy.read_text())
    cases = select_cases(data['records'])
    results = dict(scope=__doc__, sources={
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), args.occupancy, Path(P.__file__))},
        deep_gemm=dict(path=str(runtime.path), sha256=runtime.sha256,
                       default_sms=original_sms, default_alignment=original_alignment,
                       tc_util=int(module.get_tc_util())),
        configurations=configs, samples=args.samples, cache_eviction_bytes=256*1024*1024,
        capacity=8192, cases=[], success=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    flush = torch.empty(256*1024*1024, device='cuda', dtype=torch.uint8)
    e, cap, n = 32, 8192, 4096
    counts_dev = torch.empty(e, device='cuda', dtype=torch.int32)

    def save():
        args.out.write_text(json.dumps(results, indent=2) + '\n')

    try:
        for phase, k in [('gate_up', 4096), ('down', 2048)]:
            # Random weights are only for the operator speed screen. Recorded
            # counts and physical capacity/strides match the executor.
            a = torch.empty((e, cap, k), device='cuda', dtype=torch.bfloat16).normal_().to(P.FP8_DTYPE)
            w = torch.empty((e, n, k), device='cuda', dtype=torch.bfloat16).normal_().to(P.FP8_DTYPE)
            sa = P.ue8m0_scale_buffer((e,), cap, k, device=torch.device('cuda:0'))
            sa.fill_(0x7c7d7e7f)
            sw = P.pack_expert_weight_scales(torch.exp2(torch.randint(-3, 1, (e, n//128, k//128),
                                                                       device='cuda').float()))
            out = torch.empty((e, cap, n), device='cuda', dtype=torch.bfloat16)
            counts_dev.copy_(torch.tensor(cases[0]['counts'], device='cuda', dtype=torch.int32))
            graphs = []
            for cfg in configs:
                module.set_num_sms(cfg['sms'])
                module.set_mk_alignment_for_contiguous_layout(cfg['block_m'])
                assert int(module.get_mk_alignment_for_contiguous_layout()) == cfg['block_m']
                print('COMPILE', phase, cfg, flush=True)
                for _ in range(2):
                    P.grouped_gemm(a, sa, w, sw, out, counts_dev, cfg['expected'])
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    P.grouped_gemm(a, sa, w, sw, out, counts_dev, cfg['expected'])
                graphs.append(graph)
            for case in cases:
                counts = case['counts']
                counts_dev.copy_(torch.tensor(counts, device='cuda', dtype=torch.int32))
                live_ids = torch.cat([torch.arange(m, device='cuda') + expert * cap
                                      for expert, m in enumerate(counts) if m])
                graphs[0].replay()
                ref = out.view(-1, n).index_select(0, live_ids).clone()
                comparisons = []
                for graph in graphs:
                    graph.replay()
                    value = out.view(-1, n).index_select(0, live_ids)
                    diff = value.float() - ref.float()
                    comparisons.append(dict(differing_elements=int(torch.count_nonzero(diff)),
                                             max_abs=float(diff.abs().max()),
                                             finite=bool(torch.isfinite(value).all())))
                    assert comparisons[-1]['finite']
                samples = [[] for _ in graphs]
                # Alternate forward/reverse config order. Flush before the
                # start event, so eviction itself is excluded from GEMM time.
                for rep in range(args.samples + 4):
                    order = range(len(graphs)) if rep % 2 == 0 else reversed(range(len(graphs)))
                    for i in order:
                        flush.zero_()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        graphs[i].replay()
                        end.record()
                        end.synchronize()
                        if rep >= 4:
                            samples[i].append(begin.elapsed_time(end) * 1000)
                row = dict(phase=phase, **case, live_rows=sum(counts),
                    row_utilization={str(m):sum(counts)/sum((v+m-1)//m*m for v in counts) for m in (64, 128)},
                    variants=[dict(**cfg, median_us=statistics.median(times), samples_us=times, **check)
                              for cfg, times, check in zip(configs, samples, comparisons)])
                results['cases'].append(row)
                save()
                print('RESULT', phase, case['layer'], case['rank'], sum(counts),
                      [(x['expected'],x['sms'],x['block_m'],round(x['median_us'],2),x['differing_elements'])
                       for x in row['variants']], flush=True)
            del graphs, graph, a, w, sa, sw, out, ref, value, diff
            torch.cuda.synchronize()
        results['success'] = True
        save()
    finally:
        module.set_num_sms(original_sms)
        module.set_mk_alignment_for_contiguous_layout(original_alignment)


if __name__ == '__main__':
    main()
