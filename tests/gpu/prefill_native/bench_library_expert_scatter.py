"""Compare complete compact expert chains with stable and library atomic scatter.

GPU queue required. Uses existing SGLang kernels unchanged and random W8A8
operands with recorded routing counts. Measures preparation and full chain
separately; transport is excluded. This is not a full-model latency result.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--occupancy', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--samples', type=int, default=30)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo))
    import torch
    from naivert.kernels.naive_n0p5 import prefill as P
    helper_path = Path(__file__).with_name('library_expert_scatter.py')
    helper = load(helper_path, '_library_scatter')
    case_path = Path(__file__).with_name('bench_expert_configs.py')
    cases = load(case_path, '_expert_cases').select_cases(json.loads(args.occupancy.read_text())['records'])
    cases += [dict(label='all_foreign', counts=[0]*32),
              dict(label='eight_hot_experts', counts=[4096]*8+[0]*24),
              dict(label='last_expert', counts=[0]*31+[4096])]
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(20260920)
    torch.cuda.set_device(0)
    dev = torch.device('cuda:0')
    P.prepare_device(dev)
    t, e, h, inter = 4096, 32, 4096, 2048
    cap = t*P.TOPK + e*128
    codes = torch.randn((t, h), device=dev, dtype=torch.bfloat16).to(P.FP8_DTYPE)
    scales = torch.full((t, 8), 0x7c7d7e7f, device=dev, dtype=torch.int32)
    ids = torch.empty((t, P.TOPK), device=dev, dtype=torch.int64)
    weights = []
    for k in (h, inter):
        w = torch.randn((e, h, k), device=dev, dtype=torch.bfloat16).to(P.FP8_DTYPE)
        s = P.pack_expert_weight_scales(torch.exp2(torch.randint(-5, -2, (e, h//128, k//128), device=dev).float()))
        weights.append((w, s))
    buffers = {}
    for name in ('stable', 'atomic_aten_offsets', 'atomic_library_offsets'):
        buffers[name] = dict(q=torch.empty((cap, h), device=dev, dtype=P.FP8_DTYPE),
                            s=P.ue8m0_scale_buffer((), cap, h, device=dev),
                            up=torch.empty((cap, h), device=dev, dtype=torch.bfloat16),
                            qa=torch.empty((cap, inter), device=dev, dtype=P.FP8_DTYPE),
                            sa=P.ue8m0_scale_buffer((), cap, inter, device=dev),
                            dn=torch.empty((cap, h), device=dev, dtype=torch.bfloat16))
    flush = torch.empty(256*1024*1024, device=dev, dtype=torch.uint8)
    sources = (Path(__file__), helper_path, case_path, Path(P.__file__), args.occupancy,
               args.repo/'naivert/kernels/naive_n0p5/prefill_sglang/scatter.py')
    results = dict(scope=__doc__, success=False, cases=[], versions=P.check_versions(),
                   sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(results, indent=2)+'\n')

    def set_routes(counts, reverse=False):
        host = torch.full((t, P.TOPK), -1, dtype=torch.int64)
        cursor = 0
        for expert, m in enumerate(counts):
            for j in range(m):
                host[(cursor+j)%t, (cursor+j)//t] = expert
            cursor += m
        assert max(counts) <= t and cursor <= t*P.TOPK
        ids.copy_(host.flip((0, 1)) if reverse else host)

    def prepare(name):
        b = buffers[name]
        if name == 'stable':
            slots = P.expert_slots(ids, first_expert=0)
            b['counts'], b['perm'] = slots.masked_m, slots.permutation
            b['dest'], b['ends'], b['aligned'] = P.scatter_to_compact_experts(codes, scales, slots, b['q'], b['s'], alignment=128)
        else:
            b['counts'], b['perm'], b['dest'], b['ends'], b['aligned'] = helper.prepare(
                codes, scales, ids, b['q'], b['s'], library_offsets=name=='atomic_library_offsets')
        b['src'] = b['dest'].index_select(0, b['perm'])

    def run(name, chain):
        prepare(name)
        if not chain:
            return
        b = buffers[name]
        w, s = weights[0]
        P.grouped_gemm_contiguous(b['q'], b['s'], w, s, b['up'], b['ends'], 129)
        P.act_quant_masked(b['up'].unsqueeze(0), b['qa'].unsqueeze(0), b['sa'].unsqueeze(0), b['aligned'])
        w, s = weights[1]
        P.grouped_gemm_contiguous(b['qa'], b['sa'], w, s, b['dn'], b['ends'], 129)

    def returned(name, key, live):
        b = buffers[name]
        # uint8 views preserve FP8 NaNs and signed zeros; make scales row-major
        # first because dtype reinterpretation needs a unit last stride.
        value = b[key].view(torch.uint8) if b[key].dtype == P.FP8_DTYPE else b[key]
        return value.index_select(0, b['src'][:live]).contiguous().view(torch.uint8)

    def check(counts):
        live = sum(counts)
        expected = torch.tensor(counts, device=dev, dtype=torch.int32)
        stable = buffers['stable']
        result = {}
        for name, b in buffers.items():
            checks = dict(counts=bool(torch.equal(b['counts'], expected)),
                          permutation=bool(torch.equal(b['perm'], stable['perm'])),
                          ends=bool(torch.equal(b['ends'], stable['ends'])),
                          total=bool(torch.equal(b['aligned'], stable['aligned'])),
                          all_foreign_skipped=bool((b['dest'][ids.reshape(-1)<0] == -1).all()))
            # Every live pair must have its own destination within its expert.
            checks['unique_rows'] = int(torch.unique(b['src'][:live]).numel()) == live
            for key in ('q', 's', 'up', 'qa', 'sa', 'dn'):
                checks[key] = bool(torch.equal(returned('stable', key, live), returned(name, key, live))) if live else True
            checks['finite'] = bool(torch.isfinite(b['dn'].index_select(0, b['src'][:live])).all())
            result[name] = checks
        return result

    set_routes(cases[0]['counts'])
    graphs = {phase:{} for phase in ('prepare', 'chain')}
    for phase in graphs:
        for name in buffers:
            run(name, phase=='chain')
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(name, phase=='chain')
            graphs[phase][name] = graph
    # Capturing the second graph replaced host references to intermediates.
    # Checks below use the full-chain graph's outputs and maps exclusively.
    for i, case in enumerate(cases):
        counts = case['counts']
        set_routes(counts, reverse=bool(i%2))
        codes.view(torch.uint8).bitwise_xor_(128)
        scales.fill_(0x7c7d7e7f if i%2 else 0x7b7c7d7e)
        for graph in graphs['chain'].values():
            graph.replay()
        torch.cuda.synchronize()
        row = dict(**case, live_rows=sum(counts), checks=check(counts), timing={})
        results['cases'].append(row); save()
        print('CHECK', i, row['live_rows'], row['checks'], flush=True)
        assert all(all(c.values()) for c in row['checks'].values()), row
        reference_return = returned('stable', 'dn', sum(counts)).clone()
        for phase, gs in graphs.items():
            times = {name:[] for name in gs}
            for rep in range(args.samples+4):
                names = list(gs) if rep%2==0 else list(reversed(gs))
                for name in names:
                    flush.zero_()
                    a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record(); gs[name].replay(); b.record(); b.synchronize()
                    if rep >= 4:
                        times[name].append(a.elapsed_time(b)*1000)
                    if phase == 'chain':
                        assert torch.equal(returned(name, 'dn', sum(counts)), reference_return), (i,name,rep)
            row['timing'][phase] = dict(median_us={n:statistics.median(v) for n,v in times.items()}, samples_us=times)
            print('RESULT', i, row['live_rows'], phase, row['timing'][phase]['median_us'], flush=True)
        row['post_timing_checks'] = check(counts)
        assert all(all(c.values()) for c in row['post_timing_checks'].values())
        save()
    results['success'] = True; save()


if __name__ == '__main__':
    main()
