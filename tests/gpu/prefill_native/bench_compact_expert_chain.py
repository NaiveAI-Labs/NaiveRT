"""Compare complete masked/compact expert chains using existing library kernels.

Run through gpu_profile_queue. Includes GPU expert grouping, offset construction,
scatter, both GEMMs, fused quantization and the return source map. Transport is
not included. Random W8A8 data and recorded real counts are an operator screen,
not a model accuracy or latency result. No device code is changed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys


def compact_scatter(P, torch, codes, scales, slots, q_buf, s_buf):
    return P.scatter_to_compact_experts(codes, scales, slots, q_buf, s_buf, alignment=128)


def contiguous_gemm(P, a, sa, w, sw, out, ends):
    P.grouped_gemm_contiguous(a, sa, w, sw, out, ends, 129)


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
    helper_path = Path(__file__).with_name('bench_expert_configs.py')
    spec = importlib.util.spec_from_file_location('_expert_cases', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    cases = helper.select_cases(json.loads(args.occupancy.read_text())['records'])
    cases += [dict(label='all_foreign', counts=[0]*32),
              dict(label='eight_hot_experts', counts=[4096]*8 + [0]*24),
              dict(label='last_expert', counts=[0]*31 + [4096])]
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(20260920)
    torch.cuda.set_device(0)
    dev = torch.device('cuda:0')
    runtime = P.prepare_device(dev)
    assert runtime.module.get_mk_alignment_for_contiguous_layout() == 128
    t, e, cap, h, inter = 4096, 32, 8192, 4096, 2048
    # At most T*TOPK live pairs, and < E*128 alignment padding. The final
    # row is never live and is safe for the existing scale scatter's trash.
    flat_cap = t*P.TOPK + e*128
    codes = torch.randn((t, h), device=dev, dtype=torch.bfloat16).to(P.FP8_DTYPE)
    scales = torch.full((t, 8), 0x7c7d7e7f, device=dev, dtype=torch.int32)
    ids = torch.empty((t, P.TOPK), device=dev, dtype=torch.int32)
    weights = []
    for k in (h, inter):
        w = torch.randn((e, h, k), device=dev, dtype=torch.bfloat16).to(P.FP8_DTYPE)
        s = P.pack_expert_weight_scales(torch.exp2(torch.randint(-5, -2, (e, h//128, k//128), device=dev).float()))
        weights.append((w, s))
    old = dict(q=torch.empty((e+1, cap, h), device=dev, dtype=P.FP8_DTYPE),
               s=P.ue8m0_scale_buffer((e+1,), cap, h, device=dev),
               up=torch.empty((e, cap, h), device=dev, dtype=torch.bfloat16),
               qa=torch.empty((e, cap, inter), device=dev, dtype=P.FP8_DTYPE),
               sa=P.ue8m0_scale_buffer((e,), cap, inter, device=dev),
               dn=torch.empty((e, cap, h), device=dev, dtype=torch.bfloat16))
    new = dict(q=torch.empty((flat_cap, h), device=dev, dtype=P.FP8_DTYPE),
               s=P.ue8m0_scale_buffer((), flat_cap, h, device=dev),
               up=torch.empty((flat_cap, h), device=dev, dtype=torch.bfloat16),
               qa=torch.empty((flat_cap, inter), device=dev, dtype=P.FP8_DTYPE),
               sa=P.ue8m0_scale_buffer((), flat_cap, inter, device=dev),
               dn=torch.empty((flat_cap, h), device=dev, dtype=torch.bfloat16))
    flush = torch.empty(256*1024*1024, device=dev, dtype=torch.uint8)
    results = dict(scope=__doc__, success=False, cases=[], versions=P.check_versions(),
                   flat_capacity=flat_cap, masked_capacity=cap,
                   sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__), helper_path, Path(P.__file__), args.occupancy)})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(results, indent=2)+'\n')

    def set_routes(counts, reverse=False):
        # Each local expert occurs at most once per token. Round-robin token
        # assignment realizes any count vector with max<=T and sum<=T*TOPK.
        assert max(counts) <= t and sum(counts) <= t*P.TOPK
        host = torch.full((t, P.TOPK), -1, dtype=torch.int32)
        cursor = 0
        for expert, m in enumerate(counts):
            for j in range(m):
                row = (cursor+j) % t
                col = (cursor+j) // t
                host[row, col] = expert
            cursor += m
        if reverse:
            host = host.flip((0, 1))
        ids.copy_(host)

    def stage(name, phase):
        b = old if name == 'masked' else new
        if phase == 'prepare':
            b['slots'] = P.expert_slots(ids, first_expert=0)
            if name == 'masked':
                b['dest'] = P.scatter_to_experts(codes, scales, b['slots'], b['q'], b['s'])
            else:
                b['dest'], b['ends'], b['live_aligned'] = compact_scatter(P, torch, codes, scales, b['slots'], b['q'], b['s'])
        elif phase == 'gate_up':
            w, s = weights[0]
            if name == 'masked':
                P.grouped_gemm(b['q'][:e], b['s'][:e], w, s, b['up'], b['slots'].masked_m, 129)
            else:
                contiguous_gemm(P, b['q'], b['s'], w, s, b['up'], b['ends'])
        elif phase == 'quant':
            if name == 'masked':
                P.act_quant_masked(b['up'], b['qa'], b['sa'], b['slots'].masked_m)
            else:
                P.act_quant_masked(b['up'].unsqueeze(0), b['qa'].unsqueeze(0), b['sa'].unsqueeze(0), b['live_aligned'])
        elif phase == 'down':
            w, s = weights[1]
            if name == 'masked':
                P.grouped_gemm(b['qa'], b['sa'], w, s, b['dn'], b['slots'].masked_m, 129)
            else:
                contiguous_gemm(P, b['qa'], b['sa'], w, s, b['dn'], b['ends'])
        elif phase == 'return_map':
            b['src'] = b['dest'].index_select(0, b['slots'].permutation)

    phases = ('prepare', 'gate_up', 'quant', 'down', 'return_map')

    def run(name):
        for phase in phases:
            stage(name, phase)

    def check(counts):
        assert torch.equal(old['slots'].masked_m.cpu(), torch.tensor(counts, dtype=torch.int32))
        assert torch.equal(old['slots'].permutation, new['slots'].permutation)
        offset = 0
        checks = {key: True for key in ('q', 's', 'up', 'qa', 'sa', 'dn')}
        finite = True
        for expert, m in enumerate(counts):
            if not m:
                continue
            for key in checks:
                a = old[key][expert, :m].contiguous().view(torch.uint8)
                b = new[key][offset:offset+m].contiguous().view(torch.uint8)
                checks[key] &= bool(torch.equal(a, b))
            finite &= bool(torch.isfinite(new['dn'][offset:offset+m]).all())
            offset += (m+127)//128*128
        assert int(new['live_aligned'][0]) == offset < flat_cap
        pairs = sum(counts)
        old_return = old['dn'].view(-1, h).index_select(0, old['src'][:pairs])
        new_return = new['dn'].index_select(0, new['src'][:pairs])
        checks['return_rows'] = bool(torch.equal(old_return.view(torch.uint8), new_return.view(torch.uint8)))
        checks['finite'] = finite
        return checks

    # One captured full chain per layout; routing counts and destinations must
    # all change on replay. No CPU-derived offsets or cached per-case maps.
    set_routes(cases[0]['counts'])
    graphs = {}
    for name in ('masked', 'compact'):
        run(name)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(name)
        graphs[name] = graph
    for i, case in enumerate(cases):
        counts = case['counts']
        set_routes(counts, reverse=bool(i % 2))
        codes.view(torch.uint8).bitwise_xor_(128)  # changed signs, including signed zero
        scales.fill_(0x7c7d7e7f if i % 2 else 0x7b7c7d7e)
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        row = dict(**case, live_rows=sum(counts), aligned_rows=sum((m+127)//128*128 for m in counts),
                   byte_checks=check(counts))
        results['cases'].append(row)
        save()
        print('CHECK', i, row['live_rows'], row['byte_checks'], flush=True)
        assert all(row['byte_checks'].values()), row
        samples = {name: [] for name in graphs}
        for rep in range(args.samples+4):
            names = list(graphs) if rep % 2 == 0 else list(reversed(graphs))
            for name in names:
                flush.zero_()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); graphs[name].replay(); b.record(); b.synchronize()
                if rep >= 4:
                    samples[name].append(a.elapsed_time(b)*1000)
        row['median_us'] = {name:statistics.median(v) for name,v in samples.items()}
        row['samples_us'] = samples
        row['post_timing_byte_checks'] = check(counts)
        assert all(row['post_timing_byte_checks'].values())
        save()
        print('RESULT', i, row['live_rows'], row['median_us'], flush=True)
    results['success'] = True
    save()


if __name__ == '__main__':
    main()
