"""Probe existing DeepEP dispatch against our packed all-gather at 4K/TP8.

Diagnostic only: all GPU work must run under gpu_profile_queue with a timeout.
No model changes. Compare bytes, integer scale words, routing and changing graph
inputs before timing. The host adapter also tests concurrent collective cleanup;
the stock library's GIL-holding destructor requires process teardown instead.
This does not establish an executor integration or full-model speedup.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback


def run(args, report):
    import torch
    import torch.cuda.nccl as nccl
    if args.vendored:
        import deep_ep_cpp as stock_deepep  # coexistence must not register duplicate pybind types
        from naivert.prefill.native.deepep import load_extension
        deep_ep_cpp = load_extension()
        report['stock_module_coexists'] = stock_deepep.__file__ != deep_ep_cpp.__file__
        assert report['stock_module_coexists']
        args.local_adapter = True
    else:
        import deep_ep_cpp

    torch.set_num_threads(1)
    ranks, rows, hidden, experts, topk = 8, 512, 4096, 256, 8
    report.update(torch=torch.__version__, nccl=nccl.version(), rows_per_rank=rows,
                  ranks=ranks, hidden=hidden, num_sms=args.sms, inner=args.inner,
                  module=deep_ep_cpp.__file__, local_adapter=args.local_adapter,
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  module_sha256=hashlib.sha256(Path(deep_ep_cpp.__file__).read_bytes()).hexdigest(),
                  nccl_proto=os.environ.get('NCCL_PROTO'))
    streams = [torch.cuda.Stream(device=r) for r in range(ranks)]
    buffers = []
    config = deep_ep_cpp.Config(args.sms, 6, 256, 6, 128)
    nvl_bytes = max(256 << 20, config.get_nvl_buffer_size_hint(hidden, ranks))
    report['nvl_bytes'] = nvl_bytes
    report['stage'] = 'setup'
    for r in range(ranks):
        with torch.cuda.device(r):
            buffers.append(deep_ep_cpp.Buffer(r, ranks, nvl_bytes, 0, False, True))
    ids = [b.get_local_device_id() for b in buffers]
    handles = [b.get_local_ipc_handle() for b in buffers]
    for r, buffer in enumerate(buffers):
        with torch.cuda.device(r):
            if args.local_adapter:
                buffer.sync_local(buffers)
            else:
                buffer.sync(ids, handles, None)
    print('DeepEP buffers ready', flush=True)

    def sync():
        for r in range(ranks):
            torch.cuda.synchronize(r)

    cpu, inputs, packed, gathered = [], [], [], []
    for r in range(ranks):
        generator = torch.Generator().manual_seed(1234 + r)
        codes = torch.randint(0, 256, (rows, hidden), dtype=torch.uint8, generator=generator)
        scales = torch.randint(-(2**31), 2**31 - 1, (rows, 8), dtype=torch.int32, generator=generator)
        routing = torch.rand(rows, experts, generator=generator).argsort(1)[:, :topk].contiguous()
        weights = torch.rand(rows, topk, generator=generator)
        cpu.append((codes, scales, routing, weights))
        with torch.cuda.device(r), torch.cuda.stream(streams[r]):
            inputs.append(tuple(t.to(r) for t in cpu[-1]))
            packed.append(torch.empty((rows, 4192), dtype=torch.uint8, device=r))
            gathered.append(torch.empty((ranks * rows, 4192), dtype=torch.uint8, device=r))

    def pack(r):
        codes, scales, routing, weights = inputs[r]
        packed[r][:, :hidden].copy_(codes)
        packed[r][:, hidden:hidden + 32].view(torch.int32).copy_(scales)
        packed[r][:, hidden + 32:hidden + 64].view(torch.int32).copy_(routing)
        packed[r][:, hidden + 64:].view(torch.float32).copy_(weights)

    for r in range(ranks):
        with torch.cuda.device(r), torch.cuda.stream(streams[r]):
            pack(r)
    sync()
    nccl.all_gather(packed, gathered, streams=streams)
    sync()
    print('NCCL communicator ready', flush=True)
    outputs = [None] * ranks

    def dispatch():
        for r in range(ranks):
            with torch.cuda.device(r), torch.cuda.stream(streams[r]):
                codes, scales, routing, weights = inputs[r]
                per_rank, _, per_expert, in_rank, _ = buffers[r].get_dispatch_layout(
                    routing, experts, None, False, False)
                outputs[r] = buffers[r].intranode_dispatch(
                    codes, scales, routing, weights, per_rank, in_rank, per_expert,
                    0, None, None, 1, rows * ranks, config, None, False, False)

    def allgather():
        for r in range(ranks):
            with torch.cuda.device(r), torch.cuda.stream(streams[r]):
                pack(r)
        nccl.all_gather(packed, gathered, streams=streams)

    graph_events = {}

    def capture(fn, name):
        sync()
        graphs = []
        # Captured timing events need actual record nodes; the default internal
        # capture events only express dependencies and cannot be elapsed-time queried.
        events = [[torch.cuda.Event(enable_timing=True, external=True),
                   torch.cuda.Event(enable_timing=True, external=True)] for _ in range(ranks)]
        for r in range(ranks):
            with torch.cuda.device(r), torch.cuda.stream(streams[r]):
                g = torch.cuda.CUDAGraph()
                g.capture_begin(capture_error_mode='relaxed')
                graphs.append(g)
                events[r][0].record(streams[r])
        for _ in range(args.inner):
            fn()
        for r in range(ranks):
            with torch.cuda.device(r), torch.cuda.stream(streams[r]):
                events[r][1].record(streams[r])
                graphs[r].capture_end()
        graph_events[name] = events
        return graphs

    report['stage'] = 'capture'
    graph_deep = capture(dispatch, 'deepep_layout_dispatch')
    graph_gather = capture(allgather, 'pack_nccl_allgather')
    print('both captures ready', flush=True)
    launchers = ThreadPoolExecutor(max_workers=ranks)

    def replay(graphs):
        def one(r):
            with torch.cuda.device(r), torch.cuda.stream(streams[r]):
                graphs[r].replay()
        list(launchers.map(one, range(ranks)))

    def check(label):
        report['stage'] = 'check_' + label
        replay(graph_deep)
        sync()
        counts = []
        for dst in range(ranks):
            got = outputs[dst]
            out_codes, out_scales, out_ids, out_weights = [t.cpu() for t in got[:4]]
            prefix, source = got[5].cpu(), got[8].cpu()
            begin = 0
            for src in range(ranks):
                codes, scales, routing, weights = cpu[src]
                owned = (routing // 32 == dst)
                expected_src = torch.where(owned.any(1))[0]
                end = int(prefix[src, dst])
                selected = source[begin:end].long()
                assert torch.equal(selected.sort().values, expected_src), (label, dst, src, 'sources')
                assert torch.equal(out_codes[begin:end], codes[selected]), (label, dst, src, 'bytes')
                assert torch.equal(out_scales[begin:end], scales[selected]), (label, dst, src, 'scales')
                expected_ids = torch.where(owned[selected], routing[selected] - dst * 32, -1)
                expected_weights = torch.where(owned[selected], weights[selected], 0.0)
                assert torch.equal(out_ids[begin:end], expected_ids), (label, dst, src, 'ids')
                assert torch.equal(out_weights[begin:end], expected_weights), (label, dst, src, 'weights')
                begin = end
            assert (out_ids[begin:] == -1).all(), (label, dst, 'unused rows')
            counts.append(begin)
        replay(graph_gather)
        sync()
        reference = torch.cat([
            torch.cat((x, s.view(torch.uint8), i.to(torch.int32).view(torch.uint8), w.view(torch.uint8)), 1)
            for x, s, i, w in cpu
        ])
        assert all(torch.equal(t.cpu(), reference) for t in gathered), (label, 'allgather')
        report.setdefault('checks', []).append(dict(case=label, exact=True, received_rows=counts))
        print('checked', label, counts, flush=True)

    check('uniform')
    # All pairs route to rank 0; other ranks must not expose stale valid rows.
    # Update payload bytes and routing in place, retaining graph addresses.
    saved = cpu
    cpu = [(x.bitwise_xor(255), s.bitwise_not(),
            torch.arange(topk).expand(rows, -1).contiguous(), w) for x, s, _, w in cpu]
    for r in range(ranks):
        with torch.cuda.device(r), torch.cuda.stream(streams[r]):
            for dest, source in zip(inputs[r], cpu[r]):
                dest.copy_(source)
    sync()
    check('hot_rank_changed_payload')
    cpu = saved
    for r in range(ranks):
        with torch.cuda.device(r), torch.cuda.stream(streams[r]):
            for dest, source in zip(inputs[r], cpu[r]):
                dest.copy_(source)
    sync()
    check('uniform_restored')

    report['stage'] = 'timing'
    samples = {'deepep_layout_dispatch': [], 'pack_nccl_allgather': []}
    graphs = {'deepep_layout_dispatch': graph_deep, 'pack_nccl_allgather': graph_gather}
    for iteration in range(args.samples + 5):
        order = list(graphs) if iteration % 2 == 0 else list(reversed(graphs))
        for name in order:
            began = time.perf_counter()
            replay(graphs[name])
            sync()
            if iteration >= 5:
                samples[name].append(dict(wall_ms=(time.perf_counter() - began) * 1000 / args.inner,
                                         rank_ms=[a.elapsed_time(b) / args.inner for a, b in graph_events[name]]))
    report['samples'] = samples
    report['medians_ms'] = {
        name: dict(wall=statistics.median(s['wall_ms'] for s in values),
                   slowest_rank=statistics.median(max(s['rank_ms']) for s in values))
        for name, values in samples.items()
    }
    report['timing_scope'] = 'Per dispatch from repeated operations in one graph; includes amortized launch skew. Excludes expert permutation/GEMMs and return.'
    if args.local_adapter:
        report['stage'] = 'cleanup'
        if args.vendored:
            sync()
            for group in graphs.values():
                for graph in group:
                    graph.reset()
            for buffer in buffers:
                buffer.destroy(True)
            # Host setup failure must not attempt a collective with missing peers.
            with torch.cuda.device(0):
                incomplete = deep_ep_cpp.Buffer(0, ranks, 1 << 20, 0, False, True)
            sync()
            incomplete.destroy(True)
            report['partial_setup_cleanup_passed'] = True
        else:
            list(launchers.map(lambda r: buffers[r].destroy(), range(ranks)))
        report['collective_cleanup_passed'] = True
    report['success'] = True
    report['stage'] = 'complete'
    launchers.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sms', type=int, default=20)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--inner', type=int, default=16)
    parser.add_argument('--local-adapter', action='store_true')
    parser.add_argument('--vendored', action='store_true', help='build the runtime copy and check coexistence with stock DeepEP')
    args = parser.parse_args()
    report = dict(success=False)
    try:
        run(args, report)
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}), flush=True)
    # See module docstring: this is bounded compatibility/performance evidence,
    # not an executor integration with a supported cleanup lifecycle.
    os._exit(0 if report['success'] else 1)


if __name__ == '__main__':
    main()
