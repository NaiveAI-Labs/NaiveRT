"""Test unchanged FlashInfer numerical collectives with local peer pointers.

One process, eight GPUs. Host setup mirrors MixedCommHandler's unicast workspace
layout without a process group, CUDA IPC, multicast or NVSHMEM initialization.
This is an operator experiment, not an executor backend. Run through the GPU queue.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace


def run(args, report):
    import torch
    import torch.cuda.nccl as nccl
    from flashinfer.comm import mixed_comm as M

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    ranks, hidden, rows = 8, 4096, 512
    devices = [torch.device('cuda', r) for r in range(ranks)]
    streams = [torch.cuda.Stream(device=d) for d in devices]

    def sync():
        for stream in streams:
            stream.synchronize()

    # Let torch also update its peer-access bookkeeping.
    for r, dev in enumerate(devices):
        with torch.cuda.device(dev):
            x = torch.zeros(1, device=dev)
            for peer in devices:
                if peer != dev:
                    assert torch.cuda.can_device_access_peer(dev, peer)
                    x.to(peer)
    for dev in devices:
        torch.cuda.synchronize(dev)

    report['stage'] = 'load_library'
    print('loading FlashInfer mixed communication module', flush=True)
    if args.library:
        import nvidia.nvshmem
        import tvm_ffi
        lib_dir = Path(nvidia.nvshmem.__path__[0]) / 'lib'
        host = next(path for path in (lib_dir/'libnvshmem_host.so', lib_dir/'libnvshmem_host.so.3') if path.exists())
        ctypes.CDLL(str(host), mode=ctypes.RTLD_GLOBAL)
        module = tvm_ffi.load_module(str(args.library))
        report['library'] = dict(path=str(args.library), sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
    else:
        module = M.get_mixed_comm_module()
    mode = M.MixedCommMode.FUSED_OPT_WAITS_UC
    ops = (M.MixedCommOp.ALLGATHER, M.MixedCommOp.REDUCESCATTER)
    report['stage'] = 'query_kernel_attributes'
    blocks = {op: module.get_max_block_size(torch.bfloat16, 1, ranks, 1, 1, op, mode) for op in ops}
    report['max_block_sizes'] = {op.name: dict(b) for op, b in blocks.items()}
    max_block = max(int(v) for block in blocks.values() for v in block.values())
    align = lambda n: (n + 16383) // 16384 * 16384
    # Exactly the handler's pure-DP geometry: TP=1, DP=8, four signal buffers.
    base_bytes = align((ranks + 1) * max_block * 16 + 4)
    buffer_bytes = 4 * args.sms * base_bytes
    info_bytes = align(args.sms * 8)
    storage, handlers, pointer_tables = [], [], []
    for r, dev in enumerate(devices):
        with torch.cuda.device(dev), torch.cuda.stream(streams[r]):
            data = torch.empty(buffer_bytes + info_bytes, device=dev, dtype=torch.uint8)
            data[:buffer_bytes].view(torch.int16).fill_(-32768)  # Lamport sentinel: negative zero
            data[buffer_bytes:].zero_()
            storage.append(data)
    for r, dev in enumerate(devices):
        with torch.cuda.device(dev), torch.cuda.stream(streams[r]):
            pointers = [storage[(r + i + 1) % ranks].data_ptr() for i in range(ranks)]
            table = torch.tensor(pointers, device=dev, dtype=torch.int64)
            pointer_tables.append(table)
            handlers.append(SimpleNamespace(
                mixed_comm_module=module,
                uc_buffer_dict={'void_p': ctypes.c_void_p(table.data_ptr())},
                mem_buffer=ctypes.c_void_p(storage[r].data_ptr()),
                mc_buffer_dict={'full': ctypes.c_void_p()}, ns_buffer=ctypes.c_void_p(),
                vm_buffer_bytes_base=base_bytes, ns_data_bytes=0, ns_signal_bytes=0,
                grid_size=args.sms, max_block_size_dict={(op, mode): blocks[op] for op in ops},
                min_block_size=256, min_num_steps=4,
                para_info=SimpleNamespace(dp_size=ranks, local_rank=r, local_size=ranks,
                                          inter_rank=0, inter_size=1)))
    sync()
    report.update(torch=torch.__version__, sms=args.sms, workspace_bytes_per_rank=buffer_bytes+info_bytes,
                  source_sha256=hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest(),
                  mode=mode.name, max_block_sizes={op.name: dict(b) for op, b in blocks.items()}, cases=[])
    with ThreadPoolExecutor(max_workers=ranks) as launchers:
        for op in ops:
            n_in = rows if op == M.MixedCommOp.ALLGATHER else rows*ranks
            n_out = rows*ranks if op == M.MixedCommOp.ALLGATHER else rows
            inputs, outputs, controls = [], [], []
            for r, dev in enumerate(devices):
                with torch.cuda.device(dev), torch.cuda.stream(streams[r]):
                    inputs.append(torch.full((n_in, hidden), r+1, device=dev, dtype=torch.bfloat16))
                    outputs.append(torch.empty((n_out, hidden), device=dev, dtype=torch.bfloat16))
                    controls.append(torch.empty_like(outputs[-1]))
            fused = M._allgather if op == M.MixedCommOp.ALLGATHER else M._reducescatter
            control = nccl.all_gather if op == M.MixedCommOp.ALLGATHER else nccl.reduce_scatter

            def candidate():
                for r, dev in enumerate(devices):
                    with torch.cuda.device(dev), torch.cuda.stream(streams[r]):
                        fused(handlers[r], inputs[r], outputs[r], mode)

            def baseline():
                control(inputs, controls, streams=streams)

            report['stage'] = op.name + '_eager'
            sync(); baseline(); sync(); candidate(); sync()
            assert all(torch.equal(a, b) for a, b in zip(outputs, controls))
            print(op.name, 'eager integer check passed', flush=True)
            graphs, events = {}, {}
            for label, fn in (('nccl', baseline), ('flashinfer_uc', candidate)):
                group, stamps = [], []
                for r, dev in enumerate(devices):
                    with torch.cuda.device(dev), torch.cuda.stream(streams[r]):
                        g = torch.cuda.CUDAGraph()
                        a, b = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
                        g.capture_begin(capture_error_mode='relaxed')
                        a.record(streams[r]); group.append(g); stamps.append((a, b))
                for _ in range(args.inner):
                    fn()
                for r, dev in enumerate(devices):
                    with torch.cuda.device(dev), torch.cuda.stream(streams[r]):
                        stamps[r][1].record(streams[r]); group[r].capture_end()
                graphs[label], events[label] = group, stamps

            def replay(label):
                def one(r):
                    with torch.cuda.device(devices[r]), torch.cuda.stream(streams[r]):
                        graphs[label][r].replay()
                list(launchers.map(one, range(ranks)))

            # Change every rank's input after capture. Compare to an FP64 sum
            # oracle for RS: NCCL and the library can use different rounding.
            cpu = []
            for r, dev in enumerate(devices):
                data = torch.randn(n_in, hidden, generator=torch.Generator().manual_seed(721+r)).bfloat16()
                data[0, :2] = torch.tensor([0., -0.], dtype=torch.bfloat16)
                cpu.append(data)
                with torch.cuda.device(dev), torch.cuda.stream(streams[r]):
                    inputs[r].copy_(data)
            sync()
            replay('nccl'); sync(); replay('flashinfer_uc'); sync()
            reference = torch.cat(cpu) if op == M.MixedCommOp.ALLGATHER else sum(x.double() for x in cpu)
            errors = []
            for r in range(ranks):
                truth = reference if op == M.MixedCommOp.ALLGATHER else reference[r*rows:(r+1)*rows]
                got, old = outputs[r].cpu(), controls[r].cpu()
                assert torch.isfinite(got).all()
                if op == M.MixedCommOp.ALLGATHER:
                    assert torch.equal(got, truth)
                else:
                    # Broad correctness bound; precision suitability still needs
                    # comparison and a full-model numerical gate.
                    assert (got.double()-truth).square().mean().sqrt() < .03 * truth.square().mean().sqrt()
                errors.append(dict(
                    baseline_rms=float((old.double()-truth).square().mean().sqrt()),
                    candidate_rms=float((got.double()-truth).square().mean().sqrt()),
                    candidate_vs_baseline_equal=bool(torch.equal(got, old)),
                    candidate_vs_baseline_byte_equal=bool(torch.equal(got.view(torch.uint8), old.view(torch.uint8)))))
            report['stage'] = op.name + '_timing'
            samples = {label: [] for label in graphs}
            for rep in range(args.samples + 5):
                for label in (list(graphs) if rep % 2 else list(reversed(graphs))):
                    start = time.perf_counter(); replay(label); sync()
                    if rep >= 5:
                        samples[label].append(dict(wall_us=(time.perf_counter()-start)*1e6/args.inner,
                            rank_us=[a.elapsed_time(b)*1000/args.inner for a,b in events[label]]))
            result = dict(op=op.name, errors=errors, samples=samples,
                          median_us={label: dict(wall=statistics.median(s['wall_us'] for s in values),
                              slowest_rank=statistics.median(max(s['rank_us']) for s in values))
                              for label, values in samples.items()})
            report['cases'].append(result)
            print(op.name, result['median_us'], flush=True)
            sync()
            for group in graphs.values():
                for graph in group:
                    graph.reset()
            del reference, cpu, inputs, outputs, controls
    report.update(stage='complete', success=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sms', type=int, default=128)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--inner', type=int, default=16)
    parser.add_argument('--library', type=Path, help='isolated, already-built module (e.g. a corrected device link)')
    args = parser.parse_args()
    report = dict(success=False, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        run(args, report)
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
