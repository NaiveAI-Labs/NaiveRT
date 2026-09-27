"""Compare existing 64-bit composite keys with stable uint8/int32 expert keys."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics


def reference_slots(ids):
    """Frozen pre-change composite-key layout, independent of runtime adoption."""
    import torch
    from naivert.kernels.naive_n0p5 import prefill as P
    n, k = ids.shape
    local = ids.to(torch.int64) - 0
    owned = (local >= 0) & (local < 32)
    expert = torch.where(owned, local, 32)
    flat = expert.reshape(-1)
    order = torch.arange(n*k, device=ids.device, dtype=torch.int64)
    keys, perm = torch.sort(flat * (n*k) + order)
    starts = torch.searchsorted(keys, torch.arange(33, device=ids.device) * (n*k))
    rank = order - starts.index_select(0, flat.index_select(0, perm))
    slot = torch.empty_like(flat).scatter_(0, perm, rank)
    slot = torch.where(owned.reshape(-1), slot, 0).view(n, k)
    return P.ExpertSlots(expert, slot, (starts[1:] - starts[:-1]).to(torch.int32), perm)


def run(a, report):
    import torch
    from naivert.kernels.naive_n0p5 import prefill as P
    from narrow_slots_experiment import narrow_slots
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    torch.manual_seed(930)
    report.update(torch=torch.__version__, samples=a.samples, inner=a.inner,
                  baseline_source_sha256=hashlib.sha256(inspect.getsource(reference_slots).encode()).hexdigest(),
                  baseline_scope='frozen pre-change 64-bit unique composite-key implementation',
                  candidate_sha256=hashlib.sha256(Path(__file__).with_name('narrow_slots_experiment.py').read_bytes()).hexdigest(),
                  cases=[])
    variants = ('original', 'int32', 'uint8')
    for n in (256, 1024, 4096):
        ids = torch.zeros((n, 8), device='cuda', dtype=torch.int64)
        graphs, outputs = {}, {}

        def call(label):
            if label == 'original':
                outputs[label] = reference_slots(ids)
            else:
                outputs[label] = narrow_slots(ids, first_expert=0, key_dtype=getattr(torch, label))

        for label in variants:
            call(label)
            torch.cuda.synchronize()
            graphs[label] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[label]):
                for _ in range(a.inner): call(label)
        for case in ('uniform', 'hot_local', 'all_local', 'all_foreign', 'one_local', 'masked'):
            if case in ('uniform', 'hot_local', 'masked'):
                new = torch.rand(n, 256, device='cuda').topk(8, dim=-1).indices
                if case == 'hot_local':
                    new[:, 0] = 0
                    new[:, 1:] = torch.rand(n, 255, device='cuda').topk(7, dim=-1).indices + 1
                if case == 'masked': new[n//2:] = -1
            elif case == 'all_local':
                new = torch.rand(n, 32, device='cuda').topk(8, dim=-1).indices
            else:
                new = torch.rand(n, 224, device='cuda').topk(8, dim=-1).indices + 32
                if case == 'one_local': new[0, 0] = 0
            ids.copy_(new)
            for label in variants: graphs[label].replay()
            torch.cuda.synchronize()
            for label in variants[1:]:
                assert all(x.dtype == y.dtype and torch.equal(x, y)
                           for x, y in zip(outputs['original'], outputs[label])), (n, case, label)
            samples = {label: [] for label in variants}
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for rep in range(a.samples + 4):
                for label in (variants if rep % 2 == 0 else variants[::-1]):
                    begin.record(); graphs[label].replay(); end.record(); end.synchronize()
                    if rep >= 4: samples[label].append(begin.elapsed_time(end) * 1000 / a.inner)
            result = dict(rows=n, case=case, all_fields_exact=True, samples_us=samples,
                          median_us={label: statistics.median(v) for label, v in samples.items()})
            report['cases'].append(result)
            print(n, case, result['median_us'], flush=True)
        if a.profile and n == 4096:
            from collections import Counter
            from torch.profiler import profile, ProfilerActivity
            report['profile_kernel_counts_per_layout'] = {}
            for label in variants:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    graphs[label].replay()
                    torch.cuda.synchronize()
                counts = Counter(event.name for event in prof.events() if event.device_type.name == 'CUDA')
                report['profile_kernel_counts_per_layout'][label] = {name: count / a.inner for name, count in counts.items()}
        for graph in graphs.values(): graph.reset()
    report.update(success=True, cleanup_completed=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--inner', type=int, default=16)
    p.add_argument('--profile', action='store_true', help='separate post-timing kernel-count trace at 4096 rows')
    a = p.parse_args()
    report = dict(success=False, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try: run(a, report)
    except BaseException as exc: report.update(success=False, error=repr(exc)); raise
    finally:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
