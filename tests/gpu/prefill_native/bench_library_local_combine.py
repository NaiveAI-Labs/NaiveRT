"""Measure unchanged SGLang BF16 expert output -> FP32 weighted token partial.

This is local-combine overhead, not a full MoE saving. Run through the GPU queue.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


def run(a, report):
    import torch
    from naivert.kernels.naive_n0p5 import prefill as P
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    torch.manual_seed(319)
    spec = importlib.util.spec_from_file_location('library_combine_probe', a.kernel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    n, e, cap, h, k = 4096, 32, 4096, 4096, 8
    dn = torch.empty((e + 1, cap, h), device='cuda', dtype=torch.bfloat16).normal_()
    dn[e].zero_()
    out = torch.empty((n, h), device='cuda', dtype=torch.float32)
    weights = torch.rand((n, k), device='cuda', dtype=torch.float32)
    weights /= weights.sum(-1, keepdim=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    report.update(kernel_sha256=hashlib.sha256(a.kernel.read_bytes()).hexdigest(),
                  scope='local weighting/unpermutation only; BF16 inputs, FP32 weights and outputs',
                  shape=dict(tokens=n, experts=e, capacity=cap, hidden=h, topk=k), cases=[])
    for case in ('uniform', 'hot_expert', 'all_local', 'all_foreign'):
        with torch.cuda.stream(stream):
            if case == 'uniform':
                ids = torch.rand(n, 256, device='cuda').topk(k, dim=-1).indices
            elif case == 'hot_expert':
                ids = torch.arange(32, 40, device='cuda').expand(n, k).clone()
                ids[:, 0] = 0
            elif case == 'all_local':
                ids = torch.rand(n, e, device='cuda').topk(k, dim=-1).indices
            else:
                ids = torch.arange(32, 40, device='cuda').expand(n, k).contiguous()
            slots = P.expert_slots(ids, first_expert=0)
            dest = (slots.expert * cap + slots.slot).contiguous()

            def candidate():
                module.post_reorder_for_cutlass_moe(dn, out, dest, slots.expert,
                                                   weights, e, k, n, h, 1.0)

            def check():
                oracle = torch.zeros((n, h), device='cuda', dtype=torch.float64)
                for j in range(k):
                    oracle += dn[slots.expert[:, j], slots.slot[:, j]].double() * weights[:, j:j+1].double()
                error = out.double() - oracle
                finite = bool(torch.isfinite(out).all())
                max_abs = float(error.abs().max())
                assert finite and max_abs < 3e-6, (case, finite, max_abs)
                return dict(max_abs=max_abs, rms=float(error.square().mean().sqrt()))

            candidate()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(a.inner): candidate()
            # Exercise changed weights after capture, including FP32 mantissas.
            weights.copy_(torch.rand_like(weights))
            weights.div_(weights.sum(-1, keepdim=True))
            graph.replay()
            first = check()
            saved = out.clone()
            graph.replay()
            assert torch.equal(saved, out)
            # Change the source payload without changing captured pointers.
            dn.mul_(-0.5)
            graph.replay()
            changed = check()
            samples = []
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for i in range(a.samples + 5):
                begin.record(stream); graph.replay(); end.record(stream); end.synchronize()
                if i >= 5: samples.append(begin.elapsed_time(end) * 1000 / a.inner)
            report['cases'].append(dict(name=case, first=first, changed=changed,
                                        replay_exact=True, samples_us=samples,
                                        median_us=statistics.median(samples)))
            print(case, report['cases'][-1]['median_us'], flush=True)
            graph.reset()
    stream.synchronize()
    report.update(success=True, cleanup_completed=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--inner', type=int, default=16)
    a = p.parse_args()
    report = dict(success=False, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try: run(a, report)
    except BaseException as exc: report.update(success=False, error=repr(exc)); raise
    finally:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
