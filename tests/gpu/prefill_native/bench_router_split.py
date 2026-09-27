"""Test FP32 router weights decomposed into three exact BF16 terms.

Uses only torch/cuBLAS GEMMs with FP32 outputs. Run under gpu_profile_queue.
Real checkpoint weights, synthetic normalized BF16 activations, FP64 oracle.
This is a numerical/performance experiment, not a change to the runtime router.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split-k', type=int, default=1)
    args = parser.parse_args()
    assert args.split_k > 0 and 4096 % args.split_k == 0
    sys.path.insert(0, str(args.repo))
    import torch
    from safetensors import safe_open
    from naivert.kernels.naive_n0p5 import prefill as P
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    index_path = args.target / 'model.safetensors.index.json'
    mapping = json.loads(index_path.read_text())['weight_map']

    def read(key):
        with safe_open(args.target / mapping[key], framework='pt', device='cpu') as handle:
            return handle.get_tensor(key).to('cuda:0').float()

    report = dict(scope=__doc__, torch=torch.__version__, tf32=False, split_k=args.split_k,
                  allow_bf16_reduced_precision_reduction=False,
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  checkpoint_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
                  cases=[], success=False)
    for layer in range(1, 48):
        prefix = f'model.layers.{layer}.'
        gate = read(prefix + 'mlp.gate.weight')
        bias = read(prefix + 'mlp.gate.e_score_correction_bias')
        gamma = read(prefix + 'post_attention_layernorm.weight').bfloat16().float()
        remaining, terms = gate.clone(), []
        for _ in range(3):
            term = remaining.bfloat16()
            terms.append(term)
            remaining = remaining - term.float()
        reconstructed = sum(term.double() for term in terms)
        assert torch.equal(reconstructed, gate.double()), (layer, 'weight reconstruction')
        stacked = torch.cat(terms, dim=0)
        if args.split_k > 1:
            batched_weight = stacked.view(768, args.split_k, 4096 // args.split_k).permute(1, 0, 2).contiguous().transpose(1, 2)
        for pattern in ('normal', 'outliers'):
            torch.manual_seed(1234 + layer)
            raw = torch.randn((512, 4096), device='cuda:0')
            if pattern == 'outliers':
                raw[:, ::128] *= 16
            x = (raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-5) * gamma).bfloat16()
            base_out = torch.empty((512, 256), device='cuda:0', dtype=torch.float32)
            split_out = torch.empty_like(base_out)
            expanded = torch.empty((512, 768), device='cuda:0', dtype=torch.float32)
            partials = torch.empty((args.split_k, 512, 768), device='cuda:0', dtype=torch.float32)

            def baseline():
                return torch.mm(x.float(), gate.T, out=base_out)

            def candidate():
                if args.split_k == 1:
                    torch.mm(x, stacked.T, out_dtype=torch.float32, out=expanded)
                else:
                    torch.bmm(x.view(512, args.split_k, 4096 // args.split_k).transpose(0, 1),
                              batched_weight, out_dtype=torch.float32, out=partials)
                    torch.sum(partials, dim=0, out=expanded)
                torch.add(expanded[:, 256:512], expanded[:, 512:], out=split_out)
                split_out.add_(expanded[:, :256])
                return split_out

            truth = x.double() @ gate.double().T
            baseline()
            candidate()
            bw, bi = P.route(base_out, bias)
            cw, ci = P.route(split_out, bias)
            _, ti = P.route(truth.float(), bias)

            def error(y):
                delta = y.double() - truth
                return dict(rms=float(delta.square().mean().sqrt()),
                            relative_rms=float(delta.square().mean().sqrt() / truth.square().mean().sqrt()),
                            max_abs=float(delta.abs().max()))

            case = dict(layer=layer, pattern=pattern, weight_reconstruction_exact=True,
                        baseline_error=error(base_out), candidate_error=error(split_out),
                        routing_rows=512, ordered_ids_equal_rows=int((bi == ci).all(-1).sum()),
                        expert_sets_equal_rows=int((bi.sort(-1).values == ci.sort(-1).values).all(-1).sum()),
                        baseline_oracle_equal_rows=int((bi == ti).all(-1).sum()),
                        candidate_oracle_equal_rows=int((ci == ti).all(-1).sum()),
                        max_abs_route_weight_diff=float((bw - cw).abs().max()))
            if layer in (1, 16, 32, 47) and pattern == 'normal':
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graphs = {}
                with torch.cuda.stream(stream):
                    for fn in (baseline, candidate):
                        for _ in range(5):
                            fn()
                stream.synchronize()
                for name, fn in (('fp32', baseline), ('three_bf16', candidate)):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        for _ in range(32):
                            fn()
                    graphs[name] = graph
                samples = {name: [] for name in graphs}
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for rep in range(35):
                    for name in (list(graphs) if rep % 2 == 0 else list(reversed(graphs))):
                        with torch.cuda.stream(stream):
                            start.record()
                            graphs[name].replay()
                            end.record()
                        end.synchronize()
                        if rep >= 5:
                            samples[name].append(start.elapsed_time(end) * 1000 / 32)
                case['samples_us'] = samples
                case['median_us'] = {name: statistics.median(values) for name, values in samples.items()}
                torch.cuda.current_stream().wait_stream(stream)
            report['cases'].append(case)
        print('layer', layer, json.dumps({k: v for k, v in report['cases'][-2].items() if k != 'samples_us'}), flush=True)
    report['success'] = True
    report['total_rows'] = sum(c['routing_rows'] for c in report['cases'])
    report['ordered_ids_equal_rows'] = sum(c['ordered_ids_equal_rows'] for c in report['cases'])
    report['expert_sets_equal_rows'] = sum(c['expert_sets_equal_rows'] for c in report['cases'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS', report['ordered_ids_equal_rows'], '/', report['total_rows'], flush=True)


if __name__ == '__main__':
    main()
