"""Probe the existing DSA head-256 specialization, including Q/K padding cost.

No device source is changed. Run through gpu_profile_queue. This is an
operator screen with synthetic operands, not a full-model correctness gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / 'naivert').is_dir():
        sys.path.insert(0, str(parent))
        break

import torch
import torch.nn.functional as F

from naivert.kernels.naive_n0p5 import prefill as P


def oracle_errors(q, k, v, indices, outputs):
    # FP64 attention over exactly the selected BF16 operands. Sample boundary
    # rows rather than materializing the full [4096, 2048, 8] attention matrix.
    rows = [0, 1, 63, 64, 126, 127, 255, 1023, 2046, 2047, 2048, 3072, 4094, 4095]
    refs = []
    for row in rows:
        ids = indices[row]
        ids = ids[ids >= 0].long()
        scores = q[row].double() @ k[ids, 0].double().T * P.ATTN_SCALE
        refs.append(scores.softmax(-1) @ v[ids, 0].double())
    ref = torch.stack(refs)
    result = {}
    for name, out in outputs.items():
        diff = out[rows].double() - ref
        result[name] = dict(rms=float(diff.square().mean().sqrt()),
                            max_abs=float(diff.abs().max()),
                            relative_rms=float(diff.square().mean().sqrt() /
                                               ref.square().mean().sqrt()))
    return dict(rows=rows, errors=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(20260920)
    versions = P.check_versions()
    P.prepare_device(torch.device('cuda:0'))
    source = Path(P._module(P._SPARSE).__file__)
    reports = []
    for start, valid, magnitude in [(0, 199, 1.0), (0, 2047, 1.0),
                                     (0, 4095, 1.0), (0, 4096, 3.0),
                                     (4096, 1903, 1.0)]:
        rows, span = 4096, start + 4096
        q = torch.randn((rows, 8, 192), device='cuda', dtype=torch.bfloat16) * magnitude
        k = torch.randn((span, 1, 192), device='cuda', dtype=torch.bfloat16) * magnitude
        v = torch.randn((span, 1, 128), device='cuda', dtype=torch.bfloat16)
        ends = torch.arange(1, rows + 1, device='cuda', dtype=torch.int32).clamp(max=valid) + start
        counts = ends.clamp(max=P.INDEX_TOPK)
        zeros = torch.zeros_like(counts)
        slots = torch.arange(P.INDEX_TOPK, device='cuda', dtype=torch.int32)[None, :]
        priority = torch.rand(span, device='cuda')
        key_ids = torch.arange(span, device='cuda', dtype=torch.int32)
        scores = priority[None, :].expand(rows, span).clone()
        scores.masked_fill_(key_ids[None, :] >= ends[:, None], -1)
        indices = scores.topk(P.INDEX_TOPK, dim=1).indices.to(torch.int32)
        indices = torch.where(slots < counts[:, None], indices, -1)
        indices = P._ascending_rows(indices)
        del scores
        cuq = torch.tensor([0, rows], device='cuda', dtype=torch.int32)
        cuk = torch.tensor([0, start + valid], device='cuda', dtype=torch.int32)
        q256, k256 = F.pad(q, (0, 64)), F.pad(k, (0, 64))

        def run(name):
            if name == 'padded_with_copies':
                qq, kk = F.pad(q, (0, 64)), F.pad(k, (0, 64))
            elif name == 'padded_kernel_only':
                qq, kk = q256, k256
            else:
                qq, kk = q, k
            return P.sparse_attention(qq, kk, v, indices, cuq, cuk,
                max_seqlen_q=rows, max_seqlen_k=span, scale=P.ATTN_SCALE,
                selected_starts=zeros, selected_ends=counts)

        names = ['original', 'padded_with_copies', 'padded_kernel_only']
        eager = {name: run(name).clone() for name in names}
        assert torch.equal(eager[names[1]], eager[names[2]])
        graphs, outputs = {}, {}
        for name in names:
            run(name)
            torch.cuda.synchronize()
            graphs[name] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[name]):
                outputs[name] = run(name)
        samples = {name: [] for name in names}
        for rep in range(34):
            for name in (names if rep % 2 == 0 else list(reversed(names))):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                graphs[name].replay()
                b.record()
                b.synchronize()
                if rep >= 4:
                    samples[name].append(a.elapsed_time(b) * 1000)
        for name in names:
            assert torch.equal(eager[name], outputs[name]), (name, 'capture differs')
            assert bool(torch.isfinite(outputs[name]).all()), (name, 'nonfinite')
        diff = outputs[names[0]].float() - outputs[names[1]].float()
        report = dict(start=start, valid=valid, magnitude=magnitude,
            median_us={name: statistics.median(times) for name, times in samples.items()},
            samples_us=samples, differing_elements=int(torch.count_nonzero(diff)),
            elements=diff.numel(), pair_max_abs=float(diff.abs().max()),
            oracle=oracle_errors(q, k, v, indices, outputs), capture_exact=True)
        # Padded tensors must be refreshed as captured input changes.
        q.mul_(0.7)
        k.mul_(0.8)
        v.add_(0.1)
        q256[..., :192].copy_(q)
        k256[..., :192].copy_(k)
        for name in names:
            graphs[name].replay()
        torch.cuda.synchronize()
        for name in names:
            assert torch.equal(run(name), outputs[name]), (name, 'changed-input replay differs')
        assert torch.equal(outputs[names[1]], outputs[names[2]])
        report['changed_input_capture_exact'] = True
        reports.append(report)
        print(json.dumps({k: v for k, v in report.items() if k != 'samples_us'}), flush=True)
        del graphs, outputs, eager
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(versions=versions, cases=reports,
        sparse_source=str(source), sparse_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Synthetic operator screen; original attention scale retained; padding cost included.'),
        indent=2) + '\n')


if __name__ == '__main__':
    main()
