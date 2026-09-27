"""Check the installed stable selector and bounded DSA on padding/chunk boundaries.

Run through gpu_profile_queue. Uses the real vendored selector, deliberate cutoff
ties, random attention operands, and interleaved CUDA-graph operator timings.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from naivert.kernels.naive_n0p5 import prefill as P
from naivert.prefill.native.stable_dsa_topk import install_stable_epilogue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(20260919)
    versions = P.check_versions()
    P.prepare_device(torch.device('cuda:0'))
    selector = install_stable_epilogue()
    assert selector is P._module(P._TOPK)
    assert selector._indexer_topk_prefill_epilogue_kernel.returns_sorted_indices

    # All scores tie: the actual radix + replacement epilogue must choose the
    # lowest valid key IDs. A post-sort alone cannot enforce this choice.
    lengths = torch.tensor([1, 127, 2047, 2048, 2049, 4096], device='cuda', dtype=torch.int32)
    starts = torch.tensor([0, 17, 0, 17, 0, 0], device='cuda', dtype=torch.int32)
    q8 = torch.zeros((6, P.INDEX_HEADS, P.INDEX_DIM), device='cuda', dtype=P.FP8_DTYPE)
    k8 = torch.zeros((4096, P.INDEX_DIM), device='cuda', dtype=P.FP8_DTYPE)
    weights = torch.ones((6, P.INDEX_HEADS), device='cuda')
    scales = torch.ones((4096, 1), device='cuda')
    slots = torch.arange(P.INDEX_TOPK, device='cuda', dtype=torch.int32)[None, :]
    expected = torch.where(slots < lengths[:, None], slots + starts[:, None], -1)
    # A failing sentinel proves the live wrapper skips its fallback sort.
    original_sort = P._ascending_rows
    def forbidden_sort(_):
        raise AssertionError('ordered epilogue was followed by a redundant sort')
    P._ascending_rows = forbidden_sort
    try:
        for _ in range(3):
            indices = P.indexer_select(q8, weights, k8, scales, starts, starts + lengths,
                                       max_seqlen=4096)
            assert torch.equal(indices, expected), 'cutoff tie selection / padding differs'
    finally:
        P._ascending_rows = original_sort

    reports = []
    for start, valid in [(0, 1), (0, 127), (0, 2047), (0, 2048), (0, 4096), (4096, 1904)]:
        rows, span = 4096, start + 4096
        q = torch.randn((rows, 8, 192), device='cuda', dtype=torch.bfloat16) * 0.2
        k = torch.randn((span, 1, 192), device='cuda', dtype=torch.bfloat16) * 0.2
        v = torch.randn((span, 1, 128), device='cuda', dtype=torch.bfloat16)
        end = torch.arange(1, rows + 1, device='cuda', dtype=torch.int32).clamp(max=valid) + start
        counts = end.clamp(max=P.INDEX_TOPK)
        zeros = torch.zeros_like(counts)
        # Random, unique causal subsets, packed before padding. A shared random
        # key permutation sorted by membership yields a different subset per row.
        priority = torch.rand(span, device='cuda')
        key_ids = torch.arange(span, device='cuda', dtype=torch.int32)
        scores = priority[None, :].expand(rows, span).clone()
        scores.masked_fill_(key_ids[None, :] >= end[:, None], -1)
        indices = scores.topk(P.INDEX_TOPK, dim=1).indices.to(torch.int32)
        indices = torch.where(slots < counts[:, None], indices, -1)
        del scores
        cuq = torch.tensor([0, rows], device='cuda', dtype=torch.int32)
        cuk = torch.tensor([0, start + valid], device='cuda', dtype=torch.int32)
        def run(bounded):
            return P.sparse_attention(q, k, v, indices, cuq, cuk,
                max_seqlen_q=rows, max_seqlen_k=span,
                selected_starts=zeros if bounded else None,
                selected_ends=counts if bounded else None)
        original, bounded = run(False), run(True)
        assert torch.equal(original, bounded), (start, valid, 'bounded output differs')
        graphs, outputs = {}, {}
        for label in (False, True):
            run(label)
            torch.cuda.synchronize()
            graphs[label] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[label]):
                outputs[label] = run(label)
        samples = {False: [], True: []}
        for rep in range(34):
            for label in ((False, True) if rep % 2 == 0 else (True, False)):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); graphs[label].replay(); b.record(); b.synchronize()
                if rep >= 4:
                    samples[label].append(a.elapsed_time(b) * 1000)
        assert torch.equal(outputs[False], outputs[True])
        # Reuse the captured graphs with changed tensor contents.
        q.mul_(0.7); v.add_(0.1)
        graphs[False].replay(); graphs[True].replay(); torch.cuda.synchronize()
        assert torch.equal(outputs[False], outputs[True]), 'changed-input replay differs'
        report = dict(start=start, valid=valid, exact=True,
                      original_us=statistics.median(samples[False]),
                      bounded_us=statistics.median(samples[True]))
        reports.append(report)
        print(json.dumps(report), flush=True)
        del graphs, outputs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(versions=versions, selector_module=selector.__name__,
        cutoff_ties_exact=True, redundant_sort_skipped=True, cases=reports), indent=2) + '\n')


if __name__ == '__main__':
    main()
