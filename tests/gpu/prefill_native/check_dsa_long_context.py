"""Exact tiled DSA selection and capture checks; run through the GPU queue."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from naivert.kernels.naive_n0p5 import prefill as P
from naivert.prefill.native import stable_dsa_topk as S


def reference(logits, lengths, starts, global_indices):
    values = logits.cpu().numpy()
    lengths, starts = lengths.cpu().tolist(), starts.cpu().tolist()
    out_values = np.full((len(lengths), 2048), -np.inf, dtype=np.float32)
    out_indices = np.full((len(lengths), 2048), -1, dtype=np.int32)
    for row, length in enumerate(lengths):
        scores = values[row, :length]
        assert not np.isnan(scores).any()
        bits = scores.view(np.uint32)
        ordered = np.where(bits & np.uint32(0x80000000), ~bits, bits ^ np.uint32(0x80000000))
        ids = np.argsort(-ordered.astype(np.int64), kind='stable')[:2048]
        ids.sort()
        out_values[row, :len(ids)] = scores[ids]
        out_indices[row, :len(ids)] = ids + (starts[row] if global_indices else 0)
    return torch.from_numpy(out_values).cuda(), torch.from_numpy(out_indices).cuda()


def exact(actual, expected):
    values, indices = actual
    ev, ei = expected
    assert torch.equal(indices, ei), ('indices', int(torch.count_nonzero(indices != ei)))
    assert torch.equal(values.view(torch.int32), ev.view(torch.int32)), 'score bits differ'


def operands(width):
    rows = 12
    logits = (torch.randn(rows, width, device='cuda') * 4).round() / 8
    logits[1].zero_()
    logits[2].fill_(1)
    logits[3].fill_(-0.0)
    logits[3, 1::2] = 0.0
    logits[4].fill_(float('-inf'))
    logits[4, ::37] = float('inf')
    # The winners and cutoff ties cross tile boundaries, including the last tile.
    logits[5].fill_(-2)
    logits[5, ::17] = 1
    logits[5, -1000:] = 2
    logits[11].fill_(1)  # every valid key ties, including across tile boundaries
    lengths = torch.tensor([0, 1, 2047, width, width, width, min(width, 2048),
                            min(width, 2049), min(width, 32767), min(width, 32768),
                            max(0, width - 17), width], device='cuda', dtype=torch.int32)
    starts = torch.arange(rows, device='cuda', dtype=torch.int32) * 13
    positions = torch.arange(width, device='cuda')
    logits.masked_fill_(positions[None, :] >= lengths[:, None], float('nan'))
    return logits, lengths, starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--skip-indexer', action='store_true')
    parser.add_argument('--widths', default='2048,32767,32768,32769,65536,65537,69632,131072,262145,1048576')
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(20260922)
    P.prepare_device(torch.device('cuda:0'))
    selector = S.install_stable_epilogue()
    if args.candidate:
        spec = importlib.util.spec_from_file_location('dsa_candidate', args.candidate)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        if hasattr(candidate, 'select'):
            S.tiled_stable_topk = candidate.select
        if hasattr(candidate, 'indexer_select'):
            P.indexer_select = candidate.indexer_select
    radix = selector.invoke_indexer_topk_prefill
    result = dict(versions=P.check_versions(), selection=[], indexer=[],
                  source_sha256=hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(),
                  scope='Existing device kernels composed in tiles; exact ordered-FP32 / key-ID oracle.')
    if args.candidate:
        result['candidate_sha256'] = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(result, indent=2) + '\n')

    # A small final radix bucket of -inf scores used to let -FLT_MAX padding
    # outrank valid entries. Exercise the new tiled path independently of the
    # unchanged short-span vendor path, including partial and empty tiles.
    for length in [2049, 2050, 3072]:
        scores = torch.full((1, 65537), float('nan'), device='cuda')
        scores[:, :length] = -float('inf')
        scores[:, :1000] = float('inf')
        lengths = torch.tensor([length], dtype=torch.int32, device='cuda')
        starts = torch.tensor([17], dtype=torch.int32, device='cuda')
        exact(S.tiled_stable_topk(scores, lengths, starts), reference(scores, lengths, starts, True))
    result['small_infinity_buckets_exact'] = True
    save()

    for width in map(int, args.widths.split(',')):
        print('selection', width, flush=True)
        logits, lengths, starts = operands(width)
        for global_indices in [False, True]:
            expected = reference(logits, lengths, starts, global_indices)
            exact(S.tiled_stable_topk(logits, lengths, starts, return_global_indices=global_indices), expected)
        zero = torch.zeros_like(lengths)
        values = torch.empty((12, 2048), device='cuda')
        indices = torch.empty((12, 2048), device='cuda', dtype=torch.int32)

        def composed():
            raw = radix(logits, zero, lengths, 2048)
            S.stable_epilogue[(12,)](logits, raw, lengths, starts, values, indices,
                query_start=0, num_rows=12, topk=2048, max_seqlen=width,
                total_rows=12, return_global_indices=True, BLOCK_K=256)
            return values, indices

        exact(composed(), expected)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = composed()
        graph.replay()
        exact(captured, expected)
        times = []
        for rep in range(14):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); graph.replay(); b.record(); b.synchronize()
            if rep >= 4:
                times.append(a.elapsed_time(b) * 1000)
        # Change all metadata/data in place, including rows that used to be empty.
        logits.copy_((torch.randn_like(logits) * 2).round())
        lengths.fill_(width)
        lengths[1] = 17
        starts.add_(11)
        expected = reference(logits, lengths, starts, True)
        graph.replay()
        exact(captured, expected)
        row = dict(width=width, rows=12, exact=True, captured_changed_inputs_exact=True,
                   median_us=statistics.median(times), samples_us=times)
        result['selection'].append(row); save(); print(json.dumps(row), flush=True)
        graph.reset()

    if not args.skip_indexer:
        for width in [65536, 69632, 131072, 1048576]:
            print('indexer', width, flush=True)
            rows = 8
            q = (torch.randn(rows, P.INDEX_HEADS, P.INDEX_DIM, device='cuda') * .2).to(P.FP8_DTYPE)
            k = (torch.randn(width + 32, P.INDEX_DIM, device='cuda') * .2).to(P.FP8_DTYPE)
            weights = torch.randn(rows, P.INDEX_HEADS, device='cuda') * .1
            scales = torch.ones(width + 32, 1, device='cuda')
            starts = torch.tensor([0, 17, 0, 17, 0, 17, 0, 17], device='cuda', dtype=torch.int32)
            lengths = torch.tensor([1, 2047, 2048, 2049, width // 2, width - 17, width, width],
                                   device='cuda', dtype=torch.int32)
            ends = starts + lengths

            def run():
                return P.indexer_select(q, weights, k, scales, starts, ends, max_seqlen=width)

            def expected_logits():
                logits = torch.empty(rows, width, device='cuda')
                selector.indexer_logits_prefill_fwd(q, k.view(-1, 1, P.INDEX_DIM), weights, starts, ends,
                    max_seqlen=width, output=logits, k_scales=scales, is_compressed_logits=True,
                    is_indexer_topk=True)
                return logits

            def expected_indices():
                return reference(expected_logits(), ends - starts, starts, True)[1]

            def checked_run(chunk):
                # Compare the actual key-partitioned producer's score bits with
                # one unsplit library launch, not just its final selected IDs.
                recorded = []
                original = selector.indexer_logits_prefill_fwd

                def record(*a, **kw):
                    result = original(*a, **kw)
                    recorded.append(kw['output'].clone())
                    return result

                selector.indexer_logits_prefill_fwd = record
                try:
                    selected = P.indexer_select(q, weights, k, scales, starts, ends,
                                                max_seqlen=width, chunk=chunk)
                finally:
                    selector.indexer_logits_prefill_fwd = original
                tiles = (width + S.KEY_TILE - 1) // S.KEY_TILE
                parts = [x.view(tiles, -1, S.KEY_TILE).transpose(0, 1).reshape(-1, tiles * S.KEY_TILE)
                         for x in recorded]
                actual_logits = torch.cat(parts)[:, :width]
                expected = expected_logits()
                valid = torch.arange(width, device='cuda')[None, :] < (ends - starts)[:, None]
                assert torch.equal(actual_logits.view(torch.int32)[valid], expected.view(torch.int32)[valid]), 'logit bits differ'
                return selected

            check_bits = args.candidate is None and width > S.SINGLE_BLOCK_MAX_KEYS
            eager = checked_run(None) if check_bits else run()
            assert torch.equal(eager, expected_indices())
            # Exercise workspace reuse and a final partial query chunk.
            chunked = (checked_run(3) if check_bits else
                       P.indexer_select(q, weights, k, scales, starts, ends, max_seqlen=width, chunk=3))
            assert torch.equal(chunked, eager)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run()
            graph.replay()
            assert torch.equal(out, eager)
            q.zero_()  # all keys tie: lowest IDs must win, including the >64K path
            ends.copy_(starts + width)
            graph.replay()
            assert torch.equal(out, expected_indices())
            result['indexer'].append(dict(width=width, exact=True, changed_inputs_exact=True,
                                          producer_score_bits_exact=check_bits))
            save(); graph.reset()
    result['success'] = True
    save(); print('DSA_LONG_CONTEXT_OK', flush=True)


if __name__ == '__main__':
    main()
