"""Measure the complete 4096-query indexer at long spans; use the shared GPU queue."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from naivert.kernels.naive_n0p5 import prefill as P
from naivert.prefill.native import stable_dsa_topk as S


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--widths', default='65536,69632,131072,262144,1048576')
    parser.add_argument('--tiles', default='32768,65536')
    parser.add_argument('--rows', type=int, default=4096)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--phase-events', action='store_true', help='diagnostic events inside the captured graph')
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(20260922)
    P.prepare_device(torch.device('cuda:0'))
    S.install_stable_epilogue()
    if args.candidate:
        spec = importlib.util.spec_from_file_location('dsa_candidate', args.candidate)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        if hasattr(candidate, 'select'):
            S.tiled_stable_topk = candidate.select
        if hasattr(candidate, 'indexer_select'):
            P.indexer_select = candidate.indexer_select
    result = dict(rows=args.rows, measurements=[], versions=P.check_versions(),
                  source_sha256=hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(),
                  scope='One rank, complete logits + selection; random operands, not model latency.')
    if args.candidate:
        result['candidate_sha256'] = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    events = []
    if args.phase_events:
        def timed(label, fn):
            def wrapped(*a, **kw):
                start = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                start.record()
                value = fn(*a, **kw)
                end.record()
                events.append((label, start, end))
                return value
            return wrapped
        module = P._module(P._TOPK)
        module.indexer_logits_prefill_fwd = timed('logits', module.indexer_logits_prefill_fwd)
        S.tiled_stable_topk = timed('selection', S.tiled_stable_topk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for width in map(int, args.widths.split(',')):
        q = (torch.randn(args.rows, 16, 128, device='cuda') * .2).to(P.FP8_DTYPE)
        k = (torch.randn(width, 128, device='cuda') * .2).to(P.FP8_DTYPE)
        weights = torch.randn(args.rows, 16, device='cuda') * .1
        scales = torch.ones(width, 1, device='cuda')
        starts = torch.zeros(args.rows, device='cuda', dtype=torch.int32)
        ends = torch.arange(width - args.rows + 1, width + 1, device='cuda', dtype=torch.int32)
        reference = None
        for tile in map(int, args.tiles.split(',')):
            S.KEY_TILE = tile  # Host schedule experiment; device implementations stay unchanged.
            print('BEGIN', width, tile, flush=True)

            def run():
                return P.indexer_select(q, weights, k, scales, starts, ends, max_seqlen=width)

            eager = run()
            torch.cuda.synchronize()
            if reference is None:
                reference = eager.clone()
            else:
                assert torch.equal(eager, reference), 'tile sizes changed selection'
            gc.collect()
            torch.cuda.empty_cache()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            graph = torch.cuda.CUDAGraph()
            events.clear()
            with torch.cuda.graph(graph):
                output = run()
            graph.replay()
            assert torch.equal(output, reference), 'capture changed selection'
            samples = []
            for rep in range(9):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); graph.replay(); b.record(); b.synchronize()
                if rep >= 3:
                    samples.append(a.elapsed_time(b))
            padded = (width + tile - 1) // tile * tile
            query_chunk = (max(8, P.indexer_chunk_rows(args.rows, padded) // 8 * 8)
                           if width > S.SINGLE_BLOCK_MAX_KEYS else P.indexer_chunk_rows(args.rows, width))
            row = dict(width=width, key_tile=tile, query_chunk=query_chunk,
                       median_ms=statistics.median(samples), samples_ms=samples,
                       graph_extra_reserved_bytes=torch.cuda.memory_reserved() - before,
                       peak_extra_allocated_bytes=torch.cuda.max_memory_allocated() - before,
                       indices_exact=True)
            if args.phase_events:
                row['phase_ms'] = {phase: sum(a.elapsed_time(b) for label, a, b in events if label == phase)
                                   for phase in ['logits', 'selection']}
            result['measurements'].append(row)
            args.out.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(row), flush=True)
            graph.reset()
            del graph, eager, output
        del q, k, weights, scales, starts, ends, reference
        gc.collect(); torch.cuda.empty_cache()
    result['success'] = True
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
