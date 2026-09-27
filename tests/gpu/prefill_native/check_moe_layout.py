"""Compare the existing ATen dispatch with SGLang's unchanged masked permutation.

Run through gpu_profile_queue. Verify all bytes of local expert rows, all UE8M0
scale words, and the token/slot return map, including changing routing on replay.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from naivert.kernels.naive_n0p5 import prefill as P


def reference_slots(topk_ids, *, first_expert=0, experts=32):
    """Original atomic-count layout, retained as an independent reference."""
    n, k = topk_ids.shape
    pairs = n * k
    local = topk_ids.to(torch.int64) - first_expert
    owned = (local >= 0) & (local < experts)
    expert = torch.where(owned, local, experts)
    flat = expert.reshape(-1)
    order = torch.arange(pairs, device=topk_ids.device, dtype=torch.int64)
    perm = torch.argsort(flat * pairs + order)
    counts = torch.zeros(experts + 1, device=topk_ids.device, dtype=torch.int64).scatter_add_(
        0, flat, torch.ones_like(flat))
    starts = torch.cumsum(counts, 0) - counts
    rank = order - starts.index_select(0, flat.index_select(0, perm))
    slot = torch.empty_like(flat).scatter_(0, perm, rank)
    slot = torch.where(owned.reshape(-1), slot, 0).view(n, k)
    return P.ExpertSlots(expert, slot, counts[:experts].to(torch.int32), perm)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rows', type=int, default=4096)
    ap.add_argument('--capacity', type=int, default=8192)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(20260920)
    n, k, e, cap, hidden = args.rows, 8, 32, args.capacity, 4096
    assert n > 0 and cap >= n
    packed = torch.randint(0, 256, (n, P.DISPATCH_ROW_BYTES), device='cuda', dtype=torch.uint8)
    fields = P.dispatch_rows(packed)
    # All byte values are deliberate, including FP8 NaNs: permutation is a
    # byte copy, not a numeric conversion.
    codes = fields.codes.view(torch.uint8)
    ids = torch.empty((n, k), device='cuda', dtype=torch.int32)
    token = torch.arange(n, device='cuda').view(n, 1).expand(n, k).reshape(-1)
    variants = ['original', 'candidate']
    maps = {}
    qbuf = {v: torch.empty((e + 1, cap, hidden), device='cuda', dtype=torch.uint8) for v in variants}
    sbuf = {v: P.ue8m0_scale_buffer((e + 1,), cap, hidden, device=torch.device('cuda:0')) for v in variants}

    def run(label):
        slots = (reference_slots if label == 'original' else P.expert_slots)(ids, first_expert=0)
        expert, slot = slots.expert.reshape(-1), slots.slot.reshape(-1)
        dest = expert * cap + slot
        if label == 'original':
            qbuf[label].view(torch.int64).view(-1, hidden // 8).index_copy_(
                0, dest, codes.view(torch.int64).index_select(0, token))
            sbuf[label][expert, slot] = fields.scales.index_select(0, token)
            owned = expert < e
            order = torch.sort(owned.to(torch.int8), descending=True, stable=True).indices
            src = dest.index_select(0, order)
            dst = (token * k + torch.arange(k, device='cuda').repeat(n)).index_select(0, order)
            live = owned.sum().to(torch.int32)
        else:
            dest = P.scatter_to_experts(fields.codes, fields.scales, slots,
                                        qbuf[label].view(P.FP8_DTYPE), sbuf[label])
            dst = slots.permutation
            src = dest.index_select(0, dst)
            live = slots.masked_m.sum(dtype=torch.int32)
        maps[label] = (src, dst, live)
        return slots

    graphs = {}
    for label in variants:
        ids.copy_(torch.rand(n, P.EXPERTS, device='cuda').topk(k).indices)
        run(label); torch.cuda.synchronize()
        graphs[label] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[label]):
            run(label)

    reports = []
    for case in ['uniform', 'hot_local', 'all_local', 'all_foreign', 'one_local']:
        if case == 'uniform':
            routing = torch.rand(n, P.EXPERTS, device='cuda').topk(k).indices
        elif case == 'hot_local':
            routing = torch.cat((torch.zeros(n, 1, device='cuda', dtype=torch.int64),
                                 torch.rand(n, P.EXPERTS - 1, device='cuda').topk(k - 1).indices + 1), 1)
        elif case == 'all_local':
            routing = torch.rand(n, e, device='cuda').topk(k).indices
        elif case == 'all_foreign':
            routing = torch.rand(n, P.EXPERTS - e, device='cuda').topk(k).indices + e
        else:
            routing = torch.rand(n, P.EXPERTS - e, device='cuda').topk(k).indices + e
            routing[0, 0] = 0
        ids.copy_(routing)
        slots = P.expert_slots(ids, first_expert=0)
        alternative = reference_slots(ids, first_expert=0)
        assert all(torch.equal(a, b) for a, b in zip(slots, alternative)), 'sorted counts or slots differ'
        owned = slots.expert < e
        expert, slot = slots.expert[owned], slots.slot[owned]
        source_tokens = torch.arange(n, device='cuda')[:, None].expand(n, k)[owned]
        expected_codes = codes.index_select(0, source_tokens)
        expected_scales = fields.scales.index_select(0, source_tokens)
        samples = {v: [] for v in variants}
        for rep in range(34):
            for label in (variants if rep % 2 == 0 else variants[::-1]):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); graphs[label].replay(); b.record(); b.synchronize()
                if rep >= 4:
                    samples[label].append(a.elapsed_time(b) * 1000)
        for label in variants:
            assert torch.equal(qbuf[label][expert, slot], expected_codes), (case, label, 'codes')
            assert torch.equal(sbuf[label][expert, slot], expected_scales), (case, label, 'scales')
            src, dst, live = maps[label]
            count = int(live)
            assert count == int(owned.sum())
            expected_pairs = torch.nonzero(owned.reshape(-1)).reshape(-1)
            assert torch.equal(dst[:count].sort().values, expected_pairs), (case, label, 'return pairs')
            reference_dest = slots.expert.reshape(-1) * cap + slots.slot.reshape(-1)
            assert torch.equal(src[:count], reference_dest.index_select(0, dst[:count]))
        # Input bytes change with a fixed routing, then routing changes on the
        # next iteration; neither case may rely on old graph-pool contents.
        packed.bitwise_xor_(127)
        for label in variants:
            graphs[label].replay()
        torch.cuda.synchronize()
        for label in variants:
            assert torch.equal(qbuf[label][expert, slot], codes.index_select(0, source_tokens))
            assert torch.equal(sbuf[label][expert, slot], fields.scales.index_select(0, source_tokens))
        result = dict(case=case, pairs=int(owned.sum()), max_expert=int(slots.masked_m.max()),
                      exact=True, median_us={v: statistics.median(s) for v, s in samples.items()})
        reports.append(result)
        print(json.dumps(result), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(rows=n, capacity=cap, cases=reports,
                                     torch=torch.__version__), indent=2) + '\n')


if __name__ == '__main__':
    main()
