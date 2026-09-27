#!/usr/bin/env python3
"""Check the head-to-TP8-argmax mailbox handoff for one and eight rows."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from naivert.kernels.naive_n0p5.model_ops import load_extension
from naivert.kernels.packing import pack_head
from naivert.models.naive_n0p5.model import NAIVE_N0P5_GEOMETRY as SPEC


def main():
    ext = load_extension(expert_down="w8a8")
    ext.enable_tp8_peer_access()
    hidden, vocab = SPEC.hidden_size, SPEC.vocab_shard_size
    devices = [torch.device(f"cuda:{r}") for r in range(SPEC.rank_count)]
    gen = torch.Generator().manual_seed(7)
    weight = (torch.randn(vocab, hidden, generator=gen) * 0.02).to(torch.bfloat16)
    packed = pack_head(weight, hidden=hidden)
    weights = [packed.to(d) for d in devices]
    gamma = [torch.ones(hidden, device=d) for d in devices]
    inboxes = [torch.zeros(8 * 128, dtype=torch.uint8, device=d) for d in devices]
    pointers = [x.data_ptr() for x in inboxes]
    # Equal shards also test tie-breaking: the lowest global token index wins.
    for rows in (1, 8):
        x = torch.randn(rows, hidden, generator=gen).to(torch.bfloat16)
        per_rank = []
        for rank, device in enumerate(devices):
            logits = torch.empty(rows, vocab, device=device)
            last = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
            workspace = torch.zeros(rows * 64, dtype=torch.uint8, device=device)
            tokens = torch.full((rows,), -1, dtype=torch.int32, device=device)
            scores = torch.zeros(rows, device=device)
            ext.launch_rmsnorm_head_proj_rows1(x.to(device), gamma[rank], weights[rank], logits,
                                              last, workspace, hidden, vocab, pointers, rank)
            per_rank.append((logits, last, workspace, tokens, scores))
        for device in devices:
            torch.cuda.synchronize(device)
        expected = torch.cat([st[0].cpu() for st in per_rank], dim=1).argmax(dim=1)
        for rank, (logits, _, _, tokens, scores) in enumerate(per_rank):
            ext.launch_tp8_argmax_rows(logits, tokens, scores, inboxes, rank)
        for device in devices:
            torch.cuda.synchronize(device)
        for _, _, _, tokens, scores in per_rank:
            assert torch.equal(tokens.cpu().long(), expected), (rows, tokens, expected)
            assert bool((scores == 1).all()), (rows, scores)
        print(f"head + argmax rows={rows}: PASS on all eight ranks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
