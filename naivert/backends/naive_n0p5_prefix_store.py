"""Park and restore one session's decoded prefix, keyed by uuid.

One sequence is active at a time. The working caches, graphs and kernels never
change; switching sessions is a row copy. A parked prefix holds, per rank, the
first ``position`` rows of every full-attention layer's caches (K, V, index
key, index scale), the sliding-window ring, the draft's KV rows, and the token
list whose KV those rows are. ``position`` counts processed tokens: the last
sampled token has no KV row yet and is the first token the next turn prefills.
Saved prompt checkpoints retain earlier ring states; their full-attention and
draft rows are slices of the same finished prefix, without another full copy.
"""
import time

import torch

from naivert.models.naive_n0p5.model import SLIDING_WINDOW

CACHE_NAMES = ('k_cache', 'v_cache', 'ki_cache', 'ki_scale')


class PrefixStore:
    def __init__(self, weights, buffers, *, host=False):
        if len(weights) != 8 or len(buffers) != 8:
            raise ValueError('prefix store expects eight rank weight sets and draft buffers')
        self.weights, self.buffers, self.host = weights, buffers, bool(host)
        self.entries = {}

    def _park(self, x):
        if not self.host:
            return x.clone()
        y = torch.empty(x.shape, dtype=x.dtype, device='cpu', pin_memory=True)
        y.copy_(x, non_blocking=True)
        return y

    def clear(self, ranks):
        """Empty the working caches and position for a fresh session."""
        for r, rt in enumerate(ranks):
            with torch.cuda.device(r), torch.cuda.stream(rt.stream):
                rt.pos.zero_()
                for lw in self.weights[r].layer:
                    for n in CACHE_NAMES:
                        if n in lw:
                            lw[n].zero_()
                b = self.buffers[r]
                b.kv_k.zero_()
                b.kv_v.zero_()

    def checkpoint(self, ranks, *, position):
        """Snapshot the ring at an actual execution boundary, before later rows overwrite it.

        Full-attention and draft rows are append-only; save() supplies those from
        the finished sequence. Only the sliding-window state needs an early copy.
        """
        if type(position) is not int or position < 1:
            raise ValueError('checkpoint position must be positive')
        rings, nbytes = [], 0
        for r, rt in enumerate(ranks):
            with torch.cuda.device(r), torch.cuda.stream(rt.stream):
                layers = {}
                for L, lw in enumerate(self.weights[r].layer):
                    if lw['kind'] == SLIDING_WINDOW:
                        layers[L] = {n: self._park(lw[n]) for n in CACHE_NAMES if n in lw}
                        nbytes += sum(x.numel() * x.element_size() for x in layers[L].values())
                rings.append(layers)
        for rt in ranks:
            rt.stream.synchronize()
        return dict(position=position, rings=rings, bytes=nbytes)

    def save(self, uuid, ranks, *, position, tokens, checkpoints=None):
        """Park the working prefix [0, position) under ``uuid``; returns seconds."""
        if type(position) is not int or not 1 <= position <= len(tokens):
            raise ValueError('position must count processed tokens within the token list')
        checkpoints = dict(checkpoints or {})
        if any(type(p) is not int or not 1 <= p <= position or c['position'] != p
               for p, c in checkpoints.items()):
            raise ValueError('checkpoint must belong to the saved prefix')
        start = time.monotonic()
        per_rank, nbytes = [], 0
        for r, rt in enumerate(ranks):
            with torch.cuda.device(r), torch.cuda.stream(rt.stream):
                layers = []
                for lw in self.weights[r].layer:
                    kept = {}
                    for n in CACHE_NAMES:
                        if n in lw:
                            rows = min(lw[n].shape[0], position)   # ring layers keep every slot
                            kept[n] = self._park(lw[n][:rows])
                            nbytes += kept[n].numel() * kept[n].element_size()
                    layers.append(kept)
                b = self.buffers[r]
                draft = (self._park(b.kv_k[:, :position]), self._park(b.kv_v[:, :position]))
                nbytes += sum(t.numel() * t.element_size() for t in draft)
                per_rank.append((layers, draft))
        for rt in ranks:
            rt.stream.synchronize()
        self.entries[uuid] = dict(position=position, tokens=list(tokens), ranks=per_rank,
                                  checkpoints=checkpoints,
                                  bytes=nbytes + sum(c['bytes'] for c in checkpoints.values()),
                                  saved_at=time.monotonic())
        return time.monotonic() - start

    def restore(self, uuid, ranks, *, position=None):
        """Load ``uuid``'s prefix into the working caches; returns (position, tokens, seconds)."""
        e = self.entries[uuid]
        position = e['position'] if position is None else position
        if type(position) is not int or not 1 <= position <= e['position']:
            raise ValueError('restore position is outside the saved prefix')
        checkpoint = None
        if position != e['position']:
            checkpoint = e.get('checkpoints', {}).get(position)
            if checkpoint is None:
                raise ValueError('cannot rewind without the sliding-window checkpoint')
        start = time.monotonic()
        self.clear(ranks)
        for r, rt in enumerate(ranks):
            layers, (dk, dv) = e['ranks'][r]
            with torch.cuda.device(r), torch.cuda.stream(rt.stream):
                for L, (lw, kept) in enumerate(zip(self.weights[r].layer, layers)):
                    if checkpoint is not None and lw['kind'] == SLIDING_WINDOW:
                        kept = checkpoint['rings'][r][L]
                    for n, x in kept.items():
                        if lw['kind'] != SLIDING_WINDOW:
                            x = x[:position]
                        lw[n][:x.shape[0]].copy_(x, non_blocking=True)
                b = self.buffers[r]
                b.kv_k[:, :position].copy_(dk[:, :position], non_blocking=True)
                b.kv_v[:, :position].copy_(dv[:, :position], non_blocking=True)
                rt.pos.fill_(position)
        for rt in ranks:
            rt.stream.synchronize()
        e['restored_at'] = time.monotonic()
        return position, list(e['tokens']), time.monotonic() - start

    def evict(self, uuid):
        return self.entries.pop(uuid, None) is not None

    def __contains__(self, uuid):
        return uuid in self.entries

    def bytes(self, uuid=None):
        if uuid is not None:
            return self.entries[uuid]['bytes']
        return sum(e['bytes'] for e in self.entries.values())
