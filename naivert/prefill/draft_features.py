"""Kernel prefill that also publishes the draft model's per-position features.

The draft round feeds the draft model the target's residual stream: rank ``r`` wants the
residual *after* target layer ``feature_layers[r]`` (``TARGET_LAYER_IDS``) at every prompt
position -- what the decode path reads as ``rt.resid[layer + 1]``.  In the executor that
residual is ``res_part`` right after layer ``layer + 1``'s S1 (``add_rmsnorm_`` adds the MoE
output into the residual in place), scattered: rank ``q`` holds chunk rows
``[q T/8, (q+1) T/8)``.  ``_all_gather("S1b", ...)`` is the collective that follows that
segment on every layer >= 2, so this subclass gathers the residual there (one more NCCL
all-gather of ``[T, 4096]`` bf16 per feature layer per chunk) and copies the valid rows into
each wanting rank's feature buffer.  Segments and captured graphs are untouched; the extra
collective runs eagerly in the gap like the executor's own.
"""
from __future__ import annotations

import torch

from naivert.kernels.naive_n0p5 import prefill as P
from naivert.prefill import adapters
from naivert.prefill.executor import PrefillExecutor
from naivert.prefill.execution import PrefillError


class DraftFeaturePrefillExecutor(PrefillExecutor):
    def __init__(self, weights, streams, *, feature_layers, features, **kw):
        super().__init__(weights, streams, **kw)
        if len(feature_layers) != self.ranks or len(features) != self.ranks:
            raise PrefillError("one feature layer and one feature buffer per rank")
        for layer in feature_layers:
            if not 1 <= int(layer) <= self.layers - 2:
                raise PrefillError(f"feature layer {layer} has no S1b of layer {layer + 1}")
        self.feature_layers = [int(x) for x in feature_layers]
        self.features = list(features)
        for r, f in enumerate(self.features):
            if f.device != self.devices[r] or f.dtype is not torch.bfloat16 or f.shape[1:] != (P.HIDDEN,):
                raise PrefillError(f"features[{r}] must be bf16 [rows, {P.HIDDEN}] on {self.devices[r]}")
        self._wanted = {}
        for r, layer in enumerate(self.feature_layers):
            self._wanted.setdefault(layer, []).append(r)
        self._scratch = {}
        for b in self.buckets:
            self._scratch[b] = []
            for r, dev in enumerate(self.devices):
                with torch.cuda.device(dev), torch.cuda.stream(self.streams[r]):
                    self._scratch[b].append(torch.empty((b, P.HIDDEN), dtype=torch.bfloat16, device=dev))
        self._chunk = None
        self.feature_rows = 0

    def _set_chunk(self, chunk, tokens):
        super()._set_chunk(chunk, tokens)
        self._chunk = chunk

    def _all_gather(self, stage, layer, name, inputs, outputs):
        super()._all_gather(stage, layer, name, inputs, outputs)
        if stage == "S1b" and (layer - 1) in self._wanted:
            self._publish_features(layer - 1)

    def _publish_features(self, layer: int) -> None:
        chunk = self._chunk
        rows, start, valid = chunk.bucket, chunk.start, chunk.valid
        ws = [self._ws[r][rows] for r in range(self.ranks)]
        scratch = self._scratch[rows]
        self.comm.all_gather_([w.res_part for w in ws], scratch)
        self._stats["collectives"] = self._stats.get("collectives", 0) + 1
        for r in self._wanted[layer]:
            if start + valid > int(self.features[r].shape[0]):
                raise PrefillError(f"features[{r}] holds {int(self.features[r].shape[0])} rows; "
                                   f"chunk ends at {start + valid}")
            with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                self.features[r][start:start + valid].copy_(scratch[r][:valid])
        self.feature_rows = start + valid

    @torch.inference_mode()
    def begin_request(self) -> None:
        """A fresh prompt at position 0: clear the DSA plain-key shadow and its marks.

        The harness zeroes the decode caches itself (``k_cache`` ...); this covers what only
        the executor reads.  Captured graphs stay valid (same tensors, same shapes).
        """
        for r, w in enumerate(self.weights):
            with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                for lw in w.layer:
                    if lw.get("kind") == "full_dsa":
                        lw["ki_plain"].zero_()
                        lw["ki_plain_end"] = 0


    @torch.inference_mode()
    def resume_from_decode(self, position: int) -> int:
        """Decode (seq1 / verify-8) wrote rows up to ``position`` into ``ki_cache`` alone;
        rebuild the plain shadow for them so a later ``run(start=position)`` is admitted.
        Returns the rows rebuilt over all DSA layers of rank 0."""
        rebuilt = 0
        for r, w in enumerate(self.weights):
            with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                for lw in w.layer:
                    if lw.get("kind") == "full_dsa":
                        n = adapters.dsa_index_keys_rebuild(lw, position)
                        rebuilt += n if r == 0 else 0
        return rebuilt


__all__ = ["DraftFeaturePrefillExecutor"]
