"""Combine tested DeepEP dispatch with an exact-weight, split-K router experiment.

Uses the integrated executor's dispatch. No runtime default is changed.
BF16 inputs stay exact, FP32 weights reconstruct exactly from three BF16 terms,
and partial GEMM outputs, reductions, and routing all remain FP32.
"""
import hashlib
import os
from pathlib import Path

import torch

from naivert.kernels.naive_n0p5 import prefill as P
from naivert.prefill.native import executor as _parent

_parent_path = Path(_parent.__file__)


class NativePrefill(_parent.NativePrefill):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_splits = int(os.environ.get('NAIVERT_ROUTER_K_SPLITS', '16'))
        assert self.router_splits > 0 and P.HIDDEN % self.router_splits == 0
        self.router_weights, self.router_partials, self.router_expanded = [], [], []
        for r, weight in enumerate(self.weights):
            with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                prepared = {}
                for L in range(1, self.layers):
                    remaining, terms = weight.layer[L]['router'].clone(), []
                    for _ in range(3):
                        term = remaining.bfloat16()
                        terms.append(term)
                        remaining = remaining - term.float()
                    if torch.count_nonzero(remaining).item():
                        raise ValueError(f'router weight reconstruction failed at rank {r}, layer {L}')
                    prepared[L] = torch.cat(terms, dim=0).view(
                        3 * P.EXPERTS, self.router_splits, P.HIDDEN // self.router_splits
                    ).permute(1, 0, 2).contiguous().transpose(1, 2)
                self.router_weights.append(prepared)
                self.router_partials.append(torch.empty(
                    (self.router_splits, _parent.SLICE, 3 * P.EXPERTS),
                    device=self.devices[r], dtype=torch.float32))
                self.router_expanded.append(torch.empty(
                    (_parent.SLICE, 3 * P.EXPERTS), device=self.devices[r], dtype=torch.float32))
        self.versions['router-experiment'] = dict(
            k_splits=self.router_splits, weight_reconstruction_exact=True,
            parent_sha256=hashlib.sha256(_parent_path.read_bytes()).hexdigest())
        self.sync()
        print('split router setup ready', self.versions['router-experiment'], flush=True)

    def _router(self, r, L, x, gate, *, out=None):
        assert out is self.ws[r].router
        previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        try:
            torch.bmm(x.view(_parent.SLICE, self.router_splits, P.HIDDEN // self.router_splits).transpose(0, 1),
                      self.router_weights[r][L], out_dtype=torch.float32, out=self.router_partials[r])
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
        expanded = self.router_expanded[r]
        torch.sum(self.router_partials[r], dim=0, out=expanded)
        torch.add(expanded[:, P.EXPERTS:2 * P.EXPERTS], expanded[:, 2 * P.EXPERTS:], out=out)
        out.add_(expanded[:, :P.EXPERTS])
        return out

    def _post(self, r, L):
        # Experimental injection is confined to this sequential host stage.
        # Captured graphs contain only cuBLAS/ATen kernels; dispatch and graph
        # launch worker threads run after this binding has been restored.
        original = P.router_logits
        P.router_logits = lambda x, gate, *, out=None: self._router(r, L, x, gate, out=out)
        try:
            return super()._post(r, L)
        finally:
            P.router_logits = original
