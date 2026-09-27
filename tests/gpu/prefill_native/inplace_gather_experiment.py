"""Gather residual slices directly into the next attention input.

Reuses the existing library collective. The source slice aliases its rank's
destination slice; validate every rank's state and both chunk indices before
adoption. Removes the full residual landing copy and its separate allocation.
"""
from naivert.prefill.native.executor import NativePrefill as Baseline


class NativePrefill(Baseline):
    def __init__(self, *args, **kwargs):
        kwargs['collective_backend'] = 'flashinfer'
        super().__init__(*args, **kwargs)
        for ws in self.ws:
            ws.res_slices = ws.res
        self.versions['inplace-residual-gather-experiment'] = True

    def _gather_land(self, r, L, c):
        if r in self._wanted.get(L, []):
            self._feature_copy(r, c)
