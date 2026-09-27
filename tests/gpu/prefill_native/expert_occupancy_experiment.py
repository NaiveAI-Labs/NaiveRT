"""Record real per-expert row counts without changing prefill arithmetic.

Use the normal numerical harness with --native-candidate and set
NAIVERT_EXPERT_OCCUPANCY_OUT. Eager only; its timings are diagnostic.
"""
import hashlib
import json
import os
from pathlib import Path

from naivert.kernels.naive_n0p5 import prefill as P
from naivert.prefill.native import executor


class NativePrefill(executor.NativePrefill):
    def __init__(self, *args, **kwargs):
        self.audit_path = Path(os.environ['NAIVERT_EXPERT_OCCUPANCY_OUT'])
        if self.audit_path.exists():
            raise ValueError(f'refusing to overwrite {self.audit_path}')
        self.audit_records, self.audit_seen, self.audit_pending = [], set(), {}
        kwargs.update(graphs=False, chunk_graphs=False)
        super().__init__(*args, **kwargs)
        self.audit_sources = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(executor.__file__), Path(P.__file__))
        }

    def _set_prompt(self, tokens, start, valid):
        self.audit_case = dict(start=start, valid=valid,
            tokens_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
        super()._set_prompt(tokens, start, valid)

    def _experts(self, r, L):
        key = (self.audit_case['tokens_sha256'], self.audit_case['start'], L)
        if key in self.audit_seen:
            return super()._experts(r, L)
        original_prepare = P.prepare_compact_experts
        original_gemm = P.grouped_gemm_contiguous
        calls = []

        def observe_prepare(*args, **kwargs):
            prepared = original_prepare(*args, **kwargs)
            self.audit_pending[r] = prepared[0].clone()
            return prepared

        def observe_gemm(a, a_scale, weight, weight_scale, out, expert_ends, expected):
            calls.append(dict(a=list(a.shape), a_stride=list(a.stride()),
                              weight=list(weight.shape), out=list(out.shape),
                              expected_m=expected, layout='compact-psum'))
            return original_gemm(a, a_scale, weight, weight_scale, out, expert_ends, expected)

        P.prepare_compact_experts = observe_prepare
        P.grouped_gemm_contiguous = observe_gemm
        try:
            super()._experts(r, L)
        finally:
            P.prepare_compact_experts = original_prepare
            P.grouped_gemm_contiguous = original_gemm
        self.audit_shapes = calls

    def _gather_land(self, r, L, c):
        super()._gather_land(r, L, c)
        key = (self.audit_case['tokens_sha256'], self.audit_case['start'], L)
        if r != 0 or key in self.audit_seen or not self.audit_pending:
            return
        # All ranks have queued expert return and residual gather before this
        # callback, so these host reads cannot prevent peer participation.
        assert len(self.audit_pending) == self.ranks
        counts = [self.audit_pending[rank].cpu().tolist() for rank in range(self.ranks)]
        assert sum(map(sum, counts)) == executor.T * P.TOPK
        self.audit_records.append(dict(**self.audit_case, layer=L, counts=counts,
                                       gemms=self.audit_shapes))
        self.audit_seen.add(key)
        self.audit_pending.clear()
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.write_text(json.dumps(dict(
            scope='Real model routing; all padded 4096 rows included; eager diagnostic, no timing claims.',
            sources=self.audit_sources, versions=self.versions,
            records=self.audit_records), indent=2) + '\n')
        if L == self.layers - 1:
            print('EXPERT OCCUPANCY', self.audit_case, 'layers recorded', len(self.audit_records), flush=True)
