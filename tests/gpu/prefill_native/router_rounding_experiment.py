"""Trace router-weight differences through the existing BF16 MoE boundary.

Eager diagnostic on one prompt of at most 512 tokens, with baseline FP32 routing
feeding the model. Reconstruct unweighted BF16 expert outputs from the completed
tagged mailbox, then change only route weights in a counterfactual combine.
Set NAIVERT_ROUTER_ROUNDING_OUT and NAIVERT_ROUTER_K_SPLITS (default 32).
No timings from this instrumented run are performance measurements.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch

from naivert.kernels.naive_n0p5 import prefill as P
from naivert.prefill.native import executor

_path = Path(__file__).with_name('router_executor_experiment.py')
_spec = importlib.util.spec_from_file_location('_router_rounding_parent', _path)
_parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parent)


def changed_elements(a, b):
    return int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum())


class NativePrefill(_parent.NativePrefill):
    def __init__(self, *args, **kwargs):
        os.environ.setdefault('NAIVERT_ROUTER_K_SPLITS', '32')
        self.audit_out = Path(os.environ['NAIVERT_ROUTER_ROUNDING_OUT'])
        if self.audit_out.exists():
            raise ValueError(f'audit output already exists: {self.audit_out}')
        self.original_router = P.router_logits
        self.pending, self.audited, self.audit_rows = {}, set(), []
        kwargs.update(graphs=False, chunk_graphs=False)
        super().__init__(*args, **kwargs)
        self.versions['router-rounding-audit'] = dict(
            feed='baseline FP32', splits=self.router_splits,
            scope='same-input routers and same-expert-output combine; eager diagnostic',
            combine_source_sha256=hashlib.sha256(
                Path(executor.__file__).with_name('native_combine.cu').read_bytes()).hexdigest())

    def _router(self, r, L, x, gate, *, out=None):
        self.original_router(x, gate, out=out)
        if r != 0 or L in self.audited:
            return out
        ws = self.ws[r]
        rows, start = int(ws.valid.item()), int(ws.start.item())
        assert start == 0 and 0 < rows <= executor.SLICE, 'diagnostic supports one short prompt'
        baseline = out.clone()
        super()._router(r, L, x, gate, out=out)
        candidate = out.clone()
        truth = x.double() @ gate.double().T
        logits = dict(baseline=baseline, candidate=candidate, oracle=truth.float())
        routes = {name: P.route(value, self.weights[r].layer[L]['router_bias'])
                  for name, value in logits.items()}
        errors = {name: dict(rms=float((value[:rows].double()-truth[:rows]).square().mean().sqrt()),
                             max_abs=float((value[:rows].double()-truth[:rows]).abs().max()))
                  for name, value in logits.items()}
        self.pending[L] = dict(rows=rows, routes=routes, errors=errors)
        out.copy_(baseline)
        return out

    def _post(self, r, L):
        super()._post(r, L)
        if r == 0 and L in self.pending:
            rows = self.pending[L]['rows']
            self.pending[L]['res_before'] = self.ws[r].res[:rows].clone()

    def _gather_land(self, r, L, c):
        super()._gather_land(r, L, c)
        if r != 0 or L not in self.pending:
            return
        state = self.pending.pop(L)
        ws, rows = self.ws[0], state['rows']
        # Every rank's expert and residual-gather kernels have already been
        # queued before this callback, so reading the completed epoch cannot
        # prevent a peer from entering the same return collective.
        epoch = int(ws.epoch.item())
        bank_words = executor.SLICE * P.TOPK * (P.HIDDEN // 2)
        start = 16 + (epoch & 1) * bank_words
        words = ws.mailbox.view(torch.int64)[start:start+rows*P.TOPK*(P.HIDDEN//2)]
        assert bool(((words >> 32) == epoch).all()), 'mailbox epoch differs'
        experts = words.to(torch.int32).view(torch.bfloat16).view(rows, P.TOPK, P.HIDDEN)
        base_w, base_ids = (t[:rows] for t in state['routes']['baseline'])
        counters, combined, residuals = {}, {}, {}
        for name, (weights, ids) in state['routes'].items():
            weights, ids = weights[:rows], ids[:rows]
            match = base_ids[:, :, None] == ids[:, None, :]
            same_set = match.any(-1).all(-1)
            order = match.to(torch.int32).argmax(-1)
            aligned = weights.gather(1, order)
            acc = torch.zeros((rows, P.HIDDEN), device=ws.dev, dtype=torch.float32)
            # Match complete(): rounded FP32 multiply followed by rounded add,
            # in baseline's fixed top-8 slot order, then one BF16 conversion.
            for slot in range(P.TOPK):
                acc.add_(experts[:, slot].float() * aligned[:, slot, None])
            combined[name] = acc.bfloat16()
            residuals[name] = state['res_before'] + combined[name]
            counters[name] = dict(
                changed_expert_set_rows=int((~same_set).sum()),
                changed_order_rows=int((ids != base_ids).any(-1).sum()),
                max_aligned_route_weight_diff=float((aligned[same_set]-base_w[same_set]).abs().max()) if same_set.any() else None,
                counterfactual_scope='rows with unchanged expert set, in baseline slot order')
            if name == 'baseline':
                assert changed_elements(combined[name], ws.moe_out[:rows]) == 0, 'combine reconstruction differs'
                assert changed_elements(residuals[name], ws.res[:rows]) == 0, 'residual reconstruction differs'
                base_acc = acc
            else:
                live = same_set
                counters[name].update(
                    compared_rows=int(live.sum()),
                    fp32_combine_max_abs_diff=float((acc[live]-base_acc[live]).abs().max()) if live.any() else None,
                    bf16_moe_changed_elements=changed_elements(combined[name][live], combined['baseline'][live]),
                    bf16_moe_changed_rows=int((combined[name][live] != combined['baseline'][live]).any(-1).sum()),
                    bf16_residual_changed_elements=changed_elements(residuals[name][live], residuals['baseline'][live]),
                    bf16_residual_changed_rows=int((residuals[name][live] != residuals['baseline'][live]).any(-1).sum()))
                if L + 1 < self.layers:
                    gamma = self.weights[0].layer[L+1]['attn_gamma_bf16']
                    reference = P.rmsnorm(residuals['baseline'], gamma)
                    alternative = P.rmsnorm(residuals[name], gamma)
                    counters[name]['next_attn_norm_changed_elements'] = changed_elements(alternative[live], reference[live])
        report = dict(layer=L, rows=rows, elements=rows*P.HIDDEN,
                      router_error_vs_fp64=state['errors'], counters=counters,
                      baseline_combine_bytes_exact=True, baseline_residual_bytes_exact=True)
        self.audit_rows.append(report)
        self.audited.add(L)
        self._write_audit()
        print('ROUTER ROUNDING', L, json.dumps(counters['candidate']), flush=True)

    def _write_audit(self):
        self.audit_out.parent.mkdir(parents=True, exist_ok=True)
        self.audit_out.write_text(json.dumps(dict(versions=self.versions,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            cases=self.audit_rows), indent=2)+'\n')

    def close(self):
        if self._closed:
            return
        self.sync()
        self._write_audit()
        super().close()
