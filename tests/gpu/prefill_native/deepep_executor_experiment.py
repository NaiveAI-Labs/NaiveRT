"""Experimental DeepEP dispatch; use only with the isolated host-adapter build.

Load with prefill_kernel_check.py --native-candidate. No default runtime change.
Reuse baseline attention, GEMMs, quantization, and fixed-slot FP32 mailbox return.
"""
import hashlib
import os
from pathlib import Path

import torch

from naivert.kernels.naive_n0p5 import prefill as P
from naivert.prefill.native.executor import NativePrefill as Baseline, RANKS, SLICE, T


class _DispatchComm:
    def __init__(self, executor, inner):
        self.executor, self.inner = executor, inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def all_gather_(self, inputs, outputs):
        if inputs[0].dtype == torch.uint8:
            self.executor._dispatch()
        else:
            self.inner.all_gather_(inputs, outputs)


class NativePrefill(Baseline):
    def __init__(self, *args, **kwargs):
        # This experiment supplies its own transport; the integrated executor
        # now defaults to DeepEP too.
        kwargs['dispatch_backend'] = 'nccl'
        super().__init__(*args, **kwargs)
        import deep_ep_cpp
        self.deep_sms = int(os.environ.get('NAIVERT_DEEPEP_SMS', '128'))
        self.deep_config = deep_ep_cpp.Config(self.deep_sms, 6, 256, 6, 128)
        nvl_bytes = self.deep_config.get_nvl_buffer_size_hint(P.HIDDEN, RANKS)
        self.deep_buffers = []
        for r in range(RANKS):
            with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                self.deep_buffers.append(deep_ep_cpp.Buffer(r, RANKS, nvl_bytes, 0, False, True))
                self.ws[r].route_i64 = torch.empty((SLICE, P.TOPK), device=self.devices[r], dtype=torch.int64)
                self.ws[r].deep_rows = torch.arange(T, device=self.devices[r], dtype=torch.int32)
                self.ws[r].deep_topk = torch.arange(P.TOPK, device=self.devices[r], dtype=torch.int64)
        for r, buffer in enumerate(self.deep_buffers):
            with torch.cuda.device(self.devices[r]):
                buffer.sync_local(self.deep_buffers)
        self.comm = _DispatchComm(self, self.comm)
        self.versions = dict(self.versions)
        self.versions['deep-ep-experiment'] = dict(
            sms=self.deep_sms, nvl_bytes=nvl_bytes, module=deep_ep_cpp.__file__,
            module_sha256=hashlib.sha256(Path(deep_ep_cpp.__file__).read_bytes()).hexdigest())
        print('DeepEP executor setup ready', self.versions['deep-ep-experiment'], flush=True)

    def _post(self, r, L):
        if L == 0:
            return super()._post(r, L)
        ws, lw = self.ws[r], self.weights[r].layer[L]
        lo = r * SLICE
        if self.reduce_scatter_o:
            ws.res[lo:lo + SLICE].add_(ws.partial_slice)
        else:
            ws.res.add_(ws.partial)
        P.rmsnorm(ws.res[lo:lo + SLICE], lw['ffn_gamma_bf16'], out=ws.x_slice)
        P.router_logits(ws.x_slice, lw['router'], out=ws.router)
        wts, ids = P.route(ws.router, lw['router_bias'])
        ws.route_w.copy_(wts)
        ws.route_ids.copy_(ids)
        ws.route_i64.copy_(ids)
        P.quantize_dispatch_rows(ws.x_slice, ws.q_rows, ws.s_rows)

    def _dispatch_one(self, r):
        with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
            ws, buffer = self.ws[r], self.deep_buffers[r]
            per_rank, _, per_expert, in_rank, _ = buffer.get_dispatch_layout(
                ws.route_i64, P.EXPERTS, None, False, False)
            ws.deep_recv = buffer.intranode_dispatch(
                ws.q_rows, ws.s_rows, ws.route_i64, ws.route_w,
                per_rank, in_rank, per_expert, 0, None, None, 1, T,
                self.deep_config, None, False, False)

    def _dispatch(self):
        with torch.cuda.device(self.devices[0]), torch.cuda.stream(self.streams[0]):
            capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            for r in range(RANKS):
                self._dispatch_one(r)
        else:
            # The experimental binding releases the GIL during dispatch so all
            # ranks can enqueue their collective before an allocation waits.
            list(self._launchers.map(self._dispatch_one, range(RANKS)))

    def _experts(self, r, L):
        e = P.EXPERTS_PER_RANK
        ws, lw = self.ws[r], self.weights[r].layer[L]
        codes, scales, ids = ws.deep_recv[:3]
        slots = P.expert_slots(ids, first_expert=0)
        dest = P.scatter_to_experts(codes, scales, slots, ws.q_buf, ws.s_buf)
        (w13, w13_s), (w2, w2_s) = lw['ep_w13'], lw['ep_w2']
        P.grouped_gemm(ws.q_buf[:e], ws.s_buf[:e], w13, w13_s, ws.up, slots.masked_m, ws.expected)
        P.act_quant_masked(ws.up, ws.q_act, ws.s_act, slots.masked_m)
        P.grouped_gemm(ws.q_act, ws.s_act, w2, w2_s, ws.dn[:e], slots.masked_m, ws.expected)
        # DeepEP groups token rows by their source rank. Convert its local source
        # index back to our global token*topk+slot mailbox address. Unused receive
        # rows have all expert IDs -1 and never enter the live publication range.
        prefix, source = ws.deep_recv[5], ws.deep_recv[8]
        source_rank = torch.searchsorted(prefix[:, r].contiguous(), ws.deep_rows, right=True)
        global_token = source.to(torch.int64) + source_rank.to(torch.int64) * SLICE
        pair_ids = (global_token[:, None] * P.TOPK + ws.deep_topk[None, :]).reshape(-1)
        src = dest.index_select(0, slots.permutation)
        dst = pair_ids.index_select(0, slots.permutation)
        ws.live.copy_(slots.masked_m.sum(dtype=torch.int32))
        ws.owned_total.add_(ws.live.to(torch.int64))
        torch.maximum(ws.max_masked, slots.masked_m.max().reshape(1), out=ws.max_masked)
        ws.epoch.add_(1)
        self.combine.run(ws.dn.view(-1, P.HIDDEN), src, dst, [x.mailbox for x in self.ws], r,
                       ws.route_i64, ws.route_w, ws.moe_out, ws.epoch, ws.error, rows_dev=ws.live)
        lo = r * SLICE
        ws.res[lo:lo + SLICE].add_(ws.moe_out)
        ws.res_slice = ws.res[lo:lo + SLICE]

    def close(self):
        if self._closed:
            return
        self.sync()
        if self.deep_buffers:
            list(self._launchers.map(lambda r: self.deep_buffers[r].destroy(), range(RANKS)))
            self.deep_buffers.clear()
        super().close()
        print('DeepEP executor cleanup complete', flush=True)
