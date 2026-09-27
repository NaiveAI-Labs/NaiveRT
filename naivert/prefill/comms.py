"""The tensor-parallel reduction of the external prefill path: NCCL over the backend's ranks.

Every attention layer and every FFN leaves each rank a partial ``[rows, hidden]`` sum --
o_proj over the rank's 8 heads, the dense or expert down over its 1/8 of the intermediate
rows -- and the next step needs the whole sum on every rank.  Decode reduces one row through
its mailbox (``hid_a = resid + allreduce(o_proj)``); at a chunk's row counts that protocol's
traffic is unaffordable (plan section 3), so prefill uses NCCL.

Three departures from the historical plan's text were measured by the archived
communication microbenchmark (see ``profile/prefill-script-cleanup-20260915/before``):

* ``torch.cuda.nccl``, not ``torch.distributed``.  One process owns all eight devices
  (``backends/naive_n0p5.py``), and ``torch.distributed`` wants a process per rank.
  ``torch.cuda.nccl`` builds one clique communicator over a device list (``ncclCommInitAll``)
  and issues each collective as a grouped launch across the ranks' streams.
* One bf16 all-reduce per reduction point, not a reduce-scatter and an all-gather.  Every
  call from this one thread pays NCCL's host cost for all eight ranks, so two calls cost
  about twice one; the all-reduce is the faster of the two at every bucket.
* The collectives are launched eagerly, between the per-rank graphs: NCCL's documentation
  warns that capturing collectives with several GPUs per process "may result in deadlocks",
  because ``cudaGraphLaunch`` can block.

The reduced tensor is bit-identical on every rank -- NCCL reduces each element once and
copies the result out -- so the residual stream stays replicated, as decode's does.  A bf16
all-reduce rounds after each of its seven additions where decode sums in fp32 and rounds
once; the bench bounds the difference.

The SGLang-kernel prefill keeps the reference's
data flow instead: its MoE layers run SCATTERED, 1/8 of the rows on each rank, so the step
before them is a reduce-scatter and the step after the last one an all-gather
(``reduce_scatter_`` / ``all_gather_``), both launched the same way as the all-reduce.  The
MoE dispatch between them is an all-gather too: every rank's quantised rows, scales and
routing packed into one uint8 row each (``prefill.pack_dispatch_rows``), so a single launch
moves them all.
"""
from __future__ import annotations

from typing import Sequence
from contextlib import ExitStack

import torch
import torch.cuda.nccl as nccl

#: the dtypes the reductions accept; NCCL adds in the tensors' own dtype
DTYPES = (torch.bfloat16, torch.float32)
#: an all-gather only moves bytes, so it also takes packed uint8 rows
GATHER_DTYPES = DTYPES + (torch.uint8,)


class CommError(RuntimeError):
    """The collective was handed tensors it cannot reduce."""


class TP8Comm:
    """In-place sum over one tensor per rank, on the ranks' own streams."""

    def __init__(self, devices: Sequence[torch.device], streams: Sequence[torch.cuda.Stream]) -> None:
        self.devices = [torch.device(d) for d in devices]
        self.streams = list(streams)
        if len(self.devices) != len(self.streams) or not self.devices:
            raise CommError(f"{len(self.devices)} devices and {len(self.streams)} streams")
        if len({d.index for d in self.devices}) != len(self.devices):
            raise CommError(f"devices repeat: {self.devices}")
        for d, s in zip(self.devices, self.streams):
            if s.device != d:
                raise CommError(f"stream {s} is not on {d}")

    def _check(self, tensors: Sequence[torch.Tensor], dtypes: tuple = DTYPES) -> None:
        if len(tensors) != len(self.devices):
            raise CommError(f"{len(tensors)} tensors for {len(self.devices)} ranks")
        shape, dtype = tuple(tensors[0].shape), tensors[0].dtype
        if dtype not in dtypes:
            raise CommError(f"takes {dtypes}, got {dtype}")
        for t, d in zip(tensors, self.devices):
            if t.device != d:
                raise CommError(f"a partial is on {t.device}, its rank is {d}")
            if tuple(t.shape) != shape or t.dtype is not dtype:
                raise CommError(f"partials disagree: {tuple(t.shape)} {t.dtype} vs {shape} {dtype}")
            if not t.is_contiguous():
                raise CommError("NCCL reduces contiguous tensors only")

    def warmup(self) -> None:
        """Create the communicator (``ncclCommInitAll``) outside anything timed."""
        probe = []
        for d, s in zip(self.devices, self.streams):
            with torch.cuda.device(d), torch.cuda.stream(s):
                probe.append(torch.ones(256, dtype=torch.bfloat16, device=d))
        nccl.all_reduce(probe, streams=self.streams)
        for s in self.streams:
            s.synchronize()
        got = float(probe[0][0].item())
        if got != float(len(self.devices)):
            raise CommError(f"warm-up all-reduce of ones gave {got}, expected {len(self.devices)}")

    def all_reduce_(self, tensors: Sequence[torch.Tensor]) -> None:
        """Replace every rank's partial with the sum over ranks.  Asynchronous, like a kernel."""
        self._check(tensors)
        nccl.all_reduce(list(tensors), streams=self.streams)

    def _check_split(self, whole: Sequence[torch.Tensor], parts: Sequence[torch.Tensor],
                     dtypes: tuple = DTYPES) -> None:
        """``whole`` rows are ``parts`` rows times the rank count, rank ``r`` owning slice ``r``."""
        self._check(whole, dtypes)
        self._check(parts, dtypes)
        w, p = tuple(whole[0].shape), tuple(parts[0].shape)
        if (whole[0].dtype is not parts[0].dtype or not w or not p or w[1:] != p[1:]
                or w[0] != p[0] * len(self.devices)):
            raise CommError(f"{w} {whole[0].dtype} does not split into {len(self.devices)} x "
                            f"{p} {parts[0].dtype}")

    def reduce_scatter_(self, inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor]) -> None:
        """``outputs[r]`` = row slice ``r`` of the sum over ranks of ``inputs``.  Asynchronous."""
        self._check_split(inputs, outputs)
        nccl.reduce_scatter(list(inputs), list(outputs), op=nccl.SUM, streams=self.streams)

    def all_gather_(self, inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor]) -> None:
        """``outputs[r]`` = every rank's ``inputs`` concatenated in rank order.  Asynchronous.

        bf16 / fp32 rows (S1b, F1) or packed uint8 dispatch rows (X3).
        """
        self._check_split(outputs, inputs, GATHER_DTYPES)
        nccl.all_gather(list(inputs), list(outputs), streams=self.streams)

    def ordered_sum_(self, tensors: Sequence[torch.Tensor]) -> None:
        """FP32 rank-ordered sum, broadcast without intermediate BF16 rounding.

        NCCL's reduction tree is not the rank 0..7 arithmetic order. Peer
        copies here honor every producer's stream; all eight current streams
        remain installed until the broadcast has been enqueued. Input shape can
        include the selected-expert dimension and must retain it for MoE.
        """
        self._check(tensors)
        if tensors[0].dtype != torch.float32:
            raise CommError("Naive N0.5 TP completion requires FP32 partials")
        with ExitStack() as contexts:
            for stream in self.streams:
                contexts.enter_context(torch.cuda.stream(stream))
            with torch.cuda.device(self.devices[0]):
                total = tensors[0].clone()
                for partial in tensors[1:]:
                    total.add_(partial.to(self.devices[0], non_blocking=True))
                tensors[0].copy_(total)
                for partial in tensors[1:]:
                    partial.copy_(total, non_blocking=True)


__all__ = ["CommError", "DTYPES", "GATHER_DTYPES", "TP8Comm"]
