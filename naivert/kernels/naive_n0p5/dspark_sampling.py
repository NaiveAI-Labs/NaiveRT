"""TP8-sharded top-p and classic p/q sampling; no full-vocabulary gather."""
from functools import lru_cache
from pathlib import Path
import os


def temperature_flags():
    """Compile-time temperature fallback for callers without a per-request parameter block."""
    value = os.environ.get("NAIVERT_DSPARK_INV_TEMPERATURE")
    if not value:
        return (), ""
    inv = float(value)
    if not 0 < inv < 1000:
        raise ValueError("NAIVERT_DSPARK_INV_TEMPERATURE out of range")
    return (f"-DDSPARK_INV_TEMPERATURE={inv!r}f",), "_invt" + value.replace(".", "p").replace("-", "m")

import torch

from naivert.kernels.build import KernelBuildManifest, compile_extension, select_cuda_home

ROOT = Path(__file__).resolve().parents[3]
SHARD = 19072


@lru_cache(maxsize=2)
def load_sampling(persistent=False):
    cuda = select_cuda_home(
        (os.environ.get("NAIVERT_CUDA_HOME"), os.environ.get("CUDA_HOME"), "/opt/cuda132-min"),
        fallback="/opt/cuda132-min")
    tflags, tsuffix = temperature_flags()
    # NAIVERT_DSPARK_SAMPLER_STAMPS=1: per-CTA %globaltimer phase stamps in the persistent kernel,
    # read back with ext.stamps(device) (profiling builds only; the kernel is otherwise unchanged).
    stamps = persistent and os.environ.get("NAIVERT_DSPARK_SAMPLER_STAMPS") == "1"
    suffix=("_persistent" if persistent else "") + tsuffix + ("_stamps" if stamps else "")
    flags=(("-DNAIVERT_DSPARK_PERSISTENT=1", "-DPERSISTENT_GUMBEL=0", "-DPERSISTENT_READY=0")
           if persistent else ()) + tflags + (("-DPERSISTENT_STAMPS=1",) if stamps else ())
    return compile_extension(KernelBuildManifest(
        module_name="naivert_dspark_sampling_tp8"+suffix,
        sources=(ROOT / "naivert/entry/dspark_sampling_entry.cu",),
        include_paths=(ROOT,), cflags=("-O2", "-std=c++17"),
        cuda_cflags=("-O3", "-std=c++17", "--expt-relaxed-constexpr",
                     "-gencode=arch=compute_103a,code=sm_103a",*flags),
        ldflags=(), build_directory=ROOT / ("tests/build/dspark_sampling_tp8"+suffix),
        cuda_home=cuda, required_symbols=("sample", "uniforms", "scratch_size", "mailbox_size"),
        suppress_arch_flags=True))


class TargetSampler:
    """One instance owns all eight ranks' buffers for one fixed row count.

    `full` and `draft_full` retain the caller's diagnostic names but hold only
    local vocabulary shards. Any full-vocabulary diagnostic gather is outside
    the production graph. All ranks must invoke this instance in the same order.
    """
    def __init__(self, logits, buffers, outputs, *, anchor=False,persistent=False):
        if len(logits) != 8 or len(buffers) != 8 or len(outputs) != 8:
            raise ValueError("sampling requires TP8")
        rows = 1 if anchor else 8
        for r, (x, b, out) in enumerate(zip(logits, buffers, outputs)):
            if (x.shape != (rows, SHARD) or x.dtype != torch.float32 or
                    not x.is_contiguous() or x.device != b.device or
                    out.shape != (rows,) or out.dtype != torch.int32 or out.device != x.device):
                raise ValueError(f"rank {r}: incompatible target sampling buffers")
            if not anchor and (b.logit_trace.shape != (7, SHARD) or
                               b.logit_trace.dtype != torch.float32 or
                               b.logit_trace.device != x.device or
                               not b.logit_trace.is_contiguous()):
                raise ValueError(f"rank {r}: incompatible corrected draft logits")
        self.persistent=bool(persistent)
        self.ext = load_sampling(self.persistent)
        self.anchor, self.buffers, self.outputs = anchor, buffers, outputs
        self.logits = logits
        self.draft_logits = ([torch.empty(0, dtype=torch.float32, device=x.device) for x in logits]
                             if anchor else [b.logit_trace for b in buffers])
        # 16 words per rank: ready, consumed, generation, eight inbox words, pad (sync.cuh kRendezvousWords).
        self.flags = [torch.zeros(16, dtype=torch.int64, device=x.device) for x in logits]
        self.full = [torch.empty(rows, SHARD, dtype=torch.float32, device=x.device) for x in logits]
        self.draft_full = [torch.empty(0 if anchor else 7 * SHARD, dtype=torch.float32,
                                       device=x.device) for x in logits]
        # The persistent row-arrival counters must start at zero and then stay
        # owned by the kernel. Request reset must not rewind its mailbox epoch.
        self.scratch = [torch.zeros(self.ext.scratch_size(), dtype=torch.uint8, device=x.device)
                        for x in logits]
        self.mailboxes = [torch.zeros(self.ext.mailbox_size(), dtype=torch.uint8, device=x.device)
                          for x in logits]
        self.flag_ptrs = [x.data_ptr() for x in self.flags]
        self.mailbox_ptrs = [x.data_ptr() for x in self.mailboxes]

    def run(self, rank, *, defer_consumed=False):
        b = self.buffers[rank]
        self.ext.sample(self.logits[rank], self.draft_logits[rank], self.flag_ptrs,
                        self.mailbox_ptrs, self.full[rank], self.draft_full[rank],
                        self.outputs[rank], b.draft_tokens, b.sampling_state,
                        self.scratch[rank], b.wdog, b.rep_count, b.sampling_params, rank, self.anchor, defer_consumed)
