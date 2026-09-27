"""Build the synthetic TileMailbox all-reduce for one hidden width and run it on 8 GPUs.

usage: run_tile_mailbox_allreduce.py --hidden 4096 [--hidden 7168 ...]
Each rank contributes a random bf16 vector; every rank must produce the rank-order fp32
sum rounded once to bf16, bit-exact against a host reference, for two consecutive
generations (both mailbox phases).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from naivert.kernels.toolchain import SM_COUNT, scoped_cuda_toolchain  # noqa: E402

CUDA_HOME = os.environ.get("NAIVERT_CUDA_HOME", "/opt/cuda132-min")
RANKS = 8


def build(hidden: int):
    import torch
    from torch.utils.cpp_extension import load

    build_dir = ROOT / "tests/build/tile_mailbox_allreduce" / f"h{hidden}"
    build_dir.mkdir(parents=True, exist_ok=True)
    with scoped_cuda_toolchain(CUDA_HOME, suppress_arch_flags=True):
        return load(
            name=f"naivert_tile_mailbox_allreduce_h{hidden}",
            sources=[str(ROOT / "tests/gpu/tile_mailbox_allreduce.cu"),
                     str(ROOT / "tests/gpu/tile_mailbox_allreduce_bindings.cpp")],
            extra_include_paths=[str(ROOT)],
            extra_cflags=["-O2", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                               "-gencode=arch=compute_103a,code=sm_103a",
                               f"-DNAIVERT_TP_RANKS={RANKS}", f"-DNAIVERT_HIDDEN={hidden}",
                               f"-DNAIVERT_SM_COUNT={SM_COUNT}"],
            build_directory=str(build_dir),
            verbose=False,
        )


def run(hidden: int) -> bool:
    import torch

    ext = build(hidden)
    ext.enable_peer_access(RANKS)
    bytes_per_rank = ext.mailbox_bytes_per_rank()
    mailboxes = [torch.zeros(bytes_per_rank, dtype=torch.uint8, device=f"cuda:{r}") for r in range(RANKS)]
    bases = [m.data_ptr() for m in mailboxes]
    peers = [torch.tensor(bases, dtype=torch.int64, device=f"cuda:{r}") for r in range(RANKS)]
    torch.manual_seed(7)
    ok = True
    for step, generation in enumerate((0x103, 0x203, 0x303)):
        inputs = [(torch.randn(hidden, device=f"cuda:{r}") * 3).to(torch.bfloat16) for r in range(RANKS)]
        outputs = [torch.empty(hidden, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(RANKS)]
        for r in range(RANKS):
            ext.tile_allreduce(inputs[r], outputs[r], peers[r], r, generation)
        for r in range(RANKS):
            torch.cuda.synchronize(r)
        acc = torch.zeros(hidden, dtype=torch.float32)
        for r in range(RANKS):
            acc = acc + inputs[r].float().cpu()
        reference = acc.to(torch.bfloat16)
        for r in range(RANKS):
            got = outputs[r].cpu()
            mismatches = int((got.view(torch.int16) != reference.view(torch.int16)).sum())
            if mismatches:
                ok = False
                bad = torch.nonzero(got.view(torch.int16) != reference.view(torch.int16)).flatten()[:5].tolist()
                print(f"  hidden {hidden} gen {generation:#x} rank {r}: {mismatches} mismatches, first at {bad}", flush=True)
        print(f"  hidden {hidden} gen {generation:#x}: tiles {ext.tile_count()}, all ranks exact = {ok}", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, action="append", required=True)
    a = ap.parse_args()
    results = {h: run(h) for h in a.hidden}
    print("RESULT", results, flush=True)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
