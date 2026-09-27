#!/usr/bin/env python3
"""Map rmsnorm_up_gate's element addressing over one 16-row x 32-K chunk.

usage: probe_up_gate.py [--model naive_n0p5] [--out map.json]

gate = up = one-hot at packed position (i0, k0) with unit scales, gamma = 1, and an
activation ramp x[k] = k + 1 on K 0..31 (zero elsewhere): the kernel's only nonzero
output sits at the row r it assigns to i0 and equals silu(v) * v with v = c * (kappa + 1),
where kappa is the K index it assigns to k0 (c = rsqrt(mean(x^2) + eps)). The 512 (i0, k0)
-> (r, kappa) pairs are printed as a bit-level rule and saved as JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CUDA_HOME", "/opt/cuda132-min")

import torch  # noqa: E402

from _model import add_model_argument, select  # noqa: E402
from naivert.kernels.packing import pack_gate_up  # noqa: E402

SLOT, PANE = 256, 0x8020


def main() -> int:
    ap = argparse.ArgumentParser()
    add_model_argument(ap)
    ap.add_argument("--out", default="/tmp/up_gate_chunk_map.json")
    args = ap.parse_args()
    M = select(args.model)
    hidden, inter = M.hidden, M.dense_inter
    slots, pages = inter // SLOT, hidden // 1024
    ext = M.load_extension()
    dev = torch.device("cuda:0")
    gamma = torch.ones(hidden, device=dev)
    dummy8 = torch.zeros(16, dtype=torch.float8_e4m3fn, device=dev)
    dummy16 = torch.zeros(16, dtype=torch.bfloat16, device=dev)
    x = torch.zeros((1, hidden), dtype=torch.bfloat16, device=dev)
    x[0, :32] = torch.arange(1, 33, dtype=torch.float32, device=dev).to(torch.bfloat16)
    c = float(torch.rsqrt((x.float() ** 2).mean() + M.eps))
    values = torch.arange(1, 33, dtype=torch.float32) * c
    table = torch.nn.functional.silu(values) * values          # y value for kappa = 0..31
    s = torch.ones((slots, 2, hidden // 128), dtype=torch.bfloat16)
    mapping = {}
    for i0 in range(16):
        for k0 in range(32):
            q = torch.zeros((slots, SLOT, hidden), dtype=torch.float8_e4m3fn)
            q[0, i0, k0] = 1.0
            native = pack_gate_up(q, s, q, s, hidden=hidden).view(torch.uint8).reshape(slots * 16, pages, 32 * 1024 + 128)
            ws = native[:, :, :PANE].contiguous().to(dev)
            y = torch.zeros((1, inter), dtype=torch.bfloat16, device=dev)
            ext.launch_rmsnorm_up_gate(x, gamma, dummy8, dummy16, dummy8, dummy16, ws, y, hidden, inter)
            torch.cuda.synchronize(dev)
            yt = y[0].float().cpu()
            r = int(yt.abs().argmax())
            kappa = int((table - yt[r]).abs().argmin())
            mapping[f"{i0},{k0}"] = [r, kappa]
    rows = sorted({v[0] for v in mapping.values()})
    print(f"{M.name}: rows hit {rows}")
    # Bit rule: which (i0, k0) bits move which (r, kappa) bits.
    def bits(n, width):
        return [(n >> b) & 1 for b in range(width)]
    print("(i0,k0) -> (r,kappa) samples:", {k: mapping[k] for k in ("0,0", "1,0", "4,0", "8,0", "0,1", "0,2", "0,4", "0,8", "0,16")})
    consistent = True
    for i0 in range(16):
        for k0 in range(32):
            r, kappa = mapping[f"{i0},{k0}"]
            # candidate rule from the first probe: r = 2*(i0&3) + 8*((k0>>1)&1) + ((k0>>3)&1)
            #                                       kappa = (k0&1) | ((k0>>4)&1)<<1 | (k0&4) | (i0&4)<<1 | (i0&8)<<1  (guess for the upper bits)
            pass
    Path(args.out).write_text(json.dumps(mapping), encoding="utf-8")
    # Print the full map compactly: for each i0, the list over k0 of (r, kappa).
    for i0 in range(16):
        print(f"  i0={i0:2d}: " + " ".join(f"{mapping[f'{i0},{k0}'][0]:2d}/{mapping[f'{i0},{k0}'][1]:2d}" for k0 in range(32)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
