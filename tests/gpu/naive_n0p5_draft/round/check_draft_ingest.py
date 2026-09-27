"""Batched draft context ingest vs the eight-row fc_ctx_kv kernel loop, on random features.

Builds the draft alone (kernels, one checkpoint, DraftBuffers, DraftCore; no target model), writes the
context K/V for ``--rows`` positions through ``DraftRound.initialize``'s block loop (geometry set as the
tail's geometry kernel would) and through ``DraftContextIngest``, and compares the caches element by
element (exact, within one BF16 ulp, max abs) and both against an FP64 evaluation of the same math on the
first rows. Also times both paths. Prints INGEST_OK / INGEST_FAIL.
"""
import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True, help="draft checkpoint directory")
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--capacity", type=int, default=8192)
    ap.add_argument("--fp64-rows", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sweep-rows", default="", help="comma-separated row counts to time the batched path at (features from the first rows)")
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo))
    verify = _load("run_naive_n0p5_verify", a.repo / "tests/gpu/naive_n0p5_e2e/run_naive_n0p5_verify.py")
    verify.decode.enable_torch_tp8_peers()
    from safetensors import safe_open
    from naivert.backends.naive_n0p5_draft import DraftCore
    from naivert.backends.naive_n0p5_draft_ingest import DraftContextIngest, unpack_fc
    from naivert.kernels.naive_n0p5.dspark_kernel_set import prepare_draft
    from naivert.models.naive_n0p5.dspark import DraftBuffers, RMS_EPSILON, prepare_rank_weights
    torch.manual_seed(0)
    kernels = prepare_draft(range(8))
    with safe_open(a.checkpoint / "model.safetensors", framework="pt", device="cpu") as f:
        raw = {k.removeprefix("model."): f.get_tensor(k).bfloat16().contiguous() for k in f.keys()}
    head = torch.zeros(19072, 4096, dtype=torch.bfloat16)
    weights = [prepare_rank_weights(raw, head, rank=r, device=f"cuda:{r}", kernels=kernels) for r in range(8)]
    buffers = [DraftBuffers(kernels, rank=r, capacity=a.capacity, sampling=True) for r in range(8)]
    core = DraftCore(kernels, weights, buffers)
    streams = [torch.cuda.Stream(device=r) for r in range(8)]
    for r in range(8):
        torch.cuda.synchronize(r)
    # the FC tiles unpack to the checkpoint's column shard
    for r in range(8):
        shard = raw["fc.weight"][:, r * 4096:(r + 1) * 4096]
        assert torch.equal(unpack_fc(weights[r].context["fc"]).cpu(), shard), f"rank {r}: FC unpack mismatch"
    print("FC unpack round trip OK", flush=True)
    rows = a.rows
    g = torch.Generator().manual_seed(1)
    # capacity rows of features so the batched path computes its full padded step and writes only [:rows]
    features = [(torch.randn(a.capacity, 4096, generator=g) * 3).to(torch.bfloat16).to(f"cuda:{r}") for r in range(8)]

    def sync():
        for s in streams:
            s.synchronize()

    def loop(start, end):
        for s in range(start, end, 8):
            rws = min(8, end - s)
            valid = torch.tensor([1.0] * rws + [0.0] * (8 - rws), dtype=torch.bfloat16)
            pos = torch.tensor([min(s + i, a.capacity + 7) for i in range(8)], dtype=torch.int64)
            dst = torch.tensor([s + i if i < rws else a.capacity for i in range(8)], dtype=torch.int64)
            for r, b in enumerate(buffers):
                with torch.cuda.device(r), torch.cuda.stream(streams[r]):
                    b.target_hidden.zero_()
                    b.target_hidden[:rws].copy_(features[r][s:s + rws])
                    b.ctx_valid.copy_(valid)
                    b.ctx_pos.copy_(pos)
                    b.kv_dst.copy_(dst)
            for r in range(8):
                with torch.cuda.device(r), torch.cuda.stream(streams[r]):
                    core.context(r)

    loop(0, rows)
    sync()
    ref_k = [b.kv_k[:, :rows].clone() for b in buffers]
    ref_v = [b.kv_v[:, :rows].clone() for b in buffers]
    for b in buffers:
        b.kv_k[:, :rows].zero_()
        b.kv_v[:, :rows].zero_()
    sync()
    ingest = DraftContextIngest(core, streams)
    ingest.ingest(features, 0, rows)
    sync()
    bat_k = [b.kv_k[:, :rows].clone() for b in buffers]
    bat_v = [b.kv_v[:, :rows].clone() for b in buffers]

    def compare(x, y):
        xf, yf = x.float(), y.float()
        d = (xf - yf).abs()
        ulp = (yf.abs() * 2.0 ** -7).clamp(min=2.0 ** -133)
        return dict(exact=round(float((d == 0).float().mean()), 5), within_1ulp=round(float((d <= ulp * 1.0001).float().mean()), 5),
                    within_2ulp=round(float((d <= ulp * 2.0001).float().mean()), 5), max_abs=round(float(d.max()), 5),
                    ref_rms=round(float(yf.pow(2).mean().sqrt()), 4))

    report = dict(rows=rows, capacity=a.capacity, per_rank=[])
    ok = True
    for r in range(8):
        ck, cv = compare(bat_k[r], ref_k[r]), compare(bat_v[r], ref_v[r])
        report["per_rank"].append(dict(rank=r, k=ck, v=cv))
        print(f"rank {r} K {ck} V {cv}", flush=True)
        # accumulation order differs (matmul tiles, all-reduce): expect ulp-level flips, no bias
        ok &= ck["within_2ulp"] >= 0.98 and cv["within_2ulp"] >= 0.98
    # FP64 evaluation of the same math on the first rows, both implementations against it
    n64 = min(a.fp64_rows, rows)
    dev0 = torch.device("cuda:0")
    x64 = torch.cat([features[r][:n64].to(dev0).double() for r in range(8)], dim=1)     # [n, 32768]
    fc64 = raw["fc.weight"].to(dev0).double()
    ctx = x64 @ fc64.t()
    y = ctx * torch.rsqrt(ctx.square().mean(-1, keepdim=True) + RMS_EPSILON) * raw["hidden_norm.weight"].to(dev0).double()
    cos, sin = buffers[0].rope_cos[:n64].double(), buffers[0].rope_sin[:n64].double()
    fp64 = []
    for r in (0, 1, 2, 7):
        h = r // 2
        errs_k, errs_v = [], []
        for L in range(5):
            kw = raw[f"layers.{L}.self_attn.k_proj.weight"][h * 128:(h + 1) * 128].to(dev0).double()
            vw = raw[f"layers.{L}.self_attn.v_proj.weight"][h * 128:(h + 1) * 128].to(dev0).double()
            kg = raw[f"layers.{L}.self_attn.k_norm.weight"].to(dev0).double()
            k = y @ kw.t()
            kn = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + RMS_EPSILON) * kg
            rot = torch.cat((-kn[:, 64:], kn[:, :64]), dim=-1)
            kt = kn * cos + rot * sin
            vt = y @ vw.t()
            for name, got in (("kernel", ref_k[r][L, :n64, 0]), ("batched", bat_k[r][L, :n64, 0])):
                errs_k.append((name, float((got.to(dev0).double() - kt).abs().mean())))
            for name, got in (("kernel", ref_v[r][L, :n64, 0]), ("batched", bat_v[r][L, :n64, 0])):
                errs_v.append((name, float((got.to(dev0).double() - vt).abs().mean())))
        mean = lambda name, errs: round(statistics.mean(e for n, e in errs if n == name), 6)
        row = dict(rank=r, k_mean_abs_err=dict(kernel=mean("kernel", errs_k), batched=mean("batched", errs_k)),
                   v_mean_abs_err=dict(kernel=mean("kernel", errs_v), batched=mean("batched", errs_v)))
        fp64.append(row)
        print("fp64", row, flush=True)
        # the real accuracy statement: no worse than the kernel against the exact math (2 % slack)
        ok &= row["k_mean_abs_err"]["batched"] <= 1.02 * row["k_mean_abs_err"]["kernel"] + 1e-6
        ok &= row["v_mean_abs_err"]["batched"] <= 1.02 * row["v_mean_abs_err"]["kernel"] + 1e-6
    report["fp64_first_rows"] = dict(rows=n64, ranks=fp64)
    # timing
    def timed(fn):
        out = []
        for _ in range(a.reps):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            out.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(out)
    loop_ms = timed(lambda: loop(0, rows))
    batched_ms = timed(lambda: ingest.ingest(features, 0, rows))
    report["timing_ms"] = dict(rows=rows, loop=round(loop_ms, 2), batched=round(batched_ms, 2),
                               loop_per_token=round(loop_ms / rows, 4), batched_per_token=round(batched_ms / rows, 4))
    print("TIMING", json.dumps(report["timing_ms"]), flush=True)
    if a.sweep_rows:
        sweep = []
        for n in [int(x) for x in a.sweep_rows.split(",")]:
            n = min(n, rows)
            ms = timed(lambda: ingest.ingest(features, 0, n))
            sweep.append(dict(rows=n, batched_ms=round(ms, 2), per_token=round(ms / n, 4)))
            print("SWEEP", json.dumps(sweep[-1]), flush=True)
        report["sweep"] = sweep
    report["pass"] = bool(ok)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    print("INGEST_OK" if ok else "INGEST_FAIL", flush=True)


if __name__ == "__main__":
    main()
