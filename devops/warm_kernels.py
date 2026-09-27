#!/usr/bin/env python3
"""Prepare a NaiveRT serving image or host: environment check, extension compile, serve smoke.

  --check-env    print the runtime versions the serving stack was validated with and flag mismatches
  --compile      build the CUDA extensions the server loads (nvcc from NAIVERT_CUDA_HOME / CUDA_HOME);
                 the same entry points SessionEngine.build uses, so the first server start finds them
                 in tests/build/. Kernels that configure devices need visible GPUs and are reported,
                 not fatal; FlashInfer / DeepGEMM JIT kernels of the native prefill build on first use.
  --serve-smoke  start examples/serve_sessions.py with the given --serve-args, wait for /health, send
                 one request, stop. Needs the GPUs and the checkpoints (first-start warm-up on a host).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# import name -> (expected version prefix, distribution name for importlib.metadata when the module has no __version__)
EXPECTED = {"torch": ("2.11.0", "torch"), "flashinfer": ("0.6.12", "flashinfer-python"), "sgl_kernel": ("0.4.4", "sglang-kernel"),
            "deep_gemm": ("0.1.4", "sgl-deep-gemm"), "deep_ep": ("1.2.1", "deep_ep"), "triton": ("3.6.0", "triton"),
            "transformers": ("5.12.1", "transformers"), "safetensors": ("0.8.0", "safetensors"), "fastapi": ("0.139.0", "fastapi"),
            "uvicorn": ("0.51.0", "uvicorn"), "anthropic": ("0.116.0", "anthropic")}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_env() -> int:
    bad = 0
    from importlib import metadata
    for module, (want, dist) in EXPECTED.items():
        try:
            got = getattr(importlib.import_module(module), "__version__", None)
            if got is None:
                got = metadata.version(dist)
        except Exception as exc:                                       # noqa: BLE001
            try:
                got = metadata.version(dist)     # installed but not importable here (e.g. needs the CUDA driver)
                note = f" (installed; import failed without a GPU driver: {type(exc).__name__})"
            except Exception:                                          # noqa: BLE001
                print(f"{module:14s} MISSING ({type(exc).__name__})"); bad += 1; continue
        else:
            note = ""
        ok = str(got).startswith(want)
        bad += not ok
        print(f"{module:14s} {got:24s} {'ok' if ok else 'expected ' + want}{note}")
    # The native prefill executor refuses to start unless every kernel distribution it pins is installed at
    # its pin (naivert/kernels/naive_n0p5/prefill.py check_versions); the Tier 3 pod found ttfs missing from
    # the base image, so the pins are checked here, where a GPU-less build host sees them.
    try:
        sys.path.insert(0, str(ROOT))                                 # the script runs from devops/
        from naivert.kernels.naive_n0p5 import prefill as _prefill
        pins = {**_prefill.PINNED_DISTRIBUTIONS, **_prefill.PINNED_COMPONENTS, **_prefill.EXCEPTION_PINS}
        for dist, got in _prefill.installed_distributions(pins).items():
            ok = got == pins[dist]
            bad += not ok
            print(f"{'prefill:' + dist:38s} {got:16s} {'ok' if ok else 'expected ' + pins[dist] + ' (native prefill pin)'}")
    except Exception as exc:                                           # noqa: BLE001
        print(f"prefill pins   MISSING ({type(exc).__name__}: {exc})"); bad += 1
    # The radix top-k compiles through NVRTC with CCCL from the pip nvidia namespace (cuda_dep.py looks only
    # there); the pod found the base image without nvidia-cuda-cccl ("cannot open source file cstdint").
    try:
        from naivert.kernels.naive_n0p5.prefill_ext.naive_ops.core.cuda_dep import determine_cuda_include_and_lib
        inc, _ = determine_cuda_include_and_lib()
        hdr = Path(inc["cccl"], "cuda", "std", "cstdint")
        ok = hdr.is_file()
        bad += not ok
        print(f"prefill:cccl headers                   {str(inc['cccl'])[:40]:40s} {'ok' if ok else 'MISSING cuda/std/cstdint (pip nvidia-cuda-cccl)'}")
    except Exception as exc:                                           # noqa: BLE001
        print(f"prefill:cccl headers                   MISSING ({type(exc).__name__}: {exc})"); bad += 1
    home = os.environ.get("NAIVERT_CUDA_HOME") or os.environ.get("CUDA_HOME")
    other = os.environ.get("CUDA_HOME")
    if home and other and Path(home).resolve() != Path(other).resolve():
        print(f"CUDA_HOME      {other} differs from NAIVERT_CUDA_HOME {home}: the kernel builders take CUDA_HOME first; "
              "set both to the 13.2 toolkit (devops/entrypoint.sh does)"); bad += 1
    nvcc = Path(home or "/nonexistent") / "bin/nvcc"
    if nvcc.is_file():
        out = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        m = re.search(r"release\s+(\d+)\.(\d+)", subprocess.run([str(nvcc), "--version"], capture_output=True, text=True).stdout)
        ok = m is not None and tuple(map(int, m.groups())) >= (13, 1)
        bad += not ok
        print(f"nvcc           {out} {'ok' if ok else 'expected >= 13.1 (sm_103a, cuda::ptx::elect_sync)'}")
    else:
        print("nvcc           MISSING: set NAIVERT_CUDA_HOME to a CUDA 13.2 toolkit"); bad += 1
    inc = os.environ.get("NAIVERT_CUTLASS_INCLUDE")
    print(f"cutlass        {inc if inc and Path(inc, 'cutlass').is_dir() else 'MISSING: set NAIVERT_CUTLASS_INCLUDE (cutlass v4.7.1 include/)'}")
    bad += not (inc and Path(inc, "cutlass").is_dir())
    try:
        import torch
        print(f"gpus           {torch.cuda.device_count()} visible" if torch.cuda.is_available() else "gpus           none visible")
    except Exception:                                                  # noqa: BLE001
        pass
    return 1 if bad else 0


def compile_extensions() -> int:
    sys.path.insert(0, str(ROOT))
    steps = []
    def step(name, fn):
        t0 = time.monotonic()
        try:
            fn()
            steps.append((name, "built", round(time.monotonic() - t0, 1)))
        except Exception as exc:                                       # noqa: BLE001
            steps.append((name, f"skipped: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}", round(time.monotonic() - t0, 1)))
        print(*steps[-1], flush=True)
    from naivert.kernels.naive_n0p5.model_ops import load_extension
    step("model_ops fp8 / w8a8", lambda: load_extension(weights="fp8", expert_down="w8a8"))
    dsa = _load("run_attn_dsa", ROOT / "tests/gpu/attn_dsa/run_attn_dsa.py")
    dsa8 = _load("run_attn_dsa_rows", ROOT / "tests/gpu/attn_dsa/run_attn_dsa_rows.py")
    swa = _load("run_attn_swa", ROOT / "tests/gpu/attn_swa/run_attn_swa.py")
    step("attention dsa rows1", lambda: dsa.build(False, 16))
    step("attention dsa rows8", lambda: dsa8.build(8, index_heads=16))
    step("attention swa rows1", lambda: swa.build(1))
    step("attention swa rows8", lambda: swa.build(8))
    # the four draft extensions compile without a device; prepare_draft() configures them per GPU at server start
    def draft():
        from naivert.kernels.naive_n0p5 import dspark_attention, dspark_context, dspark_head_markov, dspark_mlp
        dspark_context.load_context(); dspark_attention.load_attention(); dspark_mlp.load_mlp(); dspark_head_markov.load_head_markov()
    step("draft kernel set (context, attention, mlp, head/markov)", draft)
    def probe():
        from naivert.prefill.native.combine_ext import load_combine_extension
        load_combine_extension()
    step("native prefill EP-return probe", probe)
    def tail():
        from naivert.kernels.naive_n0p5.dspark_tail import load_tail
        load_tail(no_ready=True)
    step("draft tail (no_ready, the served variant)", tail)
    def sampler():
        from naivert.kernels.naive_n0p5.dspark_sampling import load_sampling
        load_sampling(persistent=True)
    step("TP8 sampler (persistent, the served variant)", sampler)
    # the native prefill's vendored operators: their JIT builds are what a first start otherwise pays (minutes)
    def deepep():
        from naivert.prefill.native.deepep import load_extension
        load_extension()
    step("native prefill DeepEP dispatch (vendored 1.2.1)", deepep)
    def collectives():
        from naivert.prefill.native.collectives import load_extension
        load_extension()
    step("native prefill FlashInfer collectives (vendored 0.6.12)", collectives)
    def rope():
        from naivert.prefill.native.rope import load_extension
        load_extension()
    step("native prefill FlashInfer RoPE (vendored 0.6.12)", rope)
    built = sum(1 for _, s, _ in steps if s == "built")
    print(json.dumps(dict(built=built, steps=[dict(name=n, status=s, seconds=t) for n, s, t in steps])))
    return 0 if built else 1


def serve_smoke(serve_args: str, port: int, timeout_s: int) -> int:
    cmd = [sys.executable, "-u", str(ROOT / "examples/serve_sessions.py"), "--port", str(port), *shlex.split(serve_args)]
    proc = subprocess.Popen(cmd, cwd=ROOT)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print("server exited early", proc.returncode); return 1
            try:
                if urllib.request.urlopen(base + "/health", timeout=5).status == 200:
                    break
            except Exception:                                          # noqa: BLE001
                time.sleep(5)
        else:
            print("server never became healthy"); return 1
        model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]
        req = urllib.request.Request(base + "/v1/messages", data=json.dumps(dict(model=model["id"], max_tokens=32,
                                     messages=[dict(role="user", content="Say hello in one short sentence.")])).encode(),
                                     headers={"content-type": "application/json", "anthropic-version": "2023-06-01"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.load(r)
        text = "".join(b["text"] for b in body["content"] if b["type"] == "text")
        t = body.get("naivert", {}).get("timings", {})
        print(json.dumps(dict(model=model, wall_s=round(time.monotonic() - t0, 2), text=text[:120],
                              prefill_path=t.get("prefill_path"), prompt_ms=t.get("prompt_ms"), device_tok_s=t.get("decode_tok_s_device"))))
        print("SERVE_SMOKE_OK")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-env", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--serve-smoke", action="store_true")
    ap.add_argument("--serve-args", default="", help="extra examples/serve_sessions.py arguments for --serve-smoke")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--timeout", type=int, default=1200, help="seconds to wait for /health in --serve-smoke")
    a = ap.parse_args()
    rc = 0
    if a.check_env:
        rc |= check_env()
    if a.compile:
        rc |= compile_extensions()
    if a.serve_smoke:
        rc |= serve_smoke(a.serve_args, a.port, a.timeout)
    if not (a.check_env or a.compile or a.serve_smoke):
        ap.error("pick at least one of --check-env, --compile, --serve-smoke")
    sys.exit(rc)


if __name__ == "__main__":
    main()
