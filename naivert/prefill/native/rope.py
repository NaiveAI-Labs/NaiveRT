"""Joint Q/K RoPE using unchanged, vendored FlashInfer kernels."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

import torch

_SOURCE = Path(__file__).resolve().parents[1] / 'kernels/flashinfer_rope'
_LOCK = threading.Lock()
_LOADED = None


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_extension():
    global _LOADED
    with _LOCK:
        if _LOADED is not None:
            return _LOADED
        import flashinfer
        from flashinfer.jit.core import current_compilation_context, gen_jit_spec
        from flashinfer.jit.cpp_ext import get_cuda_path

        manifest = json.loads((_SOURCE/'manifest.json').read_text())
        if flashinfer.__version__ != manifest['version']:
            raise RuntimeError(f'prefill RoPE needs FlashInfer {manifest["version"]}; got {flashinfer.__version__}')
        for name, digest in manifest['files'].items():
            if _sha(_SOURCE/name) != digest:
                raise RuntimeError(f'vendored RoPE source differs from manifest: {name}')
        flags = current_compilation_context.get_nvcc_flags_list()
        cuda_home = Path(get_cuda_path())
        identity = dict(files=manifest['files'], flashinfer=flashinfer.__version__, torch=torch.__version__,
                        cuda_home=str(cuda_home), flags=flags,
                        nvcc=subprocess.check_output([str(cuda_home/'bin/nvcc'), '--version'], text=True),
                        build_env={k: os.environ.get(k) for k in ('CC', 'CXX', 'FLASHINFER_JIT_DEBUG',
                                                                  'FLASHINFER_JIT_LINEINFO')})
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        build = gen_jit_spec(f'naivert_rope_{key}',
                            [_SOURCE/'csrc/rope.cu', _SOURCE/'csrc/flashinfer_rope_binding.cu'],
                            extra_include_paths=[_SOURCE/'include', _SOURCE/'csrc'])
        module = build.build_and_load()
        version = dict(version=manifest['version'], manifest_sha256=_sha(_SOURCE/'manifest.json'),
                       library=str(build.get_library_path()), library_sha256=_sha(build.get_library_path()),
                       device_sources_unchanged=True)
        _LOADED = module, version
        return _LOADED


def apply(q, k, positions, table):
    """BF16 Q/K [T,H,192], rotary dim 64; table stores cos then sin halves.

    Inputs may be strided QKV views. Outputs are separately contiguous, matching
    the previous two-call layout. The indexer's different FMA spelling continues
    to use the existing site-specific kernel.
    """
    module, _ = load_extension()
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    module.apply_rope_pos_ids_cos_sin_cache(q, k, q_out, k_out, table, positions, True)
    return q_out, k_out
