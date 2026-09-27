"""Existing FlashInfer BF16 collectives with single-process peer ownership.

The vendored device sources are unchanged. The host adapter owns the unicast
workspace without a process group, CUDA IPC or NVSHMEM initialization. The
executor enables BF16 reduce-scatter with FP32 accumulation. NCCL handles
all-reduce and non-BF16 collectives.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading

import torch

from naivert.prefill.comms import CommError

_SOURCE = Path(__file__).resolve().parents[1] / 'kernels/flashinfer_comm'
_LOAD_LOCK = threading.Lock()
_LOADED = None
# Values are the pinned upstream MixedCommOp / MixedCommMode enums.
_GATHER, _SCATTER, _MODE = 1, 2, 1


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_extension():
    global _LOADED
    with _LOAD_LOCK:
        if _LOADED is not None:
            return _LOADED
        import flashinfer
        import nvidia.nvshmem
        from filelock import FileLock
        from flashinfer.jit import env as jit_env
        from flashinfer.jit.core import current_compilation_context, gen_jit_spec
        from flashinfer.jit.cpp_ext import get_cuda_path

        manifest = json.loads((_SOURCE / 'manifest.json').read_text())
        if flashinfer.__version__ != manifest['version']:
            raise RuntimeError(f'prefill collectives need FlashInfer {manifest["version"]}; got {flashinfer.__version__}')
        for name, digest in manifest['files'].items():
            if _sha(_SOURCE / name) != digest:
                raise RuntimeError(f'vendored collective source differs from manifest: {name}')
        nvshmem = Path(nvidia.nvshmem.__path__[0])
        host = next(p for p in (nvshmem/'lib/libnvshmem_host.so', nvshmem/'lib/libnvshmem_host.so.3') if p.exists())
        ctypes.CDLL(str(host), mode=ctypes.RTLD_GLOBAL)
        flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[9, 10])
        arch_flags = [flag for flag in flags if flag.startswith('-gencode=')]
        if not arch_flags:
            raise RuntimeError('mixed collectives require explicit device-link architecture flags')
        cuda_home = Path(get_cuda_path())
        nvcc = subprocess.check_output([str(cuda_home/'bin/nvcc'), '--version'], text=True)
        identity = dict(files=manifest['files'], flashinfer=flashinfer.__version__, torch=torch.__version__,
                        cuda_home=str(cuda_home), nvcc=nvcc, flags=flags,
                        nvshmem_device_sha256=_sha(nvshmem/'lib/libnvshmem_device.a'),
                        build_env={k: os.environ.get(k) for k in ('CC', 'CXX', 'FLASHINFER_JIT_DEBUG',
                                                                  'FLASHINFER_JIT_LINEINFO')})
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        name = f'naivert_mixed_comm_{key}'
        generated = jit_env.FLASHINFER_GEN_SRC_DIR / name
        generated.mkdir(parents=True, exist_ok=True)
        spec = importlib.util.spec_from_file_location('_naivert_mixed_comm_generate', _SOURCE/'generate.py')
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        with FileLock(str(generated/'generate.lock')):
            sources = generator.generate(_SOURCE, generated)
        build = gen_jit_spec(name, sources,
                            extra_include_paths=[_SOURCE/'include', _SOURCE/'csrc', nvshmem/'include'],
                            extra_cuda_cflags=flags + ['-rdc=true'],
                            # FlashInfer 0.6.12 drops compile flags at device link.
                            extra_ldflags=[f'-L{nvshmem / "lib"}', '-lnvshmem_device', *arch_flags],
                            needs_device_linking=True)
        module = build.build_and_load()
        version = dict(version=manifest['version'], manifest_sha256=_sha(_SOURCE/'manifest.json'),
                       library=str(build.get_library_path()), library_sha256=_sha(build.get_library_path()),
                       device_sources_unchanged=True, device_link_arch=arch_flags)
        _LOADED = module, version
        return _LOADED


class FlashInferCollectives:
    """Own scratch shared by eight ranks, after the graph owner drains them."""

    def __init__(self, inner, *, reduce_scatter=False):
        self.inner = inner
        self.devices, self.streams = inner.devices, inner.streams
        if len(self.devices) != 8:
            raise CommError('prefill FlashInfer collectives require eight ranks')
        self.module, version = load_extension()
        self.reduce_scatter = bool(reduce_scatter)
        self._closed = False
        self.storage, self.tables = [], []
        self.blocks = {op: self.module.get_max_block_size(torch.bfloat16, 1, 8, 1, 1, op, _MODE)
                       for op in (_GATHER, _SCATTER)}
        max_block = max(int(v) for block in self.blocks.values() for v in block.values())
        self.sms = int(os.environ.get('NAIVERT_FLASHINFER_SMS', '128'))
        if self.sms <= 0:
            raise CommError('collective SM count must be positive')
        align = lambda n: (n+16383)//16384*16384
        self.base = align(9*max_block*16+4)
        data_bytes, info_bytes = 4*self.sms*self.base, align(self.sms*8)
        try:
            # Establish torch's peer-access bookkeeping too, including NCCL dispatch mode.
            for r, dev in enumerate(self.devices):
                with torch.cuda.device(dev), torch.cuda.stream(self.streams[r]):
                    seed = torch.zeros(1, device=dev)
                    for peer in self.devices:
                        if peer != dev:
                            if not torch.cuda.can_device_access_peer(dev, peer):
                                raise CommError(f'no peer access from {dev} to {peer}')
                            seed.to(peer)
                    data = torch.empty(data_bytes+info_bytes, device=dev, dtype=torch.uint8)
                    data[:data_bytes].view(torch.int16).fill_(-32768)
                    data[data_bytes:].zero_()
                    self.storage.append(data)
            for r, dev in enumerate(self.devices):
                with torch.cuda.device(dev), torch.cuda.stream(self.streams[r]):
                    self.tables.append(torch.tensor([self.storage[(r+i+1)%8].data_ptr() for i in range(8)],
                                                    device=dev, dtype=torch.int64))
            for stream in self.streams:
                stream.synchronize()
        except BaseException:
            self.close()
            raise
        self.version = dict(version, sms=self.sms, workspace_bytes_per_rank=data_bytes+info_bytes,
                            mode='FUSED_OPT_WAITS_UC', reduce_scatter=self.reduce_scatter)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _run(self, op, inputs, outputs):
        if self._closed:
            raise CommError('prefill collectives are closed')
        fn = self.module.allgather if op == _GATHER else self.module.reducescatter
        for r, dev in enumerate(self.devices):
            with torch.cuda.device(dev), torch.cuda.stream(self.streams[r]):
                fn(outputs[r], inputs[r], ctypes.c_void_p(self.tables[r].data_ptr()),
                   ctypes.c_void_p(self.storage[r].data_ptr()), ctypes.c_void_p(), ctypes.c_void_p(),
                   self.base, 0, 0, self.sms, self.blocks[op], 256, 4, r, 8, 0, 1, _MODE)

    def all_gather_(self, inputs, outputs):
        self.inner._check_split(outputs, inputs, (torch.bfloat16, torch.float32, torch.uint8))
        if inputs[0].dtype != torch.bfloat16:
            return self.inner.all_gather_(inputs, outputs)
        self._run(_GATHER, inputs, outputs)

    def reduce_scatter_(self, inputs, outputs):
        self.inner._check_split(inputs, outputs)
        if not self.reduce_scatter or inputs[0].dtype != torch.bfloat16:
            return self.inner.reduce_scatter_(inputs, outputs)
        self._run(_SCATTER, inputs, outputs)

    def close(self):
        if self._closed:
            return
        for stream in self.streams:
            stream.synchronize()
        self.tables.clear()
        self.storage.clear()
        self._closed = True
