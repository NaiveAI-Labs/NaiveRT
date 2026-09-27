"""Single-process ownership of the vendored DeepEP intra-node dispatch.

No external deep_ep package or process group is used. Device kernels are the
pinned upstream sources; only host setup, bindings and cleanup are adapted.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import threading

import torch

from naivert.kernels.naive_n0p5 import prefill as P

_MODULE = None
_LOAD_LOCK = threading.Lock()
_SOURCE = Path(__file__).resolve().parents[1] / 'kernels/deepep'


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_extension():
    """Build once outside the source tree, with explicit CUDA device linking."""
    global _MODULE
    with _LOAD_LOCK:
        if _MODULE is not None:
            return _MODULE
        from torch.utils.cpp_extension import CUDA_HOME, get_default_build_root
        manifest = json.loads((_SOURCE / 'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if _sha(_SOURCE / name) != digest:
                raise RuntimeError(f'vendored DeepEP source differs from manifest: {name}')
        cuda_home = os.environ.get('NAIVERT_CUDA_HOME') or os.environ.get('CUDA_HOME') or CUDA_HOME
        if not cuda_home:
            raise RuntimeError('DeepEP dispatch needs CUDA_HOME or NAIVERT_CUDA_HOME to build')
        env = dict(os.environ, CUDA_HOME=str(cuda_home),
                   TORCH_CUDA_ARCH_LIST=os.environ.get('TORCH_CUDA_ARCH_LIST', '10.3a'))
        nvcc = subprocess.check_output([str(Path(cuda_home) / 'bin/nvcc'), '--version'], text=True)
        identity = dict(source=manifest['files'], torch=torch.__version__, python=sys.version,
                        cuda_home=str(cuda_home), nvcc=nvcc,
                        build_env={k: env.get(k) for k in ('TORCH_CUDA_ARCH_LIST', 'CC', 'CXX',
                                   'DISABLE_SM90_FEATURES', 'DISABLE_AGGRESSIVE_PTX_INSTRS')})
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        cache = Path(os.environ.get('TORCH_EXTENSIONS_DIR', get_default_build_root())) / f'naivert_deepep_{key}'
        cache.mkdir(parents=True, exist_ok=True)
        module_path = cache / ('naivert_deep_ep_cpp' + sysconfig.get_config_var('EXT_SUFFIX'))
        ready = cache / 'ready.json'
        with (cache / 'build.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (ready.exists() and module_path.exists()
                    and json.loads(ready.read_text())['module_sha256'] == _sha(module_path)):
                log = cache / 'build.log'
                with log.open('w') as out:
                    result = subprocess.run([sys.executable, str(_SOURCE / 'setup.py'), 'build_ext',
                                             '--build-lib', str(cache), '--build-temp', str(cache / 'objects')],
                                            cwd=cache, env=env, stdout=out, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'DeepEP build failed; see {log}\n{log.read_text()[-4000:]}')
                ready.write_text(json.dumps(dict(identity=identity, module_sha256=_sha(module_path)), indent=2) + '\n')
        spec = importlib.util.spec_from_file_location('naivert_deep_ep_cpp', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE = module
        return module


class DeepEPDispatch:
    """Own all rank buffers together; close after their graph owners drain."""

    def __init__(self, devices, streams, workspaces, launchers, *, tokens):
        self.devices, self.streams, self.ws = devices, streams, workspaces
        self.launchers, self.tokens = launchers, tokens
        self.ranks = len(devices)
        self.rows_per_rank = tokens // self.ranks
        self.buffers, self.ids, self.rows, self.topk = [], [], [], []
        self.received = [None] * self.ranks
        module = load_extension()
        sms = int(os.environ.get('NAIVERT_DEEPEP_SMS', '128'))
        self.config = module.Config(sms, 6, 256, 6, 128)
        nvl_bytes = self.config.get_nvl_buffer_size_hint(P.HIDDEN, self.ranks)
        self.version = dict(version='1.2.1', sms=sms, nvl_bytes=nvl_bytes,
                            manifest_sha256=_sha(_SOURCE / 'manifest.json'),
                            module=module.__file__, module_sha256=_sha(Path(module.__file__)))
        try:
            for r, dev in enumerate(devices):
                with torch.cuda.device(dev), torch.cuda.stream(streams[r]):
                    self.buffers.append(module.Buffer(r, self.ranks, nvl_bytes, 0, False, True))
                    self.ids.append(torch.empty((self.rows_per_rank, P.TOPK), device=dev, dtype=torch.int64))
                    self.rows.append(torch.arange(tokens, device=dev, dtype=torch.int32))
                    self.topk.append(torch.arange(P.TOPK, device=dev, dtype=torch.int64))
            for r, buffer in enumerate(self.buffers):
                with torch.cuda.device(devices[r]):
                    buffer.sync_local(self.buffers)
        except BaseException:
            self.close()
            raise

    def _one(self, r):
        with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
            ws, buffer = self.ws[r], self.buffers[r]
            per_rank, _, per_expert, in_rank, _ = buffer.get_dispatch_layout(
                self.ids[r], P.EXPERTS, None, False, False)
            self.received[r] = buffer.intranode_dispatch(
                ws.q_rows, ws.s_rows, self.ids[r], ws.route_w,
                per_rank, in_rank, per_expert, 0, None, None, 1, self.tokens,
                self.config, None, False, False)

    def run(self):
        with torch.cuda.device(self.devices[0]), torch.cuda.stream(self.streams[0]):
            capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            for r in range(self.ranks):
                self._one(r)
        else:
            # Allocation may wait behind the collective. All peers must enter;
            # the vendored binding releases the GIL during each dispatch call.
            list(self.launchers.map(self._one, range(self.ranks)))

    def mailbox_pairs(self, r, permutation):
        prefix, source = self.received[r][5], self.received[r][8]
        source_rank = torch.searchsorted(prefix[:, r].contiguous(), self.rows[r], right=True)
        token = source.to(torch.int64) + source_rank.to(torch.int64) * self.rows_per_rank
        pairs = (token[:, None] * P.TOPK + self.topk[r][None, :]).reshape(-1)
        return pairs.index_select(0, permutation)

    def close(self):
        if not self.buffers:
            return
        # Includes DeepEP's private communication streams, even after partial
        # construction. No rank can still reference a peer being freed below.
        for device in self.devices:
            torch.cuda.synchronize(device)
        for buffer in self.buffers:
            buffer.destroy(True)
        self.buffers.clear()
        self.received = [None] * self.ranks
        self.ids.clear()
        self.rows.clear()
        self.topk.clear()
