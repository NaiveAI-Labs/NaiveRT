"""Use existing FlashInfer BF16 collectives beside the integrated DeepEP dispatch.

Experimental only. NAIVERT_FLASHINFER_MIXED_LIBRARY selects the isolated relink.
Gather is always replaced. NAIVERT_FLASHINFER_REDUCE_SCATTER=1 additionally tests
the library reduction, whose arithmetic differs from NCCL and needs validation.
"""
import ctypes
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from naivert.prefill.native.executor import NativePrefill as Baseline


class _Collectives:
    def __init__(self, inner):
        import nvidia.nvshmem
        import tvm_ffi
        from flashinfer.comm import mixed_comm as M
        self.inner, self.M = inner, M
        self.devices, self.streams = inner.devices, inner.streams
        self.reduce_scatter = os.environ.get('NAIVERT_FLASHINFER_REDUCE_SCATTER', '0') == '1'
        path = Path(os.environ['NAIVERT_FLASHINFER_MIXED_LIBRARY'])
        lib_dir = Path(nvidia.nvshmem.__path__[0]) / 'lib'
        host = next(p for p in (lib_dir/'libnvshmem_host.so', lib_dir/'libnvshmem_host.so.3') if p.exists())
        ctypes.CDLL(str(host), mode=ctypes.RTLD_GLOBAL)
        module = tvm_ffi.load_module(str(path))
        self.mode = M.MixedCommMode.FUSED_OPT_WAITS_UC
        ops = (M.MixedCommOp.ALLGATHER, M.MixedCommOp.REDUCESCATTER)
        blocks = {op: module.get_max_block_size(torch.bfloat16, 1, 8, 1, 1, op, self.mode) for op in ops}
        max_block = max(int(v) for block in blocks.values() for v in block.values())
        align = lambda n: (n+16383)//16384*16384
        sms = int(os.environ.get('NAIVERT_FLASHINFER_SMS', '128'))
        base = align(9*max_block*16+4)
        data_bytes, info_bytes = 4*sms*base, align(sms*8)
        self.storage, self.tables, self.handlers = [], [], []
        for r, dev in enumerate(self.devices):
            with torch.cuda.device(dev), torch.cuda.stream(self.streams[r]):
                data = torch.empty(data_bytes+info_bytes, device=dev, dtype=torch.uint8)
                data[:data_bytes].view(torch.int16).fill_(-32768)
                data[data_bytes:].zero_()
                self.storage.append(data)
        for r, dev in enumerate(self.devices):
            with torch.cuda.device(dev), torch.cuda.stream(self.streams[r]):
                table = torch.tensor([self.storage[(r+i+1)%8].data_ptr() for i in range(8)],
                                     device=dev, dtype=torch.int64)
                self.tables.append(table)
                self.handlers.append(SimpleNamespace(
                    mixed_comm_module=module, uc_buffer_dict={'void_p': ctypes.c_void_p(table.data_ptr())},
                    mem_buffer=ctypes.c_void_p(self.storage[r].data_ptr()),
                    mc_buffer_dict={'full': ctypes.c_void_p()}, ns_buffer=ctypes.c_void_p(),
                    vm_buffer_bytes_base=base, ns_data_bytes=0, ns_signal_bytes=0,
                    grid_size=sms, max_block_size_dict={(op,self.mode): blocks[op] for op in ops},
                    min_block_size=256, min_num_steps=4,
                    para_info=SimpleNamespace(dp_size=8, local_rank=r, local_size=8, inter_rank=0, inter_size=1)))
        for stream in self.streams:
            stream.synchronize()
        self.version = dict(library=str(path), library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            source_sha256=hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest(),
                            sms=sms, workspace_bytes_per_rank=data_bytes+info_bytes,
                            mode=self.mode.name, reduce_scatter=self.reduce_scatter)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def all_gather_(self, inputs, outputs):
        if inputs[0].dtype != torch.bfloat16:
            return self.inner.all_gather_(inputs, outputs)
        self.inner._check_split(outputs, inputs, (torch.bfloat16,))
        for r, dev in enumerate(self.devices):
            with torch.cuda.device(dev), torch.cuda.stream(self.streams[r]):
                self.M._allgather(self.handlers[r], inputs[r], outputs[r], self.mode)

    def reduce_scatter_(self, inputs, outputs):
        if not self.reduce_scatter:
            return self.inner.reduce_scatter_(inputs, outputs)
        self.inner._check_split(inputs, outputs, (torch.bfloat16,))
        for r, dev in enumerate(self.devices):
            with torch.cuda.device(dev), torch.cuda.stream(self.streams[r]):
                self.M._reducescatter(self.handlers[r], inputs[r], outputs[r], self.mode)

    def close(self):
        for stream in self.streams:
            stream.synchronize()
        self.handlers.clear(); self.tables.clear(); self.storage.clear()


class NativePrefill(Baseline):
    def __init__(self, *args, **kwargs):
        self.flashinfer_comm = None
        super().__init__(*args, **kwargs)
        try:
            self.flashinfer_comm = _Collectives(self.comm)
        except BaseException:
            super().close()
            raise
        self.comm = self.flashinfer_comm
        self.versions['flashinfer-collectives-experiment'] = self.flashinfer_comm.version
        print('FlashInfer collectives ready', self.flashinfer_comm.version, flush=True)

    def close(self):
        if self._closed:
            return
        super().close()
        if self.flashinfer_comm is not None:
            self.flashinfer_comm.close()
