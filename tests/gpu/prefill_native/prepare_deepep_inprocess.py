"""Prepare an isolated DeepEP experiment with unchanged device kernels.

Only adapt host peer-pointer setup and collective cleanup for one process. Never
modify the installed library or its source tree. Build the emitted setup.py under
gpu_profile_queue, then put the output directory first on PYTHONPATH to test it.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


LOCAL_SYNC = r'''
void Buffer::sync_local(const std::vector<Buffer*>& peers) {
    c10::cuda::CUDAGuard guard(device_id);
    EP_HOST_ASSERT(not available and not destroyed);
    EP_HOST_ASSERT(num_rdma_bytes == 0 and not low_latency_mode);
    EP_HOST_ASSERT(num_ranks <= NUM_MAX_NVL_PEERS and num_nvl_bytes > 0);
    EP_HOST_ASSERT(peers.size() == num_ranks and peers[rank] == this);
    for (int i = 0; i < num_ranks; ++i) {
        auto* peer = peers[i];
        EP_HOST_ASSERT(peer != nullptr and not peer->destroyed);
        EP_HOST_ASSERT(peer->rank == i and peer->num_ranks == num_ranks);
        EP_HOST_ASSERT(peer->num_nvl_bytes == num_nvl_bytes);
        EP_HOST_ASSERT(peer->num_rdma_bytes == 0 and not peer->low_latency_mode);
        if (i != rank) {
            EP_HOST_ASSERT(peer->device_id != device_id);
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device_id, peer->device_id));
            EP_HOST_ASSERT(can_access);
            auto status = cudaDeviceEnablePeerAccess(peer->device_id, 0);
            if (status == cudaErrorPeerAccessAlreadyEnabled)
                (void)cudaGetLastError();
            else
                CUDA_CHECK(status);
        }
        buffer_ptrs[i] = peer->buffer_ptrs[i];
        barrier_signal_ptrs[i] = reinterpret_cast<int*>(
            static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
    }
    CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(buffer_ptrs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs,
                          sizeof(barrier_signal_ptrs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    local_peers = true;
    available = true;
}

'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    pins = {
        'csrc/deep_ep.cpp': 'de03cc5b5e2d05b3dc521ab1bfd419f3d94bfaa5385ee3c05e0052295a6e138d',
        'csrc/deep_ep.hpp': 'acd71eb8bc6301559045137f1c91bff19d8b1404671a173870af03cd81ddcfd6',
        'setup.py': '54249343fa57229d368384f4342b06851e3e62876c88bea786f6aa9826e35092',
    }
    assert all(sha(args.source / name) == digest for name, digest in pins.items()), 'Unexpected upstream source.'
    assert not args.out.exists(), 'Use a fresh experiment directory.'
    args.out.mkdir(parents=True)
    shutil.copytree(args.source / 'csrc', args.out / 'csrc')
    for name in ('setup.py', 'LICENSE'):
        shutil.copy2(args.source / name, args.out / name)
    originals = {name: (args.out / name).read_text()
                 for name in ('csrc/deep_ep.cpp', 'csrc/deep_ep.hpp', 'setup.py')}
    header = replace_once(originals['csrc/deep_ep.hpp'], 'bool available = false;',
                          'bool available = false;\n    bool local_peers = false;')
    header = replace_once(header, '    void destroy();',
                          '    void sync_local(const std::vector<Buffer*>& peers);\n\n    void destroy();')
    cpp = replace_once(originals['csrc/deep_ep.cpp'], '#include <ATen/cuda/CUDAContext.h>',
                       '#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAGuard.h>')
    cpp = replace_once(cpp, 'void Buffer::destroy() {',
                       LOCAL_SYNC + 'void Buffer::destroy() {\n    c10::cuda::CUDAGuard guard(device_id);')
    cpp = replace_once(cpp, '// Close remote IPC\n        if (is_available()) {',
                       '// Close remote IPC; local peers are borrowed CUDA allocations.\n        if (is_available() and not local_peers) {')
    cpp = replace_once(cpp, '    destroyed = true;',
                       '    if (moe_recv_rdma_counter != nullptr)\n        CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_rdma_counter)));\n\n    destroyed = true;')
    cpp = replace_once(cpp, '.def("destroy", &deep_ep::Buffer::destroy)',
                       '.def("sync_local", &deep_ep::Buffer::sync_local)\n'
                       '        .def("destroy", &deep_ep::Buffer::destroy, pybind11::call_guard<pybind11::gil_scoped_release>())')
    # Eager calls can allocate after posting a GPU collective. Other rank
    # threads must be able to enter while such an allocation synchronizes.
    cpp = replace_once(cpp, '.def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)',
                       '.def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch, pybind11::call_guard<pybind11::gil_scoped_release>())')
    # This experiment only needs intra-node kernels. Keep the upstream build
    # flags and source list, but do not initialize/link an NVSHMEM runtime.
    setup = originals['setup.py']
    start, end = setup.index('    disable_nvshmem = False'), setup.index('    cxx_flags =')
    setup = setup[:start] + '    disable_nvshmem = True\n    nvshmem_dir = None\n\n' + setup[end:]
    # Upstream only requests device linking through its NVSHMEM flags, although
    # intra-node CUDA objects also use -rdc=true. Link them explicitly here.
    setup = replace_once(setup, "name='deep_ep_cpp',", "name='deep_ep_cpp',\n                dlink=True,")
    changed = {'csrc/deep_ep.cpp': cpp, 'csrc/deep_ep.hpp': header, 'setup.py': setup}
    patch = ''
    for name, text in changed.items():
        patch += ''.join(difflib.unified_diff(originals[name].splitlines(True), text.splitlines(True),
                                             fromfile='a/' + name, tofile='b/' + name))
        (args.out / name).write_text(text)
    (args.out / 'host_adapter.patch').write_text(patch)
    device_files = sorted((args.source / 'csrc/kernels').rglob('*'))
    device_hashes = {str(p.relative_to(args.source)): sha(p) for p in device_files if p.is_file()}
    assert all(sha(args.out / name) == digest for name, digest in device_hashes.items())
    manifest = dict(source=str(args.source), scope='host setup/cleanup only; intra-node-only experimental build',
                    upstream_host_sha256={name: sha(args.source / name) for name in originals},
                    adapted_host_sha256={name: sha(args.out / name) for name in originals},
                    device_sha256=device_hashes, device_sources_unchanged=True)
    (args.out / 'host_adapter_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.out), device_sources_unchanged=True,
                          device_files=len(device_hashes))))


if __name__ == '__main__':
    main()
