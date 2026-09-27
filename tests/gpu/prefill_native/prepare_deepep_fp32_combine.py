"""Prepare an isolated FP32 instantiation of the existing DeepEP combine template.

The CUDA kernel body and communication protocol stay unchanged. One host launch
switch enables float, and the combine binding releases the GIL for eager peer
launches. This is experimental; it does not edit the runtime's vendored source.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.source/'manifest.json').read_text())
    assert all(sha(a.source/name) == digest for name,digest in manifest['files'].items())
    assert not a.out.exists(), 'Use a fresh experiment directory'
    shutil.copytree(a.source, a.out)
    originals = {}
    for name in ('csrc/kernels/intranode.cu', 'csrc/deep_ep.cpp', 'setup.py'):
        originals[name] = (a.out/name).read_text()
    name = 'csrc/kernels/intranode.cu'
    text = originals[name]
    before = '    SWITCH_TYPES(COMBINE_DTYPE_LAUNCH_CASE);'
    assert text.count(before) == 1
    replacement = ('    switch (type) {\n'
                   '        case CUDA_R_16BF: COMBINE_DTYPE_LAUNCH_CASE(nv_bfloat16);\n'
                   '        case CUDA_R_32F: COMBINE_DTYPE_LAUNCH_CASE(float);\n'
                   '        default: EP_HOST_ASSERT(false and "Unsupported type");\n'
                   '    }')
    text = text.replace(before, replacement)
    # Every byte before the host launcher is unchanged, including all kernels.
    host_start = text.index('void combine(cudaDataType_t type,')
    assert text[:host_start] == originals[name][:host_start]
    (a.out/name).write_text(text)
    name = 'csrc/deep_ep.cpp'
    before = '.def("intranode_combine", &deep_ep::Buffer::intranode_combine)'
    assert originals[name].count(before) == 1
    (a.out/name).write_text(originals[name].replace(before,
        '.def("intranode_combine", &deep_ep::Buffer::intranode_combine, pybind11::call_guard<pybind11::gil_scoped_release>())'))
    name = 'setup.py'
    assert originals[name].count("name='naivert_deep_ep_cpp'") == 1
    (a.out/name).write_text(originals[name].replace("name='naivert_deep_ep_cpp'", "name='naivert_deep_ep_fp32_probe'"))
    patch = ''.join(''.join(difflib.unified_diff(text.splitlines(True), (a.out/name).read_text().splitlines(True),
                                              fromfile='a/'+name, tofile='b/'+name))
                    for name,text in originals.items())
    (a.out/'fp32_launch.patch').write_text(patch)
    changed = {name: sha(a.out/name) for name in originals}
    record = dict(source_manifest_sha256=sha(a.source/'manifest.json'),
                  scope='float instantiation of existing intranode combine; GIL release; isolated module name',
                  kernel_bodies_unchanged=True, host_launch_dtype_added='float32',
                  upstream_sha256={name:sha(a.source/name) for name in originals}, adapted_sha256=changed)
    record['unchanged_kernel_prefix_sha256'] = hashlib.sha256(originals['csrc/kernels/intranode.cu'][:host_start].encode()).hexdigest()
    (a.out/'fp32_launch_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
