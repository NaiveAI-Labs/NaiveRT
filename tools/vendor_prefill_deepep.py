#!/usr/bin/env python3
"""Vendor pinned DeepEP 1.2.1 with the tested single-process host adapter.

The CUDA sources are copied unchanged. --check reproduces the extraction in a
temporary directory and compares every emitted file with the checked-in copy.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'naivert/prefill/kernels/deepep'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / 'deepep'
        subprocess.run([sys.executable, str(ROOT / 'tests/gpu/prefill_native/prepare_deepep_inprocess.py'),
                        '--source', str(a.source), '--out', str(dest)], check=True)
        cpp = dest / 'csrc/deep_ep.cpp'
        text = cpp.read_text()
        for name in ('Config', 'EventHandle', 'Buffer'):
            text = text.replace(f'(m, "{name}")', f'(m, "{name}", pybind11::module_local())')
        # Explicit group cleanup first drains every rank. This also permits
        # cleanup after partial host setup, before a collective can be launched.
        text = text.replace('void Buffer::destroy() {', 'void Buffer::destroy(bool all_ranks_drained) {')
        text = text.replace('        // Barrier\n        intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, comm_stream);',
                            '        // Local owners may already have drained every peer.\n'
                            '        if (not all_ranks_drained)\n'
                            '            intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, comm_stream);')
        text = text.replace('.def("destroy", &deep_ep::Buffer::destroy,',
                            '.def("destroy", &deep_ep::Buffer::destroy, py::arg("all_ranks_drained") = false,')
        cpp.write_text(text)
        header = dest / 'csrc/deep_ep.hpp'
        header.write_text(header.read_text().replace('void destroy();', 'void destroy(bool all_ranks_drained = false);'))
        setup = dest / 'setup.py'
        text = setup.read_text().replace("name='deep_ep_cpp'", "name='naivert_deep_ep_cpp'")
        text = text.replace("include_dirs.append('/usr/local/cuda/include/cccl')",
                            "include_dirs.append(str(Path(os.environ['CUDA_HOME']) / 'include/cccl'))")
        # Source paths must stay valid when the build runs in an external cache.
        text = text.replace('    # Put them together',
                            "    root = Path(__file__).resolve().parent\n"
                            "    sources = [str(root / path) for path in sources]\n"
                            "    include_dirs = [str(root / path) for path in include_dirs]\n\n"
                            '    # Put them together')
        start, end = text.index('    # noinspection PyBroadException'), text.index('    setuptools.setup(')
        text = text[:start] + "    revision = ''\n\n" + text[end:]
        text = text.replace("name='deep_ep',", "name='naivert-prefill-deepep',")
        setup.write_text(text)
        (dest / 'host_adapter_manifest.json').unlink()
        patch = ''.join(''.join(difflib.unified_diff(
            (a.source / name).read_text().splitlines(True), (dest / name).read_text().splitlines(True),
            fromfile='a/' + name, tofile='b/' + name))
            for name in ('csrc/deep_ep.cpp', 'csrc/deep_ep.hpp', 'setup.py'))
        (dest / 'host_adapter.patch').write_text(patch)
        files = {str(path.relative_to(dest)): sha(path) for path in sorted(dest.rglob('*')) if path.is_file()}
        upstream = {name: sha(a.source / name) for name in files if (a.source / name).is_file()}
        device = [name for name in files if name.startswith('csrc/kernels/')]
        assert all(files[name] == upstream[name] for name in device)
        # The device-source digests are recorded in the manifest written below.
        manifest = dict(repository='https://github.com/deepseek-ai/DeepEP', version='1.2.1',
                        revision=None, identity='source archive pinned by SHA256; installed source has no Git metadata',
                        license='MIT (LICENSE)', device_sources_unchanged=True,
                        patches='host_adapter.patch: local peer pointers, explicit cleanup, GIL release, isolated module/build',
                        upstream_sha256=upstream, files=files)
        (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        for path in sorted(dest.rglob('*')):
            if not path.is_file():
                continue
            target = OUT / path.relative_to(dest)
            if a.check:
                assert target.read_bytes() == path.read_bytes(), str(target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
        print(f'{"Checked" if a.check else "Vendored"} {len(files)} files; {len(device)} unchanged device files')


if __name__ == '__main__':
    main()
