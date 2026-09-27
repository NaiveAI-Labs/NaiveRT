#!/usr/bin/env python3
"""Vendor FlashInfer 0.6.12 mixed communication from an exported source tree.

The export contains csrc/, include/, LICENSE and upstream_jit_comm.py (the
installed flashinfer/jit/comm.py). Device sources and instantiation choices are
unchanged. The extracted generator only redirects source/output directories.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'naivert/prefill/kernels/flashinfer_comm'
FILES = ('LICENSE', 'csrc/mixed_comm.cu', 'csrc/mixed_comm_kernel_inst.jinja',
         'csrc/tvm_ffi_utils.h', 'include/flashinfer/comm/mixed_comm.cuh',
         'include/flashinfer/comm/mixed_comm_decl.cuh')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    files = {name: (args.source / name).read_bytes() for name in FILES}
    upstream = (args.source / 'upstream_jit_comm.py').read_text()
    body = upstream[upstream.index('def gen_mixed_comm_module()'):upstream.index('    import nvidia.nvshmem')]
    body = body.replace('def gen_mixed_comm_module() -> JitSpec:', 'def generate(source, gen_directory):')
    body = body.replace('    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_mixed_comm"\n', '')
    body = body.replace('jit_env.FLASHINFER_CSRC_DIR', '(source / "csrc")')
    header = upstream[:upstream.index('from .core')]
    generator = (header + 'import os\nfrom itertools import product\nfrom typing import Dict, Tuple, List, Any\n'
                 'import jinja2\n\n\ndef write_if_different(path, text):\n'
                 '    if not path.exists() or path.read_text() != text:\n'
                 '        path.write_text(text)\n\n\n' + body + '    return source_paths\n')
    files['generate.py'] = generator.encode()
    files['generator.patch'] = ''.join(difflib.unified_diff(upstream.splitlines(True), generator.splitlines(True),
                                                        fromfile='a/jit/comm.py', tofile='b/generate.py')).encode()
    digest = lambda b: hashlib.sha256(b).hexdigest()
    manifest = dict(repository='https://github.com/flashinfer-ai/flashinfer', version='0.6.12',
                    identity='installed wheel source pinned by SHA256', license='Apache-2.0 (LICENSE)',
                    device_sources_unchanged=True, patches='generator.patch: extract source generator; redirect paths only',
                    upstream_jit_comm_sha256=digest(upstream.encode()),
                    upstream_sha256={name: digest(files[name]) for name in FILES},
                    files={name: digest(data) for name, data in files.items()})
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    for name, data in files.items():
        target = OUT / name
        if args.check:
            assert target.read_bytes() == data, str(target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print(f'{"Checked" if args.check else "Vendored"} {len(files)} files; CUDA sources unchanged')


if __name__ == '__main__':
    main()
