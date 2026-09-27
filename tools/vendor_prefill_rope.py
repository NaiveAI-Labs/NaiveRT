#!/usr/bin/env python3
"""Vendor unchanged FlashInfer 0.6.12 RoPE and its local header dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'naivert/prefill/kernels/flashinfer_rope'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path,
                        help='exported FlashInfer data/ plus LICENSE and upstream_jit_rope.py')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = args.source.resolve()
    pending = [root/'csrc/rope.cu', root/'csrc/flashinfer_rope_binding.cu']
    files = {'LICENSE': (root/'LICENSE').read_bytes()}
    while pending:
        path = pending.pop()
        name = str(path.relative_to(root))
        if name in files:
            continue
        files[name] = path.read_bytes()
        for include in re.findall(r'#\s*include\s*[<"]([^>"]+)[>"]', path.read_text()):
            for candidate in (path.parent/include, root/'include'/include, root/'csrc'/include):
                if candidate.is_file() and candidate.resolve().is_relative_to(root):
                    pending.append(candidate.resolve())
                    break
    digest = lambda data: hashlib.sha256(data).hexdigest()
    manifest = dict(repository='https://github.com/flashinfer-ai/flashinfer', version='0.6.12',
                    identity='installed wheel source pinned by SHA256', license='Apache-2.0 (LICENSE)',
                    device_sources_unchanged=True, patches=[],
                    upstream_jit_rope_sha256=digest((root/'upstream_jit_rope.py').read_bytes()),
                    files={name: digest(data) for name, data in sorted(files.items())})
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    for name, data in files.items():
        target = OUT/name
        if args.check:
            assert target.read_bytes() == data, str(target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print(f'{"Checked" if args.check else "Vendored"} {len(files)} files; all upstream sources unchanged')


if __name__ == '__main__':
    main()
