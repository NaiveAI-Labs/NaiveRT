"""CPU-only collection of the completed recursive-chain trace and its controls."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_device_chain import analyze


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--unprofiled-result', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert (a.root / 'exit_code').read_text().strip() == '0'
    result = a.root / 'result.json'
    data = json.loads(result.read_text())
    baseline = json.loads(a.unprofiled_result.read_text())
    assert data['pass_checks'] and data['no_new_xids'] and data['profile_device_chain']
    assert baseline['pass_checks'] and baseline['no_new_xids']
    assert data['records'] == baseline['records'], 'collection changed request outputs/protocols'
    assert data['source_inventory'] == baseline['source_inventory']
    assert data['binaries'] == baseline['binaries']
    assert data['anchor_protocol'] == baseline['anchor_protocol']
    assert data['fields'] == baseline['fields'] == 12576
    assert data['bytes_checked'] == baseline['bytes_checked'] == 2698957056
    databases = [a.root / f'trace.{i}.sqlite' for i in (1, 2)]
    analysis = analyze(result, databases)
    artifacts = [result, *databases, *(a.root / f'trace.{i}.nsys-rep' for i in (1, 2)),
                 a.root / 'profile.sh', a.root / 'gpu.log', a.root / 'exit_code']
    output = dict(pass_checks=True, analysis=analysis, records=data['records'],
                  fields=data['fields'], bytes_checked=data['bytes_checked'],
                  binaries=data['binaries'], harness_sha256=data['harness_sha256'],
                  verify_driver_sha256=data['verify_driver_sha256'],
                  source_inventory_sha256=hashlib.sha256(
                      json.dumps(data['source_inventory'], sort_keys=True).encode()).hexdigest(),
                  unprofiled_result=dict(path=str(a.unprofiled_result), sha256=sha(a.unprofiled_result)),
                  collection_on_off_exact=True, no_new_xids=True,
                  raw_directory=str(a.root), raw_sha256={p.name: sha(p) for p in artifacts},
                  collector_sha256=sha(Path(__file__)),
                  analyzer_sha256=sha(Path(__file__).with_name('analyze_device_chain.py')))
    a.out.write_text(json.dumps(output, indent=2) + '\n')
    print('PASS recursive trace, exact collection-on/off records and unchanged runtime inventories')
