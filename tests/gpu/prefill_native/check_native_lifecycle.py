"""Construct and close the native executor without loading a checkpoint."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from naivert.prefill.native.executor import NativePrefill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    weights, features, streams = [], [], []
    for r in range(8):
        dev = torch.device('cuda', r)
        streams.append(torch.cuda.Stream(device=dev))
        with torch.cuda.device(dev):
            weights.append(SimpleNamespace(dev=dev, layers=1, layer=[{'kind': 'full_dsa'}],
                rope_swa=torch.zeros((2, 128), device=dev),
                rope_full=torch.zeros((2, 128), device=dev)))
            features.append(torch.empty((4096, 4096), device=dev, dtype=torch.bfloat16))
    executor = NativePrefill(weights, streams, feature_layers=[0] * 8, features=features)
    refs = [weakref.ref(t) for t in executor.index_rows + executor.index_queries]
    assert len(refs) == 16
    try:
        assert all(t() is not None for t in refs)
    finally:
        executor.close()
    executor.close()
    assert executor._closed and not executor._chunk_captured
    assert all(t() is None for t in refs), 'closed executor retained index workspaces'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(success=True, repeated_close_safe=True,
        index_workspaces_released=len(refs), versions=executor.versions), indent=2) + '\n')
    print('NATIVE_LIFECYCLE_OK', flush=True)


if __name__ == '__main__':
    main()
