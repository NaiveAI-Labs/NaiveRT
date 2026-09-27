"""Bounded diagnostic for DeepEP's existing intra-node buffer setup in one process.

Run under gpu_profile_queue and an external timeout. No model or executor changes.
Use the public C++ Buffer methods to exchange device IDs/IPC handles locally.
Destroy is a collective CUDA barrier in the installed library and its Python
binding holds the GIL, so this isolated process releases its CUDA contexts on exit
instead of calling destroy sequentially. This is not a production lifecycle.
"""
import argparse
import json
import os
from pathlib import Path
import traceback


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    import torch
    import deep_ep_cpp
    torch.set_num_threads(1)
    result = dict(torch=torch.__version__, ranks=8, nvl_bytes=128 << 20,
                  rdma_bytes=0, low_latency=False, stage='allocate', available=[])
    buffers = []
    try:
        for r in range(8):
            with torch.cuda.device(r):
                buffers.append(deep_ep_cpp.Buffer(r, 8, result['nvl_bytes'], 0, False, True))
            print('allocated', r, flush=True)
        ids = [b.get_local_device_id() for b in buffers]
        handles = [b.get_local_ipc_handle() for b in buffers]
        result['device_ids'] = ids
        result['stage'] = 'sync'
        for r, buffer in enumerate(buffers):
            result['sync_rank'] = r
            with torch.cuda.device(r):
                buffer.sync(ids, handles, None)
                torch.cuda.synchronize(r)
                result['available'].append(buffer.is_available())
            print('synced', r, flush=True)
        result['stage'] = 'complete'
        result['success'] = all(result['available'])
    except Exception as exc:
        result['success'] = False
        result['error'] = str(exc)
        result['traceback'] = traceback.format_exc()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    # All work above is synchronized. Process teardown releases the temporary
    # allocations; do not enter the installed GIL-holding collective destructor.
    os._exit(0)


if __name__ == '__main__':
    main()
