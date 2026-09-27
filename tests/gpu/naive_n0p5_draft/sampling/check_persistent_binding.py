"""TP8 public sampler/tail equivalence and bounded memory checking.

Captured real logits and controlled edge inputs; no model or latency claim.
Capture every rank before launching from eight host threads, so sanitizer
instrumentation cannot strand an eager rank waiting for an unlaunched peer.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def work(a):
    import numpy as np
    import torch
    sys.path.insert(0, str(a.repo))
    from naivert.backends.naive_n0p5_draft_tail import DraftTail, DraftVerifyState
    from naivert.kernels.naive_n0p5.dspark_sampling import TargetSampler, SHARD
    from naivert.runtime.cuda_backend import CudaRuntimeGraphApi
    from naivert.runtime.mtp_contract import _DecodeControlLayout as Control
    from naivert.runtime.sequencer import _SequencerOutputRecordLayout as Record
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    def xids():
        return [s for s in subprocess.check_output(['dmesg'], text=True).splitlines() if 'NVRM: Xid' in s]
    before_xids = xids()
    source = {str(p.relative_to(a.repo)): sha(p) for p in (a.repo/'naivert').rglob('*')
              if p.is_file() and p.suffix in ('.py', '.cu', '.cuh', '.hpp', '.h', '.cpp')}
    api = CudaRuntimeGraphApi.load('/opt/cuda132-min/lib64/libcudart.so')
    api.library.cudaDeviceEnablePeerAccess.argtypes = [C.c_int, C.c_uint]
    for r in range(8):
        with torch.cuda.device(r):
            for peer in range(8):
                if peer == r: continue
                rc = api.library.cudaDeviceEnablePeerAccess(peer, 0)
                assert rc in (0, 704), rc
                if rc == 704: api.library.cudaGetLastError()
    streams = [torch.cuda.Stream(device=r) for r in range(8)]
    buffers, targets, logits, embeddings, weights = [], [], [], [], []
    capacity = 256
    for r in range(8):
        def z(*shape, dtype=torch.int32): return torch.zeros(shape, dtype=dtype, device=r)
        b = SimpleNamespace(device=torch.device('cuda', r), capacity=capacity, sampling=True,
            target_hidden=z(8, 4096, dtype=torch.bfloat16), ctx_valid=z(8, dtype=torch.bfloat16),
            ctx_pos=z(8, dtype=torch.int64), kv_dst=z(8, dtype=torch.int64),
            blk_pos=z(8, dtype=torch.int64), key_add=z(capacity+8, dtype=torch.float32),
            draft_tokens=z(7, dtype=torch.int64), block_ids=z(8, dtype=torch.int64),
            resid=[z(8, 4096, dtype=torch.bfloat16)], sampling_state=z(2, dtype=torch.int64),
            wdog=z(1), logit_trace=z(7, SHARD, dtype=torch.float32))
        buffers.append(b)
        targets.append(DraftVerifyState(z(8, 4096, dtype=torch.bfloat16), z(1), z(1),
            z(1, dtype=torch.int64), z(8), z(C.sizeof(Control), dtype=torch.uint8), z(1, dtype=torch.int64)))
        logits.append(z(8, SHARD, dtype=torch.float32))
        embeddings.append(z(8*SHARD, 4096, dtype=torch.bfloat16))
        weights.append(SimpleNamespace(mask_embedding=z(4096, dtype=torch.bfloat16)))
    core = SimpleNamespace(buffers=buffers, weights=weights)
    samplers = {name: TargetSampler(logits, buffers, [t.tokens for t in targets], persistent=name=='persistent')
                for name in ('chain', 'persistent')}
    anchors = {name: TargetSampler([v[:1] for v in logits], buffers, [t.tokens[:1] for t in targets],
                                  anchor=True, persistent=name=='persistent') for name in samplers}
    host = dict(host_ring=torch.zeros(64*C.sizeof(Record), dtype=torch.uint8, pin_memory=True),
                host_flags=torch.zeros(64, dtype=torch.int32, pin_memory=True),
                cancel=torch.zeros(2, dtype=torch.int32, pin_memory=True), ring_capacity=64)
    tails = {name: DraftTail(core, targets, sampler, embeddings, **host, chain=False)
             for name, sampler in samplers.items()}
    binaries = {kind+'.'+name: dict(path=module.__file__, sha256=sha(module.__file__))
                for kind, modules in (('sampler', {n:s.ext for n,s in samplers.items()}),
                                      ('tail', {n:t.extension for n,t in tails.items()}))
                for name, module in modules.items()}
    def sync():
        for stream in streams: stream.synchronize()
        assert all(b.wdog.item()==0 for b in buffers)
        assert all(s[4].item()==0 for tail in tails.values() for s in tail.states)
    for r in range(8): torch.cuda.synchronize(r)
    graphs = {}
    for anchor in (False, True):
        for name in samplers:
            graph = []
            for r in range(8):
                with torch.cuda.device(r), torch.cuda.stream(streams[r]):
                    g = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(g, stream=streams[r]):
                        if anchor: anchors[name].run(r)
                        else:
                            samplers[name].run(r, defer_consumed=True)
                            tails[name].run(r)
                    audit = api.inspect_graph(g.raw_cuda_graph())
                    expected_nodes = (1 if anchor else 2) if name=='persistent' else 33
                    assert audit['node_count'] == expected_nodes, (name, anchor, audit)
                    graph.append(g)
            graphs[anchor, name] = graph
    sync()
    print('PASS captured public sampler/tail bindings', flush=True)
    raw = {n: torch.from_numpy(np.fromfile(a.inputs/f'{n}.bin', dtype=np.int64 if n=='proposals' else np.float32))
           for n in ('target', 'draft', 'proposals')}
    p = raw['target'].reshape(8, 8*SHARD); q = raw['draft'].reshape(7, 8*SHARD)
    generator = torch.Generator().manual_seed(20260918)
    cases = [('real', p, q), ('uniform', torch.zeros_like(p), torch.zeros_like(q))]
    if not a.focused:
        cases.append(('random', torch.randn(p.shape, generator=generator)*4,
                      torch.randn(q.shape, generator=generator)*4))
    fields = bytes_checked = 0; records = []
    def setup(name, p, q, seed):
        tail = tails[name]
        for r, (b, t) in enumerate(zip(buffers, targets)):
            with torch.cuda.device(r), torch.cuda.stream(streams[r]):
                f = tail.fields[r]
                for n, x in f.items():
                    if n not in ('embed', 'mask_embedding', 'row_pos', 'trash_row', 'kv_pos'): x.zero_()
                f['cache_len'].fill_(192); f['ctx_rows'].fill_(1)
                b.draft_tokens.copy_(raw['proposals']); b.sampling_state.copy_(torch.tensor([seed, 0]))
                logits[r].copy_(p[:, r*SHARD:(r+1)*SHARD]); b.logit_trace.copy_(q[:, r*SHARD:(r+1)*SHARD])
                t.features.fill_(r+1); t.position.fill_(193); t.generation.zero_(); t.epoch.zero_()
                t.tokens.fill_(-1); t.decode_control.zero_(); t.previous_complete_ns.zero_()
                tail.states[r][7].zero_(); tail.geometry(r)
        sync()
    def values(name, anchor):
        sampler = anchors[name] if anchor else samplers[name]
        rows = []
        for r, (b, t) in enumerate(zip(buffers, targets)):
            row = dict(tokens=sampler.outputs[r], p=sampler.full[r])
            if not anchor:
                row.update(q=sampler.draft_full[r], sampling_state=b.sampling_state,
                    position=t.position, generation=t.generation, epoch=t.epoch,
                    control=t.decode_control, counts=tails[name].states[r][7],
                    **{k:x for k,x in tails[name].fields[r].items() if k not in ('embed', 'mask_embedding')})
            rows.append({k:x.cpu().clone() for k,x in row.items()})
        return rows
    with ThreadPoolExecutor(max_workers=8) as workers:
        for label, p, q in cases:
            for seed in ((1234,) if a.focused else (1234, -1)):
                for anchor in (False, True):
                    expected = []
                    for name in samplers:
                        setup(name, p, q, seed)
                        sampler = anchors[name] if anchor else samplers[name]
                        before = [x.cpu().tolist() for x in sampler.flags]
                        assert len({row[2] for row in before}) == 1
                        for repeat in range(2):
                            def launch(r):
                                with torch.cuda.device(r), torch.cuda.stream(streams[r]):
                                    graphs[anchor, name][r].replay(); streams[r].synchronize()
                            list(workers.map(launch, range(8))); sync()
                            for r, flag in enumerate(sampler.flags):
                                generation = before[r][2]+repeat+1
                                assert flag.tolist() == [0 if name=='persistent' else generation, generation, generation]
                            actual = values(name, anchor)
                            if name=='chain': expected.append(actual)
                            else:
                                for r, row in enumerate(actual):
                                    for field, x in row.items():
                                        assert torch.equal(x.view(torch.uint8), expected[repeat][r][field].view(torch.uint8)), (label, seed, anchor, repeat, r, field)
                                        fields += 1; bytes_checked += x.numel()*x.element_size()
                        records.append(dict(input=label, seed=seed, anchor=anchor, mode=name, replays=2, exact=True))
                    print('PASS', label, seed, 'anchor' if anchor else 'sampler + tail', flush=True)
    assert before_xids == xids()
    assert all(sha(a.repo/n)==h for n,h in source.items())
    assert all(sha(v['path'])==v['sha256'] for v in binaries.values())
    result = dict(pass_checks=True, focused=a.focused, records=records, fields=fields,
                  bytes_checked=bytes_checked, source_inventory=source, binaries=binaries,
                  harness_sha256=sha(__file__), inputs={n:sha(a.inputs/f'{n}.bin') for n in raw}, no_new_xids=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print('PASS public binding check', fields, bytes_checked, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo', 'inputs', 'out'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--focused', action='store_true')
    work(parser.parse_args())
    import gc
    import torch
    gc.collect()
    for rank in range(8):
        with torch.cuda.device(rank):
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated()==0, torch.cuda.memory_summary()
            torch.cuda.empty_cache()
            assert torch.cuda.memory_reserved()==0
    torch._C._host_emptyCache()
    print('PASS allocator cleanup', flush=True)
