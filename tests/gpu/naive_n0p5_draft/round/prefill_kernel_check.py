"""Kernel prefill against the verify-8 replay on the same target: caches, features, logits, greedy.

For each prompt: prefill it once through ``--prefill verify8`` (the decode kernels) and once
through the kernel prefill (``naivert.prefill.draft_features``), both ending in one seq1 step,
then compare what the draft round would see -- the KV/index caches, the per-position draft
features, the last-position logits -- and decode 64 greedy tokens from each state (free and
teacher-forced on the verify8 continuation).  Times the kernel prefill three times per prompt
(first = eager JIT, second = capture, third = replay) and the verify8 replay once.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import statistics
from contextlib import ExitStack

from run import load, sha


def _main(resources):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dataset', type=Path, help='prompt_text jsonl; the first --prompts rows')
    p.add_argument('--prompts', type=int, default=2)
    p.add_argument('--lengths', default='200,2200,4096', help='synthetic prompt lengths (repeated text)')
    p.add_argument('--greedy', type=int, default=64)
    p.add_argument('--skip-seq1-control', action='store_true',
                   help='skip the separate one-token replay control for long prompts; verify8 comparison remains')
    p.add_argument('--require-repeat-exact', action='store_true',
                   help='fail after saving the report if native cache/features/logits differ across repeated runs')
    p.add_argument('--capacity', type=int, default=8192)
    p.add_argument('--mode', choices=('kernel', 'native'), default='kernel', help='which batched executor to compare against verify8')
    p.add_argument('--profile', action='store_true', help='CUPTI-profile the third kernel repetition: GPU kernel time per device vs wall')
    p.add_argument('--dump', action='store_true', help='save the first case: verify8 and kernel features, K/V/index caches of six layers, greedy logits')
    p.add_argument('--warm-samples', type=int, default=1, help='additional unprofiled warm replays to time')
    p.add_argument('--native-dsa', choices=('original', 'bounded', 'stable', 'both'), default='both',
                   help='audit ablation: original selector/sort, bounded scan, stable selector, or both')
    p.add_argument('--native-index-norm', choices=('reference', 'decode'),
                   help='override the executor default / NAIVERT_PREFILL_REF_NORM')
    p.add_argument('--time-all-reduces', action='store_true',
                   help='record CUDA events around layer-0 reductions inside the chunk graph')
    p.add_argument('--paired-dsa-latency', action='store_true',
                   help='fresh-process alternating baseline/candidate timing; no CUPTI or numerical probe')
    p.add_argument('--paired-native-baseline', type=Path,
                   help='compare against a saved native/executor.py, including all-rank byte equality')
    p.add_argument('--paired-correctness-only', action='store_true',
                   help='one baseline pass, then candidate first/capture/replay passes with exact state comparisons')
    p.add_argument('--paired-native-bindings', type=Path,
                   help='saved prefill.py bindings for the baseline executor (keeps its layout implementation)')
    p.add_argument('--paired-allow-numerical-differences', action='store_true',
                   help='timing only: report byte differences instead of requiring equality; requires separate full-model numerical validation')
    p.add_argument('--native-candidate', type=Path,
                   help='isolated experiment module exporting NativePrefill; leaves the installed executor unchanged')
    a = p.parse_args()
    assert not a.native_candidate or a.mode == 'native'
    assert not a.paired_allow_numerical_differences or a.paired_native_baseline
    assert not a.paired_correctness_only or (a.paired_native_baseline and not a.paired_allow_numerical_differences)
    a.out.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(a.repo))
    from naivert.kernels.naive_n0p5.model_ops import load_extension
    from naivert.models.naive_n0p5.dspark import TARGET_LAYER_IDS
    from naivert.models.naive_n0p5.checkpoint_fp8 import NaiveNCheckpoint, load_tp8_weights
    from naivert.prefill.comms import TP8Comm
    from naivert.prefill.draft_features import DraftFeaturePrefillExecutor
    from naivert.prefill.planning import PrefillPlanConfig
    from naivert.prefill.operands import kernel_prefill_weights
    from prefill import prefill_tokens
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    verify = load('run_naive_n0p5_verify', a.repo/'tests/gpu/naive_n0p5_e2e/run_naive_n0p5_verify.py'); decode = verify.decode
    dsa = load('run_attn_dsa', a.repo/'tests/gpu/attn_dsa/run_attn_dsa.py')
    dsa8 = load('run_attn_dsa_rows', a.repo/'tests/gpu/attn_dsa/run_attn_dsa_rows.py')
    swa = load('run_attn_swa', a.repo/'tests/gpu/attn_swa/run_attn_swa.py')
    ops = load_extension(weights='fp8', expert_down='w8a8')
    ext_dsa, ext_dsa8, ext_swa, ext_swa8 = dsa.build(False, 16), dsa8.build(8, index_heads=16), swa.build(1), swa.build(8)
    decode.enable_torch_tp8_peers()
    capacity = a.capacity
    checkpoint = NaiveNCheckpoint(a.target)
    t0 = time.monotonic()
    weights = load_tp8_weights(checkpoint, 48, capacity, ext_dsa.workspace_bytes(), ext_swa.workspace_bytes(),
                               expert_down='w8a8', ops=ops)
    print('target weights', round(time.monotonic()-t0, 1), 's', flush=True)
    mailboxes = [torch.zeros(verify.mailbox_bytes(ops), dtype=torch.uint8, device=r) for r in range(8)]
    inboxes = [torch.zeros(4096, dtype=torch.uint8, device=r) for r in range(8)]
    ranks = [decode.RankRuntime(r, weights[r], ops, ext_dsa, ext_swa, capacity, mailboxes, inboxes, pdl=True, fused_unproj=True) for r in range(8)]
    for rt in ranks: rt.pdl_markers = decode.PDL_W8A8_MARKERS
    verifies = [verify.VerifyRuntime(rt, ext_dsa8, ext_swa8, ext_dsa8.workspace_bytes(), ext_swa8.workspace_bytes(),
                                     pdl=True, fused_unproj=True) for rt in ranks]
    t0 = time.monotonic()
    kweights = kernel_prefill_weights(weights, checkpoint, capacity=capacity)
    print('kernel prefill operands', round(time.monotonic()-t0, 1), 's', flush=True)
    free = [torch.cuda.mem_get_info(r) for r in range(8)]
    print('free memory GiB per device', [round(f/2**30, 1) for f, _ in free], flush=True)
    tokenizer = AutoTokenizer.from_pretrained(a.target, local_files_only=True, trust_remote_code=False)
    cases = []
    if a.dataset:
        rows = [json.loads(l) for l in a.dataset.read_text().splitlines() if l.strip()][:a.prompts]
        for row in rows:
            cases.append((row['id'], tokenizer.encode(row['prompt_text'], add_special_tokens=False)))
    text = 'Explain how memory bandwidth, cache reuse, and tensor parallel communication affect GPU inference latency. '
    part = tokenizer.encode(text, add_special_tokens=False)
    for n in [int(x) for x in a.lengths.split(',') if x]:
        cases.append((f'synthetic{n}', (part*((n+len(part)-1)//len(part)))[:n]))
    max_len = max(len(t) for _, t in cases) + 8
    assert max_len + a.greedy + 8 < capacity
    features = [torch.empty(max_len, 4096, dtype=torch.bfloat16, device=r) for r in range(8)]
    streams = [rt.stream for rt in ranks]
    if a.mode == 'native':
        from naivert.prefill.native.executor import NativePrefill
        from naivert.kernels.naive_n0p5 import prefill as P
        topk_module = P._module(P._TOPK)
        original_epilogue = topk_module._indexer_topk_prefill_epilogue_kernel
        candidate_type = load('native_prefill_audit_candidate', a.native_candidate).NativePrefill if a.native_candidate else NativePrefill
        kprefill = candidate_type(kweights, streams, feature_layers=TARGET_LAYER_IDS, features=features)
        if hasattr(kprefill, 'close'):
            resources.callback(kprefill.close)
        if a.native_index_norm is not None:
            kprefill.reference_index_norm = a.native_index_norm == 'reference'
        if a.native_dsa in ('original', 'bounded'):
            topk_module._indexer_topk_prefill_epilogue_kernel = original_epilogue
        if a.native_dsa in ('original', 'stable'):
            sparse_attention = P.sparse_attention
            def unbounded_attention(*args, **kwargs):
                kwargs.pop('selected_starts', None)
                kwargs.pop('selected_ends', None)
                return sparse_attention(*args, **kwargs)
            P.sparse_attention = unbounded_attention
    else:
        kprefill = DraftFeaturePrefillExecutor(kweights, streams, comm=TP8Comm([w.dev for w in kweights], streams),
            max_seq_len=capacity, graphs=True, plan_config=PrefillPlanConfig(strategy='budget'), feature_layers=TARGET_LAYER_IDS, features=features)
    print('executor ready', kprefill.versions, flush=True)
    reduction_events = []
    if a.time_all_reduces:
        assert a.mode == 'native', 'event diagnostic is for the native chunk graph'
        assert kprefill.reduce_scatter_o, 'layer-0 event diagnostic requires MoE O reduce-scatter'
        reduce = kprefill.comm.all_reduce_
        def timed_reduce(tensors):
            begins = [torch.cuda.Event(enable_timing=True, external=True) for _ in streams]
            ends = [torch.cuda.Event(enable_timing=True, external=True) for _ in streams]
            for event, stream in zip(begins, streams):
                # PyTorch chooses cudaEventRecordExternal by inspecting the
                # current stream, even when record() receives an explicit one.
                with torch.cuda.device(stream.device), torch.cuda.stream(stream):
                    event.record(stream)
            reduce(tensors)
            for event, stream in zip(ends, streams):
                with torch.cuda.device(stream.device), torch.cuda.stream(stream):
                    event.record(stream)
            reduction_events.append((begins, ends))
        kprefill.comm.all_reduce_ = timed_reduce
    cache_names = ('k_cache', 'v_cache', 'ki_cache', 'ki_scale')

    def sync():
        for r in range(8): torch.cuda.synchronize(r)

    def phase_done(name, phase, seconds, executor=None):
        # Long-context runs can spend minutes in each pass. Persist progress and
        # graph/memory growth without adding device events to the measured pass.
        executor = executor or kprefill
        row = dict(case=name, phase=phase, seconds=seconds,
                   chunk_graph_sets=len(getattr(executor, '_chunk_captured', {})),
                   memory=[dict(rank=r, allocated=torch.cuda.memory_allocated(r),
                                reserved=torch.cuda.memory_reserved(r),
                                peak_allocated=torch.cuda.max_memory_allocated(r)) for r in range(8)])
        with (a.out/'phases.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print('PHASE', name, phase, round(seconds, 3), 's',
              'graph sets', row['chunk_graph_sets'],
              'max allocated GiB', round(max(x['allocated'] for x in row['memory'])/2**30, 2), flush=True)

    def reset():
        for r, rt in enumerate(ranks):
            with torch.cuda.device(r), torch.cuda.stream(rt.stream):
                rt.pos.zero_()
                for lw in weights[r].layer:
                    for n in cache_names:
                        if n in lw: lw[n].zero_()
        kprefill.begin_request(); sync()

    def snapshot(n):
        sync()
        return [dict(caches=[{k: lw[k][:min(lw[k].shape[0], n)].clone() for k in cache_names if k in lw} for lw in weights[r].layer],
                     logits=ranks[r].logits.clone(), features=features[r][:n].clone(),
                     pos=int(ranks[r].pos.item())) for r in range(8)]

    def full_logits():
        sync(); return torch.cat([rt.logits.float().cpu().reshape(-1) for rt in ranks])

    def step(token):
        for r, rt in enumerate(ranks):
            with torch.cuda.device(r), torch.cuda.stream(rt.stream): rt.token.fill_(token)
        for rt in ranks: rt.run_step()

    def restore(state, n):
        for r, rt in enumerate(ranks):
            with torch.cuda.device(r), torch.cuda.stream(rt.stream):
                for lw, stored in zip(weights[r].layer, state[r]['caches']):
                    for k, x in stored.items(): lw[k].zero_(); lw[k][:x.shape[0]].copy_(x)
                rt.logits.copy_(state[r]['logits']); rt.pos.fill_(n)
        sync()

    def greedy(state, n, forced=None):
        restore(state, n); out = []; lg = full_logits(); rows = [lg]
        for j in range(a.greedy):
            tok = int(lg.argmax()); out.append(tok)
            nxt = forced[j] if forced is not None else tok
            if j + 1 < a.greedy:
                step(nxt); lg = full_logits(); rows.append(lg)
        return out, torch.stack(rows)

    def rel(x, y):
        x, y = x.float(), y.float()
        return float((x-y).square().mean().sqrt() / y.square().mean().sqrt().clamp_min(1e-30))

    sync()
    for rt in ranks: rt.capture()
    if a.paired_dsa_latency or a.paired_native_baseline:
        assert a.mode == 'native' and a.native_dsa == 'both'
        assert not (a.paired_dsa_latency and a.paired_native_baseline)
        assert not a.profile and not a.time_all_reduces
        assert a.warm_samples > 0
        assert not a.paired_dsa_latency or len(cases) == 1
        baseline_type = NativePrefill
        if a.paired_native_baseline:
            baseline_module = load('native_prefill_audit_baseline', a.paired_native_baseline)
            if a.paired_native_bindings:
                spec = importlib.util.spec_from_file_location('prefill_bindings_audit_baseline', a.paired_native_bindings)
                bindings = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = bindings  # dataclasses resolve annotation strings through this entry
                spec.loader.exec_module(bindings)
                baseline_module.P = bindings
            baseline_type = baseline_module.NativePrefill
        baseline = baseline_type(kweights, streams, feature_layers=TARGET_LAYER_IDS, features=features)
        if hasattr(baseline, 'close'):
            resources.callback(baseline.close)
        baseline.reference_index_norm = kprefill.reference_index_norm
        stable_epilogue = topk_module._indexer_topk_prefill_epilogue_kernel
        bounded_attention = P.sparse_attention
        def original_attention(*args, **kwargs):
            kwargs.pop('selected_starts', None)
            kwargs.pop('selected_ends', None)
            return bounded_attention(*args, **kwargs)
        executors = {'original': baseline, 'both': kprefill}
        for name, tokens in cases:
            try:
                for label, executor in executors.items():
                    original_dsa = a.paired_dsa_latency and label == 'original'
                    topk_module._indexer_topk_prefill_epilogue_kernel = original_epilogue if original_dsa else stable_epilogue
                    P.sparse_attention = original_attention if original_dsa else bounded_attention
                    # Each executor owns its graph pool and workspaces; weights,
                    # input, decode tail and cache reset are identical.
                    for _ in range(0 if a.paired_correctness_only else 4):
                        reset(); executor.begin_request(); sync()
                        prefill_tokens(ranks, verifies, tokens, features, TARGET_LAYER_IDS, mode='native', executor=executor)
                        sync()
            finally:
                topk_module._indexer_topk_prefill_epilogue_kernel = stable_epilogue
                P.sparse_attention = bounded_attention
            samples = {label: [] for label in executors}
            launches = {label: [] for label in executors}
            for rep in range(0 if a.paired_correctness_only else a.warm_samples):
                for label in (('original', 'both') if rep % 2 == 0 else ('both', 'original')):
                    executor = executors[label]
                    reset(); executor.begin_request(); sync()
                    began = time.perf_counter()
                    prefill_tokens(ranks, verifies, tokens, features, TARGET_LAYER_IDS, mode='native', executor=executor)
                    sync(); samples[label].append((time.perf_counter() - began) * 1000)
                    launches[label].append(executor._stats['launch_ms'])
            medians = {label: statistics.median(values) for label, values in samples.items() if values}
            report = dict(name=name, prompt_tokens=len(tokens), samples_ms=samples, median_ms=medians,
                          saved_ms=medians.get('original', 0)-medians.get('both', 0), graph_launches=launches,
                          harness_sha256=sha(Path(__file__)), versions=kprefill.versions,
                          timing='alternating AB/BA, reset excluded, final seq1 token included; no profiler or internal events')
            if a.paired_native_baseline:
                states = []
                first_pass_ms = {}
                for label, executor in executors.items():
                    reset(); executor.begin_request(); sync()
                    began = time.perf_counter()
                    prefill_tokens(ranks, verifies, tokens, features, TARGET_LAYER_IDS, mode='native', executor=executor)
                    sync(); first_pass_ms[label] = (time.perf_counter() - began) * 1000
                    if a.paired_correctness_only:
                        phase_done(name, label+'_first_pass', first_pass_ms[label]/1000, executor=executor)
                    states.append(snapshot(len(tokens)))
                exact = []
                def same_bytes(left, right):
                    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
                for left, right in zip(*states):
                    exact.append(dict(
                        caches=all(same_bytes(v, right['caches'][L][key])
                                   for L, layer in enumerate(left['caches']) for key, v in layer.items()),
                        features=same_bytes(left['features'], right['features']),
                        logits=same_bytes(left['logits'], right['logits']), pos=left['pos'] == right['pos']))
                report['baseline_source_sha256'] = sha(a.paired_native_baseline)
                if a.paired_native_bindings:
                    report['baseline_bindings_sha256'] = sha(a.paired_native_bindings)
                report['baseline_path'] = str(a.paired_native_baseline)
                report['all_rank_exact'] = exact
                if a.paired_correctness_only:
                    for field in ('samples_ms', 'median_ms', 'saved_ms', 'graph_launches'):
                        report.pop(field)
                    report['first_pass_ms'] = first_pass_ms
                    report['timing'] = 'correctness gate: one baseline pass; candidate first, capture and replay; not alternating latency samples'
                    report['candidate_followups'] = []
                    # Hold the baseline state while checking capture and replay.
                    # At long spans this avoids rewarming the expensive baseline
                    # solely to repeat a previously measured latency comparison.
                    for rep in range(2):
                        reset(); kprefill.begin_request(); sync(); began = time.perf_counter()
                        prefill_tokens(ranks, verifies, tokens, features, TARGET_LAYER_IDS,
                                       mode='native', executor=kprefill)
                        sync(); seconds = time.perf_counter()-began
                        phase_done(name, f'candidate_followup_{rep}', seconds)
                        current = snapshot(len(tokens))
                        matches = [dict(rank=r,
                            caches=all(same_bytes(v, current[r]['caches'][L][key])
                                       for L, layer in enumerate(left['caches']) for key, v in layer.items()),
                            features=same_bytes(left['features'], current[r]['features']),
                            logits=same_bytes(left['logits'], current[r]['logits']),
                            pos=left['pos'] == current[r]['pos']) for r, left in enumerate(states[0])]
                        report['candidate_followups'].append(dict(seconds=seconds, all_rank_exact=matches))
                        del current
                report['equality_method'] = 'byte comparison of contiguous uint8 views; includes signed-zero representation'
                report['comparison'] = 'saved native executor vs candidate native executor'
                report['numerical_differences_allowed'] = a.paired_allow_numerical_differences
                if a.paired_allow_numerical_differences:
                    report['accuracy_gate'] = 'not evaluated here; requires separate full-model numerical validation'
                    report['last_logits'] = []
                    for r, (left, right) in enumerate(zip(*states)):
                        old, new = left['logits'].double(), right['logits'].double()
                        assert torch.isfinite(old).all() and torch.isfinite(new).all()
                        error = new-old
                        report['last_logits'].append(dict(rank=r,
                            rel_rms=float(error.square().mean().sqrt()/old.square().mean().sqrt().clamp_min(1e-30)),
                            max_abs=float(error.abs().max()), argmax_equal=bool(old.argmax()==new.argmax())))
            filename = 'paired_native_latency.json' if a.paired_native_baseline else 'paired_dsa_latency.json'
            if a.paired_correctness_only:
                filename = 'paired_native_correctness.json'
            if a.native_candidate:
                report['candidate_source_sha256'] = sha(a.native_candidate)
                report['candidate_path'] = str(a.native_candidate)
            if len(cases) > 1:
                filename = f'{name}_{filename}'
            (a.out/filename).write_text(json.dumps(report, indent=2)+'\n')
            if a.paired_native_baseline:
                assert all(rank['pos'] for rank in exact), exact
                if not a.paired_allow_numerical_differences:
                    assert all(all(rank.values()) for rank in exact), exact
                if a.paired_correctness_only:
                    assert all(all(value for key, value in rank.items() if key != 'rank')
                               for followup in report['candidate_followups']
                               for rank in followup['all_rank_exact']), report['candidate_followups']
            print('PAIRED CORRECTNESS' if a.paired_correctness_only else 'PAIRED LATENCY', name,
                  json.dumps(first_pass_ms if a.paired_correctness_only else medians), flush=True)
        for executor in (baseline, kprefill):
            if hasattr(executor, 'close'):
                executor.close()
        (a.out/'lifecycle.json').write_text(json.dumps(dict(cleanup_completed=True,
            candidate_closed=getattr(kprefill, '_closed', None),
            baseline_closed=getattr(baseline, '_closed', None)), indent=2)+'\n')
        return
    reports = []
    for name, tokens in cases:
        n = len(tokens); print('CASE', name, n, flush=True)
        reset(); t0 = time.monotonic()
        prefill_tokens(ranks, verifies, tokens, features, TARGET_LAYER_IDS, mode='verify8')
        sync(); v8_seconds = time.monotonic()-t0
        phase_done(name, 'verify8', v8_seconds)
        A = snapshot(n); lgA = full_logits()
        timings = []
        profile = None
        for rep in range(3):
            reset(); t0 = time.monotonic()
            if a.profile and rep == 2:
                from torch.profiler import profile as tprof, ProfilerActivity
                with tprof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    t0 = time.monotonic()
                    if a.mode == 'native':
                        # Match prefill_tokens: NativePrefill.run accepts one
                        # chunk, while the request may span several chunks.
                        for start in range(0, n - 1, kprefill.chunk_tokens):
                            end = min(start + kprefill.chunk_tokens, n - 1)
                            kprefill.run(tokens[start:end], start=start)
                    else:
                        kprefill.run(tokens[:n-1], start=0)
                    sync(); run_wall = time.monotonic()-t0
                busy = {}; count = {}
                for e in prof.events():
                    if e.device_type.name == 'CUDA':
                        busy[e.device_index] = busy.get(e.device_index, 0) + e.time_range.elapsed_us()
                        count[e.device_index] = count.get(e.device_index, 0) + 1
                first = {}; last = {}; complete_ms = {}; nccl_ms = {}
                for e in prof.events():
                    if e.device_type.name == 'CUDA':
                        di = e.device_index; st_, en_ = e.time_range.start, e.time_range.end
                        first[di] = min(first.get(di, st_), st_); last[di] = max(last.get(di, en_), en_)
                        if 'complete' in e.name: complete_ms[di] = complete_ms.get(di, 0) + e.time_range.elapsed_us() / 1000
                        if 'nccl' in e.name.lower(): nccl_ms[di] = nccl_ms.get(di, 0) + e.time_range.elapsed_us() / 1000
                trace_origin = min(first.values())
                timeline = dict(first_kernel_start_ms=[round((first[d_] - trace_origin) / 1000, 2) for d_ in sorted(first)],
                                last_kernel_end_ms=[round((last[d_] - trace_origin) / 1000, 2) for d_ in sorted(last)],
                                complete_ms=[round(complete_ms.get(d_, 0), 1) for d_ in sorted(first)],
                                nccl_ms=[round(nccl_ms.get(d_, 0), 1) for d_ in sorted(first)])
                print('TIMELINE', json.dumps(timeline), flush=True)
                by_name = {}
                for e in prof.events():
                    if e.device_type.name == 'CUDA' and e.device_index == 0:
                        t, c = by_name.get(e.name, (0, 0)); by_name[e.name] = (t + e.time_range.elapsed_us(), c + 1)
                top = sorted(by_name.items(), key=lambda kv: -kv[1][0])[:25]
                comm = sum(t for name, (t, c) in by_name.items() if 'nccl' in name.lower() or 'ncclDevKernel' in name)
                profile = dict(executor_run_wall_s=run_wall, timeline=timeline, launch_ms=kprefill._stats.get('launch_ms'), owned_pairs_per_rank=kprefill._stats.get('owned_pairs_per_rank'), gpu_kernel_busy_ms_per_device={k: round(v/1000, 2) for k, v in sorted(busy.items())},
                               kernels_per_device={k: v for k, v in sorted(count.items())}, nccl_ms_device0=round(comm/1000, 2),
                               top_kernels_device0=[dict(name=n[:90], ms=round(t/1000, 2), count=c) for n, (t, c) in top])
                print('PROFILE', json.dumps(profile), flush=True)
                # Keep collective arrival/duration evidence, not a multi-GB trace.
                profile['nccl_events'] = [dict(device=e.device_index, name=e.name,
                    start_ms=(e.time_range.start-trace_origin)/1000,
                    duration_ms=e.time_range.elapsed_us()/1000) for e in prof.events()
                    if e.device_type.name == 'CUDA' and 'nccl' in e.name.lower()]
                for r, rt in enumerate(ranks):
                    with torch.cuda.device(r), torch.cuda.stream(rt.stream): rt.pos.fill_(n-1)
                prefill_tokens(ranks, verifies, tokens, features, TARGET_LAYER_IDS,
                               start=n-1, mode='seq1')
                sync(); timings.append(time.monotonic()-t0); continue
            prefill_tokens(ranks, verifies, tokens, features, TARGET_LAYER_IDS, mode=a.mode, executor=kprefill)
            sync(); timings.append(time.monotonic()-t0)
            phase_done(name, f'native_pass_{rep}', timings[-1])
        warm_seconds = []
        warm_reductions_ms = []
        for _ in range(a.warm_samples):
            reset(); began = time.monotonic()
            prefill_tokens(ranks, verifies, tokens, features, TARGET_LAYER_IDS, mode=a.mode, executor=kprefill)
            sync(); warm_seconds.append(time.monotonic() - began)
            phase_done(name, f'warm_{len(warm_seconds)-1}', warm_seconds[-1])
            if reduction_events:
                warm_reductions_ms.append([[start.elapsed_time(end) for start, end in zip(begins, ends)]
                                           for begins, ends in reduction_events[-2:]])
        stats = {k: v for k, v in kprefill._stats.items() if k in ('tokens', 'chunks', 'graph_replays', 'graph_captures', 'eager_segments', 'collectives', 'seconds', 'owned_pairs', 'launch_ms', 'chunk_graphs', 'chunk_capture_refused', 'owned_pairs_per_rank', 'max_expert_rows', 'expert_cap')}
        B = snapshot(n); lgB = full_logits()
        if a.dump and not reports:
            dump_layers = [L for L in range(48) if weights[0].layer[L]['kind'] == 'full_dsa'][:8] + [1, 7, 24, 45]
            torch.save(dict(name=name, tokens=tokens, feature_layers=list(TARGET_LAYER_IDS),
                            features_A=[A[r]['features'].cpu() for r in range(8)], features_B=[B[r]['features'].cpu() for r in range(8)],
                            caches_A={L: {k: v.cpu() for k, v in A[0]['caches'][L].items()} for L in dump_layers},
                            caches_B={L: {k: v.cpu() for k, v in B[0]['caches'][L].items()} for L in dump_layers},
                            kinds={L: weights[0].layer[L]['kind'] for L in dump_layers}), a.out/'dump_first_case.pt')
        # Compare the full handoff on all ranks, including final logits.
        reset(); prefill_tokens(ranks, verifies, tokens, features, TARGET_LAYER_IDS, mode=a.mode, executor=kprefill)
        C = snapshot(n)
        repeat_by_rank = [dict(rank=r,
            caches=all(torch.equal(x.view(torch.uint8), C[r]['caches'][L][key].view(torch.uint8))
                       for L, layer in enumerate(B[r]['caches']) for key, x in layer.items()),
            features=torch.equal(B[r]['features'].view(torch.uint8), C[r]['features'].view(torch.uint8)),
            logits=torch.equal(B[r]['logits'].view(torch.uint8), C[r]['logits'].view(torch.uint8)))
            for r in range(8)]
        repeat_exact = all(row['caches'] and row['features'] and row['logits'] for row in repeat_by_rank)
        repeat_diff = dict(layers=[L for L, (lb, lc) in enumerate(zip(B[0]['caches'], C[0]['caches']))
                                   if not all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(lb.values(), lc.values()))][:12],
                           feature_ranks=[r for r in range(8) if not torch.equal(B[r]['features'].view(torch.uint8), C[r]['features'].view(torch.uint8))],
                           logits_equal=bool(torch.equal(B[0]['logits'], C[0]['logits'])))
        if not repeat_exact:
            detail = {}
            for L in repeat_diff['layers'][:2]:
                lb, lc = B[0]['caches'][L], C[0]['caches'][L]
                for key in lb:
                    x, y = lb[key].view(torch.uint8).reshape(lb[key].shape[0], -1), lc[key].view(torch.uint8).reshape(lc[key].shape[0], -1)
                    rows = (x != y).any(1).nonzero().flatten()
                    if rows.numel():
                        fx, fy = lb[key].float() if lb[key].dtype != torch.float8_e4m3fn else lb[key].view(torch.uint8).float(), lc[key].float() if lc[key].dtype != torch.float8_e4m3fn else lc[key].view(torch.uint8).float()
                        detail[f'L{L}.{key}'] = dict(rows=int(rows.numel()), first=rows[:8].tolist(), last=rows[-3:].tolist(), max_abs=float((fx - fy).abs().max()))
            for r in repeat_diff['feature_ranks'][:2]:
                x, y = B[r]['features'], C[r]['features']
                rows = (x.view(torch.uint8).reshape(x.shape[0], -1) != y.view(torch.uint8).reshape(y.shape[0], -1)).any(1).nonzero().flatten()
                detail[f'features_rank{r}'] = dict(rows=int(rows.numel()), first=rows[:8].tolist(), last=rows[-3:].tolist(), max_abs=float((x.float() - y.float()).abs().max()))
            repeat_diff['detail'] = detail
            print('REPEAT DIFF', json.dumps(repeat_diff), flush=True)
        def compare(X, lgX, label):
            layers = []
            for L, (la, lx) in enumerate(zip(A[0]['caches'], X[0]['caches'])):
                item = dict(layer=L, kind=weights[0].layer[L]['kind'])
                for k in ('k_cache', 'v_cache'):
                    item[k+'_rel_rms'] = rel(lx[k], la[k]); item[k+'_max_abs'] = float((lx[k].float()-la[k].float()).abs().max())
                if item['kind'] == 'sliding_window' and n >= 136:
                    live = torch.tensor([q % 136 for q in range(n-128, n)], device=la['k_cache'].device)
                    item['k_attended_rows_rel_rms'] = rel(lx['k_cache'][live], la['k_cache'][live])
                if 'ki_cache' in la:
                    item['ki_cache_byte_equal_fraction'] = float((lx['ki_cache'].view(torch.uint8) == la['ki_cache'].view(torch.uint8)).float().mean())
                    item['ki_scale_rel_rms'] = rel(lx['ki_scale'], la['ki_scale'])
                layers.append(item)
            feats = []
            for r in range(8):
                fa, fx = A[r]['features'][:n-1].float(), X[r]['features'][:n-1].float()
                per_pos = (fa-fx).norm(dim=1)/fa.norm(dim=1).clamp_min(1e-30)
                feats.append(dict(rank=r, layer=TARGET_LAYER_IDS[r], rel_rms=rel(fx, fa), per_position_median=float(per_pos.median()),
                                  per_position_p90=float(per_pos.quantile(0.9)), positions_over_20pct=int((per_pos > 0.2).sum()),
                                  last_row_seq1_rel_rms=rel(X[r]['features'][n-1], A[r]['features'][n-1])))
            lpA, lpX = lgA.double().log_softmax(-1), lgX.double().log_softmax(-1)
            last = dict(argmax_equal=int(lgA.argmax()) == int(lgX.argmax()), argmax=[int(lgA.argmax()), int(lgX.argmax())],
                        top10_overlap=len(set(lpA.topk(10).indices.tolist()) & set(lpX.topk(10).indices.tolist())),
                        max_abs_logit_diff=float((lgA-lgX).abs().max()), top1_logprob=[float(lpA.max()), float(lpX.max())],
                        kl_A_X=float((lpA.exp()*(lpA-lpX)).sum()))
            gA, rowsA = greedy(A, n); gX, rowsX = greedy(X, n)
            agree = sum(x == y for x, y in zip(gA, gX)); first_div = next((i for i, (x, y) in enumerate(zip(gA, gX)) if x != y), None)
            _, rowsTX = greedy(X, n, forced=gA)
            forced_agree = int((rowsA.argmax(-1) == rowsTX.argmax(-1)).sum())
            margins = rowsA.topk(2, -1).values.diff(dim=-1).abs().flatten(); decided = margins > 1.0
            forced_decided_agree = int(((rowsA.argmax(-1) == rowsTX.argmax(-1)) & decided).sum())
            lpa, lpx = rowsA.double().log_softmax(-1), rowsTX.double().log_softmax(-1)
            kl_rows = (lpa.exp() * (lpa - lpx)).sum(-1)
            forced_extra = dict(mean_kl=float(kl_rows.mean()), p90_kl=float(kl_rows.quantile(0.9)), max_kl=float(kl_rows.max()),
                                mean_abs_logit_diff=float((rowsA - rowsTX).abs().mean()),
                                forced_token_logprob_mean_abs_diff=float((lpa.gather(1, torch.tensor(gA[:rowsA.shape[0]]).view(-1, 1)) - lpx.gather(1, torch.tensor(gA[:rowsA.shape[0]]).view(-1, 1))).abs().mean()))
            full = dict(label=label, layers=layers, features=feats, last_logits=last, forced_extra=forced_extra,
                        greedy=dict(agree=agree, of=a.greedy, first_divergence=first_div, verify8=gA, other=gX,
                                    text_verify8=tokenizer.decode(gA), text_other=tokenizer.decode(gX)),
                        teacher_forced=dict(argmax_agree=forced_agree, of=int(rowsA.shape[0]), decided=int(decided.sum()), decided_agree=forced_decided_agree,
                                            max_abs_logit_diff=float((rowsA-rowsTX).abs().max())))
            brief = dict(label=label, greedy_agree=f'{agree}/{a.greedy}', forced_agree=f'{forced_agree}/{rowsA.shape[0]} decided {forced_decided_agree}/{int(decided.sum())}',
                         forced_mean_kl=round(forced_extra['mean_kl'], 5), forced_p90_kl=round(forced_extra['p90_kl'], 5), forced_token_lp_diff=round(forced_extra['forced_token_logprob_mean_abs_diff'], 5),
                         last_argmax_equal=last['argmax_equal'], kl=round(last['kl_A_X'], 6), top10_overlap=last['top10_overlap'],
                         features_rel_rms=[round(f['rel_rms'], 4) for f in feats], features_pos_median=[round(f['per_position_median'], 4) for f in feats],
                         features_pos_over20=[f['positions_over_20pct'] for f in feats],
                         k_rel_rms_max=max(l['k_cache_rel_rms'] for l in layers), v_rel_rms_max=max(l['v_cache_rel_rms'] for l in layers),
                         k_attended_max=max((l.get('k_attended_rows_rel_rms', 0) for l in layers), default=None),
                         dsa_k_rel=[(l['layer'], round(l['k_cache_rel_rms'], 4)) for l in layers if l['kind'] == 'full_dsa'])
            return full, brief
        kernel_full, kernel_brief = compare(B, lgB, 'kernel_vs_verify8')
        seq1_seconds = None
        control_full = control_brief = None
        if not a.skip_seq1_control:
            reset(); t0 = time.monotonic()
            prefill_tokens(ranks, verifies, tokens, features, TARGET_LAYER_IDS, mode='seq1')
            sync(); seq1_seconds = time.monotonic()-t0
            Cs = snapshot(n); lgC = full_logits()
            control_full, control_brief = compare(Cs, lgC, 'seq1_vs_verify8')
        rep = dict(name=name, prompt_tokens=n, verify8_prefill_seconds=v8_seconds, seq1_prefill_seconds=seq1_seconds,
                   kernel_prefill_seconds=timings, kernel_stats=stats, kernel_repeat_exact=repeat_exact, repeat_diff=repeat_diff,
                   warm_prefill_seconds=warm_seconds, repeat_by_rank=repeat_by_rank,
                   warm_all_reduce_ms=warm_reductions_ms,
                   kernel=kernel_full, control=control_full, profile=profile)
        (a.out/f'{name}.json').write_text(json.dumps(rep, indent=1)+'\n')
        brief = dict(name=name, prompt_tokens=n, verify8_prefill_seconds=round(v8_seconds, 3), seq1_prefill_seconds=round(seq1_seconds, 3) if seq1_seconds is not None else None,
                     kernel_prefill_seconds=[round(t, 3) for t in timings], kernel_repeat_exact=repeat_exact, kernel=kernel_brief, control=control_brief)
        brief['warm_prefill_median_ms'] = statistics.median(warm_seconds)*1000 if warm_seconds else None
        brief['repeat_by_rank'] = repeat_by_rank
        print('RESULT', json.dumps(brief), flush=True)
        reports.append(brief)
        if a.require_repeat_exact:
            assert repeat_exact, repeat_by_rank
    candidate = None
    if a.native_candidate:
        candidate = dict(path=str(a.native_candidate), sha256=sha(a.native_candidate))
    cleanup_completed = False
    if hasattr(kprefill, 'close'):
        kprefill.close()
        cleanup_completed = True
        if candidate is not None:
            candidate['cleanup_completed'] = True
    (a.out/'summary.json').write_text(json.dumps(dict(cases=reports, free_memory_gib_after_load=[round(f/2**30, 1) for f, _ in free],
        versions=kprefill.versions, feature_layers=list(TARGET_LAYER_IDS), harness_sha256=sha(Path(__file__)),
        native_dsa=a.native_dsa, native_index_norm=a.native_index_norm, candidate=candidate,
        cleanup_completed=cleanup_completed), indent=1)+'\n')


def main():
    with ExitStack() as resources:
        _main(resources)


if __name__ == '__main__':
    main()
