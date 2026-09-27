"""Native W8A8 down: FP32 SwiGLU input, TP8 owners, replay, and BF16 outputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/gpu/naive_n0p5_e2e"))
import torch
from naivert.kernels.naive_n0p5.model_ops import load_extension
from naivert.kernels.packing import pack_down, pack_gate_up
from naivert.models.naive_n0p5.training_moe import quantize_down_activation_reference
from run_naive_n0p5_decode import enable_torch_tp8_peers

H, K, E = 4096, 256, 8


def sync():
    for r in range(8):
        torch.cuda.synchronize(r)


def make_states(ext, weights, scales, inputs, ids, probs, residual, *, layout='umma_sw128'):
    rows = len(ids)
    slab = int(ext.expert_down_mailbox_row_bytes())
    domain_size = int(ext.expert_down_domain_size())
    states = []
    activation_dtype = torch.float32 if getattr(ext, 'expert_activation_is_fp32', lambda: False)() else torch.bfloat16
    for r in range(8):
        packed, scale = pack_down(weights[r], scales[r], hidden=H, outputs_per_tile=64, layout=layout,
                                  scale_dtype=torch.float32 if getattr(ext, "expert_weight_scales_are_fp32", lambda: False)() else torch.bfloat16)
        with torch.cuda.device(r):
            states.append(dict(stream=torch.cuda.Stream(device=r), w=packed.cuda(), s=scale.cuda(),
                               up=inputs[r].to(activation_dtype).cuda(), ids=ids.cuda(), p=probs.cuda(), res=residual.cuda(),
                               out=torch.empty_like(residual, device=r), epoch=torch.ones(1, dtype=torch.int64, device=r),
                               mb=torch.zeros(rows * slab, dtype=torch.uint8, device=r),
                               domain=torch.zeros(rows * domain_size, dtype=torch.uint8, device=r)))
    sync()
    for r, st in enumerate(states):
        with torch.cuda.device(r), torch.cuda.stream(st['stream']):
            for row in reversed(range(rows)):
                args = (
                    st['domain'][row*domain_size:(row+1)*domain_size], st['up'][row], st['w'], st['s'],
                    st['p'][row], st['ids'][row], st['out'][row], st['res'][row],
                    [s['mb'][row*slab:(row+1)*slab] for s in states], 5, 1, r, st['epoch'])
                if activation_dtype == torch.float32 and r == 0 and row == 0:
                    for invalid, reason in ((st['up'][row].bfloat16(), 'dtype'),
                                            (st['up'][row].flatten()[:-1], 'too small')):
                        try:
                            ext.prepare_expert_down_replay_domain_epoch(args[0], invalid, *args[2:])
                        except RuntimeError as exc:
                            assert reason in str(exc)
                        else:
                            raise AssertionError('down accepted an invalid activation buffer')
                ext.prepare_expert_down_replay_domain_epoch(*args)
    return states


def eager(ext, states):
    for r, st in enumerate(states):
        with torch.cuda.device(r), torch.cuda.stream(st['stream']):
            ext.launch_expert_down_rows_from_replay_domain(st['domain'], 1, r, len(st['ids']))


def replay_checks(ext, states):
    eager(ext, states); sync()
    for r, st in enumerate(states):
        with torch.cuda.device(r):
            st['graph'] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(st['graph'], stream=st['stream']):
                ext.launch_expert_down_rows_from_replay_domain(st['domain'], 1, r, len(st['ids']))
    # Capture itself does not run this kernel. Compare changed inputs in eager
    # and graph execution, with distinct epochs so stale payload cannot pass.
    for iteration in range(3):
        for r, st in enumerate(states):
            with torch.cuda.device(r), torch.cuda.stream(st['stream']):
                st['epoch'].add_(1)
                st['up'].mul_(-.5)
                st['ids'].add_(1).remainder_(E)
                st['p'].copy_(st['p'].roll(1, 1))
                st['res'].fill_(iteration - 1)
        eager(ext, states); sync()
        expected = [st['out'].cpu() for st in states]
        for r, st in enumerate(states):
            with torch.cuda.device(r), torch.cuda.stream(st['stream']):
                st['epoch'].add_(1)
                st['graph'].replay()
        sync()
        assert all(torch.equal(st['out'].cpu(), want) for st, want in zip(states, expected))
    # Save the eager answers for both residual values before the stress run.
    expected_residuals = []
    for value in (0, 1):
        for r, st in enumerate(states):
            with torch.cuda.device(r), torch.cuda.stream(st['stream']):
                st['epoch'].add_(1); st['res'].fill_(value)
        eager(ext, states); sync()
        expected_residuals.append([st['out'].cpu() for st in states])
    for r, st in enumerate(states):
        with torch.cuda.device(r):
            st['history'] = torch.empty((40, *st['out'].shape), dtype=st['out'].dtype, device=r)
    sync()
    # Back-to-back exchanges wrap both banks many times. Rotate a delayed rank
    # and retain every output, so a stale intermediate cannot hide behind the
    # last successful exchange.
    for iteration in range(40):
        for r, st in enumerate(states):
            with torch.cuda.device(r), torch.cuda.stream(st['stream']):
                st['epoch'].add_(1)
                st['res'].fill_(iteration % 2)
                if r == iteration % 8: torch.cuda._sleep(20000)
                st['graph'].replay()
                st['history'][iteration].copy_(st['out'])
    sync()
    for r, st in enumerate(states):
        history = st['history'].cpu()
        for iteration in range(40):
            assert torch.equal(history[iteration], expected_residuals[iteration % 2][r]), (r, iteration)
    got = [st['out'].cpu() for st in states]
    assert all(torch.equal(got[0], x) for x in got)
    for r, st in enumerate(states):
        with torch.cuda.device(r), torch.cuda.stream(st['stream']):
            st['epoch'].add_(1)
    eager(ext, states); sync()
    assert all(torch.equal(st['out'].cpu(), want) for st, want in zip(states, got))


def check_case(ext, rows, checkpoint=None, layer=1, *, baseline=None, benchmark=False, fp32_inputs=False):
    generator = torch.Generator().manual_seed(127 + rows + layer)
    ids = torch.stack([torch.randperm(E, generator=generator) for _ in range(rows)]).int()
    probs = torch.rand((rows, E), generator=generator)
    probs /= probs.sum(1, keepdim=True)
    residual = torch.randn((rows, H), generator=generator).bfloat16()
    weights, scales, inputs = [], [], []
    exact, magnitude = torch.zeros((rows, E, H), dtype=torch.float64), torch.zeros((rows, E, H), dtype=torch.float64)
    for r in range(8):
        if checkpoint is None:
            codes = torch.randint(0, 256, (E, H, K), generator=generator, dtype=torch.uint8)
            codes[(codes & 127) == 127] = 0
            w = codes.view(torch.float8_e4m3fn)
            s = torch.exp2(torch.randint(-18, -9, (E, H//128, 2), generator=generator).float()).bfloat16()
            s[1::2] *= 1.1875  # Also exercise the existing arbitrary BF16 scale ABI.
        else:
            pairs = [checkpoint.expert(layer, e, 2, r, 'cpu', kernel_scales=False) for e in range(E)]
            w, s = torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])
        x = torch.randn((rows, E, K), generator=generator)*3
        if not fp32_inputs:
            x = x.bfloat16()
        if checkpoint is None:
            x[:, 0] = 0
            x[:, 1] *= 1e5
            x[:, 2] *= 1e-20
            x[:, 3, 0] = 257024
        q, a = quantize_down_activation_reference(x)
        for t in range(rows):
            for slot in range(E):
                for kg in range(2):
                    ww = w[int(ids[t, slot]), :, kg*128:(kg+1)*128].double()
                    xx = q[t, slot, kg*128:(kg+1)*128].double()
                    scale = s[int(ids[t, slot]), :, kg].double().repeat_interleave(128) * a[t, slot, kg].double()
                    exact[t, slot] += (ww @ xx)*scale
                    magnitude[t, slot] += (ww.abs() @ xx.abs())*scale.abs()
        weights.append(w); scales.append(s); inputs.append(x)
    states = make_states(ext, weights, scales, inputs, ids, probs, residual)
    eager(ext, states); sync()
    # Enclose FP32 dot/reduction/scaling error, then apply both BF16 stores
    # monotonically. Exact dyadic fixtures below independently pin the stores.
    u = 2.0**-24
    bound = (150*u/(1-150*u))*magnitude
    lo, hi = (exact-bound).bfloat16().double(), (exact+bound).bfloat16().double()
    p = probs.double()[:, :, None]
    mix_bound = (15*u/(1-15*u))*(torch.maximum(lo.abs(), hi.abs())*p).sum(1)
    lo, hi = ((lo*p).sum(1)-mix_bound).bfloat16().double(), ((hi*p).sum(1)+mix_bound).bfloat16().double()
    res = residual.double()
    add_bound = (u/(1-u))*(res.abs()+torch.maximum(lo.abs(), hi.abs()))
    lo, hi = (lo+res-add_bound).bfloat16().double(), (hi+res+add_bound).bfloat16().double()
    got = [st['out'].cpu() for st in states]
    assert all(torch.equal(got[0], out) for out in got), 'ranks disagree'
    assert bool(((got[0].double() >= lo) & (got[0].double() <= hi)).all()), 'FP32/BF16 interval failure'
    assert torch.isfinite(got[0].float()).all()
    target = ((exact.bfloat16().double()*p).sum(1).bfloat16()+residual).bfloat16()
    max_abs = float((got[0].float()-target.float()).abs().max())
    extra = {}
    if baseline is not None:
        assert not fp32_inputs, 'bitwise baseline comparison requires common BF16-representable inputs'
        layout = 'qmma' if baseline.expert_down_mailbox_row_bytes() == 8 * 0x70000 else 'umma_sw128'
        old = make_states(baseline, weights, scales, inputs, ids, probs, residual, layout=layout)
        eager(baseline, old); sync()
        assert all(torch.equal(s['out'].cpu(), want) for s, want in zip(old, got)), 'baseline bitwise mismatch'
        extra['baseline_bitwise_equal'] = True
        if benchmark:
            extra['timing'] = benchmark_pair(ext, states, baseline, old)
    replay_checks(ext, states)
    return dict(rows=rows, input_values='fp32' if fp32_inputs else 'bf16-representable',
                checkpoint_layer=layer if checkpoint else None, within_fp32_bf16_bound=True,
                all_ranks_exact=True, graph_matches_eager=True, replay_bank_cycles=40,
                all_replay_outputs_checked=True, rotating_rank_delay=True, reference_max_abs=max_abs, **extra)


def benchmark_pair(ext, states, baseline, old, iterations=100, trials=6):
    # A hundred exchanges per graph amortize host rank-launch skew. Each
    # exchange advances its epoch on the GPU, so no stale mailbox can pass.
    # Timings include that epoch-update kernel; weights/routes are L2-warm.
    for module, group in ((ext, states), (baseline, old)):
        for r, st in enumerate(group):
            with torch.cuda.device(r):
                st['batch'] = torch.cuda.CUDAGraph()
                st['start'], st['end'] = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with torch.cuda.graph(st['batch'], stream=st['stream']):
                    for _ in range(iterations):
                        st['epoch'].add_(1)
                        module.launch_expert_down_rows_from_replay_domain(st['domain'], 1, r, len(st['ids']))
    def run(group):
        for r, st in enumerate(group):
            with torch.cuda.device(r), torch.cuda.stream(st['stream']):
                st['start'].record(); st['batch'].replay(); st['end'].record()
        sync()
        return max(s['start'].elapsed_time(s['end']) for s in group)*1000/iterations
    run(old); run(states)
    times = dict(baseline=[], optimized=[])
    for trial in range(trials):
        order = [('baseline', old), ('optimized', states)]
        if trial % 2: order.reverse()
        for name, group in order: times[name].append(run(group))
    assert all(torch.equal(a['out'].cpu(), b['out'].cpu()) for a, b in zip(states, old))
    return dict(epoch_update_included=True, weights='L2-warm, fixed routes', exchanges_per_graph=iterations,
                microseconds=times, median_microseconds={k: statistics.median(v) for k, v in times.items()})


def boundary_cases(ext):
    reports = []
    for case in range(3):
        weights = [torch.zeros((E, H, K), dtype=torch.float8_e4m3fn) for _ in range(8)]
        scales = [torch.ones((E, H//128, 2), dtype=torch.bfloat16) for _ in range(8)]
        inputs = [torch.zeros((1, E, K), dtype=torch.bfloat16) for _ in range(8)]
        ids = torch.arange(E, dtype=torch.int32)[None]
        p = torch.zeros(1, E); p[0, :2] = .5
        residual = torch.zeros(1, H, dtype=torch.bfloat16)
        wanted = 0.
        if case == 0:
            weights[0][0, :, 0] = 8; scales[0][0] = 32; inputs[0][0, 0, 0] = 1
            weights[1][0, :, 0] = 7; scales[1][0] = 36.5; inputs[1][0, 0, 0] = -1
            wanted = .25
        elif case == 1:
            weights[0][0, :, :2] = 1; inputs[0][0, 0, :2] = torch.tensor([1., 1./256]).bfloat16()
            weights[0][1, :, 0] = 1; inputs[0][0, 1, 0] = -1
        else:
            weights[0][:2, :, 0] = 1
            inputs[0][0, 0, 0] = 1; inputs[0][0, 1, 0] = 2
            p[0, :2] = torch.tensor([.99609375, .00390625]); residual.fill_(-1)
        states = make_states(ext, weights, scales, inputs, ids, p, residual)
        eager(ext, states); sync()
        assert all(bool((st['out'] == wanted).all()) for st in states), (case, wanted, states[0]['out'][0, :4])
        reports.append(dict(case=case, output=wanted, exact=True))
    return reports


def inspect_isa(ext, out):
    sass = subprocess.check_output(['/opt/cuda132-min/bin/cuobjdump', '--dump-sass', ext.__file__], text=True)
    sections = [part for part in sass.split('Function : ') if
                'naivert_expert_down_allreduce_rows' in part.split('\n', 1)[0]]
    assert sections
    for part in sections:
        assert 'UTCQMMA' in part and 'HMMA.' not in part, 'down must use tcgen05 FP8, without HMMA lowering'
    out.with_suffix('.sass').write_text('\nFunction : '.join(sections))
    return len(sections)


def quantization_range(ext, *, fp32_inputs=False):
    # Disjoint identity projections expose each expert's quantized values.
    # Inverse power-of-two weight scales keep even BF16's extremes finite.
    values = [0., 448., 450., 257024., -448., 2.0**-133, torch.finfo(torch.bfloat16).max, 1.]
    dtype = torch.float32 if fp32_inputs else torch.bfloat16
    x = torch.tensor(values, dtype=dtype)[None, :, None].repeat(1, 1, K)
    x[0, 7, :4] = torch.tensor([1., 1./256, -3., 2.0**-100]).bfloat16()
    if fp32_inputs:
        x[0, 1, 1] = 1.0626  # BF16 -> FP8 double-rounding would select the other code.
        x[0, 2] = 448.1      # BF16 would suppress the scale transition from 1 to 2.
        x[0, 5] = 2.0**-149  # FP32 minimum subnormal.
        x[0, 6] = torch.finfo(torch.float32).max
    q, a = quantize_down_activation_reference(x)
    weights = [torch.zeros((E, H, K), dtype=torch.float8_e4m3fn) for _ in range(8)]
    scales = [torch.ones((E, H//128, 2), dtype=torch.bfloat16) for _ in range(8)]
    k = torch.arange(K)
    for e in range(E):
        weights[0][e, e*K+k, k] = 1
        scales[0][e] = (1./a[0, e]).bfloat16()[None, :]
    inputs = [x] + [torch.zeros_like(x) for _ in range(7)]
    states = make_states(ext, weights, scales, inputs, torch.arange(E, dtype=torch.int32)[None],
                         torch.ones(1, E), torch.zeros(1, H, dtype=torch.bfloat16))
    eager(ext, states); sync()
    expected = torch.zeros(1, H, dtype=torch.bfloat16)
    expected[0, :E*K] = q.bfloat16().flatten()
    assert all(torch.equal(st['out'].cpu(), expected) for st in states), 'quantization range identity failure'
    return dict(experts=E, exact=True, input_values='fp32' if fp32_inputs else 'bf16-representable',
                includes_subnormal_and_max=True, checks_double_rounding=fp32_inputs)


def producer_precision(ext, baseline=None):
    """Sparse exact GEMMs isolate the SwiGLU epilogue and its output storage.

    Inputs and weights are dyadic E4M3 values. The short sums are exactly
    representable in FP32, so an FP64 SiLU reference measures epilogue error
    without confusing it with GEMM reduction-order error.
    """
    assert ext.expert_activation_is_fp32()
    generator = torch.Generator().manual_seed(2701)
    columns = [0, 31, 128, 511, 1024, 2047, 3072, 4095]
    gc = torch.randint(-8, 9, (E, K, len(columns)), generator=generator).float() / 2
    uc = torch.randint(-8, 9, (E, K, len(columns)), generator=generator).float() / 2
    g = torch.zeros(E, K, H, dtype=torch.float8_e4m3fn)
    u = torch.zeros_like(g)
    for j, col in enumerate(columns):
        g[:, :, col] = gc[:, :, j].to(g.dtype)
        u[:, :, col] = uc[:, :, j].to(u.dtype)
    gs = torch.full((E, K//128, H//128), .125, dtype=torch.bfloat16)
    us = torch.full_like(gs, .25)
    small = pack_gate_up(g, gs, u, us, hidden=H, scale_dtype=torch.float32)
    packed = torch.zeros((256, *small.shape[1:]), dtype=small.dtype, device=0)
    packed[:E].copy_(small)
    old_packed = packed.clone() if baseline is not None else None
    if old_packed is not None:
        old_packed[:E].copy_(pack_gate_up(g, gs, u, us, hidden=H))
    x = torch.zeros(8, H, dtype=torch.bfloat16)
    x[:, columns] = ((torch.arange(64).reshape(8, 8) % 17 - 8).float() / 4).bfloat16()
    xq = x[:, columns].to(torch.float8_e4m3fn).double()
    gate = torch.einsum('eic,tc->tei', gc.double(), xq) * .125
    up = torch.einsum('eic,tc->tei', uc.double(), xq) * .25
    reference = torch.nn.functional.silu(gate) * up
    hidden = x.cuda(0)
    scores = torch.full((8, 256), -20., device=0)
    scores[:, :E] = torch.arange(E, device=0).float() / 4
    bias = torch.zeros(256, device=0)
    reports = []
    for rows in (1, 4, 8):
        launch = getattr(ext, f'launch_expert_select_up_gate_silu_rows{rows}')
        # Guard both ends; rows and expert slots are all distinct in the oracle.
        guarded = torch.full((rows * E * K + 64,), 12345., device=0)
        y = guarded[32:-32].view(rows, E, 16, 16)
        p = torch.empty(rows, E, device=0)
        indices = torch.empty(rows, E, dtype=torch.int32, device=0)
        launch(hidden[:rows], scores[:rows], bias, packed, y, p, indices)
        torch.cuda.synchronize(0)
        assert bool((guarded[:32] == 12345).all() & (guarded[-32:] == 12345).all())
        actual = y.cpu().reshape(rows, E, K)
        selected = indices.cpu().long()
        want = reference[:rows].gather(1, selected[:, :, None].expand(rows, E, K))
        error = actual.double() - want
        rounded = actual.bfloat16().double()
        old_error = rounded - want
        relative_l2 = float(error.norm() / want.norm())
        bf16_relative_l2 = float(old_error.norm() / want.norm())
        assert relative_l2 < 5e-7, (rows, relative_l2)
        assert relative_l2 < bf16_relative_l2 / 100
        assert bool((actual != actual.bfloat16().float()).any()), 'producer still rounded to BF16'
        try:
            launch(hidden[:rows], scores[:rows], bias, packed, y.bfloat16(), p, indices)
        except RuntimeError as exc:
            assert 'dtype' in str(exc)
        else:
            raise AssertionError('selector accepted a BF16 output in the FP32 build')
        extra = {}
        if baseline is not None:
            old_y = torch.empty_like(y, dtype=torch.bfloat16)
            old_indices = torch.empty_like(indices)
            getattr(baseline, f'launch_expert_select_up_gate_silu_rows{rows}')(
                hidden[:rows], scores[:rows], bias, old_packed, old_y, p, old_indices)
            torch.cuda.synchronize(0)
            assert torch.equal(old_indices, indices)
            assert torch.equal(old_y.cpu().reshape_as(actual), actual.bfloat16())
            extra['bf16_round_matches_old_producer'] = True
        reports.append(dict(rows=rows, relative_l2_fp64=relative_l2,
                            bf16_relative_l2_fp64=bf16_relative_l2,
                            guards_intact=True, wrong_dtype_rejected=True, **extra))
    return reports


def main():
    assert os.environ.get('NAIVERT_NATIVE_W8A8_QUEUE_ACTIVE') == '1'
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--checkpoint')
    ap.add_argument('--small', action='store_true')
    ap.add_argument('--baseline-extension', type=Path)
    ap.add_argument('--benchmark', action='store_true'); args = ap.parse_args()
    if args.benchmark and not args.baseline_extension: ap.error('--benchmark requires --baseline-extension')
    torch.set_num_threads(4)
    ext = load_extension(expert_down='w8a8')
    enable_torch_tp8_peers()
    baseline = None
    if args.baseline_extension:
        spec = importlib.util.spec_from_file_location(args.baseline_extension.stem, args.baseline_extension)
        baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
        assert baseline.expert_down_mailbox_row_bytes() in (8 * 0x70000, 2 * 0x70000), 'expected ABI1 or ABI2 baseline'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    isa = inspect_isa(ext, args.out)
    cases = [check_case(ext, 1, baseline=baseline, benchmark=args.benchmark)]
    if not args.small: cases.append(check_case(ext, 8, baseline=baseline, benchmark=args.benchmark))
    cases.append(check_case(ext, 1, fp32_inputs=True))
    if not args.small: cases.append(check_case(ext, 8, fp32_inputs=True))
    if args.checkpoint:
        from naivert.models.naive_n0p5.checkpoint_fp8 import NaiveNCheckpoint
        reader = NaiveNCheckpoint(args.checkpoint)
        for layer in (1, 41):
            for rows in (1, 8): cases.append(check_case(ext, rows, reader, layer, baseline=baseline))
        reader.close()
    report = dict(status='pass', cases=cases, boundaries=boundary_cases(ext),
                  quantization=[quantization_range(ext), quantization_range(ext, fp32_inputs=True)], native_fp8_kernel_variants=isa,
                  producer_precision=producer_precision(ext, baseline),
                  mailbox_row_bytes=int(ext.expert_down_mailbox_row_bytes()))
    if args.baseline_extension:
        report['baseline_extension'] = dict(path=str(args.baseline_extension), sha256=hashlib.sha256(args.baseline_extension.read_bytes()).hexdigest())
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
