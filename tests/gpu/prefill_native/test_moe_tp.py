"""One rank, one layer: the TP-sharded grouped-GEMM MoE against an fp32 reference.

Reference: this rank's expert slices dequantised from the checkpoint's own block scales
(``dequant_block``), fp32 GEMMs, SiLU, weighted sum in fp32 -- the decode-side numerics of
the rank partial.  The candidate differs by the per-token e4m3 group quant of the activations
and the UE8M0 re-quantisation of the weights (the reference recipe's numerics), so the check
is a relative-error bound plus determinism, not bitwise.  Also times each stage.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--layer', type=int, default=1)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--tokens', type=int, default=4096)
    p.add_argument('--reps', type=int, default=20)
    a = p.parse_args()
    if not os.environ.get('GPU_QUEUE_TICKET_ID'):
        raise RuntimeError('use the GPU queue')
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.repo))
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    from naivert.models.naive_n0p5.checkpoint_fp8 import NaiveNCheckpoint, dequant_block
    from naivert.prefill.native.moe_tp import TPMoE, load_tp_experts, EXPERTS, TOPK, HIDDEN, INTER
    from naivert.prefill.ops import moe_route_fused
    dev = torch.device('cuda', a.rank)
    torch.cuda.set_device(dev)
    ck = NaiveNCheckpoint(a.target)
    t0 = time.monotonic()
    weights = load_tp_experts(ck, a.layer, a.rank, dev)
    torch.cuda.synchronize(dev)
    load_s = time.monotonic() - t0
    print('experts loaded', round(load_s, 1), 's', {k: (tuple(v[0].shape), tuple(v[1].shape), v[1].dtype) for k, v in weights.items()}, flush=True)
    # routing from the real router on random normed activations
    g = torch.Generator(device=dev).manual_seed(20260919)
    x = (torch.randn((a.tokens, HIDDEN), generator=g, device=dev) * 0.5).to(torch.bfloat16)
    router = ck.read(f'model.layers.{a.layer}.mlp.gate.weight', dev).float()
    bias = ck.read(f'model.layers.{a.layer}.mlp.gate.e_score_correction_bias', dev).float()
    logits = torch.mm(x.float(), router.mT)
    w_k, ids = moe_route_fused(logits, bias, topk=TOPK)
    ids = ids.to(torch.int32); w_k = w_k.float()
    print('routing', ids.shape, ids.dtype, 'experts hit', int(torch.unique(ids).numel()), 'max rows', int(torch.bincount(ids.reshape(-1).long(), minlength=EXPERTS).max()), flush=True)
    moe = TPMoE(a.tokens, dev)
    with torch.inference_mode():
        out = moe.forward(x, ids, w_k, weights).clone()
        torch.cuda.synchronize(dev)
        out2 = moe.forward(x, ids, w_k, weights).clone()
        torch.cuda.synchronize(dev)
    exact_repeat = torch.equal(out, out2)
    # reference: fp32 with the checkpoint's own scales
    ref = torch.zeros((a.tokens, HIDDEN), dtype=torch.float32, device=dev)
    xf = x.float()
    with torch.inference_mode():
        for e in range(EXPERTS):
            mask = (ids == e)
            tok = mask.any(1).nonzero().flatten()
            if tok.numel() == 0:
                continue
            (gq, gs), (uq, us), (dq, ds) = (ck.expert(a.layer, e, which, a.rank, dev, kernel_scales=False) for which in range(3))
            gw, uw, dw = (dequant_block(q, s).float() for q, s in ((gq, gs), (uq, us), (dq, ds)))
            h = xf.index_select(0, tok)
            act = torch.nn.functional.silu(h @ gw.mT) * (h @ uw.mT)
            y = act @ dw.mT
            wsel = (w_k * mask).sum(1).index_select(0, tok)
            ref.index_add_(0, tok, y * wsel[:, None])
    # reference 2: the candidate's own quantisation emulated in torch (fp32 accumulate):
    # requantised UE8M0 weights dequantised, activations per-token group-128 e4m3 with UE8M0
    # scales, bf16 rounding of the down partial rows as DeepGEMM writes them.
    from naivert.kernels.naive_n0p5.prefill import requantize_expert_ue8m0
    from deep_gemm.utils import per_token_cast_to_fp8
    def deq(q, s):
        return (q.float().reshape(q.shape[0] // 128, 128, q.shape[1] // 128, 128) * s.float()[:, None, :, None]).reshape(q.shape)
    xq, xs = per_token_cast_to_fp8(x, use_ue8m0=True)
    xdq = (xq.float().reshape(a.tokens, HIDDEN // 128, 128) * xs[:, :, None]).reshape(a.tokens, HIDDEN)
    ref2 = torch.zeros((a.tokens, HIDDEN), dtype=torch.float32, device=dev)
    with torch.inference_mode():
        for e in range(EXPERTS):
            mask = (ids == e)
            tok = mask.any(1).nonzero().flatten()
            if tok.numel() == 0:
                continue
            (gq, gs), (uq, us), (dq, ds) = (ck.expert(a.layer, e, which, a.rank, dev, kernel_scales=False) for which in range(3))
            w13q, w13s = requantize_expert_ue8m0(torch.cat((gq, uq)).contiguous(), torch.cat((gs, us)).float().contiguous())
            w2q, w2s = requantize_expert_ue8m0(dq.contiguous(), ds.float().contiguous())
            w13 = deq(w13q, w13s); w2 = deq(w2q, w2s)
            h = xdq.index_select(0, tok)
            gu = h @ w13.mT
            act = (torch.nn.functional.silu(gu[:, :INTER]) * gu[:, INTER:]).to(torch.bfloat16)
            aq, ascale = per_token_cast_to_fp8(act, use_ue8m0=True)
            adq = (aq.float().reshape(-1, INTER // 128, 128) * ascale[:, :, None]).reshape(-1, INTER)
            y = (adq @ w2.mT).to(torch.bfloat16).float()
            wsel = (w_k * mask).sum(1).index_select(0, tok)
            ref2.index_add_(0, tok, y * wsel[:, None])
    err2 = out - ref2
    print('vs emulated quantised path: rel_rms', float(err2.norm() / ref2.norm()), 'max_abs', float(err2.abs().max()), flush=True)
    err = (out - ref)
    rel = float(err.norm() / ref.norm()); max_abs = float(err.abs().max())
    per_tok = err.norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30)
    # timings
    ev = lambda: torch.cuda.Event(enable_timing=True)
    stages = {'layout': [], 'gather_quant': [], 'experts': [], 'combine': [], 'total': []}
    with torch.inference_mode():
        for _ in range(a.reps):
            e0, e1, e2, e3, e4 = ev(), ev(), ev(), ev(), ev()
            e0.record(); moe.layout.build(ids); e1.record(); moe.gather_quant(x); e2.record()
            moe.experts(weights); e3.record(); moe.combine(w_k); e4.record()
            torch.cuda.synchronize(dev)
            for name, (s, t) in zip(('layout', 'gather_quant', 'experts', 'combine', 'total'), ((e0, e1), (e1, e2), (e2, e3), (e3, e4), (e0, e4))):
                stages[name].append(s.elapsed_time(t))
    med = {k: round(sorted(v)[len(v)//2], 3) for k, v in stages.items()}
    rep = dict(layer=a.layer, rank=a.rank, tokens=a.tokens, load_seconds=load_s, layout_rows=moe.layout.rows,
               exact_repeat=exact_repeat, rel_rms=rel, max_abs=max_abs, per_token_rel_median=float(per_tok.median()),
               per_token_rel_max=float(per_tok.max()), ref_rms=float(ref.square().mean().sqrt()),
               stage_ms_median=med, memory_gib=round(torch.cuda.memory_allocated(dev) / 2**30, 2),
               out_dtype=str(moe.down.dtype), rel_rms_vs_emulated_quant=float(err2.norm() / ref2.norm()), max_abs_vs_emulated=float(err2.abs().max()))
    print('RESULT', json.dumps(rep), flush=True)
    (a.out / 'result.json').write_text(json.dumps(rep, indent=1) + '\n')


if __name__ == '__main__':
    main()
