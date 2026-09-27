"""Compare existing RoPE kernels at the executor's actual strided Q/K geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def run(a, report):
    import torch
    import flashinfer
    from flashinfer.rope import apply_rope_with_cos_sin_cache
    from naivert.kernels.naive_n0p5 import prefill as P
    from naivert.models.naive_n0p5.layouts import rope_table

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    torch.manual_seed(941)
    fi_root = Path(flashinfer.__file__).parent
    report.update(torch=torch.__version__, flashinfer=flashinfer.__version__, backend=a.backend,
                  samples=a.samples, inner=a.inner, cases=[], sources={})
    if a.backend == 'vendored':
        from naivert.prefill.native.rope import load_extension
        report['vendored_library'] = load_extension()[1]
    for path in (Path(P.__file__), Path(P.__file__).parents[2] / 'prefill/ops.py',
                 fi_root / 'rope.py', fi_root / 'data/include/flashinfer/pos_enc.cuh',
                 fi_root / 'data/csrc/rope.cu'):
        report['sources'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    variants = ('separate', 'legacy_joint', 'flashinfer_joint')
    for rows in (512, 4096):
        projection = torch.randn((rows, P.QKV_ROWS), device='cuda', dtype=torch.bfloat16)
        q, k, _ = P.split_qkv(projection)
        qk = projection[:, :(P.ATTN_HEADS + P.KV_HEADS) * P.HEAD_DIM].view(rows, -1, P.HEAD_DIM)
        positions = torch.arange(rows, device='cuda', dtype=torch.int64)
        table = rope_table(1e4, 64, 65536 + rows).cuda()
        fi_table = torch.cat((table[:, 0::2], table[:, 1::2]), dim=-1).contiguous()
        graphs, outputs = {}, {}

        def call(label):
            if label == 'separate':
                result = (P.rope(q, positions, table, imag='cos'),
                          P.rope(k, positions, table, imag='cos'))
            elif label == 'legacy_joint':
                out = P.rope(qk, positions, table, imag='cos')
                result = (out[:, :P.ATTN_HEADS], out[:, P.ATTN_HEADS:])
            elif a.backend == 'installed':
                qo, ko = apply_rope_with_cos_sin_cache(
                    positions, q.flatten(1), k.flatten(1), P.HEAD_DIM, fi_table, is_neox=False)
                result = (qo.view_as(q), ko.view_as(k))
            else:
                result = P.rope_qk(q, k, positions, fi_table)
            outputs[label] = result

        for label in variants:
            call(label)
            torch.cuda.synchronize()
            graphs[label] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[label]):
                for _ in range(a.inner): call(label)
        for theta in (1e4, 1e7):
            table.copy_(rope_table(theta, 64, table.shape[0]))
            fi_table.copy_(torch.cat((table[:, 0::2], table[:, 1::2]), dim=-1))
            for offset in (0, 4096, 65536):
                # Every case changes both input values and position IDs after capture.
                projection.normal_()
                projection[0, :32] = 0.0
                projection[0, 32:64] = -0.0
                positions.copy_(torch.arange(rows, device='cuda') + offset)
                for label in variants: graphs[label].replay()
                torch.cuda.synchronize()
                metrics = {}
                cos = table[positions, 0::2].double()[:, None, :]
                sin = table[positions, 1::2].double()[:, None, :]
                for h, inp in enumerate((q, k)):
                    oracle = inp.double().clone()
                    re, im = inp[..., :64:2].double(), inp[..., 1:64:2].double()
                    oracle[..., :64:2] = re*cos-im*sin
                    oracle[..., 1:64:2] = im*cos+re*sin
                    for label in variants:
                        out = outputs[label][h]
                        ref = outputs['separate'][h]
                        metrics[f'{label}_{("q", "k")[h]}'] = dict(
                            byte_different_elements=int((out.contiguous().view(torch.int16) != ref.contiguous().view(torch.int16)).sum()),
                            max_abs_vs_baseline=float((out.float()-ref.float()).abs().max()),
                            rotated_rms_vs_fp64=float((out[..., :64].double()-oracle[..., :64]).square().mean().sqrt()),
                            output_stride=list(out.stride()),
                            tail_bytes_exact=bool(torch.equal(out[..., 64:].contiguous().view(torch.int16), inp[..., 64:].contiguous().view(torch.int16))))
                        assert torch.isfinite(out).all()
                        assert metrics[f'{label}_{("q", "k")[h]}']['tail_bytes_exact']
                        if label == 'legacy_joint':
                            assert metrics[f'{label}_{("q", "k")[h]}']['byte_different_elements'] == 0
                samples = {label: [] for label in variants}
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for rep in range(a.samples + 4):
                    for label in (variants if rep % 2 == 0 else variants[::-1]):
                        begin.record(); graphs[label].replay(); end.record(); end.synchronize()
                        if rep >= 4: samples[label].append(begin.elapsed_time(end) * 1000 / a.inner)
                result = dict(rows=rows, theta=theta, offset=offset, metrics=metrics,
                              samples_us=samples, median_us={v: statistics.median(samples[v]) for v in variants})
                report['cases'].append(result)
                a.out.write_text(json.dumps(report, indent=2) + '\n')
                print(rows, theta, offset, result['median_us'],
                      {k: v['byte_different_elements'] for k, v in metrics.items()}, flush=True)
        for graph in graphs.values(): graph.reset()
    report.update(success=True, cleanup_completed=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--inner', type=int, default=16)
    p.add_argument('--backend', choices=('installed', 'vendored'), default='installed')
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(success=False, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try: run(a, report)
    except BaseException as exc: report.update(error=repr(exc)); raise
    finally: a.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
