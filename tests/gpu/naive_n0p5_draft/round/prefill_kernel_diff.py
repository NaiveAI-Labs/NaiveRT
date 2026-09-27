"""Where the kernel prefill differs from verify8: per position and per channel, from a --dump file."""
import sys
import torch

d = torch.load(sys.argv[1], weights_only=False)
n = len(d['tokens'])
print('case', d['name'], 'tokens', n)
for r, L in enumerate(d['feature_layers']):
    A, B = d['features_A'][r][:n-1].float(), d['features_B'][r][:n-1].float()
    diff = (A-B)
    rel_pos = diff.norm(dim=1)/A.norm(dim=1).clamp_min(1e-30)
    top_ch = diff.square().sum(0)
    share = float(top_ch.topk(4).values.sum()/top_ch.sum())
    print(f'layer {L:2d} rel_rms {float(diff.norm()/A.norm()):.4f}  per-position rel: median {float(rel_pos.median()):.4f} p90 {float(rel_pos.quantile(0.9)):.4f} max {float(rel_pos.max()):.4f} at {int(rel_pos.argmax())}'
          f'  positions>0.2: {int((rel_pos>0.2).sum())}  top-4 channels share of diff energy {share:.3f} ch {top_ch.topk(4).indices.tolist()}'
          f'  A rownorm median {float(A.norm(dim=1).median()):.1f} max {float(A.norm(dim=1).max()):.1f} at {int(A.norm(dim=1).argmax())}')
    if n > 300:
        for lo, hi in ((0, 128), (128, 256), (256, n-1)):
            print(f'    positions {lo}-{hi}: rel rms {float(diff[lo:hi].norm()/A[lo:hi].norm()):.4f}')
for L, kind in d['kinds'].items():
    ca, cb = d['caches_A'][L], d['caches_B'][L]
    k_a, k_b = ca['k_cache'].float(), cb['k_cache'].float()
    rows = k_a.shape[0]
    rel_pos = (k_a-k_b).norm(dim=1)/k_a.norm(dim=1).clamp_min(1e-30)
    line = f'cache layer {L:2d} {kind[:4]} rows {rows} K rel {float((k_a-k_b).norm()/k_a.norm()):.4f} per-row median {float(rel_pos.median()):.4f} p90 {float(rel_pos.quantile(0.9)):.4f} rows>0.2 {int((rel_pos>0.2).sum())}'
    if kind == 'sliding_window' and n >= 136:
        live = torch.tensor([p % 136 for p in range(n-128, n)])
        mask = torch.zeros(rows, dtype=torch.bool); mask[live] = True
        line += f'  attended-rows K rel {float((k_a[mask]-k_b[mask]).norm()/k_a[mask].norm()):.4f}  other rows {float((k_a[~mask]-k_b[~mask]).norm()/k_a[~mask].norm().clamp_min(1e-30)):.4f}'
    if 'ki_cache' in ca:
        eq = (ca['ki_cache'].view(torch.uint8) == cb['ki_cache'].view(torch.uint8)).float().mean(1)
        line += f'  index bytes equal per row median {float(eq.median()):.3f} min {float(eq.min()):.3f}'
    print(line)
