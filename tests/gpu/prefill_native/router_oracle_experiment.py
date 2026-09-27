"""Observe actual prefill router inputs against FP64; no runtime adoption.

Load with --native-candidate. Set NAIVERT_ROUTER_AUDIT_OUT to a fresh JSON path,
and NAIVERT_ROUTER_AUDIT_FEED to baseline, candidate or oracle. The selected
router feeds the model; all three are compared on that same stream of inputs.
This diagnostic forces eager execution and its latency is not meaningful.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch

from naivert.kernels.naive_n0p5 import prefill as P

_path = Path(__file__).with_name('router_executor_experiment.py')
_spec = importlib.util.spec_from_file_location('_router_oracle_parent', _path)
_parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parent)


class NativePrefill(_parent.NativePrefill):
    def __init__(self, *args, **kwargs):
        self.feed = os.environ.get('NAIVERT_ROUTER_AUDIT_FEED', 'baseline')
        assert self.feed in ('baseline', 'candidate', 'oracle')
        self.audit_out = Path(os.environ['NAIVERT_ROUTER_AUDIT_OUT'])
        if self.audit_out.exists():
            raise ValueError(f'audit output already exists: {self.audit_out}')
        self.original_router = P.router_logits
        self.audit_rows, self.audited = [], set()
        kwargs.update(graphs=False, chunk_graphs=False)
        super().__init__(*args, **kwargs)
        self.oracle_weights = []
        for r, weight in enumerate(self.weights):
            with torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]):
                self.oracle_weights.append({L: weight.layer[L]['router'].double()
                                            for L in range(1, self.layers)})
        self.versions['router-oracle'] = dict(feed=self.feed,
            scope='FP64 router GEMM only; other model arithmetic unchanged; eager diagnostic',
            experiment_sha256=hashlib.sha256(_path.read_bytes()).hexdigest())
        self.sync()

    def _router(self, r, L, x, gate, *, out=None):
        ws = self.ws[r]
        start, valid = int(ws.start.item()), int(ws.valid.item())
        key = (start, valid, r, L)
        rows = max(0, min(x.shape[0], valid - r*x.shape[0]))
        inspect = rows > 0 and key not in self.audited
        if not inspect:
            if self.feed == 'baseline':
                return self.original_router(x, gate, out=out)
            if self.feed == 'candidate':
                return super()._router(r, L, x, gate, out=out)
            return out.copy_((x.double() @ self.oracle_weights[r][L].T).float())

        self.original_router(x, gate, out=out)
        baseline = out.clone()
        super()._router(r, L, x, gate, out=out)
        candidate = out.clone()
        truth = x.double() @ self.oracle_weights[r][L].T
        bias = self.weights[r].layer[L]['router_bias']
        logits = {'baseline': baseline, 'candidate': candidate, 'oracle': truth.float()}
        routes = {name: P.route(value, bias) for name, value in logits.items()}
        errors, ordered, sets = {}, {}, {}
        for name, value in logits.items():
            diff = value[:rows].double() - truth[:rows]
            errors[name] = dict(rms=float(diff.square().mean().sqrt()), max_abs=float(diff.abs().max()))
        for left, right in (('baseline', 'candidate'), ('baseline', 'oracle'), ('candidate', 'oracle')):
            a, b = routes[left][1][:rows], routes[right][1][:rows]
            ordered[left+'_'+right] = int((a != b).any(-1).sum())
            sets[left+'_'+right] = int((a.sort(-1).values != b.sort(-1).values).any(-1).sum())
        bw, bi = routes['baseline']; cw, ci = routes['candidate']
        bsort, bperm = bi[:rows].sort(-1)
        csort, cperm = ci[:rows].sort(-1)
        same_set = (bsort == csort).all(-1)
        aligned_diff = (bw[:rows].gather(1, bperm)-cw[:rows].gather(1, cperm)).abs()
        max_aligned = float(aligned_diff[same_set].max()) if bool(same_set.any()) else None
        changed = torch.nonzero((bi[:rows] != ci[:rows]).any(-1)).flatten().tolist()
        details = []
        for i in changed[:32]:
            expert_ids = sorted(set(v[1][i].tolist()[j] for v in routes.values() for j in range(P.TOPK)))
            scores = (truth[i].sigmoid() + bias.double()).sort(descending=True).values
            details.append(dict(token=start+r*x.shape[0]+i,
                ids={name: ids[i].tolist() for name, (_, ids) in routes.items()},
                weights={name: weights[i].tolist() for name, (weights, _) in routes.items()},
                experts=expert_ids, logits={name: value[i, expert_ids].tolist() for name, value in logits.items()},
                fp64_logits=truth[i, expert_ids].tolist(), fp64_selection_gap_8_9=float(scores[7]-scores[8])))
        row = dict(start=start, valid=valid, rank=r, layer=L, rows=rows, error=errors,
                   ordered_changed_rows=ordered, expert_set_changed_rows=sets,
                   max_aligned_weight_diff_same_expert_set=max_aligned,
                   changed_rows_details=details, omitted_changed_rows=max(0,len(changed)-len(details)))
        self.audit_rows.append(row); self.audited.add(key)
        if any(sets.values()):
            print('ROUTER ORACLE', self.feed, start, valid, r, L, sets, flush=True)
        out.copy_(logits[self.feed])
        return out

    def close(self):
        if self._closed:
            return
        self.sync()
        self.audit_out.parent.mkdir(parents=True, exist_ok=True)
        self.audit_out.write_text(json.dumps(dict(feed=self.feed, versions=self.versions,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            cases=self.audit_rows), indent=2)+'\n')
        super().close()
