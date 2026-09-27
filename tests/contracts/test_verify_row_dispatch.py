"""Verify dispatch keeps fused batches and uses disjoint per-row fallback mailboxes."""
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from conftest import load_driver
from naivert.kernels.naive_n0p5.model_ops import _build_manifest


@pytest.mark.parametrize("rows", [8, 16])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("pdl", [False, True])
def test_verify_projection_dispatch(rows, fused, bf16, pdl):
    verify = load_driver("run_naive_n0p5_verify.py")
    verify.configure_rows(rows)
    ops = MagicMock(spec=list(_build_manifest().required_symbols))
    ops.launch_checkpoint_bf16_linear = MagicMock()
    layers = []
    for _ in range(2):
        layer = defaultdict(MagicMock)
        layer.update(kind="sliding_window", o_packed=torch.empty(0, dtype=torch.bfloat16 if bf16 else torch.uint8))
        layers.append(layer)
    rt = MagicMock()
    rt.mailboxes = [torch.empty(rows * 4 * verify.SLAB, dtype=torch.uint8, device="meta") for _ in range(8)]
    rt.epoch = torch.ones(1, dtype=torch.int64)
    v = verify.VerifyRuntime.__new__(verify.VerifyRuntime)
    v.rt, v.ops, v.rank = rt, ops, 0
    v.w = SimpleNamespace(layers=2, layer=layers, embedding=MagicMock(), head_gamma=MagicMock(), lm_head=MagicMock())
    v.units, v.pdl, v.fused_unproj = rows // 8, pdl, fused
    v.prepared_gate_up = False
    v.down = v.router = v.dense = v.head = v.select = "rows8"
    v.pos_units = [MagicMock() for _ in range(v.units)]
    v.ext_swa8 = MagicMock()
    v.attn_ws = {"sliding_window": [MagicMock(), MagicMock()]}
    v.proj_o = torch.empty(rows, verify.HEADS, verify.VDIM, device="meta")
    v.hid_a = torch.empty(rows, verify.HIDDEN, device="meta")
    v.resid = [torch.empty_like(v.hid_a) for _ in range(3)]
    for field in ("tokens", "dense_up", "scores", "x_mlp", "moe_up", "probs", "indices", "logits", "last_hidden", "head_ws"):
        setattr(v, field, MagicMock())
    v.domains, v.domain_size = {1: MagicMock()}, 1
    phases = v.phases(advance=False)
    projections = [(name, fn) for name, fn in phases if ".unproj" in name]
    if fused:
        assert projections == []
        assert sum("+unproj" in name for name, _ in phases) == 2 * v.units
        return
    assert len(projections) == 2 * rows
    for _, fn in projections:
        fn()
    launch = (ops.launch_checkpoint_bf16_linear if bf16 else
              ops.launch_unproj_allreduce_replay_pdl_consumer if pdl else ops.launch_unproj_allreduce_replay)
    assert launch.call_count == 2 * rows
    for i, call in enumerate(launch.call_args_list):
        args = call.args
        assert args[0].shape == (1, verify.K_UNPROJ)
        mailbox = args[4 if bf16 else 5]
        for peer in mailbox:
            assert peer.numel() == verify.SLAB
            assert peer.storage_offset() == (i % rows) * verify.SLAB
