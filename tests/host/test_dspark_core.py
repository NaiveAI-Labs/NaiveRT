"""Refuse incompatible draft geometry/topology before entering TP waits."""
from types import SimpleNamespace

import pytest
import torch

from naivert.backends.naive_n0p5_draft import DraftCore
from naivert.kernels.naive_n0p5.dspark_kernel_set import prepare_draft
from naivert.models.naive_n0p5.dspark import DraftBuffers, validate_draft_config


def config():
    return dict(hidden_size=4096, intermediate_size=4096, num_hidden_layers=5,
                num_attention_heads=32, num_key_value_heads=4, head_dim=128,
                vocab_size=152576, markov_rank=256, block_size=7, rms_norm_eps=1e-5,
                sliding_window=1024, use_sliding_window=True,
                layer_types=["sliding_attention"] * 5, hidden_act="silu",
                markov_head_type="vanilla", enable_confidence_head=True,
                confidence_head_with_markov=True,
                dflash_config=dict(target_layer_ids=[1, 7, 14, 20, 26, 32, 39, 45],
                                   mask_token_id=151675, use_mask_embedding=True),
                rope_parameters=dict(rope_theta=10000., rope_type="default"))


def test_draft_config_contract():
    validate_draft_config(config())
    for name, value in (("rms_norm_eps", 1e-6), ("block_size", 8),
                        ("sliding_window", 128), ("attention_bias", True),
                        ("layer_types", ["full_attention"] * 5), ("use_target_kv", True)):
        with pytest.raises(ValueError):
            validate_draft_config(dict(config(), **{name: value}))
    for section, field, value in (("dflash_config", "target_layer_ids", [1] * 8),
                                  ("dflash_config", "use_mask_embedding", False),
                                  ("rope_parameters", "rope_type", "yarn")):
        c = config(); c[section][field] = value
        with pytest.raises(ValueError):
            validate_draft_config(c)


@pytest.mark.parametrize("capacity", [0, 4, 9, 1048580, 16.0])
def test_bad_capacity_without_cuda(capacity):
    with pytest.raises(ValueError):
        DraftBuffers(None, rank=0, capacity=capacity)


def test_bad_topology_before_build():
    for devices in ((0,) * 8, range(7)):
        with pytest.raises(ValueError):
            prepare_draft(devices)


def test_inconsistent_ranks_fail_before_dispatch():
    k = SimpleNamespace(devices=tuple(range(8)))
    w = [SimpleNamespace(attention=(None,) * 5, mlp=(None,) * 5)] * 8
    def buffers():
        return [SimpleNamespace(rank=r, device=torch.device("cuda", r),
                                capacity=2304, sampling=False, mailbox=object()) for r in range(8)]
    b = buffers()
    DraftCore(k, w, b)
    for bad in (b[:-1], [b[0]] * 8, tuple(reversed(b))):
        with pytest.raises(ValueError):
            DraftCore(k, w, bad)
    b[7].sampling = True
    with pytest.raises(ValueError):
        DraftCore(k, w, b)
