"""Naive N0.5 import checks: TP scale axes, preserved FP8 codes, and rotary rows."""
import pytest
import torch

from naivert.model import ModelPackageError
from naivert.models.naive_n0p5.checkpoint_fp8 import NaiveNCheckpoint, dequant_block, interleave_rope, shard_fp8_block


@pytest.mark.parametrize("axis", [0, 1])
def test_tp_shards_reconstruct_codes_and_nonuniform_scale_grid(axis):
    shape = (1024, 256) if axis == 0 else (256, 1024)
    q = (torch.arange(shape[0] * shape[1]).reshape(shape) % 101 - 50).to(torch.float8_e4m3fn)
    s = (torch.arange((shape[0] // 128) * (shape[1] // 128)).reshape(shape[0] // 128, shape[1] // 128) + 1.2345) / 1000
    parts = [shard_fp8_block(q, s, r, axis=axis) for r in range(8)]
    assert torch.equal(torch.cat([v.view(torch.uint8) for v, _ in parts], axis), q.view(torch.uint8))
    assert torch.equal(torch.cat([v for _, v in parts], axis), s)
    assert torch.equal(torch.cat([dequant_block(*p) for p in parts], axis), dequant_block(q, s))
    # Effective weights really depend on scales; treating codes as BF16 input loses them.
    assert not torch.equal(dequant_block(q, s), q.float())


@pytest.mark.parametrize("rank,axis", [(-1, 0), (8, 0), (0, 2)])
def test_invalid_tp_shard_rejected(rank, axis):
    with pytest.raises(ValueError):
        shard_fp8_block(torch.zeros(1024, 256).to(torch.float8_e4m3fn), torch.ones(8, 2), rank, axis=axis)


def test_scale_shape_and_unaligned_tp_boundary_rejected():
    q = torch.zeros(256, 256).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="scale shape"):
        shard_fp8_block(q, torch.ones(1, 2), 0, axis=0)
    with pytest.raises(ValueError, match="block boundaries"):
        shard_fp8_block(q, torch.ones(2, 2), 0, axis=0)


def test_rope_projection_and_affine_follow_same_permutation():
    w = torch.arange(2 * 128 * 4, dtype=torch.float32).reshape(256, 4)
    x = torch.tensor([1., -2., 3., -4.])
    assert torch.equal(interleave_rope(w, 128, projection=True) @ x, interleave_rope(w @ x, 128))
    y = interleave_rope(torch.arange(256), 128).reshape(2, 128)
    assert y[0, :6].tolist() == [0, 32, 1, 33, 2, 34]
    assert y[0, 64:].tolist() == list(range(64, 128))
    assert y[1, :6].tolist() == [128, 160, 129, 161, 130, 162]


def test_reader_does_not_accept_an_unrelated_config(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ModelPackageError, match="config.json"):
        NaiveNCheckpoint(tmp_path)


@pytest.mark.parametrize("which", [0, 2])
def test_expert_reader_preserves_codes_and_rounds_only_scales(tmp_path, which):
    from safetensors.torch import save_file
    from naivert.kernels.packing import pack_gate_up, pack_down
    shape = (2048, 4096) if which == 0 else (4096, 2048)
    q = torch.full(shape, 2.0).to(torch.float8_e4m3fn)
    s = torch.linspace(0.0012345, 0.0789123, 512).reshape(shape[0] // 128, shape[1] // 128)
    p = f"model.layers.1.mlp.experts.0.{('gate', 'up', 'down')[which]}_proj."
    save_file({p + "weight": q, p + "weight_scale_inv": s}, tmp_path / "one.safetensors")
    # Isolate slicing/packing from the production identity check (tested above).
    reader = NaiveNCheckpoint.__new__(NaiveNCheckpoint)
    from threading import Lock
    from concurrent.futures import ThreadPoolExecutor
    reader._reader_lock = Lock()
    reader.root, reader.weight_map = tmp_path, {p + name: "one.safetensors" for name in ("weight", "weight_scale_inv")}
    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = list(pool.map(lambda rank: reader.expert(1, 0, which, rank, kernel_scales=False), range(8)))
    assert len(reader._readers) == 1
    for rank in (0, 3, 7):
        codes, fp32 = concurrent[rank]
        _, bf16 = reader.expert(1, 0, which, rank)
        expect_q, expect_s = shard_fp8_block(q, s, rank, axis=0 if which == 0 else 1)
        assert torch.equal(codes.view(torch.uint8), expect_q.view(torch.uint8))
        assert torch.equal(fp32, expect_s)
        assert torch.equal(bf16, fp32.to(torch.bfloat16))
        if which == 0:
            packed = pack_gate_up(codes[None], fp32[None], codes[None], fp32[None], hidden=4096)
            stored = packed.view(torch.uint8)[0, 0, 0, 32768:32784].view(torch.bfloat16)
            assert torch.equal(stored, bf16[0, :8])
        else:
            _, stored = pack_down(codes[None], fp32[None], hidden=4096, outputs_per_tile=64)
            assert torch.equal(stored[0, 0], bf16[0])
    reader.close()
