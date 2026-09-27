"""MX8 pages retain the FP8 weight ordering and every E8M0 scale byte."""
import pytest
import torch

from naivert.kernels.packing import pack_down, pack_down_mx, pack_gate_up, pack_gate_up_mx


def test_mx_gate_up_matches_fp8_pages():
    generator = torch.Generator().manual_seed(42)
    gate, up = [torch.randint(0, 256, (2, 256, 2048), dtype=torch.uint8, generator=generator)
                .view(torch.float8_e4m3fn) for _ in range(2)]
    scales = [torch.randint(1, 255, (2, 256, 64), dtype=torch.uint8, generator=generator)
              for _ in range(2)]
    mx = pack_gate_up_mx(gate, scales[0], up, scales[1], hidden=2048)
    fp8 = pack_gate_up(gate, torch.ones(2, 2, 16), up, torch.ones(2, 2, 16), hidden=2048)
    assert torch.equal(mx[..., :32768], fp8.view(torch.uint8)[..., :32768])
    stored = mx[..., 32768:].reshape(2, 16, 2, 2, 32, 16)
    for branch in range(2):
        restored = stored[:, :, :, branch].permute(0, 1, 4, 2, 3).reshape(2, 256, 64)
        assert torch.equal(restored, scales[branch])


@pytest.mark.parametrize('width,hidden', [(56, 896), (64, 128)])
def test_mx_down_matches_fp8_tiles(width, hidden):
    generator = torch.Generator().manual_seed(43)
    weight = torch.randint(0, 256, (2, hidden, 256), dtype=torch.uint8, generator=generator).view(torch.float8_e4m3fn)
    scales = torch.randint(1, 255, (2, hidden, 8), dtype=torch.uint8, generator=generator)
    mx = pack_down_mx(weight, scales, hidden=hidden, outputs_per_tile=width)
    fp8, _ = pack_down(weight, torch.ones(2, hidden // 128, 2), hidden=hidden, outputs_per_tile=width)
    assert torch.equal(mx[..., :width * 256].reshape_as(weight), fp8.view(torch.uint8))
    assert torch.equal(mx[..., width * 256:].reshape_as(scales), scales)
