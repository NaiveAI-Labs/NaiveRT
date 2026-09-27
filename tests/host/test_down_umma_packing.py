"""The load-time UMMA image must preserve every checkpoint byte and scale."""
import pytest
import torch

from naivert.kernels.packing import pack_down


def test_umma_panels_preserve_codes_at_hardware_addresses():
    # Include every bit pattern, including signed zero and the NaN encodings.
    raw = torch.randint(0, 256, (2, 256, 256), dtype=torch.uint8, generator=torch.Generator().manual_seed(35))
    raw[0, 0] = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    scales = torch.tensor([.5, 2., 4., 8., 16., 32., 64., 128.]).reshape(2, 2, 2)
    packed, stored = pack_down(raw.view(torch.float8_e4m3fn), scales, hidden=256, outputs_per_tile=64,
                               layout='umma_sw128')
    _, old_scales = pack_down(raw.view(torch.float8_e4m3fn), scales, hidden=256, outputs_per_tile=64)
    assert packed.dtype == torch.float8_e4m3fn and packed.is_contiguous()
    assert packed.numel() == raw.numel()
    assert torch.equal(stored.view(torch.uint8), old_scales.view(torch.uint8))
    panels = packed.view(torch.uint8).reshape(2, 4, 2, 8192)
    k = torch.arange(128)
    for e in range(2):
        for tile in range(4):
            for group in range(2):
                for row in range(64):
                    physical = row * 128 + ((k // 16) ^ (row % 8)) * 16 + k % 16
                    assert torch.equal(panels[e, tile, group, physical], raw[e, tile*64+row, group*128+k])


@pytest.mark.parametrize('layout,width', [('unknown', 64), ('umma_sw128', 32)])
def test_reject_incompatible_down_layout(layout, width):
    with pytest.raises(ValueError):
        pack_down(torch.zeros((1, 128, 256), dtype=torch.float8_e4m3fn), torch.ones(1, 1, 2),
                  hidden=128, outputs_per_tile=width, layout=layout)


@pytest.mark.parametrize('mode,layout', [('bf16', 'qmma'), ('w8a8', 'umma_sw128')])
def test_checkpoint_selects_layout_before_packing(monkeypatch, mode, layout):
    from naivert.models.naive_n0p5.checkpoint_fp8 import NaiveNCheckpoint
    from naivert.kernels import packing
    reader = NaiveNCheckpoint.__new__(NaiveNCheckpoint)
    def expert(layer, index, which, rank, device, *, kernel_scales=True):
        shape = (4096, 256) if which == 2 else (256, 4096)
        return (torch.empty(shape, dtype=torch.float8_e4m3fn, device='meta'),
                torch.empty((shape[0]//128, shape[1]//128), dtype=torch.bfloat16 if kernel_scales else torch.float32, device='meta'))
    reader.expert = expert
    reader.read = lambda *a, **k: torch.empty(1, device='meta')
    called = []
    def pack(weight, scales, **kwargs):
        called.append(kwargs['layout'])
        return weight, scales
    monkeypatch.setattr(packing, 'pack_down', pack)
    reader.mlp(1, 0, 'meta', expert_down=mode)
    assert called == [layout]


def test_fp32_scale_packing_preserves_non_bf16_checkpoint_values():
    from naivert.kernels.packing import pack_gate_up
    scale = torch.tensor([1.001, 2.003, .12345, .000123]).reshape(1,2,2)
    weights = torch.zeros(1,256,256,dtype=torch.float8_e4m3fn)
    _, stored = pack_down(weights,scale,hidden=256,outputs_per_tile=64,
                           layout='umma_sw128',scale_dtype=torch.float32)
    # Each hidden block's two K scales survive at their expanded tile addresses.
    assert stored.dtype == torch.float32
    assert torch.equal(stored.reshape(1,4,16)[0,0,:2],scale[0,0])
    assert torch.equal(stored.reshape(1,4,16)[0,2,:2],scale[0,1])
    assert not torch.equal(stored,stored.bfloat16().float())
    codes=torch.zeros(1,256,1024,dtype=torch.float8_e4m3fn)
    scales=torch.arange(16,dtype=torch.float32).reshape(1,2,8)*.00123+.01111
    pages=pack_gate_up(codes,scales,codes,scales*3,hidden=1024,scale_dtype=torch.float32)
    live=pages.view(torch.uint8)[...,32768:32768+64].contiguous().view(torch.float32)
    assert torch.equal(live[0,0,0,:8],scales[0,0])
    assert torch.equal(live[0,8,0,8:],scales[0,1]*3)
