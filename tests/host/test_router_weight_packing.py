"""The router container preserves FP32 bits and the original caller's view."""
import pytest
import torch

from naivert.kernels.packing import pack_router_weight


def test_router_container_preserves_bits_and_float4_order():
    bits = (torch.arange(3 * 128, dtype=torch.int64) * 2654435761).to(torch.int32).reshape(3, 128)
    bits[0, :8] = torch.tensor([0, 0x80000000, 0x7f800000, 0xff800000,
                                0x7fc01234, 0xffc01234, 1, 0x80000001],
                               dtype=torch.int64).to(torch.int32)
    packed = pack_router_weight(bits.view(torch.float32))
    assert packed.shape == (2, 3, 128) and packed.is_contiguous()
    assert packed[0].data_ptr() == packed.data_ptr()
    assert torch.equal(packed[0].view(torch.int32), bits)
    # Every other group of four chunks exchanges its two intact float4 halves.
    restored = packed[1].view(torch.int32).reshape(3, 4, 4, 2, 4).clone()
    restored[:, 1::2] = restored[:, 1::2].flip(-2)
    assert torch.equal(restored.reshape_as(bits), bits)
    assert torch.equal(packed[1].view(torch.int32)[0, 32:40], bits[0, [36, 37, 38, 39, 32, 33, 34, 35]])


@pytest.mark.parametrize("shape,dtype", [((64,), torch.float32), ((2, 63), torch.float32),
                                         ((2, 64), torch.bfloat16)])
def test_router_container_rejects_incompatible_inputs(shape, dtype):
    with pytest.raises(ValueError, match="router weight must be FP32"):
        pack_router_weight(torch.zeros(shape, dtype=dtype))
