"""BF16 output boundaries over explicit FP32 norm and interleaved RoPE."""
import torch
from .execution import PrefillError

NORM_EPS = 1.0e-5
ROPE_DIM = 64

def rmsnorm_fp32_gamma(x: torch.Tensor, gamma: torch.Tensor, *, eps: float = NORM_EPS) -> torch.Tensor:
    """RMSNorm with naivert's **fp32** gamma, rounding once at the end, as decode does.

    ``rmsnorm`` above is sgl-kernel's, which requires the weight in the activation dtype.
    naivert stores gamma fp32 and ``attn_common/rows_gemv_mma.cuh`` loads it fp32, folding
    it into the fp32 product before the single bf16 rounding; handing sgl-kernel a bf16
    copy pre-rounds gamma instead, which moves the normalised row by up to 2^-9 relative --
    two orders of magnitude past the 1-ulp standard the decode kernels are held to, and
    the caches this feeds are compared against decode byte for byte.  So the prefill path
    normalises in torch and pays the elementwise pass.  See section 3 of the plan.
    """
    if x.dtype is not torch.bfloat16:
        raise PrefillError(f"rmsnorm_fp32_gamma expects bf16 activations, got {x.dtype}")
    if gamma.dtype is not torch.float32:
        raise PrefillError(f"rmsnorm_fp32_gamma expects an fp32 gamma, got {gamma.dtype}")
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * inv * gamma).to(torch.bfloat16)


def rope_interleaved(x: torch.Tensor, positions: torch.Tensor, table: torch.Tensor,
                     *, rope_dim: int = ROPE_DIM) -> torch.Tensor:
    """Rotate the first ``rope_dim`` dims of every head, as adjacent (x, y) pairs.

    naivert's ``weights.interleave_rope_rows`` permutes the *output rows* of wq/wk at load
    time, so the checkpoint's NeoX half-rotation is already GPT-J interleaved pairs by the
    time a projection lands: ``attn_swa_body.cuh`` reads ``rope_freqs + position * 64`` and
    calls ``rotate_pair(x, y, rope[e], rope[e + 1])``, i.e. the table is (cos, sin)
    interleaved fp32.  This writes that rotation out -- bf16 in, fp32 rotation, one bf16
    rounding out -- with every product rounded, where the kernel's build contracts one
    product of each component into an fma (nvcc's default ``--fmad=true``).  So it lands
    off the kernel's rows in a few elements per hundred thousand -- by one bf16 step, or
    further where a component cancels and the rounded product is large against it;
    ``rope_fused`` is the byte-exact form (the stage 3 test, section 1).  Dims at or beyond
    ``rope_dim`` pass through.

    ``x`` is ``[M, H, D]`` or ``[M, D]``; ``positions`` is ``[M]`` long.
    """
    if x.dtype is not torch.bfloat16:
        raise PrefillError(f"rope_interleaved expects bf16, got {x.dtype}")
    if table.dtype is not torch.float32 or int(table.shape[1]) != rope_dim:
        raise PrefillError(
            f"rope table must be fp32 [max_seq_len, {rope_dim}], got {tuple(table.shape)} {table.dtype}")
    flat = x.unsqueeze(1) if x.dim() == 2 else x
    m = int(flat.shape[0])
    if int(positions.shape[0]) != m:
        raise PrefillError(f"rope positions {tuple(positions.shape)} do not match rows {m}")
    cs = table.index_select(0, positions.long())                       # [M, rope_dim]
    cos = cs[:, 0::2].unsqueeze(1)                                     # [M, 1, rope_dim/2]
    sin = cs[:, 1::2].unsqueeze(1)
    head = flat[..., :rope_dim].float()
    even, odd = head[..., 0::2], head[..., 1::2]
    rot = torch.empty_like(head)
    rot[..., 0::2] = even * cos - odd * sin
    rot[..., 1::2] = even * sin + odd * cos
    out = flat.clone()
    out[..., :rope_dim] = rot.to(torch.bfloat16)
    return out.squeeze(1) if x.dim() == 2 else out
