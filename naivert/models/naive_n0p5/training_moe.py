"""Numerical boundaries for NaiveTrain's Sonic block128 EP1 MoE path.

These PyTorch operations are an executable reference, not the optimized TP8
backend. In particular, partials must retain a separate expert axis until
the full down projection is complete. The separately named down-activation
quantizer specifies the initial W8A8 adaptation recipe; its equivalence to the
training installation's QuACK quantizer still needs verification.

Sources: NaiveTrain 8a7d3d71 and NaiveSonicMoE 7844c976.
"""
from __future__ import annotations

import torch


def quantize_down_activation_reference(x: torch.Tensor):
    """Explicit W8A8 adaptation recipe, not a claim about the missing QuACK fork.

    Round amax/448 upward to a power of two per 128 values, use scale 1 for
    zero blocks, and E4M3 round-to-nearest for the payload. Return FP8 values
    and FP32 representations of E8M0 scales. Uses frexp rather than the GPU
    implementation's integer bit manipulation to provide an independent oracle.
    """
    if x.dtype not in (torch.bfloat16, torch.float32) or x.shape[-1] % 128:
        raise ValueError("expected BF16 or FP32 activations with last dimension divisible by 128")
    grouped = x.float().reshape(*x.shape[:-1], -1, 128)
    amax = grouped.abs().amax(-1)
    ratio = (amax / 448).clamp_min(2.0 ** -127)
    mantissa, exponent = torch.frexp(ratio)
    exponent = exponent - (mantissa == 0.5).to(exponent.dtype)
    scale = torch.where(amax == 0, 1.0, torch.exp2(exponent.float()))
    q = (grouped / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(x.shape)
    return q, scale


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """BF16 normalized values, followed by the BF16 gamma multiplication."""
    if x.dtype != torch.bfloat16 or gamma.dtype != torch.bfloat16:
        raise ValueError("training RMSNorm expects BF16 activations and compute gamma")
    normalized = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps))
    return normalized.to(torch.bfloat16) * gamma


def route(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, topk: int = 8):
    """FP32 sigmoid router; correction bias selects experts, but does not weight them."""
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("expected activations [tokens, hidden] and router [experts, hidden]")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError("training router requires BF16 activations and original FP32 weights/bias")
    if bias.shape != (weight.shape[0],) or not 1 <= topk <= weight.shape[0]:
        raise ValueError("invalid router bias shape or top-k")
    scores = x.float() @ weight.T
    probabilities = torch.sigmoid(scores)
    indices = torch.topk(probabilities + bias, topk, dim=-1, sorted=False).indices
    selected = probabilities.gather(-1, indices)
    # Match NaiveTrain's sigmoid path, including its topk=1 behavior.
    if topk > 1:
        selected = selected / (selected.sum(-1, keepdim=True) + 1e-20)
    return scores, indices, selected


def combine(expert_outputs: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Completed BF16 expert outputs [T, E, H] -> FP32 weighted sum -> BF16 [T, H].

    The caller adds the BF16 residual AFTER this return. Multiplying the
    probabilities into down-projection inputs, incomplete K sums, or weights
    changes the rounding graph and is not this operation.
    """
    if expert_outputs.ndim != 3 or probabilities.shape != expert_outputs.shape[:2]:
        raise ValueError("expected completed experts [tokens, selected, hidden] and probabilities [tokens, selected]")
    if expert_outputs.dtype != torch.bfloat16 or probabilities.dtype != torch.float32:
        raise ValueError("combine requires completed BF16 expert outputs and FP32 probabilities")
    if expert_outputs.device != probabilities.device:
        raise ValueError("expert outputs and probabilities must share a device")
    return (expert_outputs.float() * probabilities[..., None]).sum(1).to(torch.bfloat16)


def combine_tp_partials(partials: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """FP32 [ranks, T, E, H] -> complete each expert -> BF16 -> combine.

    This specifies the required TP boundary. Summing TP partials can still
    differ from an unsharded GEMM's accumulation order; this is not a claim
    of bitwise equivalence to QuACK. A precombined [ranks, T, H] mailbox has
    already lost the information required to reproduce this boundary.
    """
    if partials.ndim != 4 or partials.dtype != torch.float32 or partials.shape[0] == 0:
        raise ValueError("expected FP32 partials [ranks, tokens, selected, hidden]")
    return combine(partials.sum(0).to(torch.bfloat16), probabilities)
