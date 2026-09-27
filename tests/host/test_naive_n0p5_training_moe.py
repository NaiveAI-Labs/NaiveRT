"""Cases where moving expert weighting or a BF16 store changes the result."""
import pytest
import torch

from naivert.models.naive_n0p5 import training_moe


def test_complete_each_expert_before_weighting():
    # 1 + 2^-8 is a BF16 midpoint and rounds to the even value 1.
    raw = torch.tensor([[[1.00390625], [-1.0]]])
    probabilities = torch.tensor([[0.5, 0.5]])
    actual = training_moe.combine(raw.bfloat16(), probabilities)
    assert actual.item() == 0.0
    delayed_round = (raw * probabilities[..., None]).sum(1).bfloat16()
    assert delayed_round.item() == 0.001953125


def test_keep_experts_separate_through_tp_completion():
    partials = torch.tensor([[[[256.0], [0.0]]], [[[-255.5], [0.0]]]])
    probabilities = torch.tensor([[0.5, 0.5]])
    actual = training_moe.combine_tp_partials(partials, probabilities)
    assert actual.item() == 0.25
    # Existing topology: weight/sum experts, round a rank's mixture, then sum ranks.
    early_mix = (partials * probabilities[None, ..., None]).sum(2).bfloat16()
    assert early_mix.float().sum(0).bfloat16().item() == 0.0


def test_round_mixture_before_residual_addition():
    experts = torch.tensor([[[1.0], [2.0]]], dtype=torch.bfloat16)
    probabilities = torch.tensor([[0.99609375, 0.00390625]])
    residual = torch.tensor([[-1.0]], dtype=torch.bfloat16)
    actual = training_moe.combine(experts, probabilities) + residual
    assert actual.item() == 0.0
    fused = ((experts.float() * probabilities[..., None]).sum(1) + residual.float()).bfloat16()
    assert fused.item() == 0.00390625


def test_router_preserves_fp32_weights_and_uses_bias_only_for_selection():
    x = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)
    weight = torch.tensor([[1.001, -1.0], [1.0005, -1.0], [0.0, 0.0]])
    bias = torch.tensor([0.0, 0.0, 1.0])
    scores, indices, probabilities = training_moe.route(x, weight, bias, 2)
    assert set(indices[0].tolist()) == {0, 2}
    assert scores[0, 0] > scores[0, 1] > scores[0, 2]
    assert torch.equal(x.float() @ weight.bfloat16().float().T, torch.zeros(1, 3))
    assert torch.all((probabilities > 0.499) & (probabilities < 0.501))
    with pytest.raises(ValueError, match="original FP32"):
        training_moe.route(x, weight.bfloat16(), bias, 2)


@pytest.mark.parametrize("rows", [1, 8])
def test_mixture_preserves_token_and_expert_axes(rows):
    experts = torch.arange(rows * 8 * 16).reshape(rows, 8, 16).bfloat16()
    probabilities = torch.zeros(rows, 8)
    probabilities[torch.arange(rows), torch.arange(rows) % 8] = 1.0
    assert torch.equal(training_moe.combine(experts, probabilities), experts[torch.arange(rows), torch.arange(rows) % 8])


def test_precombined_or_unrounded_operands_are_rejected():
    probabilities = torch.ones(1, 8) / 8
    with pytest.raises(ValueError, match="completed BF16"):
        training_moe.combine(torch.zeros(1, 8, 16), probabilities)
    with pytest.raises(ValueError, match="FP32 partials"):
        training_moe.combine_tp_partials(torch.zeros(8, 1, 16), probabilities)


def test_down_activation_quantization_range_and_block_independence():
    x = torch.zeros(1, 6 * 128, dtype=torch.bfloat16)
    x[0, 128:256] = 448
    x[0, 256:384] = 450  # Immediately above a scale-1 block's FP8 ceiling.
    x[0, 384:512] = 257024
    x[0, 512:640] = -448
    x[0, 640:] = 2.0 ** -133  # Smallest positive BF16 subnormal.
    q, scales = training_moe.quantize_down_activation_reference(x)
    assert scales.tolist() == [[1.0, 1.0, 2.0, 1024.0, 1.0, 2.0 ** -127]]
    assert torch.isfinite(q.float()).all()
    assert q[0, 0].float() == 0
    assert q[0, 128].float() == 448
    assert q[0, 512].float() == -448
    assert q[0, 640].float() == 2.0 ** -6


def test_down_activation_quantization_rejects_wrong_granularity():
    with pytest.raises(ValueError, match="divisible by 128"):
        training_moe.quantize_down_activation_reference(torch.zeros(1, 32, dtype=torch.bfloat16))


def test_fp32_down_input_avoids_double_rounding_and_scale_changes():
    # 1.0625 is an E4M3 midpoint. The BF16 intermediate rounds 1.0626
    # back to that midpoint and then E4M3 ties-to-even chooses 1.0.
    # A value just over 448 likewise needs scale 2 before any BF16 round.
    x = torch.zeros(1, 256)
    x[0, 0], x[0, 1], x[0, 128] = 448., 1.0626, 448.1
    q, s = training_moe.quantize_down_activation_reference(x)
    old_q, old_s = training_moe.quantize_down_activation_reference(x.bfloat16())
    assert s.tolist() == [[1., 2.]]
    assert old_s.tolist() == [[1., 1.]]
    assert q.float()[0, 1].item() == 1.125
    assert old_q.float()[0, 1].item() == 1.0
    assert abs(q.float()[0, 1] - x[0, 1]) < abs(old_q.float()[0, 1] - x[0, 1])
