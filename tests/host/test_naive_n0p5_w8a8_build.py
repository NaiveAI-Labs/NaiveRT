"""Native W8A8 must select a distinct extension and reject incompatible weights."""
import pytest
from naivert.kernels.naive_n0p5 import model_ops


def test_native_w8a8_build_isolated_from_default():
    default = model_ops._build_manifest()
    native = model_ops._build_manifest(expert_down="w8a8")
    flag = "-DNAIVERT_EXPERT_DOWN_W8A8=1"
    assert flag in native.cuda_cflags and flag in native.cflags
    assert flag not in default.cuda_cflags and flag not in default.cflags
    assert native.module_name != default.module_name
    assert native.build_directory != default.build_directory
    assert native.sources == default.sources


@pytest.mark.parametrize("weights", ["mx", "fp4"])
def test_native_w8a8_rejects_incompatible_weight_images(weights):
    with pytest.raises(ValueError, match="block128"):
        model_ops._build_manifest(weights=weights, expert_down="w8a8")
