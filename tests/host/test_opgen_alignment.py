"""Invalid transfer alignment must fail before generating CUDA bindings."""
from dataclasses import replace
import re
import shutil
import subprocess
import pytest
from naivert.kernels import opgen
from naivert.kernels.ops.rmsnorm_expert_proj_prepared import OP


@pytest.mark.parametrize("alignment", [0, -16, 3, 24])
def test_reject_invalid_transfer_alignment(alignment):
    params=(*OP.params[:-1],replace(OP.params[-1],alignment=alignment))
    with pytest.raises(ValueError,match="alignment must be a positive power of two"):
        replace(OP,params=params)


def test_router_alignment_dispatch_preserves_entrypoint_and_fallback(tmp_path):
    """Even an explicit wide variant must fall back for a 16-byte offset."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    header = opgen.render_all(OP)[OP.launch_header_path]
    enum = re.search(r"enum class Variant\s*\{.*?\};", header, re.S).group()
    source = tmp_path / "dispatch.cpp"
    source.write_text("""#include <cassert>
#include <cstdint>
""" + enum + """
Variant select(float* gamma, Variant variant) {
  struct { struct { float* gamma; } base; } args{{gamma}};
  return """ + OP.launch.variant_selector + """;
}
int main() {
  alignas(32) float gamma[16]{};
  for (int i = 0; i < 4; ++i) {
    auto input = static_cast<Variant>(i);
    assert(static_cast<int>(select(gamma, input)) == 2 + i % 2);
    assert(static_cast<int>(select(gamma + 4, input)) == i % 2);
  }
  assert(static_cast<int>(select(gamma, static_cast<Variant>(17))) == 17);
  assert(static_cast<int>(select(gamma + 4, static_cast<Variant>(17))) == 17);
}
""")
    binary = tmp_path / "dispatch"
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
