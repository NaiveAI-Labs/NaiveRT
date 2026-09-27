"""Stand-in for ``sglang.jit_kernel.utils.is_arch_support_pdl`` at the pinned SGLang revision.

SGLang answers False on HIP and MUSA builds of torch, and otherwise whether the compute
capability major of the current CUDA device is at least 9, treating a device that cannot be
queried as capability (0, 0).  Like SGLang's ``cache_once``, the first answer is kept for
the life of the process.
"""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def is_arch_support_pdl() -> bool:
    if getattr(torch.version, "hip", None) or getattr(torch.version, "musa", None) is not None:
        return False
    try:
        major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    except Exception:
        major = 0
    return major >= 9
