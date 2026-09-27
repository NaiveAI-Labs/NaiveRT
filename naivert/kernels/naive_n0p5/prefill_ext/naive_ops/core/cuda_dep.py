
import importlib.util
from pathlib import Path
from typing import Optional

_CACHED_CUDA_INCLUDE_LIB: Optional[tuple[dict[str, Path], Path]] = None 


def determine_cuda_include_and_lib(check_crt: bool = False):
    global _CACHED_CUDA_INCLUDE_LIB
    spec = importlib.util.find_spec("nvidia")

    if spec is not None and spec.submodule_search_locations is not None:
        pip_nvidia_path = Path(spec.submodule_search_locations[0])
        cu12_cccl_path = pip_nvidia_path / "cuda_cccl"
        cu12_runtime_path = pip_nvidia_path / "cuda_runtime"

        if cu12_cccl_path.exists() and cu12_runtime_path.exists():
            include_path_dict = {
                "cccl": cu12_cccl_path / "include",
                "cuda_runtime": cu12_runtime_path / "include",
            }
            lib = cu12_runtime_path / "lib"
            _CACHED_CUDA_INCLUDE_LIB = (include_path_dict, lib)
            return _CACHED_CUDA_INCLUDE_LIB

        dirs = list(pip_nvidia_path.iterdir())
        # find cu12 or cu13
        cuda_inc_path = None 
        # use cuda runtime api requires nvidia-cuda-crt package, which isn't included by default in pytorch package.
        for d in dirs:
            check_cuda_path = d / "include" / "cuda.h"
            check_crt_path = d / "include" / "crt"
            if check_crt:
                is_crt_exist = check_crt_path.exists()
            else:
                is_crt_exist = True
            if d.name.startswith("cu") and d.is_dir() and check_cuda_path.exists() and is_crt_exist:
                cuda_inc_path = d
                break
        if cuda_inc_path is not None:
            include_path = cuda_inc_path / "include"
            lib = cuda_inc_path / "lib"
            include_path_dict = {
                "cccl": include_path / "cccl",
                "cuda_runtime": include_path,
            }

            if include_path.exists() and lib.exists():
                _CACHED_CUDA_INCLUDE_LIB = (include_path_dict, lib)
                return _CACHED_CUDA_INCLUDE_LIB
    raise NotImplementedError("currently only support pip installed cuda libraries (should be default for pytorch)")