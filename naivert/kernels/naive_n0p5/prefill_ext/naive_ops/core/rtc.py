"""
modified from pytorch torch.cuda._utils
"""

import ctypes
import sys
from typing import Any, Optional, Union

import torch


CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
CU_FUNC_CACHE_PREFER_SHARED = 1
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8

# The _get_device_index has been moved to torch.utils._get_device_index
from torch._utils import _get_device_index as _torch_get_device_index


# Load CUDA driver and NVRTC
def _get_cuda_library() -> ctypes.CDLL:
    if sys.platform == "win32":
        return ctypes.CDLL("nvcuda.dll")
    else:  # Unix-based systems
        return ctypes.CDLL("libcuda.so.1")


# Helper: check CUDA errors
def _check_cuda(result: int) -> None:
    if result == 0:
        return
    err_str = ctypes.c_char_p()
    libcuda = _get_cuda_library()  # Get reference to CUDA library
    libcuda.cuGetErrorString(result, ctypes.byref(err_str))
    error_message = (
        err_str.value.decode() if err_str.value is not None else "Unknown CUDA error"
    )
    raise RuntimeError(f"CUDA error: {error_message}")


def _cu_func_get_attribute(func: ctypes.c_void_p, attribute: int) -> int:
    libcuda = _get_cuda_library()
    libcuda.cuFuncGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    libcuda.cuFuncGetAttribute.restype = ctypes.c_int

    value = ctypes.c_int()
    _check_cuda(libcuda.cuFuncGetAttribute(ctypes.byref(value), attribute, func))
    return value.value


def _cu_func_set_attribute(func: ctypes.c_void_p, attribute: int, value: int) -> None:
    libcuda = _get_cuda_library()
    libcuda.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    libcuda.cuFuncSetAttribute.restype = ctypes.c_int
    _check_cuda(libcuda.cuFuncSetAttribute(func, attribute, value))


def _cu_device_get_attribute(device: int, attribute: int) -> int:
    libcuda = _get_cuda_library()
    libcuda.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    libcuda.cuDeviceGetAttribute.restype = ctypes.c_int

    value = ctypes.c_int()
    _check_cuda(libcuda.cuDeviceGetAttribute(ctypes.byref(value), attribute, device))
    return value.value


def _cu_func_set_cache_config(func: ctypes.c_void_p, config: int) -> None:
    libcuda = _get_cuda_library()
    libcuda.cuFuncSetCacheConfig.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libcuda.cuFuncSetCacheConfig.restype = ctypes.c_int
    _check_cuda(libcuda.cuFuncSetCacheConfig(func, config))


def _get_nvrtc_library() -> ctypes.CDLL:
    major_version = int(torch.version.cuda.split(".")[0])  # type: ignore[union-attr]
    if sys.platform == "win32":
        nvrtc_libs = [
            f"nvrtc64_{major_version}0_0.dll",
        ]
    else:
        nvrtc_libs = [
            f"libnvrtc.so.{major_version}",
            "libnvrtc.so",  # Fallback to unversioned
        ]
    for lib_name in nvrtc_libs:
        try:
            return ctypes.CDLL(lib_name)
        except OSError:
            continue
    raise OSError("Could not find any NVRTC library")


def _get_full_source_and_options(
    kernel_source: str,
    header_code: str,
    cuda_include_dirs: Optional[list],
    nvcc_options: Optional[list],
    compute_capability: Optional[str] = None,
    add_external_c: bool = True,
):
    """this function can be used to calculate hash from user inputs.
    used to implement a cache-recompile mechanism.
    """
    if add_external_c and not kernel_source.strip().startswith('extern "C"'):
        kernel_source = f'extern "C" {kernel_source}'

    # Combine header code and kernel source
    full_source = header_code + "\n" + kernel_source if header_code else kernel_source

    # Prepare compilation options
    # Get compute capability if not provided
    if compute_capability is None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        compute_capability = f"{props.major}{props.minor}"

    # Prepare compilation options
    options: list[str] = []
    options.append(f"--gpu-architecture=sm_{compute_capability}")
    if cuda_include_dirs:
        for directory in cuda_include_dirs:
            options.append(f"-I{directory}")
    if nvcc_options:
        for option in nvcc_options:
            options.append(option)
    # TODO: Should we refactor flags into a common place?
    from torch.utils.cpp_extension import COMMON_NVCC_FLAGS

    # Filter out flags not supported by NVRTC
    nvrtc_compatible_flags = [
        flag for flag in COMMON_NVCC_FLAGS if flag != "--expt-relaxed-constexpr"
    ]
    options.extend([flag for flag in nvrtc_compatible_flags])

    return full_source, options


def _nvrtc_compile(
    kernel_source: str,
    kernel_name: str,
    compute_capability: Optional[str] = None,
    header_code: str = "",
    cuda_include_dirs: Optional[list] = None,
    nvcc_options: Optional[list] = None,
    qualnames: Optional[list[str]] = None,
    add_external_c: bool = True,
) -> tuple[bytes, dict[str, str]]:
    """
    Compiles a CUDA kernel using NVRTC and returns the PTX code.

    Args:
        kernel_source (str): The CUDA kernel source code as a string
        kernel_name (str): The name of the kernel function to compile
        compute_capability (str, None): The compute capability to target (e.g., "86").
                                           If None, will detect from current device.
        header_code (str, optional): Additional header code to prepend to the kernel source
        cuda_include_dirs (list, None): List of directories containing CUDA headers
        nvcc_options (list, None): Additional options to pass to NVRTC

    Returns:
        str: The compiled PTX code
    """
    # Ensure CUDA is initialized
    import torch.cuda

    # Load NVRTC library
    libnvrtc = _get_nvrtc_library()

    # NVRTC constants
    NVRTC_SUCCESS = 0

    # Helper: check NVRTC errors
    def check_nvrtc(result: int) -> None:
        if result != NVRTC_SUCCESS:
            err_str = ctypes.c_char_p()
            libnvrtc.nvrtcGetErrorString(result, ctypes.byref(err_str))
            error_message = (
                err_str.value.decode()
                if err_str.value is not None
                else "Unknown CUDA error"
            )
            raise RuntimeError(f"CUDA error: {error_message}")

    full_source, options = _get_full_source_and_options(
        kernel_source,
        header_code,
        cuda_include_dirs,
        nvcc_options,
        compute_capability,
        add_external_c,
    )
    source_bytes = full_source.encode("utf-8")
    options = [option.encode("utf-8") for option in options]

    # Convert options to C array
    num_options = len(options)
    options_array = (ctypes.c_char_p * num_options)(*options)

    # Create program
    prog = ctypes.c_void_p()
    check_nvrtc(
        libnvrtc.nvrtcCreateProgram(
            ctypes.byref(prog),
            source_bytes,
            f"{kernel_name}.cu".encode(),
            0,
            None,
            None,
        )
    )
    if qualnames:
        for qualname in qualnames:
            check_nvrtc(libnvrtc.nvrtcAddNameExpression(prog, qualname.encode()))

    # Compile program
    res = libnvrtc.nvrtcCompileProgram(prog, num_options, options_array)

    # Handle compilation errors
    if res != NVRTC_SUCCESS:
        # Get log
        log_size = ctypes.c_size_t()
        libnvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        log = ctypes.create_string_buffer(log_size.value)
        libnvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"Kernel compilation failed:\n{log.value.decode()}")
    qname_to_lowered: dict[str, str] = {}
    if qualnames:
        """
        const char *lowered_name;
        TV_NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog_,
                                            name.c_str(), // name expression
                                            &lowered_name // lowered name
                                            ));
        return std::string(lowered_name);

        """
        for qualname in qualnames:
            lowered_name_p = ctypes.c_char_p()

            check_nvrtc(
                libnvrtc.nvrtcGetLoweredName(
                    prog, qualname.encode(), ctypes.byref(lowered_name_p)
                )
            )
            lowered_name = lowered_name_p.value
            assert lowered_name is not None, f"Lowered name for {qualname} is None"
            qname_to_lowered[qualname] = lowered_name.decode()

    # Get PTX
    ptx_size = ctypes.c_size_t()
    check_nvrtc(libnvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size)))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    check_nvrtc(libnvrtc.nvrtcGetPTX(prog, ptx))
    libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    return ptx.value, qname_to_lowered


class _CudaModule:
    def __init__(self, module: ctypes.c_void_p) -> None:
        self._module = module
        self._kernels: dict[str, _CudaKernel] = {}

    def __getitem__(self, name: str) -> "_CudaKernel":
        if name in self._kernels:
            return self._kernels[name]

        # Import the CUDA library inside the method
        from torch.cuda._utils import _get_cuda_library

        libcuda = _get_cuda_library()

        func = ctypes.c_void_p()
        try:
            _check_cuda(
                libcuda.cuModuleGetFunction(
                    ctypes.byref(func), self._module, name.encode("utf-8")
                )
            )
            kernel = _CudaKernel(func, self._module)
            self._kernels[name] = kernel
            return kernel

        except RuntimeError as err:
            raise AttributeError(f"No kernel named '{name}' in this module") from err


class _CudaKernel:
    """
    Represents a compiled CUDA kernel that can be called with PyTorch tensors.
    """

    def __init__(self, func: ctypes.c_void_p, module: ctypes.c_void_p) -> None:
        self.func = func
        self.module = module
        self._cached_static_shared_bytes: int | None = None
        self._cached_max_dynamic_shared_bytes: int | None = None

    def _ensure_dynamic_shared_memory_capacity(self, requested_shared_mem: int) -> None:
        if requested_shared_mem <= 0:
            return

        if self._cached_max_dynamic_shared_bytes is None:
            self._cached_max_dynamic_shared_bytes = _cu_func_get_attribute(
                self.func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            )
        if requested_shared_mem <= self._cached_max_dynamic_shared_bytes:
            return

        if self._cached_static_shared_bytes is None:
            self._cached_static_shared_bytes = _cu_func_get_attribute(
                self.func, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
            )

        device = _get_device_index(torch.cuda.current_device())
        shared_default = _cu_device_get_attribute(
            device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
        )
        shared_optin = _cu_device_get_attribute(
            device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        )
        max_dynamic_shared_optin = shared_optin - self._cached_static_shared_bytes
        if requested_shared_mem > max_dynamic_shared_optin:
            raise RuntimeError(
                "Requested dynamic shared memory exceeds device opt-in limit: "
                f"requested={requested_shared_mem} max_supported={max_dynamic_shared_optin} "
                f"static_shared={self._cached_static_shared_bytes}"
            )

        # Match Triton's opt-in behavior: when the device supports a larger
        # shared-memory carveout than the portable default, prefer shared memory
        # and lift the function's dynamic shared-memory ceiling to the device max.
        if shared_optin > shared_default:
            _cu_func_set_cache_config(self.func, CU_FUNC_CACHE_PREFER_SHARED)
            _cu_func_set_attribute(
                self.func,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                max_dynamic_shared_optin,
            )

        self._cached_max_dynamic_shared_bytes = _cu_func_get_attribute(
            self.func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        )
        if requested_shared_mem > self._cached_max_dynamic_shared_bytes:
            raise RuntimeError(
                "Failed to opt in enough dynamic shared memory for kernel launch: "
                f"requested={requested_shared_mem} configured={self._cached_max_dynamic_shared_bytes}"
            )

    def __call__(
        self,
        grid: tuple[int, int, int] = (1, 1, 1),
        block: tuple[int, int, int] = (1, 1, 1),
        args: Optional[list] = None,
        shared_mem: int = 0,
        stream: Optional[Any] = None,
    ) -> None:
        """
        Call the compiled CUDA kernel

        Args:
            grid (tuple): Grid dimensions (grid_x, grid_y, grid_z)
            block (tuple): Block dimensions (block_x, block_y, block_z)
            args (list): List of arguments to pass to the kernel.
                         PyTorch tensor arguments will be automatically converted to pointers.
            shared_mem (int): Shared memory size in bytes
            stream (torch.cuda.Stream): CUDA stream to use. If None, uses current stream.
        """
        import torch

        libcuda = torch.cuda._utils._get_cuda_library()

        if not args:
            args = []

        # Process arguments and convert tensors to pointers
        processed_args: list[ctypes.c_void_p] = []
        c_args = []

        for arg in args:
            if isinstance(arg, torch.Tensor):
                if not arg.is_cuda and not (arg.is_cpu and arg.is_pinned()):
                    raise ValueError(
                        "All tensor arguments must be CUDA tensors or pinned CPU tensors"
                    )
                # Get pointer to tensor data
                ptr = ctypes.c_void_p(arg.data_ptr())
                processed_args.append(ptr)
                c_args.append(ctypes.byref(ptr))
            elif isinstance(arg, int):
                # Convert integers to C int
                c_int = ctypes.c_int(arg)
                # Store the C int for reference keeping, not in processed_args
                c_args.append(ctypes.byref(c_int))
            # TODO: Python floats are actually doubles
            elif isinstance(arg, float):
                # Convert floats to C float
                c_float = ctypes.c_float(arg)
                # Store the C float for reference keeping, not in processed_args
                c_args.append(ctypes.byref(c_float))
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        # Convert to array of void pointers
        c_args_array = (ctypes.c_void_p * len(c_args))()
        for i, arg in enumerate(c_args):
            c_args_array[i] = ctypes.cast(arg, ctypes.c_void_p)

        # Get the stream
        if stream is None:
            # Defer import to avoid circular imports
            import torch.cuda

            stream = torch.cuda.current_stream()

        self._ensure_dynamic_shared_memory_capacity(shared_mem)

        _check_cuda(
            libcuda.cuLaunchKernel(
                self.func,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_mem,
                stream._as_parameter_,
                c_args_array,
                None,
            )
        )


def _cuda_load_module(
    ptx: Union[str, bytes],
    kernel_names: Optional[list[str]] = None,
) -> Union[_CudaModule, dict[str, "_CudaKernel"]]:
    """
    Loads a CUDA module from PTX code and returns a module object that can access kernels.

    Args:
        ptx (bytes or str): The PTX code to load
        kernel_names (list, optional): List of kernel names to extract from the module.
                                      If None, will return a module object with __getattr__.

    Returns:
        object: If kernel_names is None, returns a module object with __getattr__ to access kernels.
               If kernel_names is provided, returns a dict mapping kernel names to _CudaKernel objects.
    """
    # Ensure CUDA is initialized
    import torch.cuda

    # Load CUDA driver library
    libcuda = _get_cuda_library()

    # Convert PTX to bytes if it's a string
    if isinstance(ptx, str):
        ptx = ptx.encode("utf-8")

    # Load PTX module
    module = ctypes.c_void_p()
    # Get the current stream without directly importing torch.cuda at module level
    stream = torch.cuda.current_stream()
    with stream:
        _check_cuda(libcuda.cuModuleLoadData(ctypes.byref(module), ptx))

    if not kernel_names:
        return _CudaModule(module)

    # Return specific kernels
    kernels = {}
    for name in kernel_names:
        func = ctypes.c_void_p()
        _check_cuda(
            libcuda.cuModuleGetFunction(
                ctypes.byref(func), module, name.encode("utf-8")
            )
        )
        kernels[name] = _CudaKernel(func, module)
    return kernels


def _cuda_load_module_v2(
    ptx: Union[str, bytes],
    qname_to_lowered: dict[str, str],
    qname_to_user_name: Optional[dict[str, str]] = None,
) -> Union[_CudaModule, dict[str, "_CudaKernel"]]:
    """
    Loads a CUDA module from PTX code and returns a module object that can access kernels.

    Args:
        ptx (bytes or str): The PTX code to load
        kernel_names (list, optional): List of kernel names to extract from the module.
                                      If None, will return a module object with __getattr__.

    Returns:
        object: If kernel_names is None, returns a module object with __getattr__ to access kernels.
               If kernel_names is provided, returns a dict mapping kernel names to _CudaKernel objects.
    """
    # Ensure CUDA is initialized
    import torch.cuda

    # Load CUDA driver library
    libcuda = _get_cuda_library()

    # Convert PTX to bytes if it's a string
    if isinstance(ptx, str):
        ptx = ptx.encode("utf-8")

    # Load PTX module
    module = ctypes.c_void_p()
    # Get the current stream without directly importing torch.cuda at module level
    stream = torch.cuda.current_stream()
    with stream:
        _check_cuda(libcuda.cuModuleLoadData(ctypes.byref(module), ptx))

    if not qname_to_lowered:
        return _CudaModule(module)

    # Return specific kernels
    kernels = {}
    for name, lowered in qname_to_lowered.items():
        func = ctypes.c_void_p()
        _check_cuda(
            libcuda.cuModuleGetFunction(
                ctypes.byref(func), module, lowered.encode("utf-8")
            )
        )
        if qname_to_user_name is not None and name in qname_to_user_name:
            user_name = qname_to_user_name[name]
            kernels[user_name] = _CudaKernel(func, module)
        else:
            kernels[name] = _CudaKernel(func, module)
    return kernels


def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Get the device index from :attr:`device`, which can be a torch.device object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for a CUDA device without a specified index,
    i.e., ``torch.device('cuda')``, this will return the current default CUDA
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["cuda", "cpu"]:
                raise ValueError(f"Expected a cuda or cpu device, but got: {device}")
        elif device.type != "cuda":
            raise ValueError(f"Expected a cuda device, but got: {device}")
    if not torch.jit.is_scripting():
        if isinstance(device, torch.cuda.device):
            return device.idx
    return _torch_get_device_index(device, optional, allow_cpu)


def compile_kernel(
    kernel_source: str,
    kernel_name: str,
    compute_capability: Optional[str] = None,
    header_code: str = "",
    cuda_include_dirs: Optional[list] = None,
    nvcc_options: Optional[list] = None,
):
    """
    Compiles a CUDA kernel using NVRTC and returns a callable function.

    This function is a wrapper for NVRTC that enables runtime compilation of CUDA kernels.
    Note that this returns a raw CUDA kernel that operates on raw memory pointers.
    To use this kernel as a proper PyTorch operator, you should wrap it following the guide at:
    pytorch.org/tutorials/advanced/python_custom_ops.html

    Args:
        kernel_source (str): The CUDA kernel source code as a string
        kernel_name (str): The name of the kernel function to compile
        compute_capability (str, optional): The compute capability to target (e.g., "86").
                                           If None, will detect from current device.
        header_code (str, optional): Additional header code to prepend to the kernel source
        cuda_include_dirs (list, optional): List of directories containing CUDA headers
        nvcc_options (list, optional): Additional options to pass to NVRTC

    Returns:
        callable: A Python function that can be called with PyTorch tensor arguments to execute the kernel

    Example:
        >>> # xdoctest: +SKIP
        >>> kernel_code = '''
        extern "C"
        __global__ void add_tensors(const float* a, const float* b, float* c, int n) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n)
                c[i] = a[i] + b[i];
        }
        '''
        >>> add_kernel = torch.cuda.compile_kernel(kernel_code, "add_tensors")
        >>> a = torch.randn(1024, device="cuda")
        >>> b = torch.randn(1024, device="cuda")
        >>> c = torch.empty_like(a)
        >>> add_kernel(grid=(4, 1, 1), block=(256, 1, 1), args=[a, b, c, a.numel()])
    """
    # Load the module and get the kernel
    ptx, _ = _nvrtc_compile(
        kernel_source,
        kernel_name,
        compute_capability,
        header_code,
        cuda_include_dirs,
        nvcc_options,
    )

    # Load the module and get the kernel
    result = _cuda_load_module(ptx, [kernel_name])

    if isinstance(result, dict):
        return result[kernel_name]
    else:
        # This branch shouldn't be executed if kernel_names is provided,
        # but MyPy needs this to understand type narrowing
        return getattr(result, kernel_name)


def compile_kernels(
    kernel_source: str,
    kernel_qualnames: list[str],
    compute_capability: Optional[str] = None,
    header_code: str = "",
    cuda_include_dirs: Optional[list] = None,
    nvcc_options: Optional[list] = None,
    kernel_file_name: Optional[str] = None,
    qname_to_user_name: Optional[dict[str, str]] = None,
):
    """
    Compiles a CUDA kernel using NVRTC and returns a callable function.

    This function is a wrapper for NVRTC that enables runtime compilation of CUDA kernels.
    Note that this returns a raw CUDA kernel that operates on raw memory pointers.
    To use this kernel as a proper PyTorch operator, you should wrap it following the guide at:
    pytorch.org/tutorials/advanced/python_custom_ops.html

    Args:
        kernel_source (str): The CUDA kernel source code as a string
        kernel_name (str): The name of the kernel function to compile
        compute_capability (str, optional): The compute capability to target (e.g., "86").
                                           If None, will detect from current device.
        header_code (str, optional): Additional header code to prepend to the kernel source
        cuda_include_dirs (list, optional): List of directories containing CUDA headers
        nvcc_options (list, optional): Additional options to pass to NVRTC

    Returns:
        callable: A Python function that can be called with PyTorch tensor arguments to execute the kernel

    Example:
        >>> # xdoctest: +SKIP
        >>> kernel_code = '''
        extern "C"
        __global__ void add_tensors(const float* a, const float* b, float* c, int n) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n)
                c[i] = a[i] + b[i];
        }
        '''
        >>> add_kernel = torch.cuda.compile_kernel(kernel_code, "add_tensors")
        >>> a = torch.randn(1024, device="cuda")
        >>> b = torch.randn(1024, device="cuda")
        >>> c = torch.empty_like(a)
        >>> add_kernel(grid=(4, 1, 1), block=(256, 1, 1), args=[a, b, c, a.numel()])
    """
    import ctypes

    # Compile the kernel to PTX
    ptx, qname_to_lowered = _nvrtc_compile(
        kernel_source,
        kernel_file_name or "kernel",
        compute_capability,
        header_code,
        cuda_include_dirs,
        nvcc_options,
        qualnames=kernel_qualnames,
        add_external_c=False,
    )

    # Load the module and get the kernel
    result = _cuda_load_module_v2(ptx, qname_to_lowered, qname_to_user_name)

    return result
