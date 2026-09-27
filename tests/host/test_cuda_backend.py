from __future__ import annotations

import ctypes

import pytest

from naivert.runtime.cuda_backend import (
    CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH,
    CudaBackendError,
    CudaGraphSnapshot,
    CudaRuntimeGraphApi,
)


def probe_cuda_backend(library_name, *, loader):
    """The probe is only reachable through ``load`` now; tests keep the old spelling."""
    return CudaRuntimeGraphApi.load(library_name, loader=loader).capabilities


def test_graph_snapshot_owns_canonical_linear_traversal() -> None:
    snapshot = CudaGraphSnapshot(
        node_handles=(0x102, 0x101, 0x103),
        node_types=(0, 0, 0),
        edges=((1, 0), (0, 2)),
        edge_data=(None, None),
    )

    assert snapshot.linear_order(kernel_only=True) == (1, 0, 2)


class _FakeFunction:
    def __init__(self, result: int = 0, *, output: int | None = None) -> None:
        self.result = int(result)
        self.output = output
        self.calls: list[tuple[object, ...]] = []
        self.restype = None
        self.argtypes = None

    def __call__(self, *args: object) -> int:
        self.calls.append(args)
        if self.output is not None:
            pointer = args[0]
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p)).contents.value = (
                int(self.output)
            )
        return self.result


class _FakeLibrary:
    def __init__(self, *, missing: tuple[str, ...] = ()) -> None:
        self._name = "libcudart.so.12"
        self._functions: dict[str, _FakeFunction] = {}
        for name in (
            "cudaGraphCreate",
            "cudaGraphDestroy",
            "cudaGraphAddChildGraphNode",
            "cudaGraphInstantiate",
            "cudaGraphInstantiateWithFlags",
            "cudaGraphUpload",
            "cudaGraphLaunch",
            "cudaGraphExecDestroy",
            "cudaStreamWaitEvent",
            "cudaStreamCreateWithFlags",
            "cudaStreamDestroy",
            "cudaStreamSynchronize",
            "cudaSetDevice",
            "cudaHostAlloc",
            "cudaFreeHost",
            "cudaMemcpyAsync",
            "cudaEventCreateWithFlags",
            "cudaEventRecord",
            "cudaEventDestroy",
            "cudaEventQuery",
            "cudaGraphGetNodes",
            "cudaGraphGetEdges",
            "cudaGraphNodeGetType",
            "cudaGraphKernelNodeGetParams",
        ):
            if name in missing:
                continue
            output = 0x1000 + len(self._functions) if name in {
                "cudaGraphCreate",
                "cudaGraphAddChildGraphNode",
                "cudaGraphInstantiate",
                "cudaGraphInstantiateWithFlags",
                "cudaEventCreateWithFlags",
                "cudaStreamCreateWithFlags",
                "cudaHostAlloc",
            } else None
            self._functions[name] = _FakeFunction(output=output)
            setattr(self, name, self._functions[name])


class _InspectionLibrary(_FakeLibrary):
    def __init__(self) -> None:
        super().__init__()

        class GetNodes:
            restype = None
            argtypes = None

            def __call__(self, graph: object, nodes: object, count: object) -> int:
                del graph
                ctypes.cast(count, ctypes.POINTER(ctypes.c_size_t)).contents.value = 3
                if nodes is not None:
                    for index, value in enumerate((0x101, 0x102, 0x103)):
                        nodes[index] = value
                return 0

        class GetEdges:
            restype = None
            argtypes = None

            def __call__(
                self,
                graph: object,
                from_nodes: object,
                to_nodes: object,
                count: object,
            ) -> int:
                del graph
                ctypes.cast(count, ctypes.POINTER(ctypes.c_size_t)).contents.value = 2
                if from_nodes is not None:
                    for index, value in enumerate((0x101, 0x102)):
                        from_nodes[index] = value
                        to_nodes[index] = value + 1
                return 0

        class GetType:
            restype = None
            argtypes = None

            def __call__(self, node: object, node_type: object) -> int:
                del node
                ctypes.cast(node_type, ctypes.POINTER(ctypes.c_int)).contents.value = 0
                return 0

        self.cudaGraphGetNodes = GetNodes()
        self.cudaGraphGetEdges = GetEdges()
        self.cudaGraphNodeGetType = GetType()


class _InspectionCuda13Library(_InspectionLibrary):
    def __init__(self) -> None:
        super().__init__()
        self._name = "libcudart.so.13"

        class GetEdges13:
            restype = None
            argtypes = None

            def __call__(
                self,
                graph: object,
                from_nodes: object,
                to_nodes: object,
                edge_data: object,
                count: object,
            ) -> int:
                del graph, edge_data
                ctypes.cast(count, ctypes.POINTER(ctypes.c_size_t)).contents.value = 2
                if from_nodes is not None:
                    for index, value in enumerate((0x101, 0x102)):
                        from_nodes[index] = value
                        to_nodes[index] = value + 1
                return 0

        self.cudaGraphGetEdges = GetEdges13()


def test_probe_is_symbol_only_and_reports_device_graph_abi() -> None:
    library = _FakeLibrary()
    capabilities = probe_cuda_backend("libcudart.so.12", loader=lambda _: library)
    assert capabilities.parent_graph_composition is True
    assert capabilities.persistent_device_graph is True
    assert capabilities.as_dict()["device_graph_launch"] is True
    assert capabilities.runtime_major == 12
    assert capabilities.stream_capture is False
    assert capabilities.graph_kernel_node_get_params is True


def test_probe_reports_stream_capture_only_with_both_symbols() -> None:
    library = _FakeLibrary()
    library.cudaStreamBeginCapture = _FakeFunction()
    library.cudaStreamEndCapture = _FakeFunction()
    capabilities = probe_cuda_backend("libcudart.so.12", loader=lambda _: library)
    assert capabilities.stream_begin_capture is True
    assert capabilities.stream_end_capture is True
    assert capabilities.stream_capture is True

    del library.cudaStreamEndCapture
    capabilities = probe_cuda_backend("libcudart.so.12", loader=lambda _: library)
    assert capabilities.stream_capture is False


def test_probe_fails_closed_when_runtime_library_is_missing() -> None:
    with pytest.raises(CudaBackendError, match="unable to load"):
        probe_cuda_backend(
            "libcudart.so.12",
            loader=lambda _: (_ for _ in ()).throw(OSError("not found")),
        )


def test_probe_does_not_claim_persistent_without_upload_or_flag_abi() -> None:
    library = _FakeLibrary(missing=("cudaGraphUpload",))
    capabilities = probe_cuda_backend("libcudart.so.12", loader=lambda _: library)
    assert capabilities.parent_graph_composition is True
    assert capabilities.persistent_device_graph is False


def test_graph_topology_symbols_are_optional_for_parent_launch() -> None:
    library = _FakeLibrary(
        missing=(
            "cudaGraphGetNodes",
            "cudaGraphGetEdges",
            "cudaGraphNodeGetType",
            "cudaGraphKernelNodeGetParams",
        )
    )
    capabilities = probe_cuda_backend("libcudart.so.12", loader=lambda _: library)
    assert capabilities.parent_graph_composition is True
    assert capabilities.graph_get_nodes is False
    assert capabilities.graph_get_edges is False
    assert capabilities.graph_node_get_type is False
    assert capabilities.graph_kernel_node_get_params is False


def test_probe_rejects_cuda11_legacy_graph_instantiate_abi() -> None:
    library = _FakeLibrary()
    library._name = "libcudart.so.11.8"
    with pytest.raises(CudaBackendError, match="requires CUDA 12"):
        probe_cuda_backend("libcudart.so.11.8", loader=lambda _: library)


def test_probe_rejects_versioned_alias_with_older_realpath(tmp_path) -> None:
    target = tmp_path / "libcudart.so.11.8"
    alias = tmp_path / "libcudart.so.12"
    target.touch()
    alias.symlink_to(target.name)
    library = _FakeLibrary()
    library._name = str(alias)
    with pytest.raises(CudaBackendError, match="disagrees"):
        probe_cuda_backend(str(alias), loader=lambda _: library)


@pytest.mark.parametrize(
    "missing",
    [
        ("cudaGraphDestroy",),
        ("cudaGraphExecDestroy",),
    ],
)
def test_probe_requires_cleanup_and_ordering_symbols(
    missing: tuple[str, ...],
) -> None:
    library = _FakeLibrary(missing=missing)
    capabilities = probe_cuda_backend("libcudart.so.12", loader=lambda _: library)
    assert capabilities.parent_graph_composition is False
    assert capabilities.persistent_device_graph is False


def test_graph_api_binds_device_launch_flags() -> None:
    library = _FakeLibrary()
    capabilities = probe_cuda_backend("libcudart.so.12", loader=lambda _: library)
    api = CudaRuntimeGraphApi(library, capabilities=capabilities)

    graph = ctypes.c_void_p(0x100)
    executable = api.graph_instantiate(
        graph,
        flags=CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH,
    )
    api.graph_upload(executable, ctypes.c_void_p(0x9000))
    api.graph_exec_destroy(executable)
    api.graph_destroy(graph)

    assert executable.value
    assert len(library.cudaGraphInstantiateWithFlags.calls) == 1
    assert library.cudaGraphInstantiateWithFlags.calls[0][2].value == (
        CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH
    )
    assert len(library.cudaGraphUpload.calls) == 1
    assert len(library.cudaGraphDestroy.calls) == 1


def test_graph_api_rejects_upload_flag_without_with_params_binding() -> None:
    library = _FakeLibrary()
    capabilities = probe_cuda_backend("libcudart.so.12", loader=lambda _: library)
    api = CudaRuntimeGraphApi(library, capabilities=capabilities)
    with pytest.raises(CudaBackendError, match="Upload flag"):
        api.graph_instantiate(
            0x100,
            flags=2,
        )
    assert library.cudaGraphInstantiateWithFlags.calls == []


def test_load_probes_the_same_library_handle_it_binds() -> None:
    library = _FakeLibrary()
    calls: list[str] = []

    def loader(name: str) -> _FakeLibrary:
        calls.append(name)
        return library

    api = CudaRuntimeGraphApi.load("libcudart.so.12", loader=loader)
    assert api.library is library
    assert calls == ["libcudart.so.12"]


def test_direct_constructor_cannot_bypass_abi_probe() -> None:
    library = _FakeLibrary()
    library._name = "libcudart.so.11.8"
    with pytest.raises(CudaBackendError, match="requires CUDA 12"):
        CudaRuntimeGraphApi(library)


def test_direct_constructor_rejects_unidentifiable_library() -> None:
    library = _FakeLibrary()
    library._name = "libcudart.so"
    with pytest.raises(CudaBackendError, match="determine"):
        CudaRuntimeGraphApi(library)


def test_graph_api_rejects_zero_handles_before_runtime_call() -> None:
    library = _FakeLibrary()
    api = CudaRuntimeGraphApi(library)
    with pytest.raises(CudaBackendError, match="null"):
        api.graph_upload(0, ctypes.c_void_p(0x100))
    assert library.cudaGraphUpload.calls == []


def test_graph_api_inspects_actual_nodes_edges_and_types() -> None:
    library = _InspectionLibrary()
    capabilities = probe_cuda_backend("libcudart.so.12", loader=lambda _: library)
    api = CudaRuntimeGraphApi(library, capabilities=capabilities)
    inspected = api.inspect_graph(0x5000)
    assert inspected["node_count"] == 3
    assert inspected["edge_count"] == 2
    assert inspected["canonical_node_types"] == [0, 0, 0]
    assert inspected["canonical_edges"] == [[0, 1], [1, 2]]
    assert len(str(inspected["topology_digest"])) == 64


def test_graph_api_uses_cuda13_five_argument_edge_abi() -> None:
    library = _InspectionCuda13Library()
    capabilities = probe_cuda_backend("libcudart.so.13", loader=lambda _: library)
    api = CudaRuntimeGraphApi(library, capabilities=capabilities)
    inspected = api.inspect_graph(0x5000)
    assert inspected["canonical_edges"] == [[0, 1], [1, 2]]


def test_graph_api_surfaces_cuda_error() -> None:
    library = _FakeLibrary()
    library.cudaGraphUpload.result = 719
    api = CudaRuntimeGraphApi(library)
    with pytest.raises(CudaBackendError, match="cudaGraphUpload.*719"):
        api.graph_upload(0x100, 0x200)
