"""Host-side CUDA Runtime ABI and capability boundary.

This module intentionally does not create a CUDA context while probing the
runtime.  It only loads ``libcudart`` and verifies that the entry points needed
by the concrete graph backend are present.  Graph handles are kept as opaque
``ctypes.c_void_p`` values.

The dspark draft round instantiates its captured round graph with the device
launch flag, uploads it and launches it once through this boundary; the GPU
harnesses use ``host_alloc``/``host_free`` for their completion flags. Capability failures are explicit; there is no
alternate launch path.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import ctypes.util
import os
import re
from typing import Any, Iterable, Mapping

from .contract import RuntimeContractError, canonical_contract_digest

# CUDA Runtime API constants.  These values are part of the public CUDA ABI
# and are repeated here so the host-only probe does not need to import CUDA
# headers or initialize a CUDA context.
CUDA_SUCCESS = 0
CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = 1 << 2
CUDA_STREAM_CAPTURE_MODE_RELAXED = 2
CUDA_HOST_ALLOC_DEFAULT = 0
_CUDA_RUNTIME_MAJOR_RE = re.compile(r"libcudart\.so\.(\d+)")


class CudaBackendError(RuntimeContractError):
    """Raised when the CUDA runtime backend cannot satisfy its ABI contract."""


class CudaGraphEdgeData(ctypes.Structure):
    _fields_ = [
        ("from_port", ctypes.c_ubyte),
        ("to_port", ctypes.c_ubyte),
        ("type", ctypes.c_ubyte),
        ("reserved", ctypes.c_ubyte * 5),
    ]


@dataclass(frozen=True)
class CudaGraphSnapshot:
    """One immutable CUDA graph node/type/edge enumeration."""

    node_handles: tuple[int, ...]
    node_types: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    edge_data: tuple[bytes | None, ...]

    def __post_init__(self) -> None:
        node_count = len(self.node_handles)
        if (
            len(self.node_types) != node_count
            or len(self.edges) != len(self.edge_data)
            or len(set(self.node_handles)) != node_count
            or any(handle <= 0 for handle in self.node_handles)
            or any(
                source < 0
                or source >= node_count
                or destination < 0
                or destination >= node_count
                for source, destination in self.edges
            )
        ):
            raise CudaBackendError(
                "CUDA graph snapshot contains an invalid inventory"
            )

    def linear_order(self, *, kernel_only: bool) -> tuple[int, ...]:
        if not self.node_handles:
            raise CudaBackendError("captured graph has no nodes")
        if kernel_only and any(node_type != 0 for node_type in self.node_types):
            raise CudaBackendError(
                "captured body contract contains a non-kernel node"
            )
        outgoing = [[] for _ in self.node_handles]
        incoming = [[] for _ in self.node_handles]
        for source, destination in self.edges:
            outgoing[source].append(destination)
            incoming[destination].append(source)
        roots = [
            index for index, sources in enumerate(incoming) if not sources
        ]
        if len(roots) != 1:
            raise CudaBackendError(
                f"captured graph has {len(roots)} dependency roots"
            )
        ordered = [roots[0]]
        while outgoing[ordered[-1]]:
            next_nodes = outgoing[ordered[-1]]
            if len(next_nodes) != 1:
                raise CudaBackendError(
                    "captured graph kernel dependencies are not linear"
                )
            next_node = next_nodes[0]
            if next_node in ordered:
                break
            ordered.append(next_node)
        if (
            len(ordered) != len(self.node_handles)
            or len(self.edges) != len(self.node_handles) - 1
        ):
            raise CudaBackendError(
                "captured graph kernel chain is incomplete"
            )
        return tuple(ordered)


class CudaDim3(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_uint),
        ("y", ctypes.c_uint),
        ("z", ctypes.c_uint),
    ]


class CudaKernelNodeParams(ctypes.Structure):
    """CUDA 12+ ``cudaKernelNodeParams`` ABI used for read-only inspection."""

    _fields_ = [
        ("func", ctypes.c_void_p),
        ("grid_dim", CudaDim3),
        ("block_dim", CudaDim3),
        ("shared_mem_bytes", ctypes.c_uint),
        ("kernel_params", ctypes.POINTER(ctypes.c_void_p)),
        ("extra", ctypes.POINTER(ctypes.c_void_p)),
    ]


class CudaKernelNodeAttrValue(ctypes.Union):
    """CUDA launch-attribute value ABI used by graph kernel nodes."""

    _fields_ = [
        ("padding", ctypes.c_byte * 64),
        ("cooperative", ctypes.c_int),
    ]


class CudaDriverKernelNodeParams(ctypes.Structure):
    """CUDA Driver ``CUDA_KERNEL_NODE_PARAMS_v2`` ABI."""

    _fields_ = [
        ("func", ctypes.c_void_p),
        ("grid_x", ctypes.c_uint),
        ("grid_y", ctypes.c_uint),
        ("grid_z", ctypes.c_uint),
        ("block_x", ctypes.c_uint),
        ("block_y", ctypes.c_uint),
        ("block_z", ctypes.c_uint),
        ("shared_mem_bytes", ctypes.c_uint),
        ("kernel_params", ctypes.POINTER(ctypes.c_void_p)),
        ("extra", ctypes.POINTER(ctypes.c_void_p)),
        ("kernel", ctypes.c_void_p),
        ("context", ctypes.c_void_p),
    ]


@dataclass(frozen=True)
class CudaBackendCapabilities:
    """Symbol-level capability result from a host-only runtime probe.

    ``device_graph_launch`` means that the host ABI exposes the instantiate
    flag and upload operation required to attempt a device-launched graph.  It
    does not claim that the selected GPU/driver accepts a particular graph;
    that is checked by the GPU smoke stage.
    """

    library_name: str
    library_path: str
    runtime_major: int
    graph_create: bool
    graph_destroy: bool
    graph_add_child: bool
    graph_instantiate: bool
    graph_instantiate_with_flags: bool
    graph_upload: bool
    graph_exec_destroy: bool
    graph_get_nodes: bool
    graph_get_edges: bool
    graph_get_edges_v2: bool
    graph_node_get_type: bool
    graph_kernel_node_get_params: bool
    device_graph_launch: bool
    stream_begin_capture: bool
    stream_end_capture: bool

    @property
    def parent_graph_composition(self) -> bool:
        """Whether the ABI can compose the fixed three-child graph."""

        return bool(
            self.graph_create
            and self.graph_add_child
            and self.graph_instantiate
            and self.graph_destroy
            and self.graph_exec_destroy
        )

    @property
    def persistent_device_graph(self) -> bool:
        """Whether the ABI can attempt the CUDA device-graph path."""

        return bool(
            self.parent_graph_composition
            and self.graph_instantiate_with_flags
            and self.graph_upload
            and self.device_graph_launch
        )

    @property
    def stream_capture(self) -> bool:
        """Whether the ABI can begin/end a CUDA stream capture."""

        return bool(self.stream_begin_capture and self.stream_end_capture)

    def as_dict(self) -> dict[str, bool | str]:
        return {
            "library_name": self.library_name,
            "library_path": self.library_path,
            "runtime_major": self.runtime_major,
            "graph_create": self.graph_create,
            "graph_destroy": self.graph_destroy,
            "graph_add_child": self.graph_add_child,
            "graph_instantiate": self.graph_instantiate,
            "graph_instantiate_with_flags": self.graph_instantiate_with_flags,
            "graph_upload": self.graph_upload,
            "graph_exec_destroy": self.graph_exec_destroy,
            "graph_get_nodes": self.graph_get_nodes,
            "graph_get_edges": self.graph_get_edges,
            "graph_get_edges_v2": self.graph_get_edges_v2,
            "graph_node_get_type": self.graph_node_get_type,
            "graph_kernel_node_get_params": self.graph_kernel_node_get_params,
            "device_graph_launch": self.device_graph_launch,
            "stream_begin_capture": self.stream_begin_capture,
            "stream_end_capture": self.stream_end_capture,
            "stream_capture": self.stream_capture,
            "parent_graph_composition": self.parent_graph_composition,
            "persistent_device_graph": self.persistent_device_graph,
        }


_REQUIRED_SYMBOLS: Mapping[str, str] = {
    "graph_create": "cudaGraphCreate",
    "graph_destroy": "cudaGraphDestroy",
    "graph_add_child": "cudaGraphAddChildGraphNode",
    "graph_instantiate": "cudaGraphInstantiate",
    "graph_instantiate_with_flags": "cudaGraphInstantiateWithFlags",
    "graph_upload": "cudaGraphUpload",
    "graph_exec_destroy": "cudaGraphExecDestroy",
}
_OPTIONAL_SYMBOLS: Mapping[str, str] = {
    "graph_get_nodes": "cudaGraphGetNodes",
    "graph_get_edges": "cudaGraphGetEdges",
    "graph_get_edges_v2": "cudaGraphGetEdges_v2",
    "graph_node_get_type": "cudaGraphNodeGetType",
    "graph_kernel_node_get_params": "cudaGraphKernelNodeGetParams",
    "stream_begin_capture": "cudaStreamBeginCapture",
    "stream_end_capture": "cudaStreamEndCapture",
}


def _library_candidates(requested: str | None) -> tuple[str, ...]:
    """Return deterministic libcudart candidates without touching CUDA."""

    values: list[str] = []
    if requested:
        values.append(str(requested))
    discovered = ctypes.util.find_library("cudart")
    if discovered:
        values.append(str(discovered))
    # The production graph launcher uses the CUDA 12 ABI.  Keep explicit
    # fallbacks so a minimal container with no ldconfig entry is diagnosable.
    values.extend(("libcudart.so.12", "libcudart.so"))
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return tuple(out)


def _load_library(
    requested: str | None,
    *,
    loader: type[ctypes.CDLL] = ctypes.CDLL,
) -> tuple[str, ctypes.CDLL]:
    errors: list[str] = []
    for candidate in _library_candidates(requested):
        try:
            library = loader(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        return candidate, library
    joined = "; ".join(errors) if errors else "no libcudart candidate"
    raise CudaBackendError(f"unable to load CUDA Runtime library: {joined}")


def _runtime_major(selected: str, library: ctypes.CDLL) -> int | None:
    """Infer the libcudart major and reject alias/path disagreements."""

    candidates = [
        str(selected),
        str(getattr(library, "_name", "") or ""),
    ]
    evidence: list[tuple[str, int]] = []
    for candidate in candidates:
        match = _CUDA_RUNTIME_MAJOR_RE.search(candidate)
        if match:
            evidence.append((candidate, int(match.group(1))))
        try:
            resolved = os.path.realpath(candidate)
        except (OSError, TypeError):
            resolved = ""
        if resolved and resolved != candidate:
            match = _CUDA_RUNTIME_MAJOR_RE.search(resolved)
            if match:
                evidence.append((resolved, int(match.group(1))))
    majors = {major for _, major in evidence}
    if len(majors) > 1:
        details = ", ".join(f"{name}=>{major}" for name, major in evidence)
        raise CudaBackendError(
            "libcudart ABI major disagrees between loader name and resolved "
            f"path: {details}"
        )
    return next(iter(majors), None)


def _probe_loaded_library(
    selected: str,
    library: ctypes.CDLL,
) -> CudaBackendCapabilities:
    runtime_major = _runtime_major(selected, library)
    if runtime_major is None:
        raise CudaBackendError(
            "unable to determine libcudart ABI major from the loaded library; "
            "use an explicit versioned libcudart.so.<major> path"
        )
    # cudaGraphInstantiate has a different ABI in CUDA 11.x.  The ctypes
    # bindings below intentionally use the CUDA 12+ three-argument form.
    if runtime_major < 12:
        raise CudaBackendError(
            f"unsupported libcudart ABI major {runtime_major}; "
            "the runtime backend requires CUDA 12 or newer"
        )
    present = {
        key: callable(getattr(library, symbol, None))
        for key, symbol in _REQUIRED_SYMBOLS.items()
    }
    optional = {
        key: callable(getattr(library, symbol, None))
        for key, symbol in _OPTIONAL_SYMBOLS.items()
    }
    # ``cudaGraphInstantiateFlagDeviceLaunch`` is a compile-time ABI flag.  Its
    # runtime-side operation is exposed by cudaGraphInstantiateWithFlags.
    device_launch = bool(
        present["graph_instantiate_with_flags"] and present["graph_upload"]
    )
    return CudaBackendCapabilities(
        library_name=str(selected),
        library_path=str(selected),
        runtime_major=runtime_major,
        graph_create=present["graph_create"],
        graph_destroy=present["graph_destroy"],
        graph_add_child=present["graph_add_child"],
        graph_instantiate=present["graph_instantiate"],
        graph_instantiate_with_flags=present["graph_instantiate_with_flags"],
        graph_upload=present["graph_upload"],
        graph_exec_destroy=present["graph_exec_destroy"],
        graph_get_nodes=optional["graph_get_nodes"],
        graph_get_edges=optional["graph_get_edges"],
        graph_get_edges_v2=optional["graph_get_edges_v2"],
        graph_node_get_type=optional["graph_node_get_type"],
        graph_kernel_node_get_params=optional["graph_kernel_node_get_params"],
        device_graph_launch=device_launch,
        stream_begin_capture=optional["stream_begin_capture"],
        stream_end_capture=optional["stream_end_capture"],
    )


def _as_handle(value: int | ctypes.c_void_p) -> ctypes.c_void_p:
    if isinstance(value, ctypes.c_void_p):
        if not value.value:
            raise CudaBackendError("CUDA graph handle is null")
        return value
    handle = ctypes.c_void_p(int(value))
    if not handle.value:
        raise CudaBackendError("CUDA graph handle is null")
    return handle


class CudaRuntimeGraphApi:
    """Typed, low-level bindings used by later concrete backend stages.

    Construction is host-only.  It configures function signatures but does not
    invoke any CUDA Runtime call.  Methods return opaque ``c_void_p`` handles
    and raise :class:`CudaBackendError` on non-zero ``cudaError_t`` values.
    """

    def __init__(
        self,
        library: ctypes.CDLL,
        *,
        capabilities: CudaBackendCapabilities | None = None,
    ) -> None:
        self.library = library
        loaded_name = str(getattr(library, "_name", "") or "")
        validated = _probe_loaded_library(loaded_name, library)
        if capabilities is not None:
            # A caller may pass a prior probe result as an assertion, but it
            # cannot bypass validation of this exact CDLL handle.
            if (
                capabilities.runtime_major != validated.runtime_major
                or capabilities.parent_graph_composition
                != validated.parent_graph_composition
                or capabilities.persistent_device_graph
                != validated.persistent_device_graph
                or capabilities.stream_capture != validated.stream_capture
            ):
                raise CudaBackendError(
                    "provided CUDA capabilities do not match the loaded library"
                )
        self.capabilities = validated
        self._configure()

    @classmethod
    def load(
        cls,
        library_name: str | None = None,
        *,
        loader: type[ctypes.CDLL] = ctypes.CDLL,
    ) -> "CudaRuntimeGraphApi":
        selected, library = _load_library(library_name, loader=loader)
        # Probe the exact handle that will be used for calls. Loading twice can
        # otherwise make a custom loader or alternate SONAME describe one
        # library while binding methods on another.
        capabilities = _probe_loaded_library(selected, library)
        return cls(library, capabilities=capabilities)

    def _configure(self) -> None:
        def configure(name: str, restype: object, argtypes: Iterable[object]) -> None:
            function = getattr(self.library, name, None)
            if function is None:
                return
            function.restype = restype
            function.argtypes = list(argtypes)

        configure("cudaGraphDestroy", ctypes.c_int, (ctypes.c_void_p,))
        configure(
            "cudaGraphInstantiate",
            ctypes.c_int,
            (
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_ulonglong,
            ),
        )
        configure(
            "cudaGraphInstantiateWithFlags",
            ctypes.c_int,
            (
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_ulonglong,
            ),
        )
        configure(
            "cudaGraphUpload",
            ctypes.c_int,
            (ctypes.c_void_p, ctypes.c_void_p),
        )
        configure(
            "cudaGraphLaunch",
            ctypes.c_int,
            (ctypes.c_void_p, ctypes.c_void_p),
        )
        configure("cudaGraphExecDestroy", ctypes.c_int, (ctypes.c_void_p,))
        configure(
            "cudaGraphGetNodes",
            ctypes.c_int,
            (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_size_t),
            ),
        )
        configure(
            "cudaGraphGetEdges",
            ctypes.c_int,
            (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                *(
                    (
                        ctypes.POINTER(CudaGraphEdgeData),
                        ctypes.POINTER(ctypes.c_size_t),
                    )
                    if self.capabilities.runtime_major >= 13
                    else (ctypes.POINTER(ctypes.c_size_t),)
                ),
            ),
        )
        configure(
            "cudaGraphGetEdges_v2",
            ctypes.c_int,
            (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(CudaGraphEdgeData),
                ctypes.POINTER(ctypes.c_size_t),
            ),
        )
        configure(
            "cudaGraphNodeGetType",
            ctypes.c_int,
            (ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)),
        )
        configure(
            "cudaGraphKernelNodeGetParams",
            ctypes.c_int,
            (ctypes.c_void_p, ctypes.POINTER(CudaKernelNodeParams)),
        )
        configure(
            "cudaGraphKernelNodeGetAttribute",
            ctypes.c_int,
            (
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(CudaKernelNodeAttrValue),
            ),
        )
        configure(
            "cudaHostAlloc",
            ctypes.c_int,
            (ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint),
        )
        configure(
            "cudaFreeHost",
            ctypes.c_int,
            (ctypes.c_void_p,),
        )
        configure(
            "cudaStreamBeginCapture",
            ctypes.c_int,
            (ctypes.c_void_p, ctypes.c_int),
        )
        configure(
            "cudaStreamEndCapture",
            ctypes.c_int,
            (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
        )
    def _call(self, name: str, *args: object) -> int:
        function = getattr(self.library, name, None)
        if function is None:
            raise CudaBackendError(f"CUDA Runtime symbol {name} is unavailable")
        result = int(function(*args))
        if result != CUDA_SUCCESS:
            raise CudaBackendError(f"{name} failed with cudaError_t={result}")
        return result

    def graph_launch(
        self,
        graph_exec: int | ctypes.c_void_p,
        stream: int | ctypes.c_void_p,
    ) -> None:
        """Launch one instantiated graph on its rank-local stream."""

        self._call(
            "cudaGraphLaunch",
            _as_handle(graph_exec),
            _as_handle(stream),
        )

    def host_alloc(
        self,
        size: int,
        *,
        flags: int = CUDA_HOST_ALLOC_DEFAULT,
    ) -> ctypes.c_void_p:
        """Allocate fixed page-locked host storage for asynchronous D2H."""

        if int(size) <= 0:
            raise CudaBackendError("cudaHostAlloc size must be positive")
        pointer = ctypes.c_void_p()
        self._call(
            "cudaHostAlloc",
            ctypes.byref(pointer),
            ctypes.c_size_t(int(size)),
            ctypes.c_uint(int(flags)),
        )
        if not pointer.value:
            raise CudaBackendError("cudaHostAlloc returned a null pointer")
        return pointer

    def host_free(self, pointer: int | ctypes.c_void_p) -> None:
        """Release storage allocated by :meth:`host_alloc`."""

        self._call("cudaFreeHost", _as_handle(pointer))

    def graph_destroy(self, graph: int | ctypes.c_void_p) -> None:
        self._call("cudaGraphDestroy", _as_handle(graph))

    def graph_instantiate(
        self,
        graph: int | ctypes.c_void_p,
        *,
        flags: int = 0,
    ) -> ctypes.c_void_p:
        executable = ctypes.c_void_p()
        normalized_flags = int(flags)
        if normalized_flags & ~CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH:
            raise CudaBackendError(
                "graph_instantiate only accepts "
                "CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH; "
                "the Upload flag requires cudaGraphInstantiateWithParams"
            )
        if normalized_flags:
            function_name = "cudaGraphInstantiateWithFlags"
            self._call(
                function_name,
                ctypes.byref(executable),
                _as_handle(graph),
                ctypes.c_ulonglong(normalized_flags),
            )
        else:
            self._call(
                "cudaGraphInstantiate",
                ctypes.byref(executable),
                _as_handle(graph),
                ctypes.c_ulonglong(0),
            )
        if not executable.value:
            raise CudaBackendError("cudaGraphInstantiate returned a null exec")
        return executable

    def graph_upload(
        self,
        executable: int | ctypes.c_void_p,
        stream: int | ctypes.c_void_p,
    ) -> None:
        self._call(
            "cudaGraphUpload",
            _as_handle(executable),
            _as_handle(stream),
        )

    def stream_begin_capture(
        self,
        stream: int | ctypes.c_void_p,
        *,
        mode: int = CUDA_STREAM_CAPTURE_MODE_RELAXED,
    ) -> None:
        """Begin capture on an existing production-owned stream.

        Stream ownership stays with the caller.  This method deliberately does
        not create a stream or a CUDA context; the production graph builder is
        responsible for choosing the stream and for keeping it alive through
        all child-graph replays.
        """

        self._call(
            "cudaStreamBeginCapture",
            _as_handle(stream),
            ctypes.c_int(int(mode)),
        )

    def stream_end_capture(
        self,
        stream: int | ctypes.c_void_p,
    ) -> ctypes.c_void_p:
        """End capture and return the concrete raw graph handle."""

        graph = ctypes.c_void_p()
        self._call(
            "cudaStreamEndCapture",
            _as_handle(stream),
            ctypes.byref(graph),
        )
        if not graph.value:
            raise CudaBackendError(
                "cudaStreamEndCapture returned a null graph"
            )
        return graph

    def graph_exec_destroy(self, executable: int | ctypes.c_void_p) -> None:
        self._call("cudaGraphExecDestroy", _as_handle(executable))

    def snapshot_graph(
        self,
        graph: int | ctypes.c_void_p,
    ) -> CudaGraphSnapshot:
        """Enumerate one graph once for all topology consumers."""

        if not (
            self.capabilities.graph_get_nodes
            and (
                self.capabilities.graph_get_edges
                or self.capabilities.graph_get_edges_v2
            )
            and self.capabilities.graph_node_get_type
        ):
            raise CudaBackendError(
                "CUDA graph topology inspection symbols are unavailable"
            )
        raw_graph = _as_handle(graph)
        node_count = ctypes.c_size_t(0)
        self._call(
            "cudaGraphGetNodes",
            raw_graph,
            None,
            ctypes.byref(node_count),
        )
        nodes = (ctypes.c_void_p * int(node_count.value))()
        if int(node_count.value):
            self._call(
                "cudaGraphGetNodes",
                raw_graph,
                nodes,
                ctypes.byref(node_count),
            )
        node_handles = tuple(
            int(nodes[index] or 0)
            for index in range(int(node_count.value))
        )
        if any(handle <= 0 for handle in node_handles):
            raise CudaBackendError(
                "cudaGraphGetNodes returned a null node"
            )
        node_index = {
            handle: index for index, handle in enumerate(node_handles)
        }

        edge_count = ctypes.c_size_t(0)
        if self.capabilities.runtime_major >= 13:
            edge_function = "cudaGraphGetEdges"
            edge_data_abi = True
        elif self.capabilities.graph_get_edges_v2:
            edge_function = "cudaGraphGetEdges_v2"
            edge_data_abi = True
        else:
            edge_function = "cudaGraphGetEdges"
            edge_data_abi = False
        self._call(
            edge_function,
            raw_graph,
            None,
            None,
            *(
                (None, ctypes.byref(edge_count))
                if edge_data_abi
                else (ctypes.byref(edge_count),)
            ),
        )
        sources = (ctypes.c_void_p * int(edge_count.value))()
        destinations = (ctypes.c_void_p * int(edge_count.value))()
        raw_edge_data = (
            (CudaGraphEdgeData * int(edge_count.value))()
            if edge_data_abi
            else None
        )
        if int(edge_count.value):
            self._call(
                edge_function,
                raw_graph,
                sources,
                destinations,
                *(
                    (raw_edge_data, ctypes.byref(edge_count))
                    if edge_data_abi
                    else (ctypes.byref(edge_count),)
                ),
            )
        edge_records = []
        for index in range(int(edge_count.value)):
            source = int(sources[index] or 0)
            destination = int(destinations[index] or 0)
            if source not in node_index or destination not in node_index:
                raise CudaBackendError(
                    "cudaGraphGetEdges returned an endpoint outside node list"
                )
            edge_records.append(
                (
                    node_index[source],
                    node_index[destination],
                    (
                        bytes(raw_edge_data[index])
                        if raw_edge_data is not None
                        else None
                    ),
                )
            )
        edge_records.sort(key=lambda item: (item[0], item[1]))

        node_types = []
        for node in node_handles:
            node_type = ctypes.c_int(-1)
            self._call(
                "cudaGraphNodeGetType",
                ctypes.c_void_p(node),
                ctypes.byref(node_type),
            )
            node_types.append(int(node_type.value))
        return CudaGraphSnapshot(
            node_handles=node_handles,
            node_types=tuple(node_types),
            edges=tuple(
                (source, destination)
                for source, destination, _data in edge_records
            ),
            edge_data=tuple(
                data for _source, _destination, data in edge_records
            ),
        )

    def inspect_graph(
        self, graph: int | ctypes.c_void_p
    ) -> dict[str, object]:
        """Inspect raw graph topology without changing graph state.

        The digest is based on the actual node types and edge endpoints as
        enumerated by the CUDA runtime, never on a caller-provided node-count
        declaration.
        """

        snapshot = self.snapshot_graph(graph)
        node_types = list(snapshot.node_types)
        edges = [[source, destination] for source, destination in snapshot.edges]
        type_counts: dict[str, int] = {}
        for node_type in node_types:
            name = _GRAPH_NODE_TYPE_NAMES.get(node_type, f"type_{node_type}")
            type_counts[name] = type_counts.get(name, 0) + 1
        return {
            "node_count": len(snapshot.node_handles),
            "edge_count": len(edges),
            "node_type_counts": dict(sorted(type_counts.items())),
            "canonical_node_types": node_types,
            "canonical_edges": edges,
            "topology_digest": canonical_contract_digest(
                {"node_types": node_types, "edges": edges}
            ),
        }

    def kernel_function_names(
        self, graph: int | ctypes.c_void_p
    ) -> tuple[str, ...]:
        """Ordered kernel function names of one captured linear graph."""

        snapshot = self.snapshot_graph(graph)
        ordered = tuple(
            snapshot.node_handles[index]
            for index in snapshot.linear_order(kernel_only=True)
        )
        get_node_params, get_function_name = _driver_kernel_introspection()
        names: list[str] = []
        for position, raw_node in enumerate(ordered):
            params = CudaDriverKernelNodeParams()
            status = int(
                get_node_params(ctypes.c_void_p(raw_node), ctypes.byref(params))
            )
            if status != CUDA_SUCCESS:
                raise CudaBackendError(
                    "cuGraphKernelNodeGetParams failed for captured kernel "
                    f"node {position} with CUresult={status}"
                )
            name_pointer = ctypes.c_char_p()
            status = int(
                get_function_name(
                    ctypes.byref(name_pointer),
                    ctypes.c_void_p(int(params.func or 0)),
                )
            )
            if status != CUDA_SUCCESS or not name_pointer.value:
                raise CudaBackendError(
                    f"cuFuncGetName failed for captured kernel node {position} "
                    f"with CUresult={status}"
                )
            names.append(name_pointer.value.decode("utf-8", errors="strict"))
        return tuple(names)


def _driver_kernel_introspection() -> tuple[Any, Any]:
    """Bind the two CUDA Driver symbols that name a captured kernel node."""

    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        raise CudaBackendError(
            "CUDA Driver library is unavailable for captured-kernel inspection"
        ) from exc
    get_node_params = getattr(
        driver, "cuGraphKernelNodeGetParams_v2", None
    ) or getattr(driver, "cuGraphKernelNodeGetParams", None)
    get_function_name = getattr(driver, "cuFuncGetName", None)
    if get_node_params is None or get_function_name is None:
        raise CudaBackendError(
            "CUDA Driver captured-kernel introspection symbols are unavailable"
        )
    get_node_params.restype = ctypes.c_int
    get_node_params.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(CudaDriverKernelNodeParams),
    ]
    get_function_name.restype = ctypes.c_int
    get_function_name.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_void_p,
    ]
    return get_node_params, get_function_name


_GRAPH_NODE_TYPE_NAMES = {
    0: "kernel",
    1: "memcpy",
    2: "memset",
    3: "host",
    4: "graph",
    5: "event_record",
    6: "event_wait",
    7: "external_semaphore_signal",
    8: "external_semaphore_wait",
    9: "empty",
    10: "mem_alloc",
    11: "mem_free",
    12: "batch_mem_op",
    13: "conditional",
    14: "node_type_switch",
}


__all__ = [
    "CUDA_SUCCESS",
    "CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH",
    "CUDA_STREAM_CAPTURE_MODE_RELAXED",
    "CUDA_HOST_ALLOC_DEFAULT",
    "CudaBackendCapabilities",
    "CudaBackendError",
    "CudaGraphSnapshot",
    "CudaDim3",
    "CudaGraphEdgeData",
    "CudaKernelNodeParams",
    "CudaDriverKernelNodeParams",
    "CudaRuntimeGraphApi",
]
