"""Fail-closed CUDA Graph programmatic-edge installation."""

from __future__ import annotations

import ctypes
from typing import Any, Iterable

from .contract import RuntimeContractError
from .cuda_backend import CudaGraphEdgeData


CUDA_GRAPH_NODE_TYPE_KERNEL = 0
CUDA_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC = 1
CUDA_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC = 1
CUDA_GRAPH_KERNEL_NODE_PORT_LAUNCH_COMPLETION = 2


if ctypes.sizeof(CudaGraphEdgeData) != 8:
    raise RuntimeContractError("cudaGraphEdgeData ABI is not eight bytes")


def _check(code: int, operation: str) -> None:
    if int(code) != 0:
        raise RuntimeContractError(
            f"{operation} failed with cudaError={int(code)}"
        )


def _configure(library: ctypes.CDLL, runtime_major: int) -> tuple[Any, Any]:
    edge_suffix = "" if runtime_major >= 13 else "_v2"
    remove_edges = getattr(
        library, f"cudaGraphRemoveDependencies{edge_suffix}"
    )
    add_edges = getattr(
        library, f"cudaGraphAddDependencies{edge_suffix}"
    )
    for function in (remove_edges, add_edges):
        function.restype = ctypes.c_int
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(CudaGraphEdgeData),
            ctypes.c_size_t,
        ]
    return remove_edges, add_edges


def _canonical_linear_graph(
    api: Any, graph: int
) -> tuple[list[int], list[tuple[int, int, CudaGraphEdgeData]]]:
    snapshot = api.snapshot_graph(graph)
    if any(
        node_type != CUDA_GRAPH_NODE_TYPE_KERNEL
        for node_type in snapshot.node_types
    ):
        raise RuntimeContractError("PDL edges require a kernel-only graph")
    if any(data is None for data in snapshot.edge_data):
        raise RuntimeContractError(
            "PDL edges require CUDA graph edge-data inspection"
        )
    try:
        order = snapshot.linear_order(kernel_only=True)
    except RuntimeContractError as exc:
        raise RuntimeContractError(
            "PDL edges require one linear graph"
        ) from exc
    ordered = [snapshot.node_handles[index] for index in order]
    edge_by_pair = {
        edge: data
        for edge, data in zip(snapshot.edges, snapshot.edge_data)
    }
    ordered_edges = [
        (
            ordered[index],
            ordered[index + 1],
            CudaGraphEdgeData.from_buffer_copy(
                edge_by_pair[(order[index], order[index + 1])]
            )
        )
        for index in range(len(ordered) - 1)
    ]
    return ordered, ordered_edges


def install_programmatic_edges(
    api: Any,
    graph: int,
    *,
    from_positions: Iterable[int],
    from_port: int = CUDA_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC,
) -> dict[str, object]:
    """Replace exact default linear edges before graph instantiation."""

    positions = tuple(sorted({int(value) for value in from_positions}))
    nodes, edges = _canonical_linear_graph(api, graph)
    if any(position < 0 or position >= len(edges) for position in positions):
        raise RuntimeContractError("PDL edge position is outside the graph")
    default = CudaGraphEdgeData()
    programmatic = CudaGraphEdgeData()
    if from_port not in (
        CUDA_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC,
        CUDA_GRAPH_KERNEL_NODE_PORT_LAUNCH_COMPLETION,
    ):
        raise RuntimeContractError("unsupported PDL outgoing port")
    programmatic.from_port = int(from_port)
    programmatic.type = CUDA_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC
    library = api.library
    remove_edges, add_edges = _configure(
        library, int(api.capabilities.runtime_major)
    )
    raw_graph = ctypes.c_void_p(int(graph))
    for position in positions:
        source, destination, observed = edges[position]
        if bytes(observed) != bytes(default):
            raise RuntimeContractError(
                f"PDL edge {position} was not an ordinary dependency"
            )
        sources = (ctypes.c_void_p * 1)(source)
        destinations = (ctypes.c_void_p * 1)(destination)
        old_data = (CudaGraphEdgeData * 1)(default)
        new_data = (CudaGraphEdgeData * 1)(programmatic)
        _check(
            remove_edges(
                raw_graph, sources, destinations, old_data, 1
            ),
            "cudaGraphRemoveDependencies",
        )
        try:
            _check(
                add_edges(
                    raw_graph, sources, destinations, new_data, 1
                ),
                "cudaGraphAddDependencies",
            )
        except Exception:
            _check(
                add_edges(
                    raw_graph, sources, destinations, old_data, 1
                ),
                "cudaGraphAddDependencies(rollback)",
            )
            raise

    after_nodes, after_edges = _canonical_linear_graph(api, graph)
    if after_nodes != nodes:
        raise RuntimeContractError("PDL mutation changed graph nodes")
    observed_programmatic = [
        index
        for index, (_source, _destination, data) in enumerate(after_edges)
        if bytes(data) == bytes(programmatic)
    ]
    if observed_programmatic != list(positions):
        raise RuntimeContractError(
            "PDL graph edge-data audit did not match requested positions"
        )
    return {
        "schema": "naivert_programmatic_graph_edges_v1",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "from_positions": list(positions),
        "from_port": int(programmatic.from_port),
        "dependency_type": int(programmatic.type),
    }


__all__ = ["install_programmatic_edges"]
