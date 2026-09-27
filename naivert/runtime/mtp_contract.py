"""Host-side mirror of the device decode-control block.

``_DecodeControlLayout`` is the byte layout of the per-rank ``decode_control``
buffer the dspark draft round seeds before launch and the session engine reads
back after the device loop stops.  ``DecodeMode`` tags the engine config and a
session so ``serve/context.py`` can pick the native-MTP context limit.

The validation layer that once lived here (``DecodeRuntimeSpec``,
``DecodeControlState``, ``DecodePointerManifest``, ``DecodeRuntimeABI``) served
the removed resident-sequencer engine and had no constructor left in the tree.
"""

from __future__ import annotations

import ctypes
from enum import IntEnum


class DecodeMode(IntEnum):
    """Runtime-instance mode; hot kernels receive a specialization instead."""

    SEQ1 = 0
    NATIVE_MTP = 1

    @property
    def protocol_name(self) -> str:
        return "seq1" if self is DecodeMode.SEQ1 else "native_mtp"


class _DecodeControlFields(ctypes.Structure):
    _fields_ = (
        ("decode_mode", ctypes.c_uint32),
        ("speculative_depth", ctypes.c_uint32),
        ("active_rows", ctypes.c_uint32),
        ("row_capacity", ctypes.c_uint32),
        ("row_stride", ctypes.c_uint64),
        ("iteration", ctypes.c_uint64),
        ("committed_pos", ctypes.c_uint32),
        ("speculative_base", ctypes.c_uint32),
        ("output_count", ctypes.c_uint32),
        ("max_output_tokens", ctypes.c_uint32),
        ("accepted_count", ctypes.c_uint32),
        ("main_cache_committed", ctypes.c_uint32),
        ("main_cache_speculative_end", ctypes.c_uint32),
        ("mtp_cache_committed", ctypes.c_uint32),
        ("mtp_cache_speculative_end", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
    )


class _DecodeControlLayout(ctypes.Union):
    _fields_ = (
        ("value", _DecodeControlFields),
        ("_force_alignment", ctypes.c_longdouble),
    )


__all__ = ["DecodeMode"]
