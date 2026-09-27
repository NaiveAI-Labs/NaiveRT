"""Byte layout of one device-written output record in the host output ring.

The dspark draft round (``backends/naive_n0p5_draft_round.py``) allocates the
pinned ``host_ring`` with this layout and the session engine parses tokens
out of it.  The remaining sequencer machinery that once lived here belonged to
the removed resident-sequencer path.
"""

from __future__ import annotations

import ctypes


class _SequencerOutputRecordLayout(ctypes.Structure):
    _fields_ = (
        ("token", ctypes.c_uint32),
        ("generation", ctypes.c_uint32),
        ("parent_start_ns", ctypes.c_uint64),
        ("parent_complete_ns", ctypes.c_uint64),
        ("launch_marker_ns", ctypes.c_uint64),
        ("cur_pos", ctypes.c_uint32),
        ("sampling_position", ctypes.c_uint32),
        ("rope_position", ctypes.c_uint32),
        ("output_count", ctypes.c_uint32),
    )


__all__: list[str] = []
