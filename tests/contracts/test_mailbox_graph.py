"""Mailbox allocation and graph/PDL manifest contract."""
import os

import pytest

from conftest import load_driver

M67C_PDL_CLASSES = ("pdl_secondary", "attn_swa_rows", "attn_dsa_rows")
VERIFY8_KERNEL_NODES, VERIFY8_PDL_EDGES = 196, 95


class _Ops:
    def __init__(self, row_bytes):
        self._row_bytes = row_bytes

    def expert_down_mailbox_row_bytes(self):
        return self._row_bytes


def test_verify_mailbox_allocation_formula():
    verify = load_driver("run_naive_n0p5_verify.py")
    row = 0xE0000
    assert verify.VROWS == 8
    assert verify.mailbox_bytes(_Ops(row)) == verify.MAILBOX_BYTES + 8 * row       # Rows after the attention slabs
    assert verify.mailbox_bytes(_Ops(row), rows=16) == verify.MAILBOX_BYTES + 16 * row
    assert verify.MAILBOX_BYTES == 8 * 4 * 0x70000


def test_decode_mailbox_allocation_formula():
    decode = load_driver("run_naive_n0p5_decode.py")
    row = 0xE0000
    assert decode.MAILBOX_BYTES == 4 * 0x70000
    assert decode.mailbox_bytes(_Ops(row)) == decode.MAILBOX_BYTES + row


def test_pdl_edge_classes_are_the_m67c_set():
    decode = load_driver("run_naive_n0p5_decode.py")
    assert tuple(decode.PDL_W8A8_MARKERS) == M67C_PDL_CLASSES


@pytest.mark.skipif(os.environ.get("NAIVERT_CONTRACT_GPU") != "1", reason="GPU box only")
def test_verify8_graph_manifest():
    pytest.skip(f"run run_naive_n0p5_verify.py --pdl and assert kernel_nodes == {VERIFY8_KERNEL_NODES}, "
                f"programmatic_edges == {VERIFY8_PDL_EDGES} (wire the driver's pdl report here)")
