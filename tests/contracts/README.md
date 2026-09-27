# Runtime contract checks

These host checks detect changes to shared model geometry, mailbox allocation, and residual-export
constants. Run them from the repository root:

```bash
python3 -m pytest tests/contracts -q
```

| File | Active checks | GPU placeholder |
|---|---|---|
| `test_cache_layout.py` | Model specification, cache dimensions, and kernel geometry against `golden/naive_n0p5_spec.json`. | Decode comparison after chunked versus replay prefill. |
| `test_mailbox_graph.py` | Mailbox allocation formulas and PDL edge classes. | Captured graph node and edge counts. |
| `test_residual_export.py` | Verify-row count and draft residual-export layer constants. | Device residual-export validation. |

All three GPU tests are unimplemented placeholders. They skip even when
`NAIVERT_CONTRACT_GPU=1`; that variable does not enable a working GPU validation gate.
Use the harnesses listed in the [test guide](../README.md) for GPU checks. Native and replay
prefill use different computation paths, so a future comparison must define its numerical
acceptance criteria rather than assume token-exact equality.

When intentionally changing a golden contract, regenerate its data and explain the change in
the commit message.
