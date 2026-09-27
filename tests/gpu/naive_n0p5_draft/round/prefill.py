"""Prompt ingestion for the draft harnesses: eight prompt tokens per verify-8 step, seq1 tail.

``verify8`` runs the target's eight-row verify graph over consecutive prompt tokens (the
same kernels and cache writes a draft block gets), then finishes the last 1..8 tokens with
seq1 steps so ``rt.logits`` / ``rt.resid`` hold the final position exactly as before.
``native`` uses fixed 4,096-row NativePrefill chunks, with replay bridging an
unaligned resumed prefix to the next chunk boundary, then a final seq1 token.
``seq1`` is the original one-token-per-step replay. The rows-8 kernels are not bitwise the
rows-1 kernels, so the two modes give ulp-different caches; every path of a run shares one.
``kernel`` runs the batched kernel prefill (``naivert.prefill.draft_features``) over
``tokens[start:n-1]`` -- it writes the same cache tensors and publishes the draft features
itself -- then the last token through one seq1 step, so ``rt.logits`` / ``rt.resid`` again hold
the final position from the decode kernels.  The kernel path is not bitwise either of the
decode paths (SGLang-lineage projection, rope and fp8 index rounding; see the evidence).
"""
import time

import torch


def prefill_tokens(ranks, verifies, tokens, features, layer_ids, *, start=0, mode='verify8', progress=None,
                   executor=None, timing=None):
    """Ingest ``tokens[start:]`` at positions ``start..``; ``features[r][p]`` gets rank r's feature row.

    ``timing`` (a dict) receives the phase wall times in ms: executor chunks, position fill, seq1 tail
    (synchronised); it adds one stream synchronisation per rank at the end of the tail."""
    if mode not in ('verify8', 'seq1', 'kernel', 'native'):
        raise ValueError(f'unknown prefill mode {mode!r}')
    n = len(tokens)
    if start >= n:
        return
    first_seq1 = start
    if mode in ('kernel', 'native'):
        if executor is None:
            raise ValueError(f'mode {mode} needs its executor (DraftFeaturePrefillExecutor / NativePrefill)')
        chunk = getattr(executor, 'chunk_tokens', None)        # the native executor: 4096-row chunks
        body_end = n - 1                                      # leave the last token for the seq1 tail
        pos = start
        # Reuse the parked prefix. Decode kernels fill only the leading partial
        # chunk, then the existing native executor resumes at its fixed grid.
        if mode == 'native' and chunk is not None and pos % chunk:
            end = min(body_end, (pos // chunk + 1) * chunk)
            began = time.perf_counter()
            prefill_tokens(ranks, verifies, tokens[:end], features, layer_ids,
                           start=pos, mode='verify8', progress=progress)
            if timing is not None:
                for rt in ranks:
                    rt.stream.synchronize()
                timing.update(bridge_ms=round((time.perf_counter() - began) * 1e3, 2),
                              bridge_tokens=end - pos)
            pos = end
        if pos and pos < body_end:
            executor.resume_from_decode(pos)
        executor_ms = 0.0
        chunks_run = 0
        while pos < body_end:
            end = body_end if chunk is None else min(body_end, pos + chunk)
            began = time.perf_counter()
            executor.run(tokens[pos:end], start=pos)                # returns with its streams synchronised
            executor_ms += (time.perf_counter() - began) * 1e3
            chunks_run += 1
            pos = end
            if progress:
                progress(pos)
        began = time.perf_counter()
        for r, rt in enumerate(ranks):
            with torch.cuda.device(r), torch.cuda.stream(rt.stream):
                rt.pos.fill_(body_end)
        if timing is not None:
            timing.update(executor_ms=round(executor_ms, 2), chunks=chunks_run,
                          pos_fill_ms=round((time.perf_counter() - began) * 1e3, 2))
        first_seq1 = max(start, n - 1)
    elif mode == 'verify8':
        chunks = (n - start - 1) // 8          # leave 1..8 tokens for the seq1 tail
        if chunks:
            for v in verifies:
                if v.graph is None:
                    v.capture()
        for c in range(chunks):
            p = start + 8 * c
            block = tokens[p:p + 8]
            for r, v in enumerate(verifies):
                with torch.cuda.device(r), torch.cuda.stream(v.rt.stream):
                    v.tokens.copy_(torch.as_tensor(block, dtype=torch.long, device=v.tokens.device))
            for v in verifies:
                v.replay()
            for r, v in enumerate(verifies):
                with torch.cuda.device(r), torch.cuda.stream(v.rt.stream):
                    features[r][p:p + 8].copy_(v.resid[layer_ids[r] + 1][:8])
            if progress:
                progress(p + 8)
        first_seq1 = start + 8 * chunks
    # The eight rank steps launch serially from this thread: launching them from eight threads was
    # measured slower (4.2 vs 3.1 ms for the one-token tail).
    tail_began = time.perf_counter()
    for p in range(first_seq1, n):
        token = tokens[p]
        for r, rt in enumerate(ranks):
            with torch.cuda.device(r), torch.cuda.stream(rt.stream):
                rt.token.fill_(token)
        for rt in ranks:
            rt.run_step()
        for r, rt in enumerate(ranks):
            with torch.cuda.device(r), torch.cuda.stream(rt.stream):
                features[r][p].copy_(rt.resid[layer_ids[r] + 1][0])
        if progress:
            progress(p + 1)
    if timing is not None:
        for rt in ranks:
            rt.stream.synchronize()
        timing.update(tail_ms=round((time.perf_counter() - tail_began) * 1e3, 2), tail_tokens=n - first_seq1)
