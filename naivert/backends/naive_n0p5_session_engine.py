"""SessionEngine: the persistent draft round behind ``naivert.serve``, with parked per-session prefixes.

Speaks the engine protocol ``ServerContext`` drives (``open_session`` -> session with ``step``,
``finish``, ``close``; ``config``; ``request_generation``; ``close``). One request at a time. The
device loop runs the whole request; ``step`` returns tokens as the tail publishes them to the
pinned host ring. ``finish`` parks the finished sequence's KV under the request's session id in a
``PrefixStore``; a later request resumes from a matching parked end or saved prompt checkpoint
and prefills only the remaining tokens (``prepared_prefill.start_position`` reports it).

Prompt ingestion (``prefill_path`` in the timings): both ``build`` and the serving CLI default
to native prefill. Cold prompts run through ``naivert.prefill.native.executor.NativePrefill``
from position 0 (4096-row chunks, with indexer cost growing with the visible prefix), then the last token
through one seq1 step. A resumed prefix chooses verify8 append, native append after a partial-chunk
bridge, or full recomputation by estimated cost (``resume_policy``: cost, or replay / native to force one).
"""
from __future__ import annotations

import ctypes as C
import importlib.util
import json
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from naivert.backends.naive_n0p5_draft_ingest import DraftContextIngest
from naivert.backends.naive_n0p5_prefix_store import PrefixStore
from naivert.runtime.contract import RuntimeContractError
from naivert.runtime.mtp_contract import DecodeMode, _DecodeControlLayout as Control
from naivert.runtime.sequencer import _SequencerOutputRecordLayout as Record

log = logging.getLogger(__name__)
EOS = 151645
VOCAB = 152576
CONTEXT_EXCEEDED_MARK = "past the pinned Seq1 MLA kernel's validated range"
RECORD_BYTES = C.sizeof(Record)
GENERATION_OFFSET = Record.generation.offset
TOKEN_OFFSET = Record.token.offset


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_resume(entries, tokens, session_id=None, *, checkpoints=None):
    """Pick the longest exact continuation of a parked end or saved checkpoint.

    ``entries`` maps uuid -> (position, tokens). A resume is exact when the new prompt repeats the
    parked sequence through its parked position and has at least one token past it (the last prompt
    token is always recomputed). The sliding-window ring holds only the 128 positions below the
    parked position; earlier positions require their own saved ring checkpoint. The longest exact match
    wins; with a session id that session's entry wins ties, but every entry is a candidate: an exact
    continuation is unambiguous whatever key parked it, and Claude Code's subagents carry the
    parent's key while their requests interleave with its own.
    """
    best = None
    candidates = list(entries)
    if session_id is not None and session_id in entries:
        candidates = [session_id] + [u for u in candidates if u != session_id]
    for uuid in candidates:
        position, parked = entries[uuid]
        if position <= len(tokens) - 1 and position <= len(parked) and parked[:position] == tokens[:position]:
            matched = position
        else:
            points = (checkpoints or {}).get(uuid, ())
            if not points:
                continue
            end = min(position, len(parked), len(tokens) - 1)
            common = next((i for i in range(end) if parked[i] != tokens[i]), end)
            matched = max((p for p in points if 0 < p <= common), default=0)
        if matched and (best is None or matched > best[1]):
            best = (uuid, matched)
    return best


def over_budget(total_bytes, ranks, budget_per_rank):
    """True when the parked prefixes (``total_bytes`` summed over ``ranks`` ranks) exceed the per-rank budget."""
    return int(total_bytes) > int(budget_per_rank) * int(ranks)


def best_prefix_match(entries, tokens):
    """-> (uuid, common prefix length) of the parked entry sharing the longest prefix with ``tokens``, or None.

    Only a diagnostic: when nothing resumes, the entry that came closest says where the client's
    re-rendered history first departs from what the model produced."""
    best = None
    for uuid, (position, parked) in entries.items():
        n = min(position, len(parked), len(tokens))
        common = next((i for i in range(n) if parked[i] != tokens[i]), n)
        if best is None or common > best[1]:
            best = (uuid, common)
    return best


def park_name(entries, session_id, tokens, generation):
    """-> (uuid to park ``tokens`` under, entries to evict first) for a keyed session.

    Under ``session_id`` when that key is free or holds a sequence this one extends (replacing it,
    as before); otherwise under ``session_id#<generation>`` with the key's entry kept. Entries of
    the same key family that this sequence extends are superseded and evicted. A parent
    conversation therefore keeps its parked prefix while subagent requests with the same key run.
    """
    def extended(uuid):
        position, parked = entries[uuid]
        return position <= len(tokens) and parked[:position] == tokens[:position]
    family = [u for u in entries if u == session_id or u.startswith(session_id + "#")]
    superseded = [u for u in family if extended(u)]
    if session_id not in entries or session_id in superseded:
        return session_id, superseded
    return f"{session_id}#{generation}", superseded


THINK_END = 151668


# Cost model for the resume decision, in ms, measured on eight Blackwell GPUs with the naive-n0.5 checkpoint pair:
REPLAY_MS_PER_TOKEN = 0.44        # verify-8 replay: 753 ms for a 1,712-token cold prompt, 92 ms for 209
NATIVE_FIRST_CHUNK_MS = 62.0      # executor, first 4096-row chunk including the seq1 tail
NATIVE_EXTRA_CHUNK_MS = 60.0      # each further 4096-row chunk (122.5 ms for a 4,560-token prompt)
DRAFT_INGEST_MS_PER_TOKEN = 0.065  # rd.initialize's eight-row loop (initialize_ms): 0.065 ms per prompt token
BATCHED_INGEST_MS_PER_TOKEN = 0.002  # DraftContextIngest (fused matmuls + one all-reduce per step): 3.3 ms per 2048 rows
CHUNK = 4096
NATIVE_SHORT_CHUNKS = 16
# Warm query-sharded executor, eight B300s: 1.561 / 2.501 / 26.760 s at
# 17 / 32 / 256 chunks. Rounded upward for the resume decision; not a cold
# or capture estimate.
NATIVE_LONG_BASE_CHUNK_MS = 95.0
NATIVE_LONG_SCAN_CHUNK_MS = 0.25


def prefill_estimate(start, n_tokens, ingest_ms_per_token=DRAFT_INGEST_MS_PER_TOKEN):
    """Estimated warm ms for replay append, native recomputation, and native append.

    Native append decodes the leading partial chunk, then runs aligned chunks;
    target and draft ingest costs cover only the suffix. First-use/capture and
    prefix-copy costs are excluded; both append paths restore the same prefix.
    """
    suffix = max(0, n_tokens - start)
    replay = suffix * (REPLAY_MS_PER_TOKEN + ingest_ms_per_token)
    chunks = max(1, -(-max(1, n_tokens - 1) // CHUNK))
    target = NATIVE_FIRST_CHUNK_MS + NATIVE_EXTRA_CHUNK_MS * (chunks - 1)
    if chunks > NATIVE_SHORT_CHUNKS:
        target = (NATIVE_LONG_BASE_CHUNK_MS * chunks
                  + NATIVE_LONG_SCAN_CHUNK_MS * chunks * (chunks + 1) / 2)
    native = target + n_tokens * ingest_ms_per_token
    bridge = min(max(0, n_tokens - 1 - start), (-start) % CHUNK)
    first = (start + bridge) // CHUNK
    suffix_chunks = max(0, chunks - first) if start and start + bridge < n_tokens - 1 else 0
    append_target = sum(NATIVE_EXTRA_CHUNK_MS if c < NATIVE_SHORT_CHUNKS else
                        NATIVE_LONG_BASE_CHUNK_MS + NATIVE_LONG_SCAN_CHUNK_MS * (c + 1)
                        for c in range(first, first + suffix_chunks))
    append = bridge * REPLAY_MS_PER_TOKEN + append_target + suffix * ingest_ms_per_token
    return dict(replay_ms=round(replay, 1), native_ms=round(native, 1), suffix_tokens=suffix, chunks=chunks,
                native_suffix_ms=round(append, 1), native_bridge_tokens=bridge,
                native_suffix_first_chunk=first, native_suffix_chunks=suffix_chunks)


def native_resume_ready(native, chunks, *, first_chunk=0):
    """Use warm resume estimates only when every required chunk graph exists.

    A fresh prompt still runs native, and an explicitly forced native resume may
    pay eager/capture costs. Short continuations need the same readiness check.
    """
    if native is None:
        return False
    return (native.graphs and native.chunk_graphs
            and all(c in native._chunk_captured for c in range(first_chunk, chunks)))


NATIVE_MAX_PROMPT = 1048576       # tiled selector and full-model handoff validated through the final 1M-key chunk


def prefill_path(has_native, start, n_tokens, policy="cost", ingest_ms_per_token=DRAFT_INGEST_MS_PER_TOKEN,
                 *, native_ready=True, native_suffix_ready=False):
    """Choose native recomputation, verify8 append, or native append after a short verify8 bridge.

    Cold prompts (``start == 0``) always take the executor. For a resumed prefix the ``cost`` policy
    compares ``prefill_estimate``: the replay's cost grows with the new suffix, the executor's with the
    whole prompt (its chunks start at multiples of 4096, so it recomputes rather than extends), so the
    break-even suffix grows with the prompt length. Native append only needs the new chunks.
    Native resume paths require their own chunk graphs to be ready; warm estimates exclude capture.
    ``replay`` / ``native`` force one path (A/B runs).

    Prompts longer than ``NATIVE_MAX_PROMPT`` always replay. The tiled selector and
    query-sharded indexer support up to 1M visible keys; full-model correctness,
    capture/replay and graph memory are validated through the final 1M-key chunk."""
    if not has_native:
        return "verify8"
    if n_tokens > NATIVE_MAX_PROMPT:
        return "verify8"
    if start == 0:
        return "native"
    if policy == "replay":
        return "verify8"
    if policy == "native":
        return "native"
    if policy != "cost":
        raise ValueError(f"resume policy must be cost, replay or native, got {policy!r}")
    est = prefill_estimate(start, n_tokens, ingest_ms_per_token)
    costs = {'verify8': est['replay_ms']}
    if native_ready:
        costs['native'] = est['native_ms']
    if native_suffix_ready and est['native_suffix_chunks']:
        costs['native_suffix'] = est['native_suffix_ms']
    return min(costs, key=costs.get)


def segment_stats(tokens, records):
    """Decode statistics for the whole generation and for its think / answer segments.

    ``tokens[j]`` was published by ``records[j]``; one round publishes several tokens with one
    (parent_start_ns, parent_complete_ns) interval, so rounds are the distinct intervals. The
    think segment is everything through the first ``</think>``; the round that straddles the
    boundary is counted with the segment its last token belongs to.
    """
    def stats(lo, hi):
        rounds = {}
        for j in range(lo, hi):
            key = (int(records[j].parent_start_ns), int(records[j].parent_complete_ns))
            if key[0]:
                rounds[key] = rounds.get(key, 0) + 1
        ns = sum(b - a for a, b in rounds)
        n = hi - lo
        return dict(tokens=n, rounds=len(rounds), tokens_per_round=(n / len(rounds)) if rounds else 0.0,
                    device_ms=ns / 1e6, device_tok_s=(n / (ns / 1e9)) if ns else 0.0)
    out = {}
    if THINK_END in tokens:
        cut = tokens.index(THINK_END) + 1
        out["think"], out["answer"] = stats(0, cut), stats(cut, len(tokens))
    else:
        out["answer"] = stats(0, len(tokens))
    out["total"] = stats(0, len(tokens))
    return out


class _Prefill:
    def __init__(self, **values):
        self.values = values

    def as_dict(self):
        return dict(self.values)


class _Runtime:
    def __init__(self, prefill):
        self.prepared_prefill = _Prefill(**prefill)
        self.output_reader = SimpleNamespace(progress_records=[])
        self.graph_contract_evidence = {"device_token_chain": True}
        self.abandoned_chain_positions = 0
        self.canceled_chain = False
        self.device_steps = 0
        self.reconcile_error = None
        self.device_stop_bound = None


class Session:
    """One request over the persistent round; tokens are read from the host ring as they land."""
    mode = DecodeMode.NATIVE_MTP

    def __init__(self, engine, *, tokens, start, anchor, reserved, session_id, prefill):
        self.engine, self.tokens, self.start, self.anchor = engine, list(tokens), int(start), int(anchor)
        self.max_decode_tokens, self.session_id = int(reserved), session_id
        self.runtime = _Runtime(prefill)
        self.generated = []
        self.finished = self.closed = False
        self.started = time.monotonic()
        ring = engine.rd.host_ring.numpy()
        self._ring = ring
        self._flags = engine.rd.host_flags.numpy()

    def _record(self, slot):
        return Record.from_buffer_copy(self._ring[slot * RECORD_BYTES:(slot + 1) * RECORD_BYTES].tobytes())

    def _generation(self, slot):
        return int(np.frombuffer(self._ring, dtype=np.uint32, count=1,
                                 offset=slot * RECORD_BYTES + GENERATION_OFFSET)[0])

    def stopped(self):
        return bool((self._flags & 1).any())

    def step(self):
        if self.closed:
            raise RuntimeContractError("decode session is closed")
        if len(self.generated) >= self.max_decode_tokens:
            raise RuntimeContractError("decode session exhausted max_decode_tokens")
        if not self.generated:
            self.generated.append(self.anchor)
            self.runtime.output_reader.progress_records.append(
                {"parent_start_ns": 0, "parent_complete_ns": 0, "device_ns": 0})
            return self.anchor
        slot = len(self.generated) - 1
        deadline = time.monotonic() + self.engine.step_timeout
        while self._generation(slot) != slot + 1:
            if self.stopped() and self._generation(slot) != slot + 1:
                # The device stopped without publishing this slot: only possible when the loop
                # ended (cancel/bound) before reaching it.
                time.sleep(0.001)
                if self._generation(slot) != slot + 1:
                    raise RuntimeContractError(f"device loop stopped before output {slot + 1}")
            if time.monotonic() > deadline:
                raise RuntimeContractError(f"no output {slot + 1} within {self.engine.step_timeout} s")
            time.sleep(0.00005)
        rec = self._record(slot)
        if rec.output_count != slot + 1 or rec.cur_pos != len(self.tokens) + slot + 1:
            raise RuntimeContractError(f"host ring record {slot} out of order: {rec.output_count} {rec.cur_pos}")
        self.generated.append(int(rec.token))
        # The first block's parent_start_ns is the zero previous-complete stamp; report it as an
        # untimed block (both stamps zero) so the context counts neither its tokens nor a bogus
        # interval measured from the device timer's origin.
        start_ns, complete_ns = int(rec.parent_start_ns), int(rec.parent_complete_ns)
        if start_ns == 0:
            complete_ns = 0
        self.runtime.output_reader.progress_records.append(
            {"parent_start_ns": start_ns, "parent_complete_ns": complete_ns,
             "device_ns": complete_ns - start_ns})
        return int(rec.token)

    def finish(self):
        if self.finished:
            return
        self.finished = True
        self.engine._finish(self)

    def close(self):
        if self.closed:
            return
        try:
            self.finish()
        finally:
            self.closed = True
            if self.engine._active is self:
                self.engine._active = None


class SessionEngine:
    def __init__(self, parts, *, max_decode_tokens, seed=1234, store_bytes=32 << 30, step_timeout=120.0, tokenizer=None,
                 default_sampling=(1.0, 0.0, 0.0), default_temperature=None, default_top_p=0.95,
                 resume_policy="cost", draft_ingest="batched"):
        p = parts
        self.tokenizer = tokenizer
        # Defaults for requests that set nothing: repeat penalty (theta, alpha, beta), temperature
        # (None: 1/NAIVERT_DSPARK_INV_TEMPERATURE or 1) and top-p.
        self.default_sampling = tuple(float(x) for x in default_sampling)
        self.default_temperature = None if default_temperature is None else float(default_temperature)
        self.default_top_p = float(default_top_p)
        self.ranks, self.verifies, self.buffers, self.weights = p["ranks"], p["verifies"], p["buffers"], p["weights"]
        self.core, self.target, self.rd, self.anchor_sampler = p["core"], p["target"], p["rd"], p["anchor"]
        self.features, self.capacity, self.layer_ids = p["features"], int(p["capacity"]), p["layer_ids"]
        self.prefill_tokens, self.protocol = p["prefill_tokens"], p.get("protocol")
        self.native = p.get("native_prefill")            # NativePrefill, or None: verify-8 replay for every prompt
        self.resume_policy = str(resume_policy)          # cost (default) / replay / native, see prefill_path
        if self.resume_policy not in ("cost", "replay", "native"):
            raise ValueError(f"resume policy must be cost, replay or native, got {resume_policy!r}")
        if draft_ingest not in ("batched", "loop"):
            raise ValueError(f"draft ingest must be batched or loop, got {draft_ingest!r}")
        # batched: DraftContextIngest writes the prompt's draft K/V rows in 4096-row steps before
        # initialize(); loop: initialize()'s eight-row fc_ctx_kv blocks do it (the pre-2026-09-20 path).
        self.ingest = DraftContextIngest(self.core, [rt.stream for rt in self.ranks]) if draft_ingest == "batched" else None
        self.ingest_ms_per_token = BATCHED_INGEST_MS_PER_TOKEN if self.ingest is not None else DRAFT_INGEST_MS_PER_TOKEN
        self.store = PrefixStore(self.weights, self.buffers)
        self.store_bytes, self.step_timeout = int(store_bytes), float(step_timeout)
        budget = min(int(max_decode_tokens), self.rd.capacity)
        self.config = SimpleNamespace(max_decode_tokens=budget, sampling_seed=int(seed), mode=DecodeMode.NATIVE_MTP,
                                      model_spec=SimpleNamespace(stop_token_ids=(EOS,), vocab_size=VOCAB))
        # initialize() needs position <= capacity - 8 and the tail stops at capacity - 7.
        self._backend = SimpleNamespace(native_mtp_max_cur_pos=self.capacity - 16, min_prompt_tokens=1)
        self._request_generation = 0
        self._active = None
        self._closed = False
        self.binaries = p.get("binaries", {})
        self.weights_info = p.get("weights_info", {})

    # ------------------------------------------------------------------ protocol
    @property
    def request_generation(self):
        return self._request_generation

    def _sync(self):
        for rt in self.ranks:
            rt.stream.synchronize()

    def _entries(self):
        return {uuid: (e["position"], e["tokens"]) for uuid, e in self.store.entries.items()}

    def _evict_to_budget(self):
        # store_bytes is the budget per rank (--store-gb); store.bytes() sums the eight ranks.
        # Comparing the sum against the per-rank budget evicted a 916K-token session (8.3 GB per rank)
        # the moment it was parked, and every later turn of that Claude Code session prefilled cold.
        while self.store.entries and over_budget(self.store.bytes(), len(self.ranks), self.store_bytes):
            oldest = min(self.store.entries, key=lambda u: self.store.entries[u].get("restored_at",
                                                                                     self.store.entries[u]["saved_at"]))
            log.info("prefix store evicting %s (%d MB per rank)", oldest, self.store.bytes(oldest) // len(self.ranks) >> 20)
            self.store.evict(oldest)

    def _entry_text(self, uuid, position=None):
        e = self.store.entries[uuid]
        position = e['position'] if position is None else position
        holder = e if position == e['position'] else e['checkpoints'][position]
        if "text" not in holder:
            holder["text"] = self.tokenizer.decode(e["tokens"][:position], skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False)
        return holder["text"]

    def align_prompt(self, tokens, *, text, session_id=None):
        """Incremental tokenization: keep the model's own ids for a parked history.

        ``tokens`` is the canonical encoding of the rendered prompt ``text``. When a parked sequence's
        decoded text is a prefix of ``text`` but its ids differ (the sampler produced a token split the
        encoder would not), the returned ids are the parked ids through the parked position followed by
        the encoding of the remaining text alone. That suffix starts at the special token closing the
        assistant turn, so encoding it separately is exact, and the KV continues from the tokens the
        model actually wrote. Returns ``tokens`` unchanged when the ids already match or nothing matches.
        """
        tokens = [int(t) for t in tokens]
        if self.tokenizer is None or not self.store.entries:
            return tokens
        if select_resume(self._entries(), tokens, session_id) is not None:
            return tokens
        # The key's own entry can be an older branch of the conversation.
        # Search the same entries as token-level resume, preferring the key on
        # ties; sampled token splits must not strand a valid key#N predecessor.
        candidates = list(self.store.entries)
        if session_id in self.store:
            candidates.remove(session_id)
            candidates.insert(0, session_id)
        points = [(u, p) for u in candidates
                  for p in {self.store.entries[u]['position'], *self.store.entries[u].get('checkpoints', {})}]
        # Search longest first across keys. A retry can end before the parked
        # answer, yet still share a checkpoint's sampled tokenization exactly.
        points.sort(key=lambda pair: pair[1], reverse=True)
        best = None
        for uuid, position in points:
            parked_text = self._entry_text(uuid, position)
            if not text.startswith(parked_text) or len(parked_text) == len(text):
                continue
            best = (uuid, position, parked_text)
            break
        if best is None:
            return tokens
        uuid, position, parked_text = best
        suffix_ids = [int(t) for t in self.tokenizer.encode(text[len(parked_text):], add_special_tokens=False)]
        aligned = list(self.store.entries[uuid]["tokens"][:position]) + suffix_ids
        if any(not 0 <= t < VOCAB for t in aligned) or len(suffix_ids) == 0:
            return tokens
        log.info("session %s: incremental tokenization, parked %d ids kept, %d suffix ids encoded "
                 "(canonical prompt was %d ids)", uuid, position, len(suffix_ids), len(tokens))
        return aligned

    def open_session(self, prompt_tokens, *, n_past=0, max_decode_tokens=None, session_id=None, sampling=None):
        if self._closed:
            raise RuntimeContractError("Engine is closed")
        tokens = [int(t) for t in prompt_tokens]
        if not tokens or any(not 0 <= t < VOCAB for t in tokens):
            raise RuntimeContractError("prompt is empty or outside vocabulary")
        reserved = int(self.config.max_decode_tokens)
        if max_decode_tokens is not None:
            if int(max_decode_tokens) <= 0:
                raise RuntimeContractError(f"max_decode_tokens {max_decode_tokens} must be positive")
            reserved = min(int(max_decode_tokens), reserved)
        limit = self._backend.native_mtp_max_cur_pos
        last = len(tokens) + reserved - 1
        if last > limit:
            raise RuntimeContractError(
                f"native_mtp request would decode at cur_pos {last}, {CONTEXT_EXCEEDED_MARK} (cur_pos <= {limit}); "
                f"prompt tokens plus max_decode_tokens must not exceed {limit + 2} (this request reserved {reserved})")
        if self._active is not None:
            self._active.close()
        t0 = time.monotonic_ns()
        chosen = select_resume(self._entries(), tokens, session_id,
                               checkpoints={u: e.get('checkpoints', {}) for u, e in self.store.entries.items()})
        if chosen is None and self.store.entries:
            # Nothing resumes: log where the client's re-rendered history first departs from the
            # parked entry that came closest (any key -- a keyed session's own entry may be an
            # unrelated sequence while a sibling under key#n holds the real predecessor).
            uuid, first = best_prefix_match(self._entries(), tokens)
            position, parked = self._entries()[uuid]
            log.info("session %s: no parked prefix resumes; closest is %s parked at %d, prompt %d tokens, first difference "
                     "at %d, parked %s, prompt %s", session_id, uuid, position, len(tokens), first,
                     parked[max(0, first - 4):first + 4], tokens[max(0, first - 4):first + 4])
        start = chosen[1] if chosen is not None else 0
        estimate = prefill_estimate(start, len(tokens), self.ingest_ms_per_token) if self.native is not None else None
        ready = native_resume_ready(self.native, estimate['chunks']) if estimate is not None else False
        suffix_ready = (native_resume_ready(self.native, estimate['chunks'],
                                            first_chunk=estimate['native_suffix_first_chunk'])
                        if estimate is not None else False)
        if estimate is not None:
            estimate['native_graphs_ready'] = ready
            estimate['native_suffix_graphs_ready'] = suffix_ready
        path = prefill_path(self.native is not None, start, len(tokens), self.resume_policy,
                            self.ingest_ms_per_token, native_ready=ready, native_suffix_ready=suffix_ready)
        if path == "native":
            start = 0
        if path in ("verify8", "native_suffix") and chosen is not None:
            uuid, start = chosen
            checkpoints = {p: c for p, c in self.store.entries[uuid].get('checkpoints', {}).items() if p <= start}
            position, _, restore_s = self.store.restore(uuid, self.ranks, position=start)
            assert position == start
        else:
            checkpoints = {}
            restore_s = 0.0
            self.store.clear(self.ranks)
        self._sync()
        p0 = time.monotonic_ns()
        phases = {}
        def checkpoint_prompt(position):
            if position == len(tokens) - 1 and position > 0:
                began = time.perf_counter()
                checkpoints[position] = self.store.checkpoint(self.ranks, position=position)
                phases['checkpoint_ms'] = round((time.perf_counter() - began) * 1e3, 2)
        if path in ("native", "native_suffix"):
            if path == "native":
                self.native.begin_request()
            self.prefill_tokens(self.ranks, self.verifies, tokens, self.features, self.layer_ids, start=start, mode="native",
                                executor=self.native, timing=phases, progress=checkpoint_prompt)
        else:
            self.prefill_tokens(self.ranks, self.verifies, tokens, self.features, self.layer_ids, start=start,
                                mode="verify8", progress=checkpoint_prompt)
        self._sync()
        p1 = time.monotonic_ns()
        controls = dict(sampling or {})
        seed = int(controls.get("seed", self.config.sampling_seed))
        penalty = tuple(float(x) for x in controls.get("repeat_penalty", self.default_sampling))
        temperature = controls.get("temperature", self.default_temperature)
        top_p = controls.get("top_p", self.default_top_p)
        s0 = time.perf_counter()
        self.rd.set_sampling(penalty=penalty, context_tokens=tokens, temperature=temperature, top_p=top_p)
        for r, (rt, b) in enumerate(zip(self.ranks, self.buffers)):
            with torch.cuda.device(r), torch.cuda.stream(rt.stream):
                rt.pos.fill_(len(tokens))
                b.sampling_state.copy_(torch.tensor([seed, 0], dtype=torch.int64))
        self._sync()
        s1 = time.perf_counter()
        for r in range(8):
            with torch.cuda.device(r), torch.cuda.stream(self.ranks[r].stream):
                self.anchor_sampler.run(r)
        self._sync()
        selected = {int(rt.token_out[0].item()) for rt in self.ranks}
        s2 = time.perf_counter()
        phases.update(sampling_ms=round((s1 - s0) * 1e3, 2), anchor_ms=round((s2 - s1) * 1e3, 2))
        if len(selected) != 1:
            raise RuntimeContractError(f"ranks disagree on the anchor token: {sorted(selected)}")
        anchor = selected.pop()
        # The anchor is output token one; the device loop publishes the rest. An anchor that is
        # already EOS still needs a live loop for finish(); max_tokens >= 1 keeps the contract.
        max_tokens = max(1, reserved - 1)
        i0 = time.monotonic_ns()
        si = time.perf_counter()
        # Draft context K/V for the new prompt rows [start, n-1): batched here, or by initialize()'s
        # eight-row blocks. The last prompt row is D12's in the first round either way.
        ingest_from, ingest_rows = start, len(tokens) - 1 - start
        if self.ingest is not None and ingest_rows >= self.ingest.min_rows:
            steps = []
            self.ingest.ingest(self.features, start, len(tokens) - 1, timing=steps)
            if steps:
                phases["ingest_steps"] = steps
            ingest_from = len(tokens) - 1
        self._sync()
        s3 = time.perf_counter()
        self.rd.initialize([f[:len(tokens)] for f in self.features], position=len(tokens), anchor=anchor,
                           seed=seed, max_tokens=max_tokens, eos=EOS, ingest_from=ingest_from)
        self._sync()
        s4 = time.perf_counter()
        self.rd.launch()
        i1 = time.monotonic_ns()
        phases.update(ingest_ms=round((s3 - si) * 1e3, 2), init_ms=round((s4 - s3) * 1e3, 2),
                      launch_ms=round((time.perf_counter() - s4) * 1e3, 2))
        self._request_generation += 1
        prefill = dict(start_position=start, prefill_ms=(p1 - p0) / 1e6, prefill_start_ns=p0, prefill_end_ns=p1,
                       prefill_path=path, prefill_phases=phases, initialize_ms=(i1 - i0) / 1e6, prefill_estimate=estimate,
                       draft_ingest="batched" if ingest_from != start else "loop", ingest_rows=ingest_rows,
                       repeat_penalty=list(penalty), temperature=temperature, top_p=top_p, seed=seed,
                       restore_ms=restore_s * 1e3, resumed_from=chosen[0] if (chosen and start) else None,
                       resume_checkpoint=bool(chosen and start and start < self.store.entries[chosen[0]]['position']),
                       retained_checkpoints=len(checkpoints),
                       resume_candidate=chosen[0] if chosen else None,
                       new_prompt_tokens=len(tokens) - start, setup_ms=(p0 - t0) / 1e6)
        session = Session(self, tokens=tokens, start=start, anchor=anchor, reserved=reserved,
                          session_id=session_id, prefill=prefill)
        session.checkpoints = checkpoints
        self._active = session
        return session

    def _finish(self, session):
        rd = self.rd
        if not session.stopped():
            session.runtime.canceled_chain = True
            rd.cancel[1] = 1
            deadline = time.monotonic() + self.step_timeout
            while not session.stopped():
                if time.monotonic() > deadline:
                    raise RuntimeContractError("device loop did not stop after cancel")
                time.sleep(0.0002)
        self._sync()
        controls = [Control.from_buffer_copy(bytes(t.decode_control.cpu().tolist())).value for t in self.target.states]
        count = int(controls[0].output_count)
        committed = len(session.tokens) + count
        if any(c.output_count != count or c.committed_pos != committed for c in controls):
            session.runtime.reconcile_error = "ranks disagree on committed output"
            log.error("ranks disagree on committed output: %s", [(c.output_count, c.committed_pos) for c in controls])
            self.store.clear(self.ranks)
            self._sync()
            return
        session.runtime.device_steps = int(controls[0].iteration)
        session.runtime.abandoned_chain_positions = max(0, count + 1 - len(session.generated))
        session.runtime.device_stop_bound = count
        # The serial round leaves the last committed block's draft KV to the next D12; complete it.
        for r in range(8):
            with torch.cuda.device(r), torch.cuda.stream(self.ranks[r].stream):
                rd.core.context(r)
        self._sync()
        records = [session._record(j) for j in range(count)]
        published = [int(rec.token) for rec in records]
        session.runtime.segments = segment_stats(published, records)
        full = session.tokens + [session.anchor] + published
        uuid = f"anon-{self._request_generation}"
        if session.session_id is not None:
            uuid, superseded = park_name(self._entries(), session.session_id, full, self._request_generation)
            for old in superseded:
                self.store.evict(old)
        self.store.save(uuid, self.ranks, position=committed, tokens=full, checkpoints=session.checkpoints)
        self._evict_to_budget()
        log.info("session %s parked at %d (store %d MB per rank, %d entries)", uuid, committed,
                 self.store.bytes() // 8 >> 20, len(self.store.entries))

    def close(self):
        if self._closed:
            return
        try:
            if self._active is not None:
                self._active.close()
        finally:
            self._closed = True
            self.store.entries.clear()
            if self.native is not None:
                try:
                    self.native.close()
                except Exception:                                      # noqa: BLE001
                    log.exception("native prefill executor close failed")

    # --------------------------------------------------------------------- build
    @classmethod
    def build(cls, *, repo, target, checkpoint, max_decode_tokens, capacity=1048576, seed=1234,
              store_bytes=32 << 30, output_capacity=524288, log_fn=print, tokenizer=None,
              default_sampling=(1.0, 0.0, 0.0), default_temperature=None, default_top_p=0.95,
              prefill="native", prefill_warm_chunks=3, resume_policy="cost", draft_ingest="batched"):
        """Load the model onto eight GPUs and capture the persistent round (several minutes).

        ``prefill``: ``native`` (default, the 4K-chunk executor beside the decode stack) or
        ``verify8`` (explicit replay mode). ``prefill_warm_chunks`` chunk indices are run twice
        at startup to warm and capture them. Later chunk positions can still incur first-use work."""
        if prefill not in ("verify8", "native"):
            raise ValueError(f"prefill must be verify8 or native, got {prefill!r}")
        repo, target, checkpoint = Path(repo), Path(target), Path(checkpoint)
        from safetensors import safe_open
        from naivert.backends.naive_n0p5_draft import DraftCore
        from naivert.backends.naive_n0p5_draft_target import DraftTarget
        from naivert.backends.naive_n0p5_draft_round import DraftRound
        from naivert.kernels.naive_n0p5.dspark_kernel_set import prepare_draft
        from naivert.kernels.naive_n0p5.dspark_sampling import TargetSampler
        from naivert.kernels.naive_n0p5.model_ops import load_extension
        from naivert.models.naive_n0p5.dspark import DraftBuffers, prepare_rank_weights, validate_draft_config, TARGET_LAYER_IDS
        from naivert.models.naive_n0p5.checkpoint_fp8 import NaiveNCheckpoint, load_tp8_weights
        from naivert.runtime.cuda_backend import CudaRuntimeGraphApi
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        round_dir = repo / "tests/gpu/naive_n0p5_draft/round"
        verify = _load("run_naive_n0p5_verify", repo / "tests/gpu/naive_n0p5_e2e/run_naive_n0p5_verify.py")
        decode = verify.decode
        dsa = _load("run_attn_dsa", repo / "tests/gpu/attn_dsa/run_attn_dsa.py")
        dsa8 = _load("run_attn_dsa_rows", repo / "tests/gpu/attn_dsa/run_attn_dsa_rows.py")
        swa = _load("run_attn_swa", repo / "tests/gpu/attn_swa/run_attn_swa.py")
        prefill_module = _load("naivert_round_prefill", round_dir / "prefill.py")
        ops = load_extension(weights="fp8", expert_down="w8a8")
        ext_dsa, ext_dsa8 = dsa.build(False, 16), dsa8.build(8, index_heads=16)
        ext_swa, ext_swa8 = swa.build(1), swa.build(8)
        decode.enable_torch_tp8_peers()
        kernels = prepare_draft(range(8))
        log_fn("extensions ready")
        weights = load_tp8_weights(NaiveNCheckpoint(target), 48, capacity, ext_dsa.workspace_bytes(),
                                   ext_swa.workspace_bytes(), expert_down="w8a8", ops=ops)
        log_fn("target weights ready")
        validate_draft_config(json.loads((checkpoint / "config.json").read_text()))
        with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as f:
            raw = {k.removeprefix("model."): f.get_tensor(k).bfloat16().contiguous() for k in f.keys()}
        draft_weights = []
        for r in range(8):
            draft_weights.append(prepare_rank_weights(raw, weights[r].lm_head_ref.cpu(), rank=r,
                                                      device=f"cuda:{r}", kernels=kernels))
            torch.cuda.synchronize(r)
        del raw
        mailboxes = [torch.zeros(verify.mailbox_bytes(ops), dtype=torch.uint8, device=r) for r in range(8)]
        inboxes = [torch.zeros(4096, dtype=torch.uint8, device=r) for r in range(8)]
        ranks = [decode.RankRuntime(r, weights[r], ops, ext_dsa, ext_swa, capacity, mailboxes, inboxes,
                                    pdl=True, fused_unproj=True) for r in range(8)]
        for rt in ranks:
            rt.pdl_markers = decode.PDL_W8A8_MARKERS
        verifies = [verify.VerifyRuntime(rt, ext_dsa8, ext_swa8, ext_dsa8.workspace_bytes(), ext_swa8.workspace_bytes(),
                                         pdl=True, fused_unproj=True) for rt in ranks]
        buffers = [DraftBuffers(kernels, rank=r, capacity=capacity, sampling=True) for r in range(8)]
        core = DraftCore(kernels, draft_weights, buffers)
        target_rt = DraftTarget(core, verifies)
        api = CudaRuntimeGraphApi.load(decode.torch_cudart_path())
        rd = DraftRound(core, target_rt, api, output_capacity=output_capacity, staged=False,
                        persistent_sampler=True, draft_pdl=True)
        anchor = TargetSampler([rt.logits for rt in ranks], buffers, [rt.token_out[:1] for rt in ranks],
                               anchor=True, persistent=True)
        for r in range(8):
            torch.cuda.synchronize(r)
        for rt in ranks:
            rt.capture()
        rd.capture()
        for v in verifies:
            if v.graph is None:
                v.capture()
        for rt in ranks:
            rt.stream.synchronize()
        features = [torch.empty(capacity, 4096, dtype=torch.bfloat16, device=r) for r in range(8)]
        log_fn("persistent round captured")
        native = None
        if prefill == "native":
            from naivert.prefill.native.executor import NativePrefill
            from naivert.prefill.operands import kernel_prefill_weights
            t0 = time.monotonic()
            kweights = kernel_prefill_weights(weights, NaiveNCheckpoint(target), capacity=capacity)
            log_fn(f"native prefill operands ready {time.monotonic() - t0:.1f} s")
            native = NativePrefill(kweights, [rt.stream for rt in ranks], feature_layers=TARGET_LAYER_IDS, features=features)
            warm = min(int(prefill_warm_chunks), capacity // 4096)
            if warm > 0:
                log_fn(f"native prefill warm chunks {native.warm_chunks(warm)}")
            for rt in ranks:
                rt.stream.synchronize()
            log_fn(f"native prefill executor ready {native.versions}")
        parts = dict(ranks=ranks, verifies=verifies, buffers=buffers, weights=weights, core=core, target=target_rt,
                     rd=rd, anchor=anchor, features=features, capacity=capacity, layer_ids=TARGET_LAYER_IDS,
                     prefill_tokens=prefill_module.prefill_tokens, native_prefill=native,
                     weights_info=dict(target=str(target), target_variant=getattr(weights[0].checkpoint, "variant", None),
                                       draft=str(checkpoint), prefill=prefill),
                     binaries={"persistent.tail": rd.tail.extension.__file__, "persistent.sampler": anchor.ext.__file__})
        return cls(parts, max_decode_tokens=max_decode_tokens, seed=seed, store_bytes=store_bytes, tokenizer=tokenizer,
                   default_sampling=default_sampling, default_temperature=default_temperature, default_top_p=default_top_p,
                   resume_policy=resume_policy, draft_ingest=draft_ingest)


__all__ = ["CONTEXT_EXCEEDED_MARK", "Session", "SessionEngine", "prefill_estimate", "prefill_path", "select_resume"]
