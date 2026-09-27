#!/usr/bin/env python
"""Naive N Messages-API server over the persistent draft round with parked per-session prefixes.

Run on an eight-GPU host. Temperature, top-p, seed, and penalties are supplied per request;
omitted values use the configured server defaults. One request runs at a time; a request whose prompt extends a sequence this server finished earlier, keyed by the
``x-session-id`` header (else the Messages API ``metadata.user_id``, else by content), resumes
from that sequence's parked KV.

Weights are selected with ``--target`` and ``--draft``: a directory, or a short name resolved
against the known stores (``--list-weights`` prints what is available on this box).
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from naivert.serve.http import build_app, run_app

# Checkpoint discovery: a name is a directory, or a short name looked up as <root>/<name> and <root>/<name>-copy under
# the colon-separated roots below; a directory without the marker file resolves to its newest step_* child.
MODEL_ROOTS = tuple(p for p in os.environ.get("NAIVERT_MODEL_ROOTS", "/local_models").split(":") if p)
DRAFT_ROOTS = tuple(p for p in os.environ.get(
    "NAIVERT_DRAFT_ROOTS", "/local_models/drafts:/local_models").split(":") if p)
TARGET_MARK, DRAFT_MARK = "model.safetensors.index.json", "model.safetensors"
from naivert.models.registry import DEFAULT_MODEL, MODELS, resolve_model

DEFAULT_TARGET, DEFAULT_DRAFT = MODELS[DEFAULT_MODEL].target, MODELS[DEFAULT_MODEL].draft


def step_child(d: Path) -> Path | None:
    steps = sorted(c for c in d.glob("step_*") if c.is_dir()) if d.is_dir() else []
    return steps[-1] if steps else None


def checkpoint_dir(d: Path, mark: str) -> Path | None:
    for cand in (d, step_child(d)):
        if cand is not None and (cand / mark).is_file():
            return cand.resolve()
    return None


def resolve(name: str, roots, mark: str) -> Path:
    candidates = [Path(name).expanduser()] + [Path(root, name + suffix) for root in roots for suffix in ("", "-copy")]
    for d in candidates:
        found = checkpoint_dir(d, mark)
        if found is not None:
            return found
    raise SystemExit(f"no checkpoint named {name!r} (looked for {mark} under {', '.join(roots)})")


def list_weights() -> str:
    lines = ["targets (--target):"]
    for base in map(Path, MODEL_ROOTS):
        if base.is_dir():
            for d in sorted(base.iterdir()):
                cand = checkpoint_dir(d, TARGET_MARK)
                if cand is not None:
                    lines.append(f"  {d.name.removesuffix('-copy'):48s} {cand}")
    lines.append("drafts (--draft):")
    for base in map(Path, DRAFT_ROOTS):
        if base.is_dir():
            for d in sorted(base.iterdir()):
                if (d / DRAFT_MARK).is_file() and (d / "config.json").is_file():
                    lines.append(f"  {d.name:48s} {d}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--cors-origin", action="append", default=[],
                    help="allowed Studio browser origin, e.g. https://studio.example.com (repeatable)")
    ap.add_argument("--cors-credentials", action="store_true",
                    help="allow browser cookies with explicit --cors-origin values; does not enable authentication")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS),
                    help="public model name; picks the target and draft checkpoints unless --target / --draft override them")
    ap.add_argument("--target", default=os.environ.get("NAIVERT_TARGET_CHECKPOINT") or None,
                    help="target checkpoint: directory or short name (default: NAIVERT_TARGET_CHECKPOINT, then the model's)")
    ap.add_argument("--draft", "--checkpoint", dest="draft", default=os.environ.get("NAIVERT_DRAFT_CHECKPOINT") or None,
                    help="draft checkpoint: directory or short name (default: NAIVERT_DRAFT_CHECKPOINT, then the model's)")
    ap.add_argument("--list-weights", action="store_true", help="print the checkpoints found on this box and exit")
    ap.add_argument("--served-model-name", default=None, help="model id clients send and /v1/models lists (default: --model)")
    ap.add_argument("--max-decode-tokens", type=int, default=524288)
    ap.add_argument("--capacity", type=int, default=1048576, help="cache rows per sequence (prompt + decode)")
    ap.add_argument("--output-capacity", type=int, default=None,
                    help="output ring per request (default: --max-decode-tokens, at least 24576); "
                         "--max-decode-tokens is capped to it")
    ap.add_argument("--seed", type=int, default=1234, help="sampler seed for requests that send none")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default temperature (none: 1/NAIVERT_DSPARK_INV_TEMPERATURE or 1); requests may override")
    ap.add_argument("--top-p", type=float, default=0.95, help="default nucleus mass; requests may override")
    ap.add_argument("--store-gb", type=float, default=32.0, help="parked prefix budget per rank")
    ap.add_argument("--queue-depth", type=int, default=8)
    ap.add_argument("--dump-tokens")
    ap.add_argument("--repetition-penalty", type=float, default=1.0, help="default when the request sets none (1 = off)")
    ap.add_argument("--frequency-penalty", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--prefill", choices=("verify8", "native"), default="native",
                    help="prompt ingestion: the native 4K-chunk executor (default, ~55 ms per 4096 tokens; +~40 s at start-up "
                         "for its operands and chunk warm-up) or the verify-8 replay (~0.7 s per 1,000 tokens)")
    ap.add_argument("--prefill-warm-chunks", type=int, default=3,
                    help="native: chunk indices run twice at start-up so no request pays the eager first use (default 3)")
    ap.add_argument("--resume-policy", choices=("cost", "replay", "native"), default="cost",
                    help="native: how a resumed session ingests its new suffix: cost (default) picks the cheaper of extending the "
                         "parked rows by replay and re-prefilling the whole prompt with the executor; replay / native force one")
    ap.add_argument("--draft-ingest", choices=("batched", "loop"), default="batched",
                    help="draft context K/V for the prompt: batched matmuls per 4096 rows (default) or the eight-row kernel loop")
    args = ap.parse_args()
    if args.list_weights:
        print(list_weights())
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log = logging.getLogger("serve_sessions")
    entry = resolve_model(args.model)
    target = resolve(args.target or entry.target, MODEL_ROOTS, TARGET_MARK)
    draft = resolve(args.draft or entry.draft, DRAFT_ROOTS, DRAFT_MARK)
    target_name = target.parent.name.removesuffix("-copy") if target.name.startswith("step_") else target.name
    served = args.served_model_name or entry.key
    custom = bool(args.target or args.draft)
    display = entry.display if not custom else f"{entry.display} [{target_name}+{draft.name}]"
    log.info("model  %s (%s): %s", entry.key, entry.display, entry.lineage)
    log.info("target %s (%s)", target_name, target)
    log.info("draft  %s (%s)", draft.name, draft)
    log.info("served model name %s, display %s", served, display)

    def factory():
        from naivert.backends.naive_n0p5_session_engine import SessionEngine
        from naivert.serve.chat import ChatFormatter
        from naivert.serve.context import ServerContext
        formatter = ChatFormatter.from_path(target)
        engine = SessionEngine.build(repo=ROOT, target=target, checkpoint=draft,
                                     max_decode_tokens=args.max_decode_tokens, capacity=args.capacity,
                                     output_capacity=args.output_capacity or max(24576, args.max_decode_tokens),
                                     seed=args.seed, store_bytes=int(args.store_gb * (1 << 30)), log_fn=log.info,
                                     tokenizer=formatter.tokenizer,
                                     default_sampling=(args.repetition_penalty, args.frequency_penalty, args.presence_penalty),
                                     default_temperature=args.temperature, default_top_p=args.top_p,
                                     prefill=args.prefill, prefill_warm_chunks=args.prefill_warm_chunks,
                                     resume_policy=args.resume_policy, draft_ingest=args.draft_ingest)
        engine.weights_info.update(model=entry.key, display=display, lineage=entry.lineage)
        log.info("weights %s", engine.weights_info)
        return ServerContext(engine, formatter)

    app = build_app(factory, served_model=served, display_name=display, queue_depth=args.queue_depth,
                    dump_tokens=args.dump_tokens, cors_origins=args.cors_origin, cors_credentials=args.cors_credentials)
    run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
