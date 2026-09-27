"""Multi-session, multi-round prefix cache hit rate through a running serve_sessions server.

``--sessions N --rounds K``: N sessions from the dataset's user prompts, driven round-robin
(S1r1 .. SNr1, S1r2 .. SNr2, ...) so every later round follows N-1 other sessions' decodes.
Each session is keyed by ``x-session-id``. Round r>1 appends the assistant's previous answer and
a follow-up prompt. ``--thinking-sessions M`` makes the first M sessions run with thinking
enabled (their history carries the thinking blocks back, as the Anthropic API requires).

A round-r request is a *hit* when ``usage.cache_read_input_tokens`` covers everything except the
new user turn, i.e. cache_read >= prompt_len - new_turn_tokens (new_turn_tokens measured with
count_tokens on the follow-up alone; prompt_len = usage.input_tokens + cache_read + cache_creation, reported as
``input_tokens`` in the rows); token hit rate is sum(cache_read) / sum(prompt_len) over
rounds > 1. Also reports think / answer segment statistics per mode.
"""
import argparse
import json
import re
import sys
import time

import anthropic
import httpx

SYSTEM = "You are Naive N0.5, a helpful AI assistant."
FOLLOW_UPS = [
    "Now make a second, different version of the same piece: switch to a dark color theme and add at least one animated element. Return the complete code again inside a single fenced code block.",
    "Give the latest version a short title as a comment on its first line, and make one more visible change of your choice. Return the complete code again.",
    "Reduce the latest version to its five most important elements without breaking it. Return the complete code again.",
    "Add a short legend or caption inside the latest version explaining what it shows. Return the complete code again.",
]


def user_texts(dataset, n):
    return [u for _, u in prompt_pairs(dataset, n)]


def prompt_pairs(dataset, n):
    """(system text, user text) per row: the last system block and the user block of prompt_text.

    A template-injected first system block (the release checkpoint's "Reasoning Effort") is skipped: the server's
    template adds it again.
    """
    out = []
    for line in open(dataset):
        row = json.loads(line)
        systems = re.findall(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", row["prompt_text"], re.S)
        m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", row["prompt_text"], re.S)
        if m:
            out.append((systems[-1] if systems else None, m.group(1)))
        if len(out) == n:
            break
    if len(out) < n:
        sys.exit(f"dataset has only {len(out)} prompts")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--thinking-sessions", type=int, default=0)
    ap.add_argument("--session-indices", default=None, help="comma-separated dataset prompt indices to use as sessions")
    ap.add_argument("--thinking-indices", default=None, help="comma-separated dataset prompt indices that run with thinking")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--think-max-tokens", type=int, default=8192)
    ap.add_argument("--think-budget", type=int, default=6144)
    ap.add_argument("--system", default=SYSTEM)
    ap.add_argument("--system-from-dataset", action="store_true", help="use each row's own (last) system block as the system prompt")
    ap.add_argument("--context-limit", type=int, default=0,
                    help="if set, clamp each request's max_tokens so prompt + max_tokens stays within this many tokens")
    ap.add_argument("--out")
    a = ap.parse_args()
    deadline = time.monotonic() + 3600
    while True:
        try:
            if httpx.get(a.url + "/health", timeout=5).status_code == 200:
                break
        except Exception:
            pass
        if time.monotonic() > deadline:
            sys.exit("server never became healthy")
        time.sleep(5)
    model = httpx.get(a.url + "/v1/models", timeout=10).json()["data"][0]["id"]
    print("MODEL", model, flush=True)
    client = anthropic.Anthropic(base_url=a.url, api_key="none", timeout=1800, max_retries=0)
    indices = [int(x) for x in a.session_indices.split(",")] if a.session_indices else list(range(a.sessions))
    think_indices = {int(x) for x in a.thinking_indices.split(",")} if a.thinking_indices else set(range(a.thinking_sessions))
    pairs = prompt_pairs(a.dataset, max(indices) + 1)
    prompts = [u for _, u in pairs]
    sessions = []
    for i in indices:
        sessions.append(dict(sid=f"s{i:02d}", thinking=i in think_indices, history=[dict(role="user", content=prompts[i])],
                             system=(pairs[i][0] if a.system_from_dataset and pairs[i][0] else a.system), dead=False))
    a.sessions, a.thinking_sessions = len(sessions), sum(s["thinking"] for s in sessions)
    rows = []
    for r in range(1, a.rounds + 1):
        for s in sessions:
            if s["dead"]:
                continue
            sid = s["sid"]
            if r > 1:
                follow = FOLLOW_UPS[(r - 2) % len(FOLLOW_UPS)]
                s["history"].append(dict(role="user", content=follow))
                # tokens the new user turn adds: count the whole prompt with and without it
                kw_count = dict(model=model, system=s["system"], messages=s["history"])
                if s["thinking"]:
                    kw_count["thinking"] = {"type": "enabled", "budget_tokens": a.think_budget}
                with_turn = client.messages.count_tokens(**kw_count).input_tokens
                kw_count["messages"] = s["history"][:-1] + [dict(role="user", content="x")]
                new_turn = with_turn - client.messages.count_tokens(**kw_count).input_tokens + 8
            else:
                new_turn = None
            max_tokens = a.think_max_tokens if s["thinking"] else a.max_tokens
            if a.context_limit:
                kw_count = dict(model=model, system=s["system"], messages=s["history"])
                if s["thinking"]:
                    kw_count["thinking"] = {"type": "enabled", "budget_tokens": a.think_budget}
                prompt_n = client.messages.count_tokens(**kw_count).input_tokens
                max_tokens = max(64, min(max_tokens, a.context_limit - prompt_n - 8))
            kw = dict(model=model, max_tokens=max_tokens, system=s["system"],
                      messages=s["history"], extra_headers={"x-session-id": sid})
            if s["thinking"]:
                kw["thinking"] = {"type": "enabled", "budget_tokens": a.think_budget}
            t0 = time.monotonic()
            try:
                msg = client.messages.create(**kw)
            except anthropic.APIStatusError as exc:
                print("ERROR", sid, r, exc.status_code, str(exc)[:300], flush=True)
                rows.append(dict(sid=sid, round=r, thinking=s["thinking"], error=str(exc)[:300]))
                s["dead"] = True
                continue
            wall = time.monotonic() - t0
            raw = msg.model_dump()
            timings = raw.get("naivert", {}).get("timings", {})
            segments = {k: {n: round(v, 2) if isinstance(v, float) else v for n, v in seg.items()}
                        for k, seg in (timings.get("segments") or {}).items()}
            cache = getattr(msg.usage, "cache_read_input_tokens", 0) or 0
            # prompt length under Anthropic semantics: uncached + cache read + cache written
            prompt_len = msg.usage.input_tokens + cache + (getattr(msg.usage, "cache_creation_input_tokens", 0) or 0)
            # The new turn also carries the turn framing (end of the previous turn, user header, the
            # assistant header): allow 12 tokens beyond the follow-up's own count.
            hit = None if r == 1 else (cache > 0 and prompt_len - cache <= new_turn + 12)
            row = dict(sid=sid, round=r, thinking=s["thinking"], max_tokens=max_tokens, input_tokens=prompt_len, cache_read=cache,
                       new_turn_tokens=new_turn, hit=hit, output_tokens=msg.usage.output_tokens, stop=msg.stop_reason,
                       message_id=msg.id, text_tail=repr("".join(b.text for b in msg.content if b.type == "text")[-80:]),
                       wall_s=round(wall, 2), prompt_ms=round(timings.get("prompt_ms", 0), 1),
                       initialize_ms=round(timings.get("initialize_ms", 0), 1), prefill_path=timings.get("prefill_path"),
                       device_tok_s=round(timings.get("decode_tok_s_device", 0)), segments=segments,
                       blocks=[b.type for b in msg.content])
            rows.append(row)
            print("RESULT", json.dumps(row), flush=True)
            # history for the next round: the assistant turn exactly as returned (thinking blocks included)
            s["history"].append(dict(role="assistant", content=[b.model_dump(exclude_none=True) for b in msg.content]))
    ok = [r for r in rows if "error" not in r]
    later = [r for r in ok if r["round"] > 1]

    def summarize(sel):
        if not sel:
            return None
        hits = sum(1 for r in sel if r["hit"])
        return dict(requests=len(sel), hits=hits, request_hit_rate=round(hits / len(sel), 4),
                    token_hit_rate=round(sum(r["cache_read"] for r in sel) / max(1, sum(r["input_tokens"] for r in sel)), 4),
                    prefilled_tokens=sum(r["input_tokens"] - r["cache_read"] for r in sel),
                    prompt_ms_mean=round(sum(r["prompt_ms"] for r in sel) / len(sel), 1),
                    initialize_ms_mean=round(sum(r.get("initialize_ms", 0) for r in sel) / len(sel), 1))

    def segs(sel, name):
        xs = [r["segments"][name] for r in sel if name in r["segments"]]
        if not xs:
            return None
        tok = sum(x["tokens"] for x in xs); rounds = sum(x["rounds"] for x in xs); ms = sum(x["device_ms"] for x in xs)
        return dict(requests=len(xs), tokens=tok, tokens_per_round=round(tok / max(1, rounds), 3),
                    device_tok_s=round(tok / max(1e-9, ms / 1e3)))
    first = [r for r in ok if r["round"] == 1]
    summary = dict(model=model, sessions=a.sessions, rounds=a.rounds, thinking_sessions=a.thinking_sessions,
                   completed=len(ok), errors=len(rows) - len(ok),
                   round1=None if not first else dict(requests=len(first), prompt_tokens_mean=round(sum(r["input_tokens"] for r in first) / len(first)),
                                                     prompt_ms_mean=round(sum(r["prompt_ms"] for r in first) / len(first), 1),
                                                     initialize_ms_mean=round(sum(r.get("initialize_ms", 0) for r in first) / len(first), 1)),
                   hit_all=summarize(later), hit_nothink=summarize([r for r in later if not r["thinking"]]),
                   hit_think=summarize([r for r in later if r["thinking"]]),
                   by_round={str(r): summarize([x for x in later if x["round"] == r]) for r in range(2, a.rounds + 1)},
                   nothink_answer=segs([r for r in ok if not r["thinking"]], "answer"),
                   think_think=segs([r for r in ok if r["thinking"]], "think"),
                   think_answer=segs([r for r in ok if r["thinking"]], "answer"),
                   stops={s: sum(1 for r in ok if r["stop"] == s) for s in set(r["stop"] for r in ok)})
    print("SUMMARY", json.dumps(summary), flush=True)
    print("HITRATE_DONE", flush=True)
    if a.out:
        json.dump(dict(summary=summary, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
