"""Interleaved sessions against a running serve_sessions server.

A1 B1 A2 B2 keyed by the x-session-id header, A3 unkeyed (content match), then C1 with thinking
enabled and keyed by metadata.user_id. Reports per request: prompt tokens, cache_read_input_tokens
(the parked position resumed from), prefill ms, output tokens, device tok/s and the think / answer
segment statistics (tokens, tokens per round, device tok/s).
"""
import argparse
import json
import re
import sys
import time

import anthropic
import httpx

SYSTEM = "You are Naive N0.5, a helpful AI assistant."
FOLLOW_UP = ("Now make a second, different version of the same piece: switch to a dark color theme and add "
             "at least one animated element. Return the complete code again inside a single fenced code block.")
FOLLOW_UP_3 = "Give the second version a short title as an HTML comment on its first line and return it again."


def user_texts(dataset, n):
    out = []
    for line in open(dataset):
        row = json.loads(line)
        m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", row["prompt_text"], re.S)
        if m:
            out.append(m.group(1))
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--think-budget", type=int, default=6144)
    ap.add_argument("--out")
    a = ap.parse_args()
    deadline = time.monotonic() + 1800
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
    client = anthropic.Anthropic(base_url=a.url, api_key="none", timeout=900, max_retries=0)
    prompts = user_texts(a.dataset, 3)
    history = {"A": [], "B": [], "C": []}
    rows = []

    def turn(label, sid, messages, key="header", thinking=None):
        kw = dict(model=model, max_tokens=a.max_tokens, system=SYSTEM, messages=messages)
        if key == "header":
            kw["extra_headers"] = {"x-session-id": sid}
        elif key == "metadata":
            kw["metadata"] = {"user_id": sid}
        if thinking:
            kw["thinking"] = {"type": "enabled", "budget_tokens": thinking}
        t0 = time.monotonic()
        msg = client.messages.create(**kw)
        wall = time.monotonic() - t0
        raw = msg.model_dump()
        timings = raw.get("naivert", {}).get("timings", {})
        text = "".join(b.text for b in msg.content if b.type == "text")
        segments = {k: {n: round(v, 2) if isinstance(v, float) else v for n, v in s.items()}
                    for k, s in (timings.get("segments") or {}).items()}
        row = dict(label=label, session=sid, key=key, thinking=bool(thinking),
                   prompt_tokens=msg.usage.input_tokens + (getattr(msg.usage, "cache_read_input_tokens", 0) or 0)
                   + (getattr(msg.usage, "cache_creation_input_tokens", 0) or 0),
                   cache_read=getattr(msg.usage, "cache_read_input_tokens", None),
                   output_tokens=msg.usage.output_tokens, stop=msg.stop_reason, wall_s=round(wall, 2),
                   prompt_ms=round(timings.get("prompt_ms", 0), 1), device_tok_s=round(timings.get("decode_tok_s_device", 0)),
                   n_cache=timings.get("n_prompt_tokens_cache"), segments=segments, text_chars=len(text),
                   blocks=[b.type for b in msg.content])
        rows.append(row)
        print("RESULT", json.dumps(row), flush=True)
        return text

    for sid, prompt in zip("AB", prompts):
        history[sid] = [dict(role="user", content=prompt)]
        text = turn(f"{sid}1", sid, history[sid])
        history[sid].append(dict(role="assistant", content=text))
    for sid in "AB":
        history[sid].append(dict(role="user", content=FOLLOW_UP))
        text = turn(f"{sid}2", sid, history[sid])
        history[sid].append(dict(role="assistant", content=text))
    history["A"].append(dict(role="user", content=FOLLOW_UP_3))
    turn("A3-unkeyed", "A", history["A"], key=None)
    history["C"] = [dict(role="user", content=prompts[2])]
    turn("C1-think", "C", history["C"], key="metadata", thinking=a.think_budget)
    reuse = all(r["cache_read"] and r["cache_read"] > 0 for r in rows if r["label"] in ("A2", "B2", "A3-unkeyed"))
    think = any("think" in r["segments"] for r in rows if r["label"] == "C1-think")
    print("SESSIONS_OK" if reuse else "SESSIONS_NO_REUSE", "THINK_SPLIT_OK" if think else "THINK_SPLIT_MISSING", flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
