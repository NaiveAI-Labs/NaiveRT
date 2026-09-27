"""Server-sent events on a running serve_sessions server.

1. non-streaming reference for prompt P (session sse-ref)
2. the same prompt streamed on a fresh session: event order, time to first text delta, inter-chunk
   gaps, tokens per chunk, text identical to the reference (seeded decode)
3. thinking streamed: thinking_delta before text_delta, segments in message_delta
4. disconnect after N text deltas, then a short request: the server must recover, and the request
   log must show the cancel
5. streamed second turn of the session from step 2: cache_read_input_tokens in message_start
"""
import argparse
import json
import re
import statistics as st
import sys
import time

import anthropic
import httpx

SYSTEM = "You are Naive N0.5, a helpful AI assistant."
FOLLOW_UP = ("Now make a second, different version of the same piece: switch to a dark color theme and add "
             "at least one animated element. Return the complete code again inside a single fenced code block.")


def user_texts(dataset, n):
    out = []
    for line in open(dataset):
        m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", json.loads(line)["prompt_text"], re.S)
        if m:
            out.append(m.group(1))
        if len(out) == n:
            break
    return out


def stream(url, body, headers, *, stop_after=None):
    """Raw SSE: returns events [(name, data, t)], text, thinking, timing."""
    events, text, thinking = [], [], []
    t_send = time.monotonic()
    first_delta = None
    with httpx.Client(timeout=900) as c:
        with c.stream("POST", url + "/v1/messages", json=dict(body, stream=True),
                      headers={"content-type": "application/json", **headers}) as r:
            status = r.status_code
            name = None
            for line in r.iter_lines():
                if line.startswith("event: "):
                    name = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    t = time.monotonic()
                    events.append((name, data, t))
                    if name == "content_block_delta":
                        d = data["delta"]
                        if d["type"] == "text_delta":
                            text.append(d["text"])
                            if first_delta is None:
                                first_delta = t
                        elif d["type"] == "thinking_delta":
                            thinking.append(d["thinking"])
                            if first_delta is None:
                                first_delta = t
                        if stop_after is not None and len(text) >= stop_after:
                            r.close()
                            break
                    elif name == "error":
                        break
    return dict(status=status, events=events, text="".join(text), thinking="".join(thinking),
                t_send=t_send, first_delta=first_delta)


def gaps_ms(events):
    ts = [t for n, _, t in events if n == "content_block_delta"]
    return [(b - a) * 1e3 for a, b in zip(ts, ts[1:])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
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
    client = anthropic.Anthropic(base_url=a.url, api_key="none", timeout=900, max_retries=0)
    prompts = user_texts(a.dataset, 2)
    report = {}

    # 1. reference
    body = dict(model=model, max_tokens=a.max_tokens, system=SYSTEM, messages=[dict(role="user", content=prompts[0])])
    t0 = time.monotonic()
    ref = client.messages.create(**body, extra_headers={"x-session-id": "sse-ref"})
    ref_text = "".join(b.text for b in ref.content if b.type == "text")
    report["reference"] = dict(output_tokens=ref.usage.output_tokens, stop=ref.stop_reason, wall_s=round(time.monotonic() - t0, 2))
    print("REF", json.dumps(report["reference"]), flush=True)

    # 2. streamed, fresh session
    s = stream(a.url, body, {"x-session-id": "sse-b"})
    names = [n for n, _, _ in s["events"]]
    order_ok = (names[0] == "message_start" and names[1] == "content_block_start" and names[-2:] == ["message_delta", "message_stop"]
                and all(n == "content_block_delta" for n in names[2:-3]) and names[-3] == "content_block_stop")
    md = next(d for n, d, _ in s["events"] if n == "message_delta")
    ms = next(d for n, d, _ in s["events"] if n == "message_start")
    g = gaps_ms(s["events"])
    ndelta = names.count("content_block_delta")
    tokens = md["usage"]["output_tokens"]
    report["stream"] = dict(status=s["status"], events=len(names), order_ok=order_ok, text_deltas=ndelta,
                            output_tokens=tokens, tokens_per_delta=round(tokens / max(1, ndelta), 2),
                            ttft_ms=round((s["first_delta"] - s["t_send"]) * 1e3, 1),
                            total_s=round((s["events"][-1][2] - s["t_send"]), 2),
                            gap_ms=dict(median=round(st.median(g), 2), p99=round(sorted(g)[int(len(g) * .99)], 2), max=round(max(g), 1)),
                            text_equals_reference=s["text"] == ref_text, stop=md["delta"]["stop_reason"],
                            cache_read_start=ms["message"]["usage"].get("cache_read_input_tokens"),
                            device_tok_s=round(md["naivert"]["timings"]["decode_tok_s_device"]),
                            segments=md["naivert"]["timings"].get("segments"))
    print("STREAM", json.dumps(report["stream"]), flush=True)

    # 3. thinking streamed
    body_t = dict(body, messages=[dict(role="user", content=prompts[1])], max_tokens=8192,
                  thinking={"type": "enabled", "budget_tokens": 6144})
    s3 = stream(a.url, body_t, {"x-session-id": "sse-think"})
    kinds = [d["delta"]["type"] for n, d, _ in s3["events"] if n == "content_block_delta"]
    starts = [d["content_block"]["type"] for n, d, _ in s3["events"] if n == "content_block_start"]
    md3 = next(d for n, d, _ in s3["events"] if n == "message_delta")
    first_text = kinds.index("text_delta") if "text_delta" in kinds else None
    report["thinking_stream"] = dict(status=s3["status"], blocks=starts, thinking_deltas=kinds.count("thinking_delta"),
                                     text_deltas=kinds.count("text_delta"),
                                     thinking_before_text=first_text is None or all(k == "thinking_delta" for k in kinds[:first_text]),
                                     thinking_chars=len(s3["thinking"]), text_chars=len(s3["text"]),
                                     stop=md3["delta"]["stop_reason"], segments=md3["naivert"]["timings"].get("segments"))
    print("THINK", json.dumps(report["thinking_stream"]), flush=True)

    # 4. disconnect mid-stream, then recover
    s4 = stream(a.url, dict(body, messages=[dict(role="user", content=prompts[0])]), {"x-session-id": "sse-cut"}, stop_after=40)
    t_cut = time.monotonic()
    quick = client.messages.create(model=model, max_tokens=32, system=SYSTEM, messages=[dict(role="user", content="Say hi.")])
    report["disconnect"] = dict(deltas_before_cut=[n for n, _, _ in s4["events"]].count("content_block_delta"),
                                recovery_wall_s=round(time.monotonic() - t_cut, 2), recovery_tokens=quick.usage.output_tokens,
                                recovery_text=quick.content[0].text[:60])
    print("DISCONNECT", json.dumps(report["disconnect"]), flush=True)

    # 5. streamed continuation of session sse-b
    body5 = dict(body, messages=[dict(role="user", content=prompts[0]), dict(role="assistant", content=s["text"]),
                                 dict(role="user", content=FOLLOW_UP)])
    s5 = stream(a.url, body5, {"x-session-id": "sse-b"})
    ms5 = next(d for n, d, _ in s5["events"] if n == "message_start")
    md5 = next(d for n, d, _ in s5["events"] if n == "message_delta")
    u5 = ms5["message"]["usage"]
    report["continuation_stream"] = dict(prompt_tokens=u5["input_tokens"] + (u5.get("cache_read_input_tokens") or 0)
                                         + (u5.get("cache_creation_input_tokens") or 0),
                                         cache_read=ms5["message"]["usage"].get("cache_read_input_tokens"),
                                         ttft_ms=round((s5["first_delta"] - s5["t_send"]) * 1e3, 1),
                                         output_tokens=md5["usage"]["output_tokens"], stop=md5["delta"]["stop_reason"])
    print("CONT", json.dumps(report["continuation_stream"]), flush=True)
    ok = (order_ok and report["stream"]["text_equals_reference"] and report["thinking_stream"]["thinking_before_text"]
          and report["disconnect"]["recovery_tokens"] > 0 and (report["continuation_stream"]["cache_read"] or 0) > 0)
    print("SSE_OK" if ok else "SSE_FAIL", flush=True)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
