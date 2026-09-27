"""Repeat penalty through the API: the same prompts with the penalty off and on.

Reports per request: output tokens, tokens per round, device tok/s, and repetition measures from
the server's token dump (distinct-token ratio, fraction of repeated 4-grams). Also checks that an
out-of-range value is rejected with 400.
"""
import argparse
import json
import re
import sys
import time

import anthropic
import httpx

SYSTEM = "You are Naive N0.5, a helpful AI assistant."


def user_texts(dataset, n):
    out = []
    for line in open(dataset):
        m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", json.loads(line)["prompt_text"], re.S)
        if m:
            out.append(m.group(1))
        if len(out) == n:
            break
    return out


def repetition(tokens):
    grams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    return dict(distinct_ratio=round(len(set(tokens)) / max(1, len(tokens)), 4),
                repeated_4gram=round(1 - len(set(grams)) / max(1, len(grams)), 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dump", required=True, help="the server's --dump-tokens file")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out")
    ap.add_argument("--settings", default=None,
                    help='JSON list of [name, {field: value}] pairs; default: off, rep1.3, freq0.5, pres0.8')
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
    client = anthropic.Anthropic(base_url=a.url, api_key="none", timeout=900, max_retries=0)
    prompts = user_texts(a.dataset, a.n)
    settings = [("off", {}), ("rep1.3", {"repetition_penalty": 1.3}), ("freq0.5", {"frequency_penalty": 0.5}),
                ("pres0.8", {"presence_penalty": 0.8})]
    if a.settings:
        settings = [(name, extra) for name, extra in json.loads(a.settings)]
    rows = []
    for i, prompt in enumerate(prompts):
        for name, extra in settings:
            t0 = time.monotonic()
            msg = client.messages.create(model=model, max_tokens=a.max_tokens, system=SYSTEM,
                                         messages=[dict(role="user", content=prompt)], extra_body=extra)
            wall = time.monotonic() - t0
            timings = msg.model_dump().get("naivert", {}).get("timings", {})
            seg = (timings.get("segments") or {}).get("total", {})
            rows.append(dict(prompt=i, setting=name, message_id=msg.id, output_tokens=msg.usage.output_tokens,
                             stop=msg.stop_reason, wall_s=round(wall, 2), tokens_per_round=round(seg.get("tokens_per_round", 0), 3),
                             device_tok_s=round(seg.get("device_tok_s", 0)), text_chars=sum(len(b.text) for b in msg.content if b.type == "text")))
            print("RESULT", json.dumps(rows[-1]), flush=True)
    try:
        client.messages.create(model=model, max_tokens=16, messages=[dict(role="user", content="x")], extra_body={"repetition_penalty": 0})
        rejected = False
    except anthropic.BadRequestError as exc:
        rejected = "invalid_repeat_penalty" in str(exc)
    print("REJECTS_OUT_OF_RANGE", rejected, flush=True)
    dump = {}
    for line in open(a.dump):
        r = json.loads(line)
        dump[r["id"]] = r["tokens"]
    for r in rows:
        r.update(repetition(dump.get(r["message_id"], [])))
    by = {}
    for r in rows:
        by.setdefault(r["setting"], []).append(r)
    summary = {}
    for name, rs in by.items():
        n = len(rs)
        summary[name] = dict(requests=n, output_tokens=sum(r["output_tokens"] for r in rs),
                             tokens_per_round=round(sum(r["tokens_per_round"] for r in rs) / n, 3),
                             device_tok_s=round(sum(r["device_tok_s"] for r in rs) / n),
                             distinct_ratio=round(sum(r["distinct_ratio"] for r in rs) / n, 4),
                             repeated_4gram=round(sum(r["repeated_4gram"] for r in rs) / n, 4),
                             stops=sum(r["stop"] == "end_turn" for r in rs))
    print("SUMMARY", json.dumps(summary), flush=True)
    print("PENALTY_HTTP_DONE", flush=True)
    if a.out:
        json.dump(dict(summary=summary, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
