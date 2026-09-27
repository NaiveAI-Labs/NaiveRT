"""Resume path A/B against a running serve_sessions server: one parked prefix, follow-ups of growing length.

For each suffix length a fresh session runs turn 1 (the dataset prompt, parked at its end) and turn 2
(a follow-up padded to about the requested number of tokens). Reports per turn 2: the path the server
took (``prefill_path``), prefill ms, draft-ingest ms (``initialize_ms``), their sum (the pre-first-token
cost) and the server's own estimate. Run it against ``--resume-policy replay`` and ``native`` servers
to measure both curves and against ``cost`` to see the choice.
"""
import argparse
import json
import re
import sys
import time

import anthropic
import httpx

FILLER = ("Please also keep the following notes in mind while answering; they restate the requirements in more detail. ")



def prompt_tokens(usage) -> int:
    """Prompt length under Anthropic semantics: uncached + cache read + cache written."""
    return int(usage.input_tokens) + int(getattr(usage, "cache_read_input_tokens", 0) or 0) + \
        int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

def user_text(dataset, index):
    rows = [json.loads(line) for line in open(dataset)]
    row = rows[index]
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", row["prompt_text"], re.S)
    return m.group(1) if m else row["prompt_text"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--index", type=int, default=0, help="dataset row for turn 1")
    ap.add_argument("--suffixes", default="32,128,256,512,1024,2048,4096", help="approximate follow-up lengths in tokens")
    ap.add_argument("--max-tokens", type=int, default=48)
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
    first = user_text(a.dataset, a.index)
    rows = []

    def turn(sid, messages):
        t0 = time.monotonic()
        msg = client.messages.create(model=model, max_tokens=a.max_tokens, messages=messages,
                                     extra_headers={"x-session-id": sid})
        wall = time.monotonic() - t0
        timings = msg.model_dump().get("naivert", {}).get("timings", {})
        return msg, wall, timings

    for suffix in [int(x) for x in a.suffixes.split(",")]:
        sid = f"resume-{suffix}"
        m1, wall1, t1 = turn(sid, [dict(role="user", content=first)])
        answer = "".join(b.text for b in m1.content if b.type == "text")
        # about 18 tokens per FILLER copy; the server reports the exact count
        follow = "Thanks. " + FILLER * max(1, suffix // 18) + " Now give one more example."
        m2, wall2, t2 = turn(sid, [dict(role="user", content=first), dict(role="assistant", content=answer),
                                   dict(role="user", content=follow)])
        cache = m2.usage.cache_read_input_tokens or 0
        row = dict(suffix_requested=suffix, prompt_tokens=prompt_tokens(m2.usage), cache_read=cache,
                   new_tokens=prompt_tokens(m2.usage) - cache, path=t2.get("prefill_path"),
                   prompt_ms=round(t2.get("prompt_ms", 0), 1), initialize_ms=round(t2.get("initialize_ms", 0), 1),
                   first_token_ms=round(t2.get("prompt_ms", 0) + t2.get("initialize_ms", 0), 1),
                   estimate=t2.get("prefill_estimate"), phases=t2.get("prefill_phases"), wall_s=round(wall2, 2),
                   turn1=dict(prompt_tokens=prompt_tokens(m1.usage), path=t1.get("prefill_path"),
                              prompt_ms=round(t1.get("prompt_ms", 0), 1), initialize_ms=round(t1.get("initialize_ms", 0), 1),
                              phases=t1.get("prefill_phases")))
        rows.append(row)
        print("RESULT", json.dumps(row), flush=True)
    if a.out:
        json.dump(dict(model=model, rows=rows), open(a.out, "w"), indent=1)
    print("RESUME_POLICY_DONE", flush=True)


if __name__ == "__main__":
    main()
