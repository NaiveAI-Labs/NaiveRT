"""Temperature sweep through the API: acceptance, speed, completion and repetition per setting."""
import argparse, json, re, sys, time
import anthropic, httpx


def prompt_pairs(dataset, n):
    out = []
    for line in open(dataset):
        row = json.loads(line)
        systems = re.findall(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", row["prompt_text"], re.S)
        m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", row["prompt_text"], re.S)
        if m: out.append((systems[-1] if systems else None, m.group(1)))
        if len(out) == n: break
    return out


def repeated_4grams(text):
    words = text.split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return round(1 - len(set(grams)) / max(1, len(grams)), 3) if grams else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080"); ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperatures", default="0.1,0.2,0.4,0.6,0.8,1.0"); ap.add_argument("--out")
    a = ap.parse_args()
    deadline = time.monotonic() + 3600
    while True:
        try:
            if httpx.get(a.url + "/health", timeout=5).status_code == 200: break
        except Exception: pass
        if time.monotonic() > deadline: sys.exit("server never became healthy")
        time.sleep(5)
    model = httpx.get(a.url + "/v1/models", timeout=10).json()["data"][0]["id"]
    client = anthropic.Anthropic(base_url=a.url, api_key="none", timeout=900, max_retries=0)
    temps = [float(x) for x in a.temperatures.split(",")]
    rows = []
    for i, (system, user) in enumerate(prompt_pairs(a.dataset, a.n)):
        for T in temps:
            msg = client.messages.create(model=model, max_tokens=a.max_tokens, system=system,
                                         messages=[dict(role="user", content=user)], extra_body={"temperature": T, "seed": 1})
            text = "".join(b.text for b in msg.content if b.type == "text")
            seg = msg.model_dump().get("naivert", {}).get("timings", {}).get("segments", {}).get("total", {})
            rows.append(dict(prompt=i, T=T, tokens=msg.usage.output_tokens, stop=msg.stop_reason,
                             rounds=seg.get("rounds", 0), device_ms=seg.get("device_ms", 0),
                             tokens_per_round=round(seg.get("tokens_per_round", 0), 3), device_tok_s=round(seg.get("device_tok_s", 0)),
                             repeated_4gram=repeated_4grams(text), tail=repr(text[-100:])))
            print("ROW", json.dumps(rows[-1]), flush=True)
    summary = {}
    for T in temps:
        rs = [r for r in rows if r["T"] == T]
        tok = sum(r["tokens"] - 1 for r in rs); rounds = sum(r["rounds"] for r in rs); ms = sum(r["device_ms"] for r in rs)
        summary[str(T)] = dict(requests=len(rs), tokens=tok, tokens_per_round=round(tok / max(1, rounds), 3),
                               device_tok_s=round(tok / max(1e-9, ms / 1e3)), end_turn=sum(r["stop"] == "end_turn" for r in rs),
                               repeated_4gram=round(sum(r["repeated_4gram"] for r in rs) / len(rs), 3))
    print("SUMMARY", json.dumps(summary), flush=True)
    print("TEMP_SWEEP_DONE", flush=True)
    if a.out: json.dump(dict(summary=summary, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__": main()
