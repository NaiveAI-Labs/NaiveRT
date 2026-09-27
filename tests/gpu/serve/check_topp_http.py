"""top-p sweep through the API: acceptance, speed and a text sample per setting."""
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

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080"); ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=2); ap.add_argument("--out")
    ap.add_argument("--top-ps", default="0.95,0.99,0.999,1.0")
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
    rows = []
    for i, (system, user) in enumerate(prompt_pairs(a.dataset, a.n)):
        for tp in [float(x) for x in a.top_ps.split(",")]:
            msg = client.messages.create(model=model, max_tokens=1024, system=system, messages=[dict(role="user", content=user)],
                                         extra_body={"top_p": tp, "seed": 1})
            text = "".join(b.text for b in msg.content if b.type == "text")
            seg = msg.model_dump().get("naivert", {}).get("timings", {}).get("segments", {}).get("total", {})
            rows.append(dict(prompt=i, top_p=tp, tokens=msg.usage.output_tokens, tokens_per_round=round(seg.get("tokens_per_round", 0), 2),
                             device_tok_s=round(seg.get("device_tok_s", 0)), tail=repr(text[-160:])))
            print("TOPP", json.dumps(rows[-1]), flush=True)
    print("TOPP_SWEEP_DONE", flush=True)
    if a.out: json.dump(rows, open(a.out, "w"), indent=1)

if __name__ == "__main__": main()
