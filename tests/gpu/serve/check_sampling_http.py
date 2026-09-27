"""Runtime sampling controls through the API on a running serve_sessions server.

1. temperature 0.4 on the first N CC-D prompts: output token counts must equal the build-time T=0.4
   run recorded in hitrate_ccd04.json round 1 (same seed, same math).
2. seed: same prompt with seed 1 twice -> identical text; seed 2 -> different.
3. top_p: 0.01 makes the answer independent of the seed (nucleus = argmax); 1.0 differs from 0.95.
4. rejections: top_k, temperature 0, top_p 0, seed -1 -> 400.
"""
import argparse
import json
import re
import sys
import time

import anthropic
import httpx


def prompt_pairs(dataset, n):
    out = []
    for line in open(dataset):
        row = json.loads(line)
        systems = re.findall(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", row["prompt_text"], re.S)
        m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", row["prompt_text"], re.S)
        if m:
            out.append((systems[-1] if systems else None, m.group(1)))
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reference", required=True, help="hitrate_ccd04.json from the build-time T=0.4 run")
    ap.add_argument("--n", type=int, default=6)
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
    client = anthropic.Anthropic(base_url=a.url, api_key="none", timeout=900, max_retries=0)
    pairs = prompt_pairs(a.dataset, a.n)
    ref = {r["sid"]: r for r in json.load(open(a.reference))["rows"] if r["round"] == 1}
    report = {}

    def ask(system, user, **extra):
        msg = client.messages.create(model=model, max_tokens=extra.pop("max_tokens", 8192), system=system,
                                     messages=[dict(role="user", content=user)], extra_body=extra)
        text = "".join(b.text for b in msg.content if b.type == "text")
        seg = msg.model_dump().get("naivert", {}).get("timings", {}).get("segments", {}).get("total", {})
        return dict(tokens=msg.usage.output_tokens, stop=msg.stop_reason, text=text,
                    tokens_per_round=round(seg.get("tokens_per_round", 0), 3), device_tok_s=round(seg.get("device_tok_s", 0)))

    # 1. temperature 0.4 vs the build-time run
    rows = []
    for i, (system, user) in enumerate(pairs):
        r = ask(system, user, temperature=0.4)
        exp = ref[f"s{i:02d}"]["output_tokens"]
        rows.append(dict(prompt=i, tokens=r["tokens"], reference=exp, match=r["tokens"] == exp, stop=r["stop"],
                         tokens_per_round=r["tokens_per_round"], device_tok_s=r["device_tok_s"]))
        print("T0.4", json.dumps(rows[-1]), flush=True)
    report["temperature_0.4_vs_buildtime"] = dict(rows=rows, matches=sum(r["match"] for r in rows), n=len(rows))
    # 2. seed
    system, user = pairs[0]
    s1 = ask(system, user, seed=1, max_tokens=1024); s1b = ask(system, user, seed=1, max_tokens=1024); s2 = ask(system, user, seed=2, max_tokens=1024)
    report["seed"] = dict(same_seed_identical=s1["text"] == s1b["text"], different_seed_differs=s1["text"] != s2["text"])
    print("SEED", json.dumps(report["seed"]), flush=True)
    # 3. top-p
    g1 = ask(system, user, top_p=0.01, seed=1, max_tokens=1024); g2 = ask(system, user, top_p=0.01, seed=2, max_tokens=1024)
    p1 = ask(system, user, top_p=1.0, seed=1, max_tokens=1024)
    report["top_p"] = dict(top_p_0_01_seed_independent=g1["text"] == g2["text"], top_p_1_differs_from_0_95=p1["text"] != s1["text"])
    print("TOPP", json.dumps(report["top_p"]), flush=True)
    # 4. rejections
    rejected = {}
    for name, extra in (("top_k", {"top_k": 5}), ("temperature_0", {"temperature": 0}), ("top_p_0", {"top_p": 0}), ("seed_neg", {"seed": -1})):
        try:
            client.messages.create(model=model, max_tokens=8, messages=[dict(role="user", content="x")], extra_body=extra)
            rejected[name] = False
        except anthropic.BadRequestError:
            rejected[name] = True
    report["rejections"] = rejected
    print("REJECT", json.dumps(rejected), flush=True)
    ok = (report["temperature_0.4_vs_buildtime"]["matches"] == len(rows) and all(report["seed"].values())
          and all(report["top_p"].values()) and all(rejected.values()))
    print("SAMPLING_HTTP_OK" if ok else "SAMPLING_HTTP_FAIL", flush=True)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
