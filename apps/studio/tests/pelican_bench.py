"""Compare direct/proxied pelican requests. GPU inference goes to the explicitly supplied server."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

PROMPT = "Generate an SVG of a pelican riding a bicycle. Reply with only the complete SVG markup, no explanation."


def run(endpoint, seed, stream=True):
    url = urlsplit(endpoint)
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=1200)
    request = {"model": "naive-n0.5", "max_tokens": 131072, "seed": seed,
               "thinking": {"type": "disabled", "budget_tokens": 16384},
               "messages": [{"role": "user", "content": PROMPT}], "stream": stream}
    start = time.perf_counter()
    connection.request("POST", "/v1/messages", json.dumps(request),
                       {"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    response = connection.getresponse()
    headers_ms = (time.perf_counter() - start) * 1000
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read(2048)!r}")
    first_ms = last_ms = None
    output = []
    usage = {}
    timings = None
    complete = False
    deltas = 0
    stop_reason = None
    if stream:
        data = []
        while line := response.readline():
            if line.startswith(b"data:"):
                data.append(line[5:].strip())
            elif line in (b"\n", b"\r\n") and data:
                event = json.loads(b"\n".join(data))
                data = []
                now_ms = (time.perf_counter() - start) * 1000
                kind = event["type"]
                if kind == "error":
                    raise RuntimeError(event["error"])
                if kind == "message_start":
                    usage.update(event["message"]["usage"])
                elif kind == "content_block_delta" and event["delta"].get("text"):
                    output.append(event["delta"]["text"])
                    first_ms = now_ms if first_ms is None else first_ms
                    last_ms = now_ms
                    deltas += 1
                elif kind == "message_delta":
                    usage.update(event["usage"])
                    timings = event.get("naivert", {}).get("timings")
                    stop_reason = event["delta"].get("stop_reason")
                elif kind == "message_stop":
                    complete = True
    else:
        message = json.loads(response.read())
        output = [block.get("text", "") for block in message["content"]]
        usage = message["usage"]
        timings = message.get("naivert", {}).get("timings")
        stop_reason = message["stop_reason"]
        complete = True
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.close()
    connection.close()
    if not complete or not timings:
        raise RuntimeError("Incomplete response or missing backend metrics")
    text = "".join(output)
    return {"endpoint": endpoint, "request": request, "usage": usage, "timings": timings,
            "client": {"headers_ms": headers_ms, "first_content_ms": first_ms, "last_content_ms": last_ms,
                       "elapsed_ms": elapsed_ms, "request_tok_s": usage["output_tokens"] / (elapsed_ms / 1000),
                       "delta_events": deltas}, "stop_reason": stop_reason,
            "response_sha256": hashlib.sha256(text.encode()).hexdigest(), "response_chars": len(text)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--nonstream", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for iteration in range(args.runs):
        for index, endpoint in enumerate(args.endpoint):
            result = run(endpoint, args.seed, stream=not args.nonstream)
            name = f"{'json' if args.nonstream else 'sse'}-{index}-{iteration}.json"
            (args.out / name).write_text(json.dumps(result, indent=2) + "\n")
            segments = result["timings"].get("segments") or {}
            print(json.dumps({"file": name, "endpoint": endpoint, "tokens": result["usage"]["output_tokens"],
                              "elapsed_ms": round(result["client"]["elapsed_ms"], 2),
                              "ttft_ms": result["client"]["first_content_ms"],
                              "backend_wall_tps": result["timings"]["predicted_per_second"],
                              "device_tps": segments.get("answer", {}).get("device_tok_s"),
                              "sha256": result["response_sha256"]}), flush=True)
