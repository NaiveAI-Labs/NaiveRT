"""Ceiling of the HTTP streaming path without a GPU.

A fake engine hands out tokens in bursts of ``--burst`` (one commit round) with an optional
per-round delay, through the real ServerContext, Worker, FastAPI app and uvicorn on loopback. The
client parses SSE and reports tokens per second as seen on the wire, deltas per second, and tokens
per delta. ``--round-ms 0`` measures the pure host ceiling; ``--round-ms 3.4`` mimics the device.
"""
import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/host"))
from naivert.serve.context import ServerContext  # noqa: E402
from naivert.serve.http import build_app  # noqa: E402
from test_serve_context import EOS, FakeFormatter  # noqa: E402


class BurstSession:
    def __init__(self, n, burst, round_s):
        self.n, self.burst, self.round_s, self.i = n, burst, round_s, 0
        self.max_decode_tokens = n + 1
        self.runtime = SimpleNamespace(prepared_prefill=SimpleNamespace(as_dict=lambda: {"prefill_ms": 1.0, "start_position": 0}),
                                       output_reader=SimpleNamespace(progress_records=[]),
                                       graph_contract_evidence={"device_token_chain": True})
        self.mode = 1
        self.t_round = time.monotonic_ns()

    def step(self):
        if self.i >= self.n:
            self.i += 1
            return EOS
        if self.i % self.burst == 0 and self.round_s:
            time.sleep(self.round_s)
            self.t_round = time.monotonic_ns()
        self.i += 1
        self.runtime.output_reader.progress_records.append(
            {"parent_start_ns": self.t_round - 1, "parent_complete_ns": self.t_round})
        return 1 + (self.i % 26)

    def finish(self):
        pass

    def close(self):
        pass


class BurstEngine:
    def __init__(self, n, burst, round_s):
        self.n, self.burst, self.round_s = n, burst, round_s
        self.config = SimpleNamespace(max_decode_tokens=n + 2, model_spec=SimpleNamespace(stop_token_ids=(EOS,)), mode=1)
        self._backend = SimpleNamespace(native_mtp_max_cur_pos=10 ** 9, min_prompt_tokens=1)
        self.request_generation = 0

    def open_session(self, tokens, *, n_past=0, max_decode_tokens=None, **_):
        self.request_generation += 1
        return BurstSession(self.n, self.burst, self.round_s)

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=40000)
    ap.add_argument("--burst", type=int, default=6)
    ap.add_argument("--round-ms", type=float, default=0.0)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    app = build_app(lambda: ServerContext(BurstEngine(a.tokens, a.burst, a.round_ms / 1e3), FakeFormatter()),
                    served_model="naive-n0.5", queue_depth=4)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True).start()
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    while True:
        try:
            if httpx.get(url + "/health", timeout=2).status_code == 200:
                break
        except Exception:
            time.sleep(.05)
    body = dict(model="naive-n0.5", max_tokens=a.tokens + 1, messages=[dict(role="user", content="x")], stream=True)
    for rep in range(a.repeat):
        deltas = tokens = 0
        t0 = time.monotonic()
        first = None
        with httpx.Client(timeout=600) as c, c.stream("POST", url + "/v1/messages", json=body) as r:
            for line in r.iter_lines():
                if line.startswith("event: content_block_delta"):
                    deltas += 1
                    if first is None:
                        first = time.monotonic()
                elif line.startswith("data: ") and '"message_delta"' in line:
                    tokens = json.loads(line[6:])["usage"]["output_tokens"]
        dt = time.monotonic() - first
        print(json.dumps(dict(rep=rep, tokens=tokens, deltas=deltas, seconds=round(dt, 3), tok_s=round(tokens / dt),
                              deltas_s=round(deltas / dt), tokens_per_delta=round(tokens / max(1, deltas), 2))), flush=True)
    body["stream"] = False
    t0 = time.monotonic()
    r = httpx.post(url + "/v1/messages", json=body, timeout=600).json()
    dt = time.monotonic() - t0
    print(json.dumps(dict(non_stream=True, tokens=r["usage"]["output_tokens"], seconds=round(dt, 3), tok_s=round(r["usage"]["output_tokens"] / dt))), flush=True)
    app.state.worker.stop()
    server.should_exit = True


if __name__ == "__main__":
    main()
