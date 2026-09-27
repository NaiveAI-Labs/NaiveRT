"""Loopback-only UI fixtures. No model imports, GPU work, or real throughput measurements."""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time

MODEL = "UI fixture · not inference"

SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 670">
<rect width="1000" height="670" fill="#eeeacd"/>
<path d="M0 595 Q230 510 510 600 T1000 545 V670 H0Z" fill="#d9ddbb"/>
<g fill="#385642" font-family="Georgia,serif">
<text x="65" y="70" font-size="15" letter-spacing="5">THE LITTLE FLOWER SHOP</text>
<text x="57" y="255" font-size="181" letter-spacing="-9">Fresh</text>
<text x="65" y="414" font-size="181" letter-spacing="-9" font-style="italic">ideas.</text>
<text x="73" y="485" font-size="20">A little wild. A little wonderful.</text>
<text x="73" y="518" font-size="20">Something good is growing here.</text>
<text x="75" y="612" font-size="12" letter-spacing="3">PICK A STEM. MAKE SOMEONE'S DAY.</text>
</g>
<g fill="none" stroke="#506d43" stroke-width="12" stroke-linecap="round">
<path d="M770 610 Q710 430 786 230 M780 611 Q900 472 908 370 M770 605 Q671 531 650 441"/>
</g>
<g fill="#62844e"><ellipse cx="729" cy="448" rx="30" ry="71" transform="rotate(-45 729 448)"/><ellipse cx="839" cy="491" rx="27" ry="66" transform="rotate(40 839 491)"/></g>
<g transform="translate(793 216)" fill="#e37950">
<ellipse cy="-56" rx="30" ry="56"/><ellipse cy="56" rx="30" ry="56"/><ellipse cx="-56" rx="56" ry="30"/><ellipse cx="56" rx="56" ry="30"/>
<ellipse rx="31" ry="73" transform="rotate(45)"/><ellipse rx="31" ry="73" transform="rotate(-45)"/><circle r="30" fill="#684a32"/><circle r="15" fill="#f3b850"/>
</g>
<g transform="translate(908 368)" fill="#e5af42"><ellipse cy="-32" rx="21" ry="35"/><ellipse cy="32" rx="21" ry="35"/><ellipse cx="-32" rx="35" ry="21"/><ellipse cx="32" rx="35" ry="21"/><circle r="20" fill="#a06640"/></g>
<g transform="translate(650 439)" fill="#d497a0"><ellipse cy="-32" rx="21" ry="35"/><ellipse cy="32" rx="21" ry="35"/><ellipse cx="-32" rx="35" ry="21"/><ellipse cx="32" rx="35" ry="21"/><circle r="19" fill="#e2b354"/></g>
<path d="M657 578 H889 L860 670 H690Z" fill="#bf7153"/>
<text x="727" y="637" fill="#f4dfb6" font-size="22" font-family="Georgia,serif" font-style="italic">bloom.</text>
</svg>'''

HTML = '''<!doctype html><html><head><style>
body{background:#f4ecdf;color:#785446;font-family:Georgia,serif;text-align:center;padding:70px 20px}
p{font:11px system-ui;letter-spacing:3px}h1{font-size:34px;font-weight:400}#timer{font-size:100px;margin:25px}
button{background:#a96751;color:white;border:0;border-radius:30px;padding:14px 30px;margin:6px;cursor:pointer}
small{display:block;margin-top:30px;font:12px system-ui;color:#9a8271}
</style></head><body><p>A MOMENT FOR WHAT MATTERS</p><h1>A little focus goes a long way.</h1>
<div id="timer">25:00</div><button id="start">Start focus</button><button id="reset">Reset</button>
<small id="status">UI fixture · interactive preview, not model output</small>
<script>
let remaining=1500,clock=null;const timer=document.getElementById('timer'),button=document.getElementById('start');
function paint(){timer.textContent=String(Math.floor(remaining/60)).padStart(2,'0')+':'+String(remaining%60).padStart(2,'0')}
button.onclick=()=>{if(clock){clearInterval(clock);clock=null;button.textContent='Resume'}else{button.textContent='Pause';clock=setInterval(()=>{remaining=Math.max(0,remaining-1);paint()},1000)}};
document.getElementById('reset').onclick=()=>{clearInterval(clock);clock=null;remaining=1500;paint();button.textContent='Start focus'};
try{parent.document.body.dataset.previewEscaped='true'}catch(e){document.getElementById('status').dataset.isolated='true'}
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", self.server.cors_origin)
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, x-naivert-token-progress, X-Session-Id, Authorization, x-api-key")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def json_response(self, value, status=200):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        return self.json_response({"error": {"message": "Fixture only exposes POST /v1/messages"}}, 404)

    def do_POST(self):
        if self.path != "/v1/messages":
            return self.json_response({"error": {"message": "Not found"}}, 404)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        mode = self.server.require_auth
        if mode:
            expected = {'bearer': ('Authorization', 'Bearer studio-test-token'),
                        'api-key': ('x-api-key', 'studio-test-token'),
                        'basic': ('Authorization', 'Basic ' + base64.b64encode(b'fixture:studio-test-token').decode())}[mode]
            if self.headers.get(expected[0]) != expected[1]:
                return self.json_response({"error": {"message": "Fixture authentication required"}}, 401)
        request = json.loads(body)
        prompt = request["messages"][-1]["content"].lower()
        if prompt == "error":
            return self.json_response({"error": {"message": "Fixture: simulated capacity error. No model was called."}}, 400)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.close_connection = True
        try:
            self.stream(request, prompt)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def event(self, kind, **data):
        value = {"type": kind, **data}
        self.wfile.write(("event: " + kind + "\ndata: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode())
        self.wfile.flush()

    def stream(self, request, prompt):
        self.wfile.write(b": keepalive\n\n")
        self.wfile.flush()
        time.sleep(.25)
        if self.server.queue_wait_ms:
            time.sleep(self.server.queue_wait_ms / 1000)
        self.event("message_start", naivert={"timings": {"queue_wait_ms": self.server.queue_wait_ms}}, message={"id": "fixture", "usage": {"input_tokens": 0, "output_tokens": 0,
                   "cache_read_input_tokens": 400 if len(request["messages"]) > 1 else 0, "cache_creation_input_tokens": 120}})
        thinking = request.get("thinking", {}).get("type") != "disabled"
        blocks = []
        if thinking:
            blocks.append(("thinking", "This is a UI fixture, not model output or a speed measurement.\nI'll use a warm palette, a clear composition, and a self-contained artifact. 🌷"))
            if 'curve' in prompt:
                blocks[0] = ('thinking', blocks[0][1] * 8)
            if 'scroll' in prompt:
                blocks[0] = ('thinking', blocks[0][1] * 40)
        html = any(word in prompt for word in ("html", "app", "pendulum", "timer"))
        output = f"Here is a {'tiny working timer' if html else 'little something in bloom'}.\n\n```{'html' if html else 'svg'}\n{HTML if html else SVG}\n```"
        if 'tetris' in prompt:
            # Deliberately omit preventDefault: the preview host must retain app key events
            # while preventing arrow/space scrolling. This is a UI fixture, not model output.
            output = '''```html
<!doctype html><html><body style="min-height:1400px;padding:20px">
<h1>Keyboard fixture · not model output</h1>
<p id="position">Piece x: 3, y: 0, rotation: 0</p>
<canvas width="220" height="180" style="background:#e7ecdf"></canvas>
<label>Test input <input aria-label="Test input" value="editable"></label>
<script>
let x=3,y=0,rotation=0;
addEventListener('keydown', event => {
  if(event.target.matches('input')) return;
  if(event.key==='ArrowLeft') x--;
  if(event.key==='ArrowRight') x++;
  if(event.key==='ArrowDown') y++;
  if(event.key==='ArrowUp') rotation++;
  if(event.key===' ') y+=10;
  document.getElementById('position').textContent=`Piece x: ${x}, y: ${y}, rotation: ${rotation}`;
});
</script></body></html>
```'''
        if 'typing' in prompt:
            output = output.replace('\n<svg', '\n' * 50 + '<svg').replace('</svg>',
                '\n'.join(f'<!-- Streaming fixture line {i}: test automatic code following. -->' for i in range(100)) + '\n</svg>')
        if "plain" in prompt:
            output = "A little focus goes a long way.\n\nThis is a plain-text fixture with no artifact.\nLiteral HTML stays text: <img src=x onerror=alert(1)>"
        if "truncate" in prompt:
            output = "```html\n<h1>This code block is unfinished"
        if self.server.answer_file:
            output = self.server.answer_file.read_text()
        blocks.append(("text", output))
        for index, (kind, text) in enumerate(blocks):
            field = "thinking" if kind == "thinking" else "text"
            self.event("content_block_start", index=index, content_block={"type": kind, field: ""})
            for offset in range(0, len(text), 90):
                progress = {}
                if self.headers.get('x-naivert-token-progress') == '1':
                    # Synthetic token frontiers for fixture-only transport arithmetic checks.
                    base = 150 if thinking and kind == 'text' else 0
                    segment_tokens = 150 if kind == 'thinking' else 900
                    count = base + round(segment_tokens * min(offset + 90, len(text)) / len(text))
                    progress = {"naivert": {"generated_tokens": count}}
                self.event("content_block_delta", index=index, delta={"type": field + "_delta", field: text[offset:offset + 90]}, **progress)
                time.sleep(.5 if "slow" in prompt else .025)
                if "disconnect" in prompt:
                    return
            self.event("content_block_stop", index=index)
        segments = {"answer": {"tokens": 900, "device_ms": 1500, "device_tok_s": 600}}
        if thinking:
            segments["think"] = {"tokens": 150, "device_ms": 500, "device_tok_s": 300}
        self.event("message_delta", delta={"stop_reason": "max_tokens" if "truncate" in prompt else "end_turn"},
                   usage={"output_tokens": 1050 if thinking else 900}, naivert={"timings": {"segments": segments, "prompt_ms": 50, "queue_wait_ms": self.server.queue_wait_ms}})
        self.event("message_stop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--cors-origin", default="http://127.0.0.1:8110")
    parser.add_argument("--require-auth", choices=("bearer", "api-key", "basic"), help="require synthetic test credentials")
    parser.add_argument("--answer-file", type=Path, help="Replay saved answer text; no inference")
    parser.add_argument("--queue-wait-ms", type=float, default=0, help="Simulated queue wait, excluded from simulated TTFT")
    args = parser.parse_args()
    print(f"UI FIXTURE ONLY — API http://127.0.0.1:{args.port} — all output and metrics are test data", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.cors_origin = args.cors_origin
    server.require_auth = args.require_auth
    server.answer_file = args.answer_file
    server.queue_wait_ms = args.queue_wait_ms
    server.serve_forever()
