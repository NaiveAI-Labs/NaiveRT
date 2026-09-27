"""Claude-shaped Messages requests over the real engine's HTTP worker."""
import socket
import threading
import time

import httpx
import uvicorn

from naivert.serve.chat import ChatFormatter
from naivert.serve.context import ServerContext
from naivert.serve.http import build_app


def check(engine, tokenizer):
    engine.store.entries.clear()
    app = build_app(lambda: ServerContext(engine, ChatFormatter(tokenizer)),
                    served_model='naive-n0.5', queue_depth=8, ping_seconds=.2)
    sock = socket.socket(); sock.bind(('127.0.0.1', 0))
    url = f'http://127.0.0.1:{sock.getsockname()[1]}'
    server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='on'))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    rows = []
    try:
        deadline = time.monotonic() + 30
        while not server.started:
            assert time.monotonic() < deadline, 'HTTP worker startup timed out'
            time.sleep(.1)
        with httpx.Client(base_url=url, timeout=180) as client:
            def request(label, messages):
                body = dict(model='naive-n0.5', max_tokens=32, temperature=.01,
                    system='Give a short answer to the user.', messages=messages, tools=[],
                    thinking=dict(type='adaptive'), metadata=dict(user_id='shared-parent-key'))
                r = client.post('/v1/messages', json=body)
                assert r.status_code == 200, (r.status_code, r.text[:500])
                d = r.json()
                usage = d['usage']
                rows.append(dict(label=label, usage=usage, stop_reason=d['stop_reason'],
                                 timings=d.get('naivert', {}).get('timings', {})))
                # Claude Code can omit empty / whitespace-only thinking or text.
                blocks = [b for b in d['content'] if b['type'] not in ('thinking', 'text')
                          or b.get('thinking', b.get('text', '')).strip()]
                if not blocks: blocks = [dict(type='text', text='')]
                return blocks
            parent = [dict(role='user', content='Remember this label: parent-cedar. Reply briefly.')]
            answer = request('parent-first', parent)
            sub = [dict(role='user', content='Remember this label: subagent-birch. Reply briefly.')]
            sub_answer = request('subagent-first-same-key', sub)
            # Retry the identical request after another branch used the same key.
            request('parent-identical-retry', parent)
            parent += [dict(role='assistant', content=answer),
                       dict(role='user', content='What label did I give you?')]
            request('parent-followup', parent)
            sub += [dict(role='assistant', content=sub_answer),
                    dict(role='user', content='What label did I give you?')]
            request('subagent-followup', sub)
        assert all(r['usage'].get('cache_read_input_tokens', 0) > 0 for r in rows[2:]), rows
        return rows
    finally:
        server.should_exit = True
        thread.join(timeout=30)
        assert not thread.is_alive(), 'HTTP worker did not stop'
        sock.close()
