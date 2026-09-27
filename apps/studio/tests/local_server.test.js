import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {createStudioServer} from '../local_server.mjs';

async function listen(server, t) {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  return `http://127.0.0.1:${server.address().port}`;
}

test('local proxy forwards auth and body, preserves 401, and rejects foreign origins and redirects', async t => {
  let received;
  const upstream = await listen(http.createServer((req, res) => {
    received = {path: req.url, headers: req.headers};
    if (req.headers.authorization === 'Bearer redirect') return res.writeHead(302, {Location: 'http://example.com'}).end();
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      received.body = body;
      if (req.headers.authorization !== 'Bearer synthetic-test-key') return res.writeHead(401).end('Authentication required');
      res.writeHead(200, {'Content-Type': 'application/json'}).end('{"ok":true}');
    });
  }), t);
  const proxy = await listen(createStudioServer({upstream}), t);
  const denied = await fetch(`${proxy}/v1/messages`, {method:'POST', headers:{Origin:'https://foreign.example'}, body:'{}'});
  assert.equal(denied.status, 403);
  assert.equal(received, undefined);
  const missing = await fetch(`${proxy}/v1/messages`, {method:'POST', body:'{}'});
  assert.equal(missing.status, 401);
  await missing.text();
  const result = await fetch(`${proxy}/v1/messages`, {method:'POST', headers:{Origin:proxy,
    Authorization:'Bearer synthetic-test-key', 'x-api-key':'synthetic-api-key', Cookie:'must-not-forward=1',
    'Content-Type':'application/json', 'x-naivert-token-progress':'1'}, body:'{"stream":true}'});
  assert.equal(result.status, 200);
  await result.text();
  assert.equal(received.path, '/v1/messages');
  assert.equal(received.body, '{"stream":true}');
  assert.equal(received.headers.authorization, 'Bearer synthetic-test-key');
  assert.equal(received.headers['x-api-key'], 'synthetic-api-key');
  assert.equal(received.headers.cookie, undefined);
  assert.equal(received.headers.origin, undefined);
  const redirect = await fetch(`${proxy}/v1/messages`, {method:'POST', headers:{Authorization:'Bearer redirect'}, body:'{}'});
  assert.equal(redirect.status, 502);
  assert.match(await redirect.text(), /redirected/);
});

test('SSE is delivered before completion and cancellation closes upstream', async t => {
  let disconnected;
  const closed = new Promise(resolve => disconnected = resolve);
  const upstream = await listen(http.createServer((req, res) => {
    res.writeHead(200, {'Content-Type':'text/event-stream'});
    res.write('event: ping\ndata: {}\n\n');
    res.on('close', disconnected);
  }), t);
  const proxy = await listen(createStudioServer({upstream}), t);
  const response = await fetch(`${proxy}/v1/messages`, {method:'POST', body:'{}'});
  const reader = response.body.getReader();
  const first = await reader.read();
  assert.match(new TextDecoder().decode(first.value), /event: ping/);
  await reader.cancel();
  await Promise.race([closed, new Promise((_, reject) => {
    const timer = setTimeout(() => reject(new Error('upstream not cancelled')), 2000);
    timer.unref();
  })]);
});

test('local proxy forwards optional health and does not expose other API routes', async t => {
  const upstream = await listen(http.createServer((req, res) => {
    assert.equal(req.method, 'GET');
    assert.equal(req.url, '/health');
    assert.equal(req.headers.authorization, 'Bearer synthetic-health-key');
    assert.equal(req.headers['x-session-id'], 'studio-health-session');
    res.writeHead(200, {'Content-Type':'application/json'}).end('{"status":"ok","queued":3,"slot":"GENERATING"}');
  }), t);
  const proxy = await listen(createStudioServer({upstream}), t);
  const health = await fetch(`${proxy}/health`, {headers:{Authorization:'Bearer synthetic-health-key', 'X-Session-Id':'studio-health-session'}});
  assert.equal(health.status, 200);
  assert.equal(health.headers.get('cache-control'), 'no-store');
  assert.equal((await health.json()).queued, 3);
  assert.equal((await fetch(`${proxy}/ui/config`)).status, 404);
});

test('local Studio serves the bundled Markdown modules and license without a CDN', async t => {
  const proxy = await listen(createStudioServer({upstream:'http://127.0.0.1:9',
    assetRoot:new URL('../', import.meta.url)}), t);
  for (const file of ['markdown.js', 'vendor/markdown-it/markdown-it.mjs']) {
    const response = await fetch(`${proxy}/${file}`);
    assert.equal(response.status, 200);
    assert.match(response.headers.get('content-type'), /text\/javascript/);
    assert.match(await response.text(), /markdown-it/);
  }
  const license = await fetch(`${proxy}/vendor/markdown-it/LICENSE`);
  assert.equal(license.status, 200);
  assert.match(await license.text(), /Permission is hereby granted/);
});
