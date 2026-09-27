// Local-only frontend host and fixed-upstream streaming proxy. No runtime imports.
import http from 'node:http';
import https from 'node:https';
import {readFile} from 'node:fs/promises';
import {parseArgs} from 'node:util';
import {pathToFileURL} from 'node:url';
import {normalizeEndpoint} from './endpoint.js';

const assets = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/index.html', ['index.html', 'text/html; charset=utf-8']],
  ...['studio.js', 'core.js', 'markdown.js', 'vendor/markdown-it/markdown-it.mjs', 'curve.js', 'endpoint.js', 'queue.js', 'connection.js'].map(name => [`/${name}`, [name, 'text/javascript; charset=utf-8']]),
  ['/vendor/markdown-it/LICENSE', ['vendor/markdown-it/LICENSE', 'text/plain; charset=utf-8']],
  ['/studio.css', ['studio.css', 'text/css; charset=utf-8']],
  ['/logo.png', ['logo.png', 'image/png']],
]);
const requestHeaders = ['content-type', 'authorization', 'x-api-key', 'x-naivert-token-progress', 'anthropic-version', 'x-session-id'];
const responseHeaders = ['content-type', 'x-naivert-warning', 'x-naivert-ignored', 'retry-after'];

export function createStudioServer({upstream, assetRoot = new URL('./dist/', import.meta.url)}) {
  upstream = normalizeEndpoint(upstream);
  if (!upstream) throw new Error('An upstream origin is required.');
  const server = http.createServer(async (req, res) => {
    const port = server.address().port;
    const hosts = [`127.0.0.1:${port}`, `localhost:${port}`];
    const host = req.headers.host;
    if (!hosts.includes(host) || (req.headers.origin && req.headers.origin !== `http://${host}`)) {
      res.writeHead(403).end('This local proxy only accepts its own browser origin.');
      return;
    }
    if ((req.method === 'POST' && req.url === '/v1/messages') || (req.method === 'GET' && req.url === '/health')) {
      const target = new URL(req.url, upstream);
      const headers = Object.fromEntries(requestHeaders.filter(key => req.headers[key] !== undefined).map(key => [key, req.headers[key]]));
      const transport = target.protocol === 'https:' ? https : http;
      let upstreamResponse;
      const request = transport.request(target, {method: req.method, headers}, response => {
        upstreamResponse = response;
        // Do not follow or forward redirects, cookies, or authentication challenges.
        // Credentials are scoped to the fixed configured upstream.
        if (response.statusCode >= 300 && response.statusCode < 400) {
          response.destroy();
          res.writeHead(502, {'Content-Type': 'application/json'}).end(JSON.stringify({error: {message: 'Upstream redirected. Configure its final endpoint instead.'}}));
          return;
        }
        const outgoing = Object.fromEntries(responseHeaders.filter(key => response.headers[key] !== undefined).map(key => [key, response.headers[key]]));
        res.writeHead(response.statusCode, {...outgoing, 'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'});
        res.flushHeaders();
        response.on('error', () => res.destroy());
        response.pipe(res);
      });
      request.setTimeout(req.method === 'GET' ? 5000 : 1_200_000, () => request.destroy(new Error('Upstream timeout')));
      request.on('error', () => {
        if (res.destroyed || res.writableEnded) return;
        if (res.headersSent) res.destroy();
        else res.writeHead(502, {'Content-Type': 'application/json'}).end(JSON.stringify({error: {message: 'Local proxy could not reach the configured upstream.'}}));
      });
      req.on('aborted', () => request.destroy());
      res.on('close', () => {
        if (!res.writableEnded) { upstreamResponse?.destroy(); request.destroy(); }
      });
      req.pipe(request);
      return;
    }
    const asset = assets.get(req.url);
    if (req.method !== 'GET' || !asset) {
      res.writeHead(404).end('Not found');
      return;
    }
    try {
      let body = await readFile(new URL(asset[0], assetRoot));
      if (asset[0] === 'index.html') {
        const escaped = upstream.replaceAll('&', '&amp;').replaceAll('"', '&quot;');
        body = body.toString().replace('<html lang="en">', `<html lang="en" data-local-proxy="${escaped}">`);
      }
      res.writeHead(200, {'Content-Type': asset[1], 'Cache-Control': 'no-store'}).end(body);
    } catch {
      res.writeHead(500).end('Build Studio first: node build.mjs');
    }
  });
  return server;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const {values} = parseArgs({options: {upstream: {type: 'string'}, port: {type: 'string', default: '8110'}}});
  const port = Number(values.port);
  if (!Number.isInteger(port) || port < 1 || port > 65535 || !values.upstream) {
    throw new Error('Usage: node local_server.mjs --upstream http://host:port [--port 8110]');
  }
  const upstream = normalizeEndpoint(values.upstream);
  const server = createStudioServer({upstream});
  server.listen(port, '127.0.0.1', () => console.log(`Studio: http://127.0.0.1:${port}/ — fixed API upstream: ${upstream}`));
}
