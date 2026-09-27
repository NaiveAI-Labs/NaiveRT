import test from 'node:test';
import assert from 'node:assert/strict';
import {checkConnection} from '../connection.js';
import {parseQuickSetup, messagesOptions} from '../endpoint.js';

test('quick setup accepts URL, bearer key and optional model without evaluating commands', () => {
  assert.deepEqual(parseQuickSetup(' https://example.com/v1/messages sk-fixture model-test '),
    {endpoint:'https://example.com', auth:{type:'bearer',secret:'sk-fixture'}, model:'model-test'});
  assert.deepEqual(parseQuickSetup('http://localhost:8090'),
    {endpoint:'http://localhost:8090',auth:{type:'none'},model:null});
  assert.equal(parseQuickSetup('https://example.com sk-fixture').model, null);
  for (const value of ['', 'curl https://example.com', 'https://user:secret@example.com',
    'https://example.com?key=secret', 'https://example.com key model extra']) assert.throws(() => parseQuickSetup(value));
});

const settings = {endpoint:'https://example.com',model:'model-test',auth:{type:'bearer',secret:'synthetic'},sessionId:'test-session'};
const stream = (...events) => new Response(events.map(e => `data: ${JSON.stringify(e)}\n\n`).join(''),
  {headers:{'Content-Type':'text/event-stream'}});
const finish = {type:'message_delta',delta:{stop_reason:'max_tokens'},usage:{output_tokens:1}};

test('connection probe uses the generation headers and verifies a complete inference stream', async () => {
  const result = await checkConnection({...settings, fetcher:async (url, options) => {
    assert.equal(url, 'https://example.com/v1/messages');
    assert.equal(options.method, 'POST');
    assert.deepEqual(options.headers, messagesOptions(settings.auth, settings.sessionId).headers);
    assert.equal(options.headers.Authorization, 'Bearer synthetic');
    assert.equal(options.headers['X-Session-Id'], 'test-session');
    assert.equal(options.redirect, 'error');
    assert.equal(options.credentials, 'omit');
    const body = JSON.parse(options.body);
    assert.equal(body.max_tokens, 1);
    assert.equal(body.model, 'model-test');
    assert.equal(body.thinking.type, 'disabled');
    assert.equal(body.metadata.user_id, 'test-session');
    return stream({type:'message_start',message:{usage:{input_tokens:2}}}, finish, {type:'message_stop'});
  }});
  assert.equal(result.ok, true);
  assert.match(result.message,/streaming verified/);
});

test('connection probe distinguishes auth, overload, loading and browser network failures', async () => {
  for (const [status, message] of [[401,/Authentication/],[403,/Authentication/],[429,/queue/],[529,/queue/],[503,/loading/],[400,/model/]]) {
    const result = await checkConnection({...settings, fetcher:async () => new Response('error',{status})});
    assert.equal(result.ok,false);
    assert.match(result.message,message);
  }
  const result = await checkConnection({...settings,fetcher:async () => { throw new TypeError('fetch failed'); }});
  assert.equal(result.ok,false);
  assert.match(result.message,/CORS \/ OPTIONS/);
});

test('HTTP 200 with HTML, incomplete SSE or a stream error is not healthy', async () => {
  for (const response of [new Response('<html>login</html>'), stream(finish), stream({type:'error',error:{message:'secret must not be echoed'}})]) {
    const result = await checkConnection({...settings,fetcher:async () => response});
    assert.equal(result.ok,false);
    assert.doesNotMatch(result.message,/secret/);
  }
});

test('connection probe times out and aborts the request', async () => {
  let signal;
  const result = await checkConnection({...settings, timeoutMs:5, fetcher:(_, options) => new Promise((resolve,reject) => {
    signal = options.signal;
    signal.addEventListener('abort', () => reject(new DOMException('aborted','AbortError')));
  })});
  assert.equal(signal.aborted,true);
  assert.equal(result.ok,false);
  assert.match(result.message,/timed out/);
});
