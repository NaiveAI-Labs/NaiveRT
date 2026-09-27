import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeEndpoint, apiURL, loadCredentials, saveCredentials, clearCredentials, healthOptions, messagesOptions} from '../endpoint.js';

test('health polling uses the same authenticated session as generation without a GET body', () => {
  const auth = {type:'bearer', secret:'synthetic'};
  const health = healthOptions(auth, 'studio-chat-1');
  const messages = messagesOptions(auth, 'studio-chat-1');
  assert.equal(health.method, 'GET');
  assert.equal(health.body, undefined);
  assert.equal(health.headers.Authorization, messages.headers.Authorization);
  assert.equal(health.headers['X-Session-Id'], messages.headers['X-Session-Id']);
  assert.equal(healthOptions(auth, 'studio-chat-2').headers['X-Session-Id'], 'studio-chat-2');
  assert.equal(health.redirect, 'error');
});

function memoryStorage() {
  const values = new Map();
  return {getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key)};
}

test('applied credentials survive reload only for the same API origin; clear removes them', () => {
  const storage = memoryStorage();
  for (const auth of [
    {type: 'bearer', secret: 'synthetic-token', username: '', cookies: false},
    {type: 'api-key', secret: 'synthetic-key', username: '', cookies: true},
    {type: 'basic', secret: 'synthetic-password', username: 'user', cookies: false},
  ]) {
    assert.equal(saveCredentials('https://api.example.com/v1/messages', auth, storage), true);
    assert.deepEqual(loadCredentials('https://api.example.com', storage), auth);
    for (const other of ['http://api.example.com', 'https://api.example.com:20000', 'https://other.example.com', '']) {
      assert.deepEqual(loadCredentials(other, storage), {});
    }
    assert.equal(clearCredentials(storage), true);
    assert.deepEqual(loadCredentials('https://api.example.com', storage), {});
  }
});

test('new endpoint replaces the previous credential and None discards secrets', () => {
  const storage = memoryStorage();
  saveCredentials('https://one.example.com', {type: 'bearer', secret: 'old-key'}, storage);
  saveCredentials('https://two.example.com', {type: 'none', secret: 'unused', username: 'unused'}, storage);
  assert.deepEqual(loadCredentials('https://one.example.com', storage), {});
  assert.deepEqual(loadCredentials('https://two.example.com', storage), {type: 'none', secret: '', username: '', cookies: false});
  assert.ok(!storage.getItem('naivert-studio-credentials').includes('old-key'));
  assert.ok(!storage.getItem('naivert-studio-credentials').includes('unused'));
});

test('unavailable or malformed credential storage does not break connection setup', () => {
  const unavailable = {getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); }, removeItem() { throw new Error('blocked'); }};
  assert.deepEqual(loadCredentials('https://api.example.com', unavailable), {});
  assert.equal(saveCredentials('https://api.example.com', {type: 'bearer', secret: 'test'}, unavailable), false);
  assert.equal(clearCredentials(unavailable), false);
  const storage = memoryStorage();
  for (const value of ['invalid JSON', 'null', JSON.stringify({endpoint: 'https://api.example.com', auth: {type: 'bearer'}}),
    JSON.stringify({endpoint: 'https://api.example.com', auth: {type: 'bearer', secret: 'a\nb', username: '', cookies: false}})]) {
    storage.setItem('naivert-studio-credentials', value);
    assert.deepEqual(loadCredentials('https://api.example.com', storage), {});
  }
});

test('server origins normalize and route all API calls without changing schemes or ports', () => {
  assert.equal(normalizeEndpoint('  https://API.example.com/ '), 'https://api.example.com');
  assert.equal(normalizeEndpoint(''), '');
  assert.equal(normalizeEndpoint('http://127.0.0.1:18090/v1/messages'), 'http://127.0.0.1:18090');
  assert.equal(normalizeEndpoint('http://127.0.0.1:18090/v1/messages/'), 'http://127.0.0.1:18090');
  const endpoint = normalizeEndpoint('http://127.0.0.1:18090/');
  for (const path of ['/health', '/ui/config', '/v1/messages']) {
    assert.equal(apiURL(endpoint, path), `http://127.0.0.1:18090${path}`);
    assert.equal(apiURL('', path), path);
  }
});

test('endpoint input rejects credentials, non-HTTP schemes, and unrelated paths', () => {
  for (const value of ['localhost:18090', 'javascript:alert(1)', 'file:///tmp/a',
    'https://user:secret@example.com', 'https://example.com/v1/models',
    'https://example.com/?key=secret', 'https://example.com/#fragment']) {
    assert.throws(() => normalizeEndpoint(value), Error);
  }
});

test('credentials choose exactly one auth header and cookies are opt-in', async () => {
  const {authOptions} = await import('../endpoint.js');
  assert.deepEqual(authOptions(), {headers: {}, credentials: 'omit', redirect: 'error'});
  assert.deepEqual(authOptions({type: 'bearer', secret: 'test-token'}).headers, {Authorization: 'Bearer test-token'});
  assert.deepEqual(authOptions({type: 'api-key', secret: 'test-key'}).headers, {'x-api-key': 'test-key'});
  const basic = authOptions({type: 'basic', username: 'user', secret: 'páss:word'});
  assert.equal(Buffer.from(basic.headers.Authorization.slice(6), 'base64').toString('utf8'), 'user:páss:word');
  assert.equal(authOptions({cookies: true}).credentials, 'include');
  assert.deepEqual(authOptions({type: 'none', secret: 'discarded'}).headers, {});
});

test('credential validation rejects empty secrets, header injection, and invalid basic usernames', async () => {
  const {authOptions} = await import('../endpoint.js');
  for (const auth of [{type: 'bearer'}, {type: 'api-key', secret: 'a\r\nb'},
    {type: 'basic', secret: 'pw'}, {type: 'basic', username: 'a:b', secret: 'pw'}, {type: 'unknown'}]) {
    assert.throws(() => authOptions(auth), Error);
  }
});
