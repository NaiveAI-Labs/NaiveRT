import test from 'node:test';
import assert from 'node:assert/strict';
import {setImmediate as tick} from 'node:timers/promises';
import {QueueMonitor, queueSnapshot} from '../queue.js';

const response = body => ({ok:true, json:async () => body});

test('queue failures distinguish HTTP rejection, invalid JSON, and blocked network requests', async t => {
  for (const [fetcher, detail] of [
    [async () => ({ok:false, status:401}), 'Queue request failed (HTTP 401)'],
    [async () => ({ok:true, json:async () => { throw new SyntaxError('invalid'); }}), 'Queue endpoint returned invalid JSON'],
    [async () => { throw new TypeError('Failed to fetch'); }, 'Queue request blocked or unreachable; check gateway CORS and network'],
  ]) {
    const updates = [];
    const monitor = new QueueMonitor(value => updates.push(value), {fetcher, intervalMs:60_000});
    t.after(() => monitor.stop());
    monitor.start('https://example.com/health');
    await tick();
    monitor.stop();
    assert.deepEqual(updates.at(-1), {queued:null, detail});
  }
});

test('queue count is waiting requests; zero is distinct from unavailable', () => {
  assert.deepEqual(queueSnapshot({status:'ok', queued:0, slot:'GENERATING'}), {queued:0, detail:'Server busy'});
  assert.deepEqual(queueSnapshot({status:'ok', queued:3, slot:'IDLE'}), {queued:3, detail:'Server idle'});
  for (const body of [null, {}, {status:'ok'}, {status:'ok', queued:-1}, {status:'ok', queued:'2'}, {status:'ok', queued:true}]) {
    assert.equal(queueSnapshot(body).queued, null);
  }
  assert.equal(queueSnapshot({status:'loading'}).detail, 'Server loading');
});

test('queue polling applies credentials and clears a previously valid count on failure', async t => {
  const updates = [];
  let polls = 0, finish;
  const done = new Promise(resolve => finish = resolve);
  const monitor = new QueueMonitor(value => {
    updates.push(value);
    if (polls === 2) { monitor.stop(); finish(); }
  }, {intervalMs:5, fetcher: async (url, options) => {
    assert.equal(url, 'https://example.com/health');
    assert.equal(options.headers.Authorization, 'Bearer synthetic');
    assert.equal(options.redirect, 'error');
    if (++polls === 1) return response({status:'ok', queued:4});
    throw new Error('unavailable');
  }});
  t.after(() => monitor.stop());
  monitor.start('https://example.com/health', {headers:{Authorization:'Bearer synthetic'}, redirect:'error'});
  await done;
  assert.deepEqual(updates.map(value => value.queued), [null,4,null]);
});

test('changing endpoints aborts the old poll and ignores its late response', async t => {
  const pending = new Map(), updates = [];
  const monitor = new QueueMonitor(value => updates.push(value), {intervalMs:60_000,
    fetcher: (url, options) => new Promise(resolve => pending.set(url, {resolve, signal:options.signal}))});
  t.after(() => monitor.stop());
  monitor.start('first');
  monitor.start('second');
  assert.equal(pending.get('first').signal.aborted, true);
  pending.get('second').resolve(response({status:'ok', queued:2}));
  await tick();
  pending.get('first').resolve(response({status:'ok', queued:99}));
  await tick();
  assert.deepEqual(updates.map(value => value.queued), [null,null,2]);
});
