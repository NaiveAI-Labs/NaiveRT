import test from 'node:test';
import assert from 'node:assert/strict';
import {consumeSSE, Reply, StreamRate, splitAnswer, previewDocument, withNewSeed, drawSeed} from '../core.js';

test('stream rate excludes the entire first receive batch and freezes at last content', () => {
  const rate = new StreamRate();
  rate.observe(2, 4000);
  rate.observe(8, 4000); // second SSE event in the same fetch read
  rate.observe(18, 4100);
  rate.observe(38, 4200);
  assert.equal(rate.snapshot().tokens, 30);
  assert.equal(rate.snapshot().interval_ms, 200);
  assert.equal(rate.snapshot().tok_s, 150);
  const reply = new Reply();
  reply.streamRates.total = rate;
  reply.accept({type: 'message_delta', delta: {stop_reason: 'end_turn'}, usage: {output_tokens: 39}}, 9000);
  reply.accept({type: 'message_stop'}, 9500);
  assert.equal(reply.streamRates.total.snapshot().tok_s, 150);
});

test('one batch, missing counters, and regressing counters never produce a fabricated rate', () => {
  const rate = new StreamRate();
  rate.observe(10, 1);
  rate.observe(20, 1);
  assert.equal(rate.snapshot().tok_s, null);
  rate.observe(undefined, 2);
  rate.observe(30, 3);
  assert.equal(rate.snapshot().tok_s, null);
  const regressing = new StreamRate();
  regressing.observe(20, 1);
  regressing.observe(10, 2);
  assert.equal(regressing.snapshot().tok_s, null);
});

test('curve uses rolling receive frontiers and retains a bounded window', () => {
  const rate = new StreamRate();
  for (let ms = 0; ms <= 2000; ms += 20) rate.observe(ms, ms);
  assert(rate.curve().length <= 11);
  assert(rate.curve().every(point => point.tok_s === 1000));
  assert(rate.window.length <= 52);
  rate.observe(2001, 3000); // a real gap in delivery lowers the rolling rate
  assert.equal(rate.curve().at(-1).tok_s, 1);
  const before = structuredClone(rate.curve());
  rate.snapshot(); // asking for metrics after generation does not append finalization time
  assert.deepEqual(rate.curve(), before);
});

test('short or unsupported streams have no fabricated curve; equal timestamps coalesce', () => {
  const rate = new StreamRate();
  rate.observe(4, 100);
  rate.observe(8, 100);
  rate.observe(108, 200);
  assert.deepEqual(rate.curve(), []);
  rate.observe(208, 300);
  assert.equal(rate.curve()[0].tok_s, 1000);
  rate.observe(308, 300);
  assert.equal(rate.curve().length, 1);
  assert.equal(rate.curve()[0].tok_s, 1500);
  rate.observe(undefined, 400);
  assert.deepEqual(rate.curve(), []);
});

test('thinking and answer rates use separate receive intervals', () => {
  const reply = new Reply();
  for (const [index, kind, field, samples] of [[0, 'thinking', 'thinking', [[5, 1000], [15, 1100]]], [1, 'text', 'text', [[20, 1200], [50, 1300]]]]) {
    reply.accept({type:'content_block_start',index,content_block:{type:kind,[field]:''}}, samples[0][1]);
    for (const [count, time] of samples) reply.accept({type:'content_block_delta',index,
      delta:{type:`${field}_delta`,[field]:'chunk'}, naivert:{generated_tokens:count}}, time);
  }
  assert.equal(reply.streamRates.thinking.snapshot().tok_s, 100);
  assert.equal(reply.streamRates.text.snapshot().tok_s, 300);
  assert.equal(reply.streamRates.total.snapshot().tok_s, 150);
});

function responseOf(parts) {
  return new Response(new ReadableStream({start(controller) {
    for (const part of parts) controller.enqueue(part);
    controller.close();
  }}));
}

test('SSE handles every-byte boundaries, UTF-8, CRLF, comments and multiline data', async () => {
  const source = ': keepalive\r\n\r\nevent: ping\r\ndata: {"type":"ping"}\r\n\r\ndata: {"type":"text",\r\ndata: "text":"你好 🌷"}\r\n\r\n';
  const bytes = new TextEncoder().encode(source);
  const events = [];
  await consumeSSE(responseOf([...bytes].map(byte => Uint8Array.of(byte))), event => events.push(event));
  assert.deepEqual(events, [{type: 'ping'}, {type: 'text', text: '你好 🌷'}]);
});

test('SSE cancels the body when an error event aborts processing', async () => {
  let cancelled = false;
  const response = new Response(new ReadableStream({
    start(controller) { controller.enqueue(new TextEncoder().encode('data: {"type":"error"}\n\n')); },
    cancel() { cancelled = true; },
  }));
  await assert.rejects(consumeSSE(response, () => { throw new Error('server failure'); }), /server failure/);
  assert.equal(cancelled, true);
});

test('SSE events decoded from one read share the receive timestamp', async () => {
  const times = [];
  await consumeSSE(responseOf([new TextEncoder().encode('data: {"type":"ping"}\n\ndata: {"type":"ping"}\n\n')]),
    (_, meta) => times.push(meta.receivedAt));
  assert.equal(times.length, 2);
  assert.equal(times[0], times[1]);
});

function addBlock(reply, type, text, index, time) {
  const field = type === 'thinking' ? 'thinking' : 'text';
  reply.accept({type: 'content_block_start', index, content_block: {type, [field]: ''}}, time - 1);
  reply.accept({type: 'content_block_delta', index, delta: {type: `${field}_delta`, [field]: text}}, time);
  reply.accept({type: 'content_block_stop', index}, time + 1);
}

test('thinking is first response; ordered content blocks survive multi-turn replay', () => {
  const reply = new Reply();
  reply.accept({type: 'message_start', message: {id: 'fixture', usage: {input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 30, cache_creation_input_tokens: 20}}}, 10);
  reply.accept({type: 'ping'}, 15);
  assert.equal(reply.ttft, null);
  assert.equal(reply.usage.output_tokens, undefined);
  addBlock(reply, 'thinking', 'Consider 🌷', 0, 25);
  addBlock(reply, 'text', 'First answer', 1, 90);
  addBlock(reply, 'text', 'More text', 2, 100);
  assert.equal(reply.ttft, 25);
  assert.equal(reply.firstAnswer, 90);
  assert.equal(reply.inputTokens, 50);
  assert.deepEqual(reply.blocks, [{type: 'thinking', thinking: 'Consider 🌷'}, {type: 'text', text: 'First answer'}, {type: 'text', text: 'More text'}]);
  assert.equal(reply.text, 'First answer\nMore text');
  reply.accept({type: 'message_delta', delta: {stop_reason: 'end_turn'}, usage: {output_tokens: 42}, naivert: {timings: {segments: {answer: {tokens: 32}}}}}, 120);
  assert.equal(reply.valid, false);
  reply.accept({type: 'message_stop'}, 121);
  assert.equal(reply.valid, true);
  assert.equal(reply.timings.segments.answer.tokens, 32);
});

test('missing final usage or stop rejects a truncated response', () => {
  const reply = new Reply();
  addBlock(reply, 'text', 'partial', 0, 10);
  reply.accept({type: 'message_stop'}, 11);
  assert.equal(reply.valid, false);
  assert.throws(() => reply.accept({type: 'error', error: {message: 'Out of capacity'}}, 12), /Out of capacity/);
});

test('answer-only responses and empty deltas do not start TTFT early', () => {
  const reply = new Reply();
  addBlock(reply, 'text', '', 0, 10);
  assert.equal(reply.ttft, null);
  reply.accept({type: 'content_block_delta', index: 0, delta: {type: 'text_delta', text: 'Hello'}}, 200);
  assert.equal(reply.ttft, 200);
  assert.equal(reply.firstAnswer, 200);
  assert.equal(reply.inputTokens, null);
});

test('artifacts cannot render until their closing fence arrives', () => {
  const text = 'Here it is.\n```svg\n<svg><text>你好</text></svg>\n';
  assert.equal(splitAnswer(text).at(-1).complete, false);
  assert.equal(splitAnswer(text, true).at(-1).complete, false);
  const parts = splitAnswer(text + '```\nEnjoy.');
  assert.equal(parts[1].kind, 'svg');
  assert.equal(parts[1].complete, true);
  assert.equal(parts[1].code, '<svg><text>你好</text></svg>');
  assert.equal(parts[2].text.trim(), 'Enjoy.');
});

test('HTML, SVG, ordinary code, longer fences and multiple artifacts', () => {
  const parts = splitAnswer('~~~~html\n<h1>Hi</h1>\n~~~\n~~~~\n```js\nalert(1);\n```\n```xml\n<svg/>\n```');
  const code = parts.filter(p => p.type === 'code');
  assert.equal(code[0].kind, 'html');
  assert.equal(code[0].code, '<h1>Hi</h1>\n~~~');
  assert.equal(code[1].kind, null);
  assert.equal(code[2].language, 'xml');
  assert.equal(code[2].kind, 'svg');
});

test('fences must be on their own line, allowing up to three spaces', () => {
  const parts = splitAnswer('Inline ```html is text.\n   ```html\n<script>const a="```";</script>\n   ```', true);
  assert.equal(parts[0].type, 'text');
  assert.equal(parts[1].kind, 'html');
  assert.equal(parts[1].complete, true);
});

test('a partial closing line cannot briefly execute a preview', () => {
  const text = '```html\n<h1>Hi</h1>\n```';
  assert.equal(splitAnswer(text)[0].complete, false);
  assert.equal(splitAnswer(text + 'still code\n')[0].complete, false);
  assert.equal(splitAnswer(text + '\n')[0].complete, true);
  assert.equal(splitAnswer(text, true)[0].complete, true);
});

test('raw SVG and HTML only preview at completion with a closed root', () => {
  const svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>Raw</text></svg>';
  assert.equal(splitAnswer(svg)[0].complete, false);
  assert.equal(splitAnswer(svg, true)[0].complete, true);
  assert.equal(splitAnswer('<!doctype html><html><body>Hi</body></html>', true)[0].kind, 'html');
  assert.equal(splitAnswer('<html><body>Unfinished', true)[0].complete, false);
  assert.equal(splitAnswer('<svg xmlns="http://www.w3.org/2000/svg"/>', true)[0].complete, true);
});

test('model output remains literal; HTML preview policy precedes all output', () => {
  const code = '<script>parent.document.body.textContent="bad"</script>';
  assert.deepEqual(splitAnswer(code, true), [{type: 'text', text: code}]);
  const document = previewDocument(code);
  assert(document.indexOf('Content-Security-Policy') < document.indexOf(code));
  assert(document.includes("connect-src 'none'"));
  assert(document.includes("form-action 'none'"));
  assert(document.includes("default-src 'none'"));
});


test('queue time is explicit, separate from browser TTFT, and may be unknown', () => {
  const reply = new Reply();
  assert.equal(reply.queueWaitMs, null);
  reply.accept({type:'message_start', message:{id:'x'}, naivert:{timings:{queue_wait_ms:1500}}}, 1800);
  assert.equal(reply.ttft, null); // admission / prefill completion is not first content
  addBlock(reply, 'text', 'Hi', 0, 2000);
  assert.equal(reply.ttft, 2000);
  assert.equal(reply.ttftAfterQueue, 500);
  reply.accept({type:'message_delta', delta:{}, usage:{output_tokens:1}, naivert:{timings:{decode_ms:50}}}, 4000);
  assert.equal(reply.queueWaitMs, 1500);
  assert.equal(reply.ttftAfterQueue, 500);
  for (const invalid of [-1, null, '1500', Infinity]) {
    reply.accept({type:'ping', naivert:{timings:{queue_wait_ms:invalid}}}, 4000);
    assert.equal(reply.queueWaitMs, 1500);
  }
  const old = new Reply();
  addBlock(old, 'text', 'Hi', 0, 2000);
  assert.equal(old.ttftAfterQueue, null);
  old.accept({type:'ping', naivert:{timings:{queue_wait_ms:3000}}}, 4000);
  assert.equal(old.ttftAfterQueue, null); // never invent negative or zero TTFT
});

test('lucky draw starts a new session with only the selected prompt and preserves settings', () => {
  const request = {model:'test', seed:4, max_tokens:131072, thinking:{type:'enabled', budget_tokens:1024},
    metadata:{user_id:'old-session'}, system:'format',
    messages:[{role:'user', content:'first'}, {role:'assistant',content:[{type:'text',text:'hello'}]},
      {role:'user',content:'Write a tetris'}]};
  const seed = drawSeed(4);
  assert(Number.isSafeInteger(seed) && seed >= 0 && seed !== 4);
  const retry = withNewSeed(request, seed, 'new-session');
  assert.deepEqual(retry, {...request, seed, metadata:{user_id:'new-session'},
    messages:[{role:'user',content:'Write a tetris'}]});
  retry.messages[0].content = 'different';
  retry.thinking.budget_tokens = 2048;
  assert.equal(request.messages.at(-1).content, 'Write a tetris');
  assert.equal(request.thinking.budget_tokens, 1024);
  assert.equal(request.metadata.user_id, 'old-session');
  assert.equal(request.messages.length, 3);
  assert.throws(() => withNewSeed(request, -1, 'new-session'));
});
