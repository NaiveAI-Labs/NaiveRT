// Browser-independent protocol and artifact parsing. No token counts are inferred from text.
export async function consumeSSE(response, onEvent) {
  if (!response.body) throw new Error('This browser did not expose a streaming response.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const {value, done} = await reader.read();
      const receivedAt = performance.now();
      buffer += done ? decoder.decode() : decoder.decode(value, {stream: true});
      let match;
      while ((match = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        const data = frame.split(/\r?\n/).filter(line => line.startsWith('data:'))
          .map(line => line.slice(5).replace(/^ /, '')).join('\n');
        if (data) onEvent(JSON.parse(data), {receivedAt});
      }
      if (done) break;
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export class StreamRate {
  constructor() {
    this.first = null;
    this.last = null;
    this.missing = false;
    this.window = [];
    this.points = [];
  }

  observe(count, ms) {
    if (!Number.isSafeInteger(count) || count < 0 || !Number.isFinite(ms)) {
      this.missing = true;
      return;
    }
    if (this.last && (count < this.last.count || ms < this.last.ms)) {
      this.missing = true;
      return;
    }
    const sample = {count, ms};
    // All SSE records parsed from the first fetch read belong to the first receive boundary.
    // Exclude the entire first batch, not just its first SSE event.
    if (!this.first || ms === this.first.ms) this.first = sample;
    this.last = sample;
    if (this.window.at(-1)?.ms === ms) this.window[this.window.length - 1] = sample;
    else this.window.push(sample);
    // Retain the receive boundary immediately before the trailing one-second window.
    while (this.window.length > 2 && this.window[1].ms <= ms - 1000) this.window.shift();
    const anchor = this.window[0];
    const duration = ms - anchor.ms;
    if (duration >= 200) {
      const point = {ms, tok_s: (count - anchor.count) * 1000 / duration};
      const lastPoint = this.points.at(-1);
      // A 200 ms bucket bounds chart storage; replace its last observation while it is live.
      if (lastPoint && Math.floor((lastPoint.ms - this.first.ms) / 200) === Math.floor((ms - this.first.ms) / 200)) this.points[this.points.length - 1] = point;
      else this.points.push(point);
    }
  }

  curve() { return this.missing ? [] : this.points; }

  snapshot() {
    const elapsed = this.first && this.last ? this.last.ms - this.first.ms : 0;
    const tokens = this.first && this.last ? this.last.count - this.first.count : 0;
    return {available: !this.missing && elapsed > 0 && tokens > 0,
      tokens: !this.missing ? tokens : null, interval_ms: elapsed,
      tok_s: !this.missing && elapsed > 0 && tokens > 0 ? tokens * 1000 / elapsed : null,
      first: this.first, last: this.last, missing_token_progress: this.missing};
  }
}

export class Reply {
  constructor() {
    this.blocks = [];
    this.usage = {};
    this.timings = null;
    this.stopReason = null;
    this.completed = false;
    this.gotFinalUsage = false;
    this.ttft = null;
    this.queueWaitMs = null;
    this.firstAnswer = null;
    this.streamRates = {thinking: new StreamRate(), text: new StreamRate(), total: new StreamRate()};
  }

  accept(event, elapsed) {
    if (event.type === 'error') throw new Error(event.error?.message || 'The server could not complete this request.');
    const queueMs = event.naivert?.timings?.queue_wait_ms;
    if (Number.isFinite(queueMs) && queueMs >= 0) this.queueWaitMs = queueMs;
    if (event.type === 'message_start') {
      this.id = event.message.id;
      const {output_tokens, ...inputUsage} = event.message.usage || {};
      Object.assign(this.usage, inputUsage);
    } else if (event.type === 'content_block_start') {
      const block = event.content_block;
      if (!['text', 'thinking'].includes(block.type)) throw new Error(`Unsupported response block: ${block.type}`);
      this.blocks[event.index] = {...block};
      if (block.text || block.thinking) this.markContent(block.type, elapsed);
    } else if (event.type === 'content_block_delta') {
      const block = this.blocks[event.index];
      if (!block) throw new Error('Received content before its block was opened.');
      const delta = event.delta;
      if (delta.type === 'signature_delta') {
        block.signature = (block.signature || '') + delta.signature;
      } else {
        const field = delta.type === 'thinking_delta' ? 'thinking' : delta.type === 'text_delta' ? 'text' : null;
        if (!field || block.type !== (field === 'text' ? 'text' : 'thinking')) throw new Error('Unexpected content delta.');
        block[field] = (block[field] || '') + (delta[field] || '');
        if (delta[field]) {
          this.markContent(block.type, elapsed);
          const count = event.naivert?.generated_tokens;
          this.streamRates[block.type].observe(count, elapsed);
          this.streamRates.total.observe(count, elapsed);
        }
      }
    } else if (event.type === 'message_delta') {
      Object.assign(this.usage, event.usage);
      this.stopReason = event.delta?.stop_reason;
      this.timings = event.naivert?.timings || null;
      this.gotFinalUsage = Number.isFinite(event.usage?.output_tokens);
    } else if (event.type === 'message_stop') {
      this.completed = true;
    }
  }

  markContent(kind, elapsed) {
    if (this.ttft === null) this.ttft = elapsed;
    if (kind === 'text' && this.firstAnswer === null) this.firstAnswer = elapsed;
  }

  get ttftAfterQueue() {
    return this.ttft !== null && this.queueWaitMs !== null && this.ttft >= this.queueWaitMs
      ? this.ttft - this.queueWaitMs : null;
  }

  get text() { return this.blocks.filter(b => b.type === 'text').map(b => b.text).join('\n'); }
  get thinking() { return this.blocks.filter(b => b.type === 'thinking').map(b => b.thinking).join('\n'); }
  get valid() { return this.completed && this.gotFinalUsage && Boolean(this.stopReason); }
  get inputTokens() {
    const keys = ['input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'];
    return keys.some(key => Number.isFinite(this.usage[key]))
      ? keys.reduce((sum, key) => sum + (this.usage[key] || 0), 0) : null;
  }
}

function previewKind(language, code) {
  if (language === 'svg' || ((!language || language === 'xml') && /^\s*(?:<\?xml[^>]*>\s*)?<svg[\s/>]/i.test(code))) return 'svg';
  if (['html', 'htm'].includes(language) || (!language && /^\s*(?:<!doctype html|<html)[\s>]/i.test(code))) return 'html';
  return null;
}

// Keep unfinished fences as code. A closed fence is the only streaming preview boundary.
export function splitAnswer(text, finished = false) {
  const parts = [];
  const opening = /^ {0,3}(`{3,}|~{3,})([^\n]*)\r?\n/gm;
  let cursor = 0;
  let match;
  while ((match = opening.exec(text))) {
    if (match.index > cursor) parts.push({type: 'text', text: text.slice(cursor, match.index)});
    const fence = match[1];
    const language = match[2].trim().split(/\s+/)[0].toLowerCase();
    const start = opening.lastIndex;
    const closing = new RegExp(`^ {0,3}${fence[0]}{${fence.length},}[ \\t]*\\r?$`, 'gm');
    closing.lastIndex = start;
    let end = closing.exec(text);
    // At a chunk boundary, ``` may still be followed by non-fence text on the same line.
    // Wait for a newline or message completion before treating that line as closed.
    if (end && closing.lastIndex === text.length && !finished) end = null;
    const code = text.slice(start, end ? end.index : text.length).replace(/\r?\n$/, '');
    parts.push({type: 'code', language, code, kind: previewKind(language, code), complete: Boolean(end)});
    cursor = end ? closing.lastIndex : text.length;
    opening.lastIndex = cursor;
    if (!end) break;
  }
  if (!parts.length) {
    const code = text.trim();
    const kind = previewKind('', code);
    const closed = kind === 'svg' ? /<\/svg>\s*$/i.test(code) || /^(?:<\?xml[^>]*>\s*)?<svg\b[^>]*\/>\s*$/i.test(code)
      : kind === 'html' && /<\/html>\s*$/i.test(code);
    if (kind) return [{type: 'code', language: kind, code, kind, complete: finished && Boolean(closed)}];
  }
  if (cursor < text.length) parts.push({type: 'text', text: text.slice(cursor)});
  return parts;
}

// An opaque-origin sandbox allows self-contained scripts without access to the host page.
// CSP precedes model output; a later policy cannot relax it. External assets are not supported.
export function previewDocument(code) {
  const diagnostics = `<script>
(() => {
  const report = message => parent.postMessage({type:"naivert-preview-error", message:String(message).slice(0, 1000)}, "*");
  addEventListener("error", event => { if (event.message) report(event.message); });
  addEventListener("unhandledrejection", event => report(event.reason?.message || event.reason || "Unhandled promise rejection"));
})();
<\/script>`;
  const keyboard = `<script>
addEventListener('DOMContentLoaded', () => {
  document.body.tabIndex = -1;
  document.addEventListener('pointerdown', event => {
    if (!event.target.closest('input,textarea,select,button,a,summary,[tabindex]:not(body),[contenteditable]')) {
      document.body.focus({preventScroll:true});
    }
  });
  // Let the app receive keys normally; cancel only the browser's scrolling action.
  addEventListener('keydown', event => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing) return;
    const target = event.target;
    if (target.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"])')) return;
    if (event.key === ' ' && target.closest('button,a,summary')) return;
    if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown',' '].includes(event.key)) event.preventDefault();
  });
});
<\/script>`;
  return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; form-action 'none'; base-uri 'none';"><style>body{margin:0}*{box-sizing:border-box}</style>${diagnostics}</head><body>${code}${keyboard}</body></html>`;
}

// Start a fresh conversation with only the selected prompt and the same generation settings.
export function withNewSeed(request, seed, sessionId) {
  if (!Number.isSafeInteger(seed) || seed < 0) throw new Error('Invalid seed');
  const fresh = structuredClone(request);
  return {...fresh, seed, messages: [fresh.messages.at(-1)],
    metadata: {...fresh.metadata, user_id: sessionId}};
}

export function drawSeed(previous) {
  let seed = crypto.getRandomValues(new Uint32Array(1))[0];
  if (seed === previous) seed = (seed + 1) >>> 0;
  return seed;
}
