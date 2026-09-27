import {consumeSSE, Reply, splitAnswer, previewDocument, withNewSeed, drawSeed} from './core.js';
import {renderMarkdown} from './markdown.js';
import {drawRateCurve} from './curve.js';
import {normalizeEndpoint, apiURL, authOptions, parseQuickSetup, messagesOptions, healthOptions, loadCredentials, saveCredentials, clearCredentials} from './endpoint.js';
import {QueueMonitor} from './queue.js';
import {checkConnection} from './connection.js';

const $ = id => document.getElementById(id);
const set = (id, text) => { $(id).textContent = text; };
const fmt = n => Number.isFinite(n) ? n.toLocaleString('en-US', {maximumFractionDigits: 0}) : '—';
const seconds = ms => Number.isFinite(ms) ? `${(ms / 1000).toFixed(2)} s` : '—';
const history = [];
const views = [];
let config = null;
let ready = false;
let busy = false;
let checking = false;
let controller = null;
let lastRequest = null;
let toastTimer;
let conversationId = newConversationId();
const endpointStorageKey = 'naivert-studio-endpoint';
const modelStorageKey = 'naivert-studio-model';
const localProxy = document.documentElement.dataset.localProxy || '';
let endpoint = '';
let model = 'naive-n0.5';
let auth = {};
const queueMonitor = new QueueMonitor(snapshot => {
  set('queue-count', snapshot.queued === null ? '—' : fmt(snapshot.queued));
  set('queue-detail', snapshot.detail);
});
try {
  endpoint = normalizeEndpoint(localStorage.getItem(endpointStorageKey) || '');
  model = localStorage.getItem(modelStorageKey) || model;
} catch { /* Storage may be disabled; settings remain usable in memory. */ }
if (localProxy) endpoint = normalizeEndpoint(localProxy);
auth = loadCredentials(endpoint);
if (localProxy) auth.cookies = false;
$('auth-type').value = auth.type || 'none';
$('auth-secret').value = auth.secret || '';
$('auth-username').value = auth.username || '';
$('auth-cookies').checked = Boolean(auth.cookies);
authFields();
$('endpoint-url').value = endpoint;
$('endpoint-url').readOnly = Boolean(localProxy);
$('endpoint-model').value = model;
$('endpoint-settings').open = !endpoint || Boolean(localProxy);
if (localProxy) {
  $('endpoint-help').textContent = `Local proxy forwards requests to ${endpoint}. Choose authentication and click Use endpoint. To change the upstream, restart the local server.`;
  $('auth-cookies').closest('label').hidden = true;
}

const examples = {
  tetris: 'Write a tetris',
  poster: 'Create a playful SVG poster for a small flower shop called Fresh Ideas. Use a warm cream background, oversized dark green lettering, orange flowers, and a charming hand-drawn feel. Make it a complete, polished composition in a single SVG code block.',
  app: 'Build a beautiful, working Pomodoro timer as one self-contained HTML file. Use a warm cream and terracotta palette, a large timer, start/pause/reset controls, and a short-break mode. Include all CSS and JavaScript inline; no external libraries or assets. Return the complete HTML in one code block.',
  explain: 'Create an interactive explanation of how a pendulum works as one self-contained HTML file. Include an animated pendulum and a length slider that changes its period, with a simple readable explanation. Use a restrained, editorial visual style. Include all CSS and JavaScript inline; no external libraries. Return the complete HTML in one code block.',
};
const system = 'You are a helpful creative assistant. When asked to make a visual, use a complete SVG in a fenced svg code block. For an interactive page or app, use one self-contained HTML document in a fenced html code block with inline CSS and JavaScript. The preview has no external network access: do not use CDNs, imports, remote images, or external fonts. Keep explanations brief unless requested. For ordinary questions, answer normally.';

function newConversationId() {
  return `studio-${globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`}`;
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function toast(message) {
  set('toast', message);
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 2600);
}

function controls() {
  const locked = busy || checking;
  $('send').disabled = locked || !ready;
  $('lucky-draw').disabled = locked || !ready || (!$('prompt').value.trim() && !lastRequest);
  $('send').hidden = busy;
  $('stop').hidden = !busy;
  $('new-chat').disabled = locked;
  $('endpoint-url').disabled = $('endpoint-model').disabled = $('endpoint-connect').disabled = locked;
  for (const id of ['auth-type', 'auth-secret', 'auth-username', 'auth-cookies', 'auth-clear', 'quick-setup', 'quick-connect']) $(id).disabled = locked;
  $('check-connection').disabled = locked || !ready;
  $('check-connection').textContent = checking ? 'Checking…'
    : $('connection-result').dataset.state === 'ready' ? 'Connected · Recheck' : 'Check connection';
  $('max-tokens').disabled = busy;
  for (const id of ['seed', 'thinking-budget', 'artifact-guidance']) $(id).disabled = busy;
  $('pelican-preset').disabled = locked || !ready;
  $('pelican-thinking-preset').disabled = locked || !ready;
  $('thinking').disabled = busy || !config;
  document.querySelectorAll('[data-example]').forEach(button => { button.disabled = locked; });
}

function resetConnectionCheck() {
  set('connection-result', 'Not checked');
  $('connection-result').dataset.state = 'waiting';
  $('check-connection').textContent = 'Check connection';
}

async function runConnectionCheck() {
  if (!ready || busy || checking) return;
  checking = true;
  controls();
  set('connection-result', 'Checking authentication and streaming…');
  $('connection-result').dataset.state = 'waiting';
  try {
    const result = await checkConnection({endpoint: localProxy ? location.origin : endpoint,
      model, auth, sessionId: `check-${newConversationId()}`});
    set('connection-result', result.message);
    $('connection-result').dataset.state = result.ok ? 'ready' : 'error';
    if (!result.ok) $('endpoint-settings').open = true;
    set('connection', result.ok ? 'Connected' : 'Connection check failed');
    $('connection').dataset.state = result.ok ? 'ready' : 'error';
  } finally { checking = false; controls(); }
}
$('check-connection').addEventListener('click', runConnectionCheck);

function configureEndpoint() {
  ready = Boolean(endpoint && model.trim());
  config = ready ? {model} : null;
  set('endpoint-current', endpoint || 'Not configured');
  $('endpoint-current').title = endpoint;
  set('connection', ready ? 'Endpoint configured' : 'Choose an endpoint');
  $('connection').dataset.state = ready ? 'ready' : 'waiting';
  set('model-name', ready ? model : 'Configure your API endpoint');
  set('token-limit', 'Output limit is validated by the API');
  controls();
  resetConnectionCheck();
  refreshQueue();
}

function refreshQueue() {
  if (ready) queueMonitor.start(apiURL(localProxy ? location.origin : endpoint, '/health'), healthOptions(auth, conversationId));
  else {
    queueMonitor.stop();
    set('queue-count', '—');
    set('queue-detail', 'Apply connection settings to see the queue');
  }
}
window.addEventListener('pagehide', () => queueMonitor.stop());
window.addEventListener('pageshow', event => { if (event.persisted) refreshQueue(); });

class Artifact {
  constructor(index) {
    this.element = node('section', 'artifact');
    const toolbar = node('div', 'artifact-toolbar');
    this.title = node('span', 'artifact-title');
    const actions = node('div', 'artifact-actions');
    const tabs = node('div', 'artifact-tabs');
    tabs.setAttribute('role', 'tablist');
    tabs.setAttribute('aria-label', `Artifact ${index + 1} view`);
    this.codeTab = this.button('Code', () => this.select('code', true));
    this.previewTab = this.button('Preview', () => this.select('preview', true));
    for (const tab of [this.codeTab, this.previewTab]) tab.setAttribute('role', 'tab');
    this.code = node('pre');
    this.codeText = node('code');
    this.code.append(this.codeText);
    this.followCode = true;
    this.code.addEventListener('scroll', () => {
      this.followCode = this.code.scrollHeight - this.code.clientHeight - this.code.scrollTop <= 32;
    });
    this.preview = node('div', 'preview');
    const uid = `artifact-${views.length}-${index}`;
    for (const [tab, panel, name] of [[this.codeTab, this.code, 'code'], [this.previewTab, this.preview, 'preview']]) {
      tab.id = `${uid}-${name}-tab`;
      panel.id = `${uid}-${name}`;
      panel.setAttribute('role', 'tabpanel');
      panel.setAttribute('aria-labelledby', tab.id);
      tab.setAttribute('aria-controls', panel.id);
    }
    tabs.append(this.codeTab, this.previewTab);
    tabs.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const target = event.key === 'Home' ? this.codeTab : event.key === 'End' ? this.previewTab
        : document.activeElement === this.codeTab ? this.previewTab : this.codeTab;
      if (!target.disabled && !target.hidden) { target.click(); target.focus(); }
    });
    const copy = this.button('Copy', async () => {
      try { await navigator.clipboard.writeText(this.part.code); toast('Code copied'); }
      catch { toast('Clipboard unavailable. Select and copy from the Code tab.'); }
    });
    const download = this.button('↓', () => {
      const url = URL.createObjectURL(new Blob([this.part.code], {type: this.part.kind === 'svg' ? 'image/svg+xml' : this.part.kind === 'html' ? 'text/html' : 'text/plain'}));
      const link = node('a');
      link.href = url;
      link.download = `naivert-${index + 1}.${this.part.kind || 'txt'}`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, 'Download code');
    const fullscreen = this.button('⛶', async () => {
      try {
        if (document.fullscreenElement === this.element) await document.exitFullscreen();
        else await this.element.requestFullscreen();
      } catch { toast('Fullscreen is unavailable in this browser.'); }
    }, 'Toggle fullscreen');
    this.interact = this.button('Play / interact', () => this.focusPreview());
    this.interact.hidden = true;
    this.restart = this.button('Restart', () => {
      if (!this.frame) return;
      this.previewError = null;
      this.fix.hidden = true;
      this.note.dataset.error = 'false';
      this.note.textContent = 'Restarting preview…';
      this.frame.srcdoc = previewDocument(this.part.code);
      this.focusPreview();
    });
    this.restart.hidden = true;
    this.fix = this.button('Ask to fix', () => {
      $('prompt').value = `Fix this generated HTML. Its preview fails with: ${this.previewError}. Return the complete corrected self-contained HTML.\n\n\`\`\`html\n${this.part.code}\n\`\`\``;
      $('prompt').focus();
      controls();
    });
    this.fix.hidden = true;
    this.onPreviewMessage = event => {
      if (!this.frame || event.source !== this.frame.contentWindow || event.data?.type !== 'naivert-preview-error' || typeof event.data.message !== 'string') return;
      this.previewError = event.data.message.slice(0, 1000);
      this.showPreviewError();
    };
    window.addEventListener('message', this.onPreviewMessage);
    actions.append(tabs, this.interact, this.restart, this.fix, copy, download, fullscreen);
    toolbar.append(this.title, actions);
    this.note = node('p', 'artifact-note');
    this.element.append(toolbar, this.code, this.preview, this.note);
    this.select('code');
  }

  button(text, action, label = text) {
    const button = node('button', '', text);
    button.type = 'button';
    button.setAttribute('aria-label', label);
    button.title = label;
    button.addEventListener('click', action);
    return button;
  }

  select(view, manual = false) {
    if (manual) this.manual = true;
    this.code.hidden = view !== 'code';
    this.preview.hidden = view !== 'preview';
    for (const [tab, name] of [[this.codeTab, 'code'], [this.previewTab, 'preview']]) {
      tab.setAttribute('aria-selected', String(view === name));
      tab.tabIndex = view === name ? 0 : -1;
    }
    if (view === 'preview' && manual) this.focusPreview();
    if (view === 'code' && this.followCode) this.code.scrollTop = this.code.scrollHeight;
  }

  showPreviewError() {
    this.note.textContent = `Generated app error: ${this.previewError}`;
    this.note.dataset.error = 'true';
    this.fix.hidden = false;
  }

  focusPreview() {
    if (!this.frame) return;
    this.focusRequested = true;
    this.select('preview');
    this.frame.focus({preventScroll: true});
    this.frame.contentWindow?.focus();
  }

  update(part, finished) {
    this.part = part;
    this.title.textContent = part.language || part.kind || 'code';
    this.code.classList.toggle('streaming-code', !finished && !part.complete);
    if (this.codeText.textContent !== part.code) {
      const previous = this.codeText.textContent;
      if (part.code.startsWith(previous) && this.codeText.firstChild) {
        this.codeText.firstChild.appendData(part.code.slice(previous.length));
      } else this.codeText.textContent = part.code;
      if (this.followCode && !this.code.hidden) this.code.scrollTop = this.code.scrollHeight;
    }
    this.previewTab.hidden = !part.kind;
    this.previewTab.disabled = !part.complete;
    this.note.textContent = !part.complete ? (finished ? 'Incomplete code block. Preview is unavailable; ask the model to finish it.' : 'Writing code… preview opens when the code block is complete.')
      : part.kind === 'html' ? 'Click the preview or Play / interact to use the keyboard. Tab moves focus out of the preview.'
      : part.kind === 'svg' ? 'SVG preview · download the source to keep or edit it.' : 'Code view';
    // Never reload an already-rendered artifact while subsequent text is streaming.
    if (part.complete && part.kind && !this.rendered) {
      this.rendered = true;
      if (part.kind === 'svg') {
        this.url = URL.createObjectURL(new Blob([part.code], {type: 'image/svg+xml'}));
        const img = node('img');
        img.alt = 'Generated SVG';
        img.src = this.url;
        img.addEventListener('error', () => {
          this.note.textContent = 'The SVG could not be rendered. Open Code to inspect it or ask for a correction.';
          this.select('code');
        });
        this.preview.append(img);
      } else {
        const frame = node('iframe');
        this.frame = frame;
        this.interact.hidden = this.restart.hidden = false;
        frame.title = 'Generated HTML preview';
        frame.tabIndex = 0;
        frame.setAttribute('sandbox', 'allow-scripts');
        frame.setAttribute('referrerpolicy', 'no-referrer');
        frame.addEventListener('load', () => {
          if (!this.previewError) this.note.textContent = 'Click the preview or Play / interact to use the keyboard. Tab moves focus out of the preview.';
          if (this.focusRequested && !this.preview.hidden
            && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) this.focusPreview();
        });
        frame.srcdoc = previewDocument(part.code);
        this.preview.append(frame);
      }
      if (!this.manual) this.select('preview');
    }
    if (this.previewError) this.showPreviewError();
  }

  dispose() { window.removeEventListener('message', this.onPreviewMessage); if (this.url) URL.revokeObjectURL(this.url); }
}

function createTurn(prompt, seed) {
  $('welcome').hidden = true;
  const element = node('article', 'turn');
  element.dataset.status = 'working';
  const question = node('div', 'user-message', prompt);
  const label = node('div', 'assistant-label', `${config.model} · Seed ${seed ?? 'server default'}`);
  const thoughts = node('details', 'thoughts');
  thoughts.hidden = true;
  thoughts.open = true;
  const thoughtSummary = node('summary', '', 'Thinking…');
  const thoughtText = node('div', 'thinking-text');
  thoughts.append(thoughtSummary, thoughtText);
  const answer = node('div', 'answer');
  const status = node('div', 'turn-status', 'Waiting for the first response…');
  status.setAttribute('role', 'status');
  element.append(question, label, thoughts, answer, status);
  $('conversation').append(element);
  const view = {element, thoughts, thoughtSummary, thoughtText, answer, status, parts: [], sawAnswer: false};
  views.push(view);
  return view;
}

function render(view, reply, finished) {
  const follow = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 180;
  const thought = reply.thinking;
  view.thoughts.hidden = !thought;
  if (view.thoughtText.textContent !== thought) {
    const text = view.thoughtText;
    const followThought = text.scrollHeight - text.clientHeight - text.scrollTop <= 32;
    text.textContent = thought;
    if (followThought && view.thoughts.open) text.scrollTop = text.scrollHeight;
  }
  view.thoughtSummary.textContent = reply.firstAnswer !== null || finished ? 'Thought process' : 'Thinking…';
  if (reply.firstAnswer !== null && !view.sawAnswer) {
    view.thoughts.open = false;
    view.sawAnswer = true;
  }
  const parts = splitAnswer(reply.text, finished);
  parts.forEach((part, index) => {
    let current = view.parts[index];
    if (!current || current.type !== part.type) {
      current?.artifact?.dispose();
      const artifact = part.type === 'code' ? new Artifact(view.parts.slice(0, index).filter(p => p.artifact).length) : null;
      const element = artifact ? artifact.element : node('div', 'answer-text');
      if (current) current.element.replaceWith(element);
      else view.answer.append(element);
      current = {type: part.type, element, artifact};
      view.parts[index] = current;
    }
    if (current.artifact) current.artifact.update(part, finished);
    else if (current.text !== part.text) {
      current.element.innerHTML = renderMarkdown(part.text);
      current.text = part.text;
    }
  });
  while (view.parts.length > parts.length) {
    const removed = view.parts.pop();
    removed.artifact?.dispose();
    removed.element.remove();
  }
  if (!finished) view.status.textContent = reply.firstAnswer !== null ? 'Writing…' : reply.ttft !== null ? 'Thinking…' : 'Waiting for the first response…';
  if (follow) window.scrollTo({top: document.documentElement.scrollHeight, behavior: 'instant'});
}

function metrics(reply, elapsed, phase, thinkingEnabled) {
  const segments = reply.valid ? reply.timings?.segments : null;
  const answer = segments?.answer || (!thinkingEnabled && reply.valid ? {device_tok_s: reply.timings?.decode_tok_s_device, tokens: reply.usage.output_tokens} : null);
  set('thinking-tps', fmt(segments?.think?.device_tok_s));
  set('answer-tps', fmt(answer?.device_tok_s));
  const detail = segment => segment ? `${fmt(segment.tokens)} tokens${Number.isFinite(segment.device_ms) ? ` · ${seconds(segment.device_ms)} device` : ''}` : reply.valid ? 'Not reported by this engine' : 'Available after completion';
  set('thinking-detail', thinkingEnabled ? detail(segments?.think) : 'Thinking disabled');
  set('answer-detail', detail(answer));
  const streamDetail = rate => Number.isFinite(rate.tok_s) ? `Streamed ${fmt(rate.tok_s)} tok/s`
    : rate.missing_token_progress ? 'Streamed TPS unavailable on this server' : 'Streamed TPS awaits two receive batches';
  set('thinking-stream', thinkingEnabled ? streamDetail(reply.streamRates.thinking.snapshot()) : '');
  set('answer-stream', streamDetail(reply.streamRates.text.snapshot()));
  set('token-count', reply.valid ? fmt(reply.usage.output_tokens) : '—');
  set('elapsed', `Elapsed ${(elapsed / 1000).toFixed(1)} s`);
  set('queue-wait', seconds(reply.queueWaitMs));
  set('ttft-label', reply.ttftAfterQueue !== null ? 'TTFT after queue' : 'TTFT incl. queue');
  set('ttft', seconds(reply.ttftAfterQueue ?? reply.ttft));
  set('phase', phase);
  drawRateCurve(reply);
}

function measurementDetails(reply, browserMetrics = null, request = null) {
  const values = [
    ['Prompt tokens (total)', fmt(reply.inputTokens)],
    ['Cached input', fmt(reply.usage.cache_read_input_tokens)],
    ['Cache creation', fmt(reply.usage.cache_creation_input_tokens)],
    ['Prefill', seconds(reply.timings?.prompt_ms)],
    ['Queue wait (runtime)', seconds(reply.queueWaitMs)],
    ['Total time to first token', seconds(reply.ttft)],
    ['TTFT after queue', seconds(reply.ttftAfterQueue)],
    ['First answer', seconds(reply.firstAnswer)],
    ['Stop reason', reply.stopReason || '—'],
    ['Backend decode wall TPS', fmt(reply.timings?.predicted_per_second)],
    ['Streamed generation TPS', fmt(reply.streamRates.total.snapshot().tok_s)],
    ['Streamed generation interval', seconds(reply.streamRates.total.snapshot().available ? reply.streamRates.total.snapshot().interval_ms : null)],
    ['Max pending DOM update', seconds(browserMetrics?.max_render_delay_ms)],
    ['Total DOM update work', seconds(browserMetrics?.render_ms)],
  ];
  $('detail-values').replaceChildren(...values.map(([label, value]) => {
    const row = node('div');
    row.append(node('dt', '', label), node('dd', '', value));
    return row;
  }));
  set('details-summary', reply.valid ? 'Final server counts' : 'Request incomplete');
  if (browserMetrics) set('measurement-json', JSON.stringify({
    request: {model: request.model, max_tokens: request.max_tokens, seed: request.seed ?? null,
      thinking: request.thinking, added_system: Boolean(request.system), message_count: request.messages.length},
    usage: reply.usage, timings: reply.timings, browser: browserMetrics,
    streamed: Object.fromEntries(Object.entries(reply.streamRates).map(([key, rate]) => [key, rate.snapshot()])),
    curve: {window_ms: 1000, bucket_ms: 200, thinking: reply.streamRates.thinking.curve(), output: reply.streamRates.text.curve()},
    displayed: {peak_streamed_tps: $('peak-tps').textContent, thinking_tps: $('thinking-tps').textContent, output_tps: $('answer-tps').textContent},
  }, null, 2));
}

async function generate(retry = null, fresh = false) {
  const prompt = retry ? retry.messages.at(-1).content : $('prompt').value.trim();
  const maxTokens = retry?.max_tokens ?? Number($('max-tokens').value);
  const thinkingEnabled = retry ? retry.thinking.type === 'enabled' : $('thinking').checked;
  const thinkingBudget = retry?.thinking.budget_tokens ?? Number($('thinking-budget').value);
  const seed = !fresh && retry ? retry.seed : ($('seed').value === '' ? NaN : Number($('seed').value));
  if (busy || checking || !ready || !prompt) return;
  $('notice').hidden = true;
  if (!Number.isSafeInteger(maxTokens) || maxTokens < 1 || (thinkingEnabled && (!Number.isInteger(thinkingBudget) || thinkingBudget < 1024 || maxTokens <= thinkingBudget))) {
    set('notice', thinkingEnabled ? 'The thinking budget must be at least 1,024 and below the output limit.' : 'Choose a positive integer output limit.');
    $('notice').hidden = false;
    return;
  }
  if (!Number.isSafeInteger(seed) || seed < 0) {
    set('notice', 'Use a non-negative integer seed up to 9,007,199,254,740,991.');
    $('notice').hidden = false;
    return;
  }
  let request = retry || {model: config.model, max_tokens: maxTokens, stream: true,
    thinking: {type: thinkingEnabled ? 'enabled' : 'disabled', budget_tokens: thinkingBudget},
    seed, ...($('artifact-guidance').checked ? {system} : {}),
    metadata: {user_id: conversationId}, messages: [...history, {role: 'user', content: prompt}]};
  if (fresh) {
    newChat();
    request = withNewSeed(request, seed, conversationId);
    refreshQueue();
  }
  lastRequest = structuredClone(request);
  busy = true;
  controller = new AbortController();
  controls();
  $('prompt').value = '';
  const view = createTurn(prompt, request.seed);
  const reply = new Reply();
  const start = performance.now();
  let elapsed = 0;
  let phase = 'Waiting for first response';
  let dirty = true;
  let pendingSince = null;
  const browserMetrics = {render_count: 0, render_ms: 0, max_render_ms: 0, max_render_delay_ms: 0, first_content_ms: null, last_content_ms: null, final_event_ms: null};
  const paint = finished => {
    const before = performance.now();
    if (pendingSince !== null) browserMetrics.max_render_delay_ms = Math.max(browserMetrics.max_render_delay_ms, before - pendingSince);
    render(view, reply, finished);
    const duration = performance.now() - before;
    browserMetrics.render_count++;
    browserMetrics.render_ms += duration;
    browserMetrics.max_render_ms = Math.max(browserMetrics.max_render_ms, duration);
    pendingSince = null;
  };
  $('performance-panel').hidden = $('metrics').hidden = $('request-details').hidden = $('rate-panel').hidden = false;
  $('detail-values').replaceChildren();
  set('measurement-json', 'Waiting for completion…');
  set('details-summary', 'Waiting for final counts');
  metrics(reply, 0, phase, thinkingEnabled);
  view.element.scrollIntoView({block: 'start', behavior: 'smooth'});
  const timer = setInterval(() => {
    elapsed = performance.now() - start;
    if (dirty) { paint(false); dirty = false; }
    phase = reply.firstAnswer !== null ? 'Writing' : reply.ttft !== null ? 'Thinking' : 'Waiting for first response';
    metrics(reply, elapsed, phase, thinkingEnabled);
  }, 100);
  try {
    const response = await fetch(apiURL(localProxy ? location.origin : endpoint, '/v1/messages'), {
      ...messagesOptions(auth, conversationId), signal: controller.signal,
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new Error(body?.error?.message || `The server returned HTTP ${response.status}.`);
    }
    await consumeSSE(response, (event, {receivedAt: now}) => {
      reply.accept(event, now - start);
      if (event.type === 'content_block_delta' && (event.delta.text || event.delta.thinking)) {
        browserMetrics.first_content_ms ??= now - start;
        browserMetrics.last_content_ms = now - start;
        pendingSince ??= now;
      }
      if (event.type === 'message_stop') browserMetrics.final_event_ms = now - start;
      dirty = true;
    });
    if (!reply.valid) throw new Error('The stream ended before its final result arrived. This turn was not added to conversation history.');
    history.splice(0, history.length, ...structuredClone(request.messages), {role: 'assistant', content: reply.blocks.map(block => ({...block}))});
    phase = reply.stopReason === 'max_tokens' ? 'Output limit reached' : 'Completed';
    view.status.textContent = reply.stopReason === 'max_tokens' ? 'Output limit reached. You can ask the model to continue, or start again with a larger limit.' : 'Completed';
    view.element.dataset.status = 'complete';
  } catch (error) {
    const cancelled = controller.signal.aborted;
    phase = cancelled ? 'Stopped' : 'Request failed';
    view.status.textContent = `${phase}. This turn is not included in follow-up context.`;
    view.element.dataset.status = 'incomplete';
    if (!cancelled) {
      set('notice', error instanceof TypeError ? 'Cannot reach the API. Check the endpoint URL, HTTPS, and CORS settings; a localhost tunnel may require local-network permission.' : error.message);
      $('notice').hidden = false;
    }
    if (!$('prompt').value) $('prompt').value = prompt;
  } finally {
    clearInterval(timer);
    elapsed = performance.now() - start;
    paint(true);
    browserMetrics.elapsed_ms = elapsed;
    browserMetrics.final_dom_ms = performance.now() - start;
    browserMetrics.response_chars = reply.text.length;
    metrics(reply, elapsed, phase, thinkingEnabled);
    measurementDetails(reply, browserMetrics, request);
    if (reply.valid && globalThis.crypto?.subtle) {
      const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(reply.text)).catch(() => null);
      if (digest) {
        browserMetrics.response_sha256 = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
        measurementDetails(reply, browserMetrics, request);
      }
    }
    controller = null;
    busy = false;
    controls();
    const interactive = view.parts.map(part => part.artifact).find(artifact => artifact?.frame && !artifact.preview.hidden);
    if (interactive && !$('prompt').value.trim()) interactive.focusPreview();
    else if (!interactive) $('prompt').focus({preventScroll: true});
  }
}

$('composer').addEventListener('submit', event => { event.preventDefault(); generate(); });
$('lucky-draw').addEventListener('click', () => {
  if (busy || checking || !ready) return;
  const retry = !$('prompt').value.trim() && lastRequest;
  const seed = drawSeed(retry ? lastRequest.seed : Number($('seed').value));
  $('seed').value = seed;
  generate(retry ? lastRequest : null, true);
});
$('prompt').addEventListener('input', controls);

$('prompt').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $('composer').requestSubmit();
  }
});
$('stop').addEventListener('click', () => controller?.abort());
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
  $('prompt').value = examples[button.dataset.example];
  if (button.dataset.example === 'tetris') {
    $('seed').value = 4;
    $('artifact-guidance').checked = true;
  }
  $('prompt').focus();
  controls();
}));
function pelicanPreset(thinking) {
  $('prompt').value = 'Generate an SVG of a pelican riding a bicycle. Reply with only the complete SVG markup, no explanation.';
  $('thinking').checked = thinking;
  $('thinking-budget').value = 16384;
  $('artifact-guidance').checked = false;
  $('seed').value = 42;
  $('max-tokens').value = 524288;
  $('prompt').focus();
  controls();
}
$('pelican-preset').addEventListener('click', () => pelicanPreset(false));
$('pelican-thinking-preset').addEventListener('click', () => pelicanPreset(true));
function newChat() {
  if (busy) return;
  for (const view of views) for (const part of view.parts) part.artifact?.dispose();
  views.length = history.length = 0;
  lastRequest = null;
  conversationId = newConversationId();
  $('conversation').replaceChildren();
  $('welcome').hidden = false;
  $('notice').hidden = $('performance-panel').hidden = $('metrics').hidden = $('request-details').hidden = $('rate-panel').hidden = true;
  $('request-details').open = false;
  $('prompt').value = '';
  $('prompt').focus();
  controls();
  window.scrollTo({top: 0, behavior: 'smooth'});
}
$('new-chat').addEventListener('click', () => { newChat(); refreshQueue(); });
function authFields() {
  const type = $('auth-type').value;
  $('auth-user-field').hidden = type !== 'basic';
  $('auth-secret-field').hidden = type === 'none';
  set('auth-secret-label', type === 'basic' ? 'Password' : type === 'api-key' ? 'API key' : 'Token');
}
function clearAuth() {
  if (!clearCredentials()) toast('Browser storage unavailable; saved credentials could not be removed.');
  auth = {};
  $('auth-type').value = 'none';
  $('auth-secret').value = $('auth-username').value = '';
  $('auth-cookies').checked = false;
  $('quick-setup').value = '';
  resetConnectionCheck();
  authFields();
}
$('auth-type').addEventListener('change', () => {
  $('auth-secret').value = $('auth-username').value = '';
  authFields();
});
$('auth-clear').addEventListener('click', () => { clearAuth(); configureEndpoint(); });
$('endpoint-url').addEventListener('input', () => {
  clearAuth();
  ready = false;
  refreshQueue();
  set('connection', 'Apply endpoint settings');
  $('connection').dataset.state = 'waiting';
  controls();
});
function applyEndpoint() {
  if (busy || checking) return false;
  let next;
  const nextAuth = {type: $('auth-type').value, secret: $('auth-secret').value,
    username: $('auth-username').value, cookies: $('auth-cookies').checked};
  try {
    next = normalizeEndpoint($('endpoint-url').value);
    if (!next) throw new Error('Enter the URL of your API server.');
    if (localProxy && next !== localProxy) throw new Error('Restart the local server to change its upstream.');
    if (!$('endpoint-model').value.trim()) throw new Error('Enter a model name.');
    authOptions(nextAuth);
  } catch (error) {
    set('endpoint-error', error.message);
    $('endpoint-error').hidden = false;
    return false;
  }
  const nextModel = $('endpoint-model').value.trim();
  if (next !== endpoint || nextModel !== model || JSON.stringify(nextAuth) !== JSON.stringify(auth)) newChat();
  endpoint = next;
  model = nextModel;
  auth = nextAuth;
  $('endpoint-url').value = endpoint;
  try {
    localStorage.setItem(endpointStorageKey, endpoint);
    localStorage.setItem(modelStorageKey, model);
  } catch { /* Storage may be disabled. */ }
  if (!saveCredentials(endpoint, auth)) toast('Browser storage unavailable; credentials will last only until reload.');
  $('endpoint-error').hidden = true;
  $('endpoint-settings').open = false;
  configureEndpoint();
  return true;
}
$('endpoint-form').addEventListener('submit', event => {
  event.preventDefault();
  applyEndpoint();
});
$('quick-setup-form').addEventListener('submit', event => {
  event.preventDefault();
  if (busy || checking) return;
  const line = $('quick-setup').value;
  $('quick-setup').value = '';
  try {
    const next = parseQuickSetup(line);
    if (localProxy && next.endpoint !== localProxy) throw new Error('Restart the local server to change its upstream.');
    $('endpoint-url').value = next.endpoint;
    $('auth-type').value = next.auth.type;
    $('auth-secret').value = next.auth.secret || '';
    $('auth-username').value = '';
    $('auth-cookies').checked = false;
    if (next.model) $('endpoint-model').value = next.model;
    authFields();
    if (applyEndpoint()) void runConnectionCheck();
  } catch (error) {
    set('endpoint-error', error.message);
    $('endpoint-error').hidden = false;
  }
});
configureEndpoint();
