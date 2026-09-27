import test from 'node:test';
import assert from 'node:assert/strict';
import {renderMarkdown} from '../markdown.js';
import {splitAnswer} from '../core.js';

test('basic answer Markdown renders semantic blocks and inline formatting', () => {
  const html = renderMarkdown('# Limits\n\n**Output** includes *thinking* and `answer`.\n\n- One\n  - Nested\n- Two\n\n1. First\n2. Second\n\n> Shared context\n\n---\n\n| Setting | Value |\n| --- | ---: |\n| Cap | 1048576 |');
  for (const tag of ['h1','strong','em','code','ul','ol','blockquote','hr','table','th','td']) assert.match(html, new RegExp(`<${tag}[ >]`));
  assert.match(html, /<ul>\n<li>One\n<ul>/);
  assert.match(html, /text-align:right/);
});

test('answer Markdown never executes HTML or fetches model-supplied image URLs', () => {
  const html = renderMarkdown('<img src=x onerror=alert(1)>\n\n<script>alert(1)</script>\n\n[x](javascript:alert(1))\n\n[x](data:text/html,test)\n\n![photo](https://example.com/tracker.png)\n\n[Safe](https://example.com/docs "Title")');
  assert.doesNotMatch(html, /<(script|img)\b/);
  assert.doesNotMatch(html, /href="(?:javascript|data):/);
  assert.match(html, /&lt;script&gt;/);
  assert.match(html, /<p>photo<\/p>/);
  assert.match(html, /href="https:\/\/example.com\/docs"[^>]*target="_blank"[^>]*rel="noopener noreferrer"/);
});

test('streaming Markdown can close emphasis and tables without changing fenced artifacts', () => {
  assert.match(renderMarkdown('**still writing'), /\*\*still writing/);
  assert.match(renderMarkdown('**still writing**'), /<strong>still writing<\/strong>/);
  const parts = splitAnswer('## Configuration\n\n```yaml\n- name: NAIVERT_MAX_DECODE_TOKENS\n  value: "1048576"\n```\n\n**Meaning:** the output cap.\n\n```html\n<h1>App</h1>\n```\n', true);
  assert.equal(parts[1].language, 'yaml');
  assert.equal(parts[1].kind, null);
  assert.match(parts[1].code, /value: "1048576"/);
  assert.equal(parts[3].kind, 'html');
  assert.equal(parts[3].complete, true);
  assert.match(renderMarkdown(parts[0].text), /<h2>Configuration<\/h2>/);
  assert.match(renderMarkdown(parts[2].text), /<strong>Meaning:<\/strong>/);
});
