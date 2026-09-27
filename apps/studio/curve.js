// SVG paths contain numeric coordinates only; generated model markup never enters this chart.
let latestReply;
let resizeObserver;
export function drawRateCurve(reply) {
  const $ = id => document.getElementById(id);
  latestReply = reply;
  const chart = $('rate-chart');
  if (!resizeObserver && typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(() => drawRateCurve(latestReply));
    resizeObserver.observe(chart);
  }
  const thinking = reply.streamRates.thinking.curve();
  const output = reply.streamRates.text.curve();
  const all = [...thinking, ...output];
  const missing = reply.streamRates.total.missing;
  const present = all.length > 0 && !missing;
  $('rate-chart').toggleAttribute('hidden', !present);
  $('curve-empty').hidden = present;
  $('curve-empty').textContent = missing ? 'The server has not provided token-progress counters for this request.'
    : reply.completed ? 'Not enough streamed content for a curve (minimum 200 ms per phase).'
      : 'The curve starts after 200 ms of measured generation.';
  if (!present) { $('peak-tps').textContent = '—'; return; }
  const start = reply.ttft ?? all[0].ms;
  const end = Math.max(start + 1, ...all.map(point => point.ms));
  const peak = Math.max(...all.map(point => point.tok_s));
  const top = Math.max(250, Math.ceil(peak / 250) * 250);
  // Use CSS-pixel coordinates so the compact chart keeps readable labels at any width.
  const width = Math.max(180, chart.clientWidth);
  const height = chart.clientHeight || 64;
  const left = 38, right = width - 8, topY = 9, bottom = height - 17;
  chart.setAttribute('viewBox', `0 0 ${width} ${height}`);
  $('curve-grid').setAttribute('d', `M${left} ${topY}H${right} M${left} ${bottom}H${right}`);
  for (const [id, x, y] of [
    ['curve-top', left - 6, topY + 3], ['curve-zero', left - 6, bottom + 3],
    ['curve-start', left, height - 2], ['curve-end', right, height - 2],
  ]) {
    $(id).setAttribute('x', x);
    $(id).setAttribute('y', y);
  }
  const x = ms => left + (ms - start) / (end - start) * (right - left);
  const y = tps => bottom - tps / top * (bottom - topY);
  const path = points => points.map((point, index) => `${index ? 'L' : 'M'}${x(point.ms).toFixed(2)},${y(point.tok_s).toFixed(2)}`).join(' ');
  $('curve-thinking').setAttribute('d', path(thinking));
  $('curve-output').setAttribute('d', path(output));
  for (const [id, points] of [['curve-thinking-dot', thinking], ['curve-output-dot', output]]) {
    const dot = $(id);
    dot.toggleAttribute('hidden', points.length !== 1);
    if (points.length === 1) { dot.setAttribute('cx', x(points[0].ms)); dot.setAttribute('cy', y(points[0].tok_s)); }
  }
  $('curve-top').textContent = top.toLocaleString('en-US');
  $('curve-mid').textContent = (top / 2).toLocaleString('en-US');
  $('curve-end').textContent = `${((end - start) / 1000).toFixed(1)} s`;
  $('peak-tps').textContent = Math.round(peak).toLocaleString('en-US');
  $('rate-chart').setAttribute('aria-label', `Streamed generation speed. Thinking in gray, output in orange. Peak sampled rate ${Math.round(peak)} tokens per second over ${((end - start) / 1000).toFixed(1)} seconds.`);
}
