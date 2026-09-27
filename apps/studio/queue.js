export function queueSnapshot(body) {
  if (!body || body.status !== 'ok' || !Number.isSafeInteger(body.queued) || body.queued < 0) {
    return {queued: null, detail: body?.status === 'loading' ? 'Server loading' : 'Queue status unavailable'};
  }
  return {queued: body.queued, detail: body.slot === 'IDLE' ? 'Server idle' : body.slot ? 'Server busy' : 'Waiting requests; running request excluded'};
}

// Optional telemetry. Failure never gates generation; changing connections drops stale results.
export class QueueMonitor {
  constructor(onUpdate, {fetcher = globalThis.fetch.bind(globalThis), intervalMs = 2000, timeoutMs = 4000} = {}) {
    Object.assign(this, {onUpdate, fetcher, intervalMs, timeoutMs, revision: 0});
  }

  start(url, options = {}) {
    this.stop();
    this.url = url;
    this.options = options;
    this.onUpdate({queued: null, detail: 'Checking queue…'});
    void this.poll(this.revision);
  }

  stop() {
    this.revision++;
    clearTimeout(this.timer);
    this.controller?.abort();
    this.url = this.options = null;
  }

  async poll(revision) {
    const controller = new AbortController();
    this.controller = controller;
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await this.fetcher(this.url, {...this.options, signal: controller.signal, cache: 'no-store'});
      if (!response.ok && response.status !== 503) {
        if (revision === this.revision) this.onUpdate({queued: null, detail: `Queue request failed (HTTP ${response.status})`});
        return;
      }
      const body = await response.json();
      if (revision !== this.revision) return;
      this.onUpdate(response.ok || body?.status === 'loading' ? queueSnapshot(body) : queueSnapshot(null));
    } catch (error) {
      if (revision === this.revision) this.onUpdate({queued: null, detail: controller.signal.aborted
        ? 'Queue request timed out' : error instanceof SyntaxError ? 'Queue endpoint returned invalid JSON'
          : 'Queue request blocked or unreachable; check gateway CORS and network'});
    } finally {
      clearTimeout(timeout);
      if (revision === this.revision) this.timer = setTimeout(() => { void this.poll(revision); }, this.intervalMs);
    }
  }
}
