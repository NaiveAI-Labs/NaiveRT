import {consumeSSE, Reply} from './core.js';
import {apiURL, messagesOptions} from './endpoint.js';

// Check the actual inference route: gateways may not expose /health or /v1/models.
export async function checkConnection({endpoint, model, auth, sessionId,
  fetcher = globalThis.fetch.bind(globalThis), timeoutMs = 20000}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  const start = performance.now();
  try {
    const response = await fetcher(apiURL(endpoint, '/v1/messages'), {
      ...messagesOptions(auth, sessionId), signal: controller.signal,
      body: JSON.stringify({model, max_tokens: 1, stream: true, thinking: {type: 'disabled'},
        metadata: {user_id: sessionId}, messages: [{role: 'user', content: 'Reply OK.'}]}),
    });
    if (!response.ok) {
      await response.body?.cancel().catch(() => {});
      const reason = response.status === 401 || response.status === 403 ? 'Authentication rejected; check credentials and access.'
        : response.status === 429 || response.status === 529 ? 'Server reachable, but its queue or rate limit is full.'
        : response.status === 503 ? 'Server reachable, but currently unavailable or loading.'
        : 'Request rejected; check the model and gateway configuration.';
      return {ok: false, message: `HTTP ${response.status}. ${reason}`};
    }
    const reply = new Reply();
    await consumeSSE(response, event => reply.accept(event, performance.now() - start));
    if (!reply.valid) return {ok: false, message: 'Server responded, but the Messages stream did not complete correctly.'};
    return {ok: true, message: `Connected · streaming verified · ${((performance.now() - start) / 1000).toFixed(2)} s`};
  } catch (error) {
    if (controller.signal.aborted) return {ok: false, message: `Check timed out after ${timeoutMs / 1000} s; the server may be queued or unavailable.`};
    if (error instanceof TypeError) return {ok: false, message: 'Browser cannot reach the API. Check gateway CORS / OPTIONS, HTTPS, and network access.'};
    return {ok: false, message: 'Server responded, but returned an invalid or failed Messages stream.'};
  } finally { clearTimeout(timeout); }
}
