// Accept a server origin or its complete Messages URL. Empty means unconfigured.
export function normalizeEndpoint(value) {
  const raw = value.trim();
  if (!raw) return '';
  let url;
  try { url = new URL(raw); } catch { throw new Error('Enter a full endpoint URL, such as https://api.example.com.'); }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || !['/', '/v1/messages', '/v1/messages/'].includes(url.pathname)) {
    throw new Error('Use an HTTP(S) origin or /v1/messages URL without credentials, a query, or fragment.');
  }
  return url.origin;
}

export function apiURL(endpoint, path) {
  return `${endpoint}${path}`;
}

const credentialsStorageKey = 'naivert-studio-credentials';

// Keep one applied credential, bound to its API origin (including scheme/port).
// Access storage inside try: some browsers disable even its property getter.
export function loadCredentials(endpoint, storage) {
  try {
    const saved = JSON.parse((storage ?? globalThis.localStorage).getItem(credentialsStorageKey));
    if (!endpoint || saved?.endpoint !== normalizeEndpoint(endpoint)) return {};
    const auth = saved.auth;
    if (!auth || typeof auth.type !== 'string' || typeof auth.secret !== 'string'
      || typeof auth.username !== 'string' || typeof auth.cookies !== 'boolean') return {};
    authOptions(auth);
    return auth;
  } catch { return {}; }
}

export function saveCredentials(endpoint, auth, storage) {
  try {
    const origin = normalizeEndpoint(endpoint);
    if (!origin) return false;
    authOptions(auth);
    const clean = {type: auth.type || 'none', secret: auth.type && auth.type !== 'none' ? auth.secret : '',
      username: auth.type === 'basic' ? auth.username : '', cookies: Boolean(auth.cookies)};
    (storage ?? globalThis.localStorage).setItem(credentialsStorageKey, JSON.stringify({endpoint: origin, auth: clean}));
    return true;
  } catch { return false; }
}

export function clearCredentials(storage) {
  try {
    (storage ?? globalThis.localStorage).removeItem(credentialsStorageKey);
    return true;
  } catch { return false; }
}

// Parse data only: never evaluate pasted shell commands or persist this line.
export function parseQuickSetup(value) {
  const parts = value.trim().split(/\s+/);
  if (!value.trim() || parts.length > 3) throw new Error('Paste URL API_KEY [MODEL], or just a URL without authentication.');
  const endpoint = normalizeEndpoint(parts[0]);
  const auth = parts[1] ? {type: 'bearer', secret: parts[1]} : {type: 'none'};
  authOptions(auth);
  return {endpoint, auth, model: parts[2] || null};
}

export function messagesOptions(auth, sessionId) {
  const options = healthOptions(auth, sessionId);
  return {...options, method: 'POST', headers: {...options.headers, 'Content-Type': 'application/json',
    'x-naivert-token-progress': '1'}};
}

export function healthOptions(auth, sessionId) {
  const options = authOptions(auth);
  return {...options, method: 'GET', headers: {...options.headers, 'X-Session-Id': sessionId}};
}

export function authOptions({type = 'none', secret = '', username = '', cookies = false} = {}) {
  const headers = {};
  if (![ 'none', 'bearer', 'api-key', 'basic' ].includes(type)) throw new Error('Choose a supported authentication method.');
  if (type !== 'none') {
    if (!secret || /[\r\n]/.test(secret) || /[\r\n]/.test(username)) throw new Error('Enter a credential without line breaks.');
    if (type === 'bearer') headers.Authorization = `Bearer ${secret}`;
    if (type === 'api-key') headers['x-api-key'] = secret;
    if (type === 'basic') {
      if (!username || username.includes(':')) throw new Error('Basic authentication needs a username without a colon.');
      const bytes = new TextEncoder().encode(`${username}:${secret}`);
      headers.Authorization = `Basic ${btoa(Array.from(bytes, byte => String.fromCharCode(byte)).join(''))}`;
    }
  }
  return {headers, credentials: cookies ? 'include' : 'omit', redirect: 'error'};
}
