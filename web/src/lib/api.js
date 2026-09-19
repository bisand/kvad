// Talking to kvad-serve.
//
// One wrapper, so that every call fails the same way. The server answers
// errors as `{"error": "..."}`; anything else — a proxy's HTML error page, a
// dead connection — becomes a message that at least says the status, because
// "undefined" in a toast tells nobody anything.
//
// Every message it throws is a whole sentence about what went wrong. Callers
// show it as it is and do not add a prefix of their own, or the result reads
// "Could not reach the server: could not reach the server: Failed to fetch".

/**
 * @param {string} path e.g. "/api/health"
 * @param {RequestInit} [options]
 */
export async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: { accept: "application/json", ...options.headers },
      ...options,
    });
  } catch (e) {
    // `fetch` rejects for a refused connection, DNS, CORS and an offline
    // machine alike, and says "Failed to fetch" for all of them. Name what we
    // were doing, and keep the browser's words in case they are the useful
    // half.
    throw new Error(`The server did not answer ${path} (${e.message}).`);
  }

  const body = await response.text();
  let parsed = null;
  try {
    parsed = body ? JSON.parse(body) : null;
  } catch {
    // Not JSON. Only interesting if the request also failed.
  }

  if (!response.ok) {
    const why = parsed?.error ?? `${response.status} ${response.statusText}`;
    throw new Error(`${path} failed: ${why}.`);
  }
  return parsed;
}

export const health = () => api("/api/health");
