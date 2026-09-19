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

/**
 * Read a stream of Server-Sent Events.
 *
 * Not `EventSource`, for two reasons: it can only issue GET, so it cannot
 * carry the body that loading a model or asking for a completion needs; and
 * it reconnects on its own, which for a job that has ended means replaying
 * the whole history forever. `fetch` does neither.
 *
 * `on` maps an event name to a handler. The default name for an event with no
 * `event:` line is "message", which is what /v1/chat/completions sends.
 *
 * @param {string} path
 * @param {object} [body] omitted for a GET
 * @param {Record<string, (data: string) => void>} on
 * @param {AbortSignal} [signal]
 * @param {"POST"|"GET"} [method]
 */
export async function sse(path, body, on, signal, method = "POST") {
  const response = await fetch(path, {
    method,
    headers:
      method === "GET"
        ? { accept: "text/event-stream" }
        : { "content-type": "application/json", accept: "text/event-stream" },
    body: method === "GET" ? undefined : JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    // A failure before the stream opened is an ordinary error response, and
    // says more than "the stream would not start".
    let why = `${response.status} ${response.statusText}`;
    try {
      why = (await response.json())?.error ?? why;
    } catch {}
    throw new Error(`${path} failed: ${why}.`);
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    // Events are separated by a blank line. A chunk can end mid-event, so
    // whatever is after the last separator stays in the buffer.
    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      let name = "message";
      const lines = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) lines.push(line.slice(5).trimStart());
        // A line starting with ":" is a keep-alive comment; ignore it.
      }
      if (lines.length) on[name]?.(lines.join("\n"));
    }
  }
}
