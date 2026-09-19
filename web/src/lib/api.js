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
 * POST something and read the Server-Sent Events it answers with.
 *
 * Not `EventSource`, which can only issue GET and so cannot carry a body.
 * Loading a model, pulling one and streaming a reply all need a request body,
 * so all three are POST with an SSE response and all three are read here.
 *
 * `on` maps an event name to a handler. The default name for an event without
 * one is "message", which is what /v1/chat/completions sends.
 *
 * @param {string} path
 * @param {object} body
 * @param {Record<string, (data: string) => void>} on
 * @param {AbortSignal} [signal]
 */
export async function sse(path, body, on, signal) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify(body),
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
