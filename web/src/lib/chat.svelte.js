// A conversation, and the turn being generated into it.
//
// Two things are going on and it is worth being clear about which is which.
// The *reply* comes from `/v1/chat/completions`, which is stateless, knows
// nothing about conversations, and is exactly the endpoint any other client
// would use — so our own UI exercises the compatible path every day rather
// than a private one. The *record* of what was said goes to /api/conversations
// afterwards. Nothing is saved as a side effect of asking a question.

import { api, sse } from "./api.js";
import { toasts } from "./toasts.svelte.js";

/** What the sampler does when nobody touches the controls. Matches the server. */
export const DEFAULTS = { temperature: 0.7, top_p: 0.95, top_k: 40, max_tokens: 512 };

class Chat {
  /** Every conversation, newest first. */
  list = $state([]);
  /** The open one: `{ id, title, system, ... }`. */
  current = $state(null);
  messages = $state([]);
  /** The reply as it arrives, before it is a message. */
  streaming = $state(null);
  /**
   * A reasoning model's working, as it arrives.
   *
   * Shown while the reply is being written and then dropped: the working is
   * scaffolding, it contradicts itself on the way to an answer, and it is
   * often longer than the answer. The message that is kept is the answer.
   */
  thinking = $state(null);
  sending = $state(false);
  sampler = $state({ ...DEFAULTS });

  #abort = null;

  async refresh() {
    try {
      this.list = await api("/api/conversations");
    } catch (e) {
      toasts.error(e.message);
    }
  }

  async open(id) {
    try {
      const full = await api(`/api/conversations/${id}`);
      const { messages, ...conversation } = full;
      this.current = conversation;
      this.messages = messages;
      this.streaming = null;
    } catch (e) {
      toasts.error(e.message);
      this.current = null;
      this.messages = [];
    }
  }

  async create(system = null) {
    try {
      const made = await api("/api/conversations", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ system }),
      });
      await this.refresh();
      this.current = made;
      this.messages = [];
      this.streaming = null;
      return made;
    } catch (e) {
      toasts.error(e.message);
      return null;
    }
  }

  async remove(id) {
    try {
      await api(`/api/conversations/${id}`, { method: "DELETE" });
      if (this.current?.id === id) {
        this.current = null;
        this.messages = [];
      }
      await this.refresh();
    } catch (e) {
      toasts.error(e.message);
    }
  }

  async setSystem(text) {
    if (!this.current) return;
    try {
      this.current = await api(`/api/conversations/${this.current.id}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ system: text || null }),
      });
      await this.refresh();
    } catch (e) {
      toasts.error(e.message);
    }
  }

  async rename(id, title) {
    try {
      await api(`/api/conversations/${id}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ title }),
      });
      if (this.current?.id === id) this.current = { ...this.current, title };
      await this.refresh();
    } catch (e) {
      toasts.error(e.message);
    }
  }

  /** The transcript as /v1/chat/completions wants it. */
  #turns(next) {
    const turns = [];
    if (this.current?.system) turns.push({ role: "system", content: this.current.system });
    for (const m of this.messages) turns.push({ role: m.role, content: m.content });
    turns.push({ role: "user", content: next });
    return turns;
  }

  async #save(role, content, stats, titleIfUnnamed = false) {
    const stored = await api(`/api/conversations/${this.current.id}/messages`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ role, content, stats, title_if_unnamed: titleIfUnnamed }),
    });
    this.messages.push(stored);
    return stored;
  }

  /** Ask, stream the reply, then save both halves. */
  async send(text, model, dataset = null) {
    if (this.sending) return;
    const question = text.trim();
    if (!question) return;
    if (!this.current && !(await this.create())) return;

    const turns = this.#turns(question);
    this.sending = true;
    this.streaming = "";
    this.#abort = new AbortController();

    let reply = "";
    let working = "";
    let stats = null;
    let failed = null;

    try {
      // Saved before the reply, so a generation that dies halfway does not
      // lose the question as well.
      await this.#save("user", question, null, true);
      await sse(
        "/v1/chat/completions",
        // `dataset` is kvad's own: the server searches it with the question
        // and puts what it finds in front of the model. Left out entirely
        // when there is none, so the request stays a plain OpenAI one.
        {
          model,
          messages: turns,
          stream: true,
          ...(dataset ? { dataset } : {}),
          ...this.sampler,
        },
        {
          message: (data) => {
            if (data === "[DONE]") return;
            const chunk = JSON.parse(data);
            const delta = chunk.choices?.[0]?.delta;
            // The server tells the two apart as they stream; see
            // `kvad::chat::Thinking`.
            if (delta?.reasoning_content) {
              working += delta.reasoning_content;
              this.thinking = working;
            }
            if (delta?.content) {
              reply += delta.content;
              this.streaming = reply;
            }
            if (chunk.kvad) {
              stats = {
                model: chunk.model ?? null,
                backend: chunk.kvad.backend ?? null,
                prompt_tokens: chunk.usage?.prompt_tokens ?? 0,
                cached_tokens: chunk.kvad.cached_tokens ?? 0,
                generated_tokens: chunk.usage?.completion_tokens ?? 0,
                prefill_secs: chunk.kvad.prefill_secs ?? 0,
                decode_secs: chunk.kvad.decode_secs ?? 0,
              };
            }
          },
          error: (data) => {
            failed = JSON.parse(data).error;
          },
        },
        this.#abort.signal,
      );
    } catch (e) {
      // An abort is the stop button, not a failure. Whatever arrived before
      // it is still a reply and is kept.
      if (e.name !== "AbortError") failed = e.message;
    } finally {
      this.#abort = null;
      this.streaming = null;
      this.thinking = null;
      this.sending = false;
    }

    if (reply) await this.#save("assistant", reply, stats).catch((e) => toasts.error(e.message));
    if (failed) toasts.error(failed);
    await this.refresh();
  }

  /** Stop the generation. The engine is told as well as the stream. */
  async stop() {
    if (!this.sending) return;
    try {
      await api("/api/generation", { method: "DELETE" });
    } catch (e) {
      toasts.error(e.message);
    }
    // Aborting the fetch alone would leave the engine generating into a
    // stream nobody reads; telling the engine alone would leave this request
    // waiting for tokens that never come. Both.
    this.#abort?.abort();
  }
}

export const chat = new Chat();

/** The one line of numbers under a reply. */
export function describeStats(s) {
  if (!s) return null;
  const decode = s.generated_tokens / Math.max(s.decode_secs, 1e-6);
  const computed = s.prompt_tokens - s.cached_tokens;
  const prefill = computed / Math.max(s.prefill_secs, 1e-6);
  const parts = [
    `${s.generated_tokens} tokens at ${decode.toFixed(1)}/s`,
    `prefill ${computed} at ${prefill.toFixed(0)}/s`,
  ];
  // Only worth saying when there was some: it is the KV cache doing its job,
  // and on the first turn of a conversation there is nothing to reuse.
  if (s.cached_tokens > 0) parts.push(`${s.cached_tokens} cached`);
  if (s.backend) parts.push(s.backend);
  return parts.join(" · ");
}
