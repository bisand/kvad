// Raw completion, and the two views that come free with it.
//
// The Chat page's store keeps conversations; this one keeps a single
// throwaway. That is the difference between the pages: chat is something you
// come back to, the playground is something you change one knob at a time
// until you understand what the knob does.

import { api, sse } from "./api.js";
import { toasts } from "./toasts.svelte.js";
import { models } from "./models.svelte.js";

/** The sampler, as the playground presents it. */
const DEFAULTS = {
  temperature: 0.8,
  top_k: 40,
  top_p: 0.95,
  max_tokens: 128,
  seed: 1337,
  fixSeed: true,
  explain: 0,
};

class Playground {
  prompt = $state("The history of the transformer architecture begins with");
  settings = $state({ ...DEFAULTS });

  /** Tokens as they arrive: `{ text, id, top }`. */
  tokens = $state([]);
  stats = $state(null);
  running = $state(false);
  error = $state(null);
  #stop = null;

  /** The tokeniser inspector's last answer. */
  split = $state(null);
  splitting = $state(false);

  get text() {
    return this.tokens.map((t) => t.text).join("");
  }

  reset() {
    this.tokens = [];
    this.stats = null;
    this.error = null;
  }

  async complete() {
    if (this.running) return;
    this.reset();
    this.running = true;
    this.#stop = new AbortController();
    const s = this.settings;
    try {
      await sse(
        "/api/playground/complete",
        {
          prompt: this.prompt,
          temperature: s.temperature,
          top_k: s.top_k,
          top_p: s.top_p,
          max_tokens: s.max_tokens,
          // An unfixed seed continues the engine's own generator, so two
          // identical requests differ — which is the right default for
          // chatting and the wrong one for experimenting.
          seed: s.fixSeed ? s.seed : null,
          explain: s.explain,
          model: models.loaded?.id ?? null,
        },
        {
          token: (data) => {
            const t = JSON.parse(data);
            this.tokens.push({ text: t.text, id: t.id ?? null, top: t.top ?? null });
          },
          done: (data) => {
            this.stats = JSON.parse(data);
          },
          error: (data) => {
            throw new Error(JSON.parse(data).error);
          },
        },
        this.#stop.signal,
      );
    } catch (e) {
      if (e.name !== "AbortError") {
        this.error = e.message;
        toasts.error(e.message);
      }
    } finally {
      this.running = false;
      this.#stop = null;
    }
  }

  /** Stop generating. The engine's flag, through the same route chat uses. */
  async stop() {
    try {
      await api("/api/generation", { method: "DELETE" });
    } catch (e) {
      toasts.error(e.message);
    }
    this.#stop?.abort();
  }

  async tokenize(text) {
    this.splitting = true;
    try {
      this.split = await api("/api/playground/tokenize", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text, model: models.loaded?.id ?? null }),
      });
      this.error = null;
    } catch (e) {
      this.split = null;
      this.error = e.message;
    } finally {
      this.splitting = false;
    }
  }
}

export const playground = new Playground();

/**
 * A colour for a probability, from the daisyUI palette.
 *
 * Four bands rather than a gradient: a continuous scale looks precise and
 * reads as nothing, and the useful question is "was the model sure?", which
 * has about four answers.
 */
export function confidence(p) {
  if (p >= 0.6) return "text-success";
  if (p >= 0.25) return "text-info";
  if (p >= 0.05) return "text-warning";
  return "text-error";
}

export function percent(p) {
  if (p == null) return "—";
  if (p >= 0.1) return `${(p * 100).toFixed(0)}%`;
  if (p >= 0.001) return `${(p * 100).toFixed(1)}%`;
  return "<0.1%";
}
