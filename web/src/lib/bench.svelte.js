// Benchmark runs: starting one, watching it, and reading the ones before it.
//
// The numbers here are the project's own claim about itself, so the store
// keeps every sample rather than a running median. A summary that cannot be
// recomputed from what was measured is a summary nobody can check.

import { api } from "./api.js";
import { watchJob } from "./jobwatch.js";
import { toasts } from "./toasts.svelte.js";

class Bench {
  runs = $state([]);
  /** The run being looked at: `{ job, samples, summary, log, progress }`. */
  open = $state(null);
  #watching = null;

  async refresh() {
    try {
      this.runs = await api("/api/bench/runs");
    } catch (e) {
      toasts.error(e.message);
    }
  }

  async watch(id) {
    if (this.#watching?.id === id) return;
    this.#watching?.stop.abort();
    let run;
    try {
      run = await api(`/api/bench/runs/${id}`);
    } catch (e) {
      toasts.error(e.message);
      return;
    }
    const { samples, summary, ...job } = run;
    this.open = { job, samples, summary, log: [], progress: null };
    this.#watching = {
      id,
      stop: watchJob(id, {
        onUpdate: (u) => this.#apply(id, u),
        onError: (e) => toasts.error(e.message),
      }),
    };
  }

  #apply(id, u) {
    if (this.open?.job.id !== id) return;
    switch (u.kind) {
      case "timing": {
        // Keyed by variant and round: the history is replayed before the live
        // tail, so the same sample can arrive twice.
        const at = this.open.samples.findIndex(
          (s) => s.variant === u.variant && s.round === u.round,
        );
        if (at >= 0) this.open.samples[at] = u;
        else this.open.samples.push(u);
        this.open.summary = summarise(this.open.samples);
        break;
      }
      case "progress":
        this.open.progress = u;
        break;
      case "status":
        this.open.log.push(u.message);
        if (this.open.log.length > 100) this.open.log.shift();
        break;
      case "ended":
        this.open.job = { ...this.open.job, state: u.state, error: u.error ?? null };
        if (u.state === "failed") toasts.error(u.error ?? "the benchmark failed");
        this.refresh();
        api(`/api/bench/runs/${id}`)
          .then(({ samples, summary, ...job }) => {
            if (this.open?.job.id === id) this.open = { ...this.open, job, samples, summary };
          })
          .catch(() => {});
        break;
    }
  }

  async start(request) {
    try {
      const job = await api("/api/bench/run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(request),
      });
      await this.refresh();
      await this.watch(job.id);
      return job;
    } catch (e) {
      toasts.error(e.message);
      return null;
    }
  }

  async cancel(id) {
    try {
      await api(`/api/jobs/${id}`, { method: "DELETE" });
      toasts.info("Asked it to stop. It finishes the generation it is on.");
    } catch (e) {
      toasts.error(e.message);
    }
  }
}

export const bench = new Bench();

/**
 * The same summary the server computes, from the samples in hand.
 *
 * Done twice on purpose: the server's copy is what a finished run stores, and
 * this one is what a run still going can show. Both take the middle of the
 * sorted samples and the two ends, so a watcher and a reader of history see
 * the same arithmetic.
 */
export function summarise(samples) {
  const names = [...new Set(samples.map((s) => s.variant))];
  return names.map((variant) => {
    const mine = samples.filter((s) => s.variant === variant);
    const decode = mine.map((s) => s.decode_per_sec).sort((a, b) => a - b);
    const ttft = mine.map((s) => s.ttft_millis).sort((a, b) => a - b);
    return {
      variant,
      runs: mine.length,
      decode_median: middle(decode),
      decode_low: decode[0],
      decode_high: decode[decode.length - 1],
      ttft_median: middle(ttft),
      ttft_low: ttft[0],
      ttft_high: ttft[ttft.length - 1],
    };
  });
}

/** The middle of a sorted list, and with an even count one of the two. */
function middle(sorted) {
  return sorted.length ? sorted[Math.floor(sorted.length / 2)] : null;
}
