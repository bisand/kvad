// Runs, and the text they are trained on.
//
// A run is a job on the server, so this store is mostly about *watching* one:
// subscribe to its events, fold them into a chart, and keep working when the
// tab is closed and reopened because the server never stopped.

import { api, sse } from "./api.js";
import { toasts } from "./toasts.svelte.js";

class Training {
  jobs = $state([]);
  datasets = $state([]);
  options = $state(null);

  /** The run being looked at: `{ job, metrics, samples }`. */
  open = $state(null);
  /** Lines of text from the run, newest last. Not persisted; the chart is. */
  log = $state([]);
  #watching = null;

  async refresh() {
    try {
      [this.jobs, this.datasets, this.options] = await Promise.all([
        api("/api/jobs"),
        api("/api/datasets"),
        api("/api/train/options"),
      ]);
    } catch (e) {
      toasts.error(e.message);
    }
  }

  get running() {
    return this.jobs.find((j) => j.kind === "train" && (j.state === "running" || j.state === "queued"));
  }

  /** Watch a job: its history first, then whatever happens next. */
  async watch(id) {
    if (this.#watching?.id === id) return;
    this.#watching?.stop.abort();

    let job;
    try {
      job = await api(`/api/jobs/${id}`);
    } catch (e) {
      toasts.error(e.message);
      return;
    }
    // The metrics and samples from the request are dropped in favour of the
    // ones the stream replays, so there is one path that builds them and one
    // rule for duplicates.
    this.open = { job: { ...job, metrics: undefined, samples: undefined }, metrics: [], samples: [] };
    this.log = [];

    const stop = new AbortController();
    this.#watching = { id, stop };

    sse(
      `/api/jobs/${id}/events`,
      undefined,
      {
        update: (data) => this.#apply(JSON.parse(data)),
      },
      stop.signal,
      "GET",
    ).catch((e) => {
      if (e.name !== "AbortError") toasts.error(e.message);
    });
  }

  #apply(u) {
    if (!this.open) return;
    switch (u.kind) {
      case "metric": {
        // Keyed by step: the subscription is taken before the history is
        // read, so the same step can arrive twice and the second one wins.
        const at = this.open.metrics.findIndex((m) => m.step === u.step);
        const saved = u.saved || (at >= 0 && this.open.metrics[at].saved);
        // A `saved` update carries no losses of its own — it only marks a
        // step the chart already has.
        const merged = Number.isFinite(u.train_loss)
          ? { ...u, saved }
          : { ...(this.open.metrics[at] ?? u), saved: true };
        if (at >= 0) this.open.metrics[at] = merged;
        else this.open.metrics.push(merged);
        break;
      }
      case "sample": {
        const at = this.open.samples.findIndex((s) => s.step === u.step);
        if (at >= 0) this.open.samples[at] = u;
        else this.open.samples.push(u);
        break;
      }
      case "status":
        this.log.push(u.message);
        if (this.log.length > 200) this.log.shift();
        break;
      case "pace":
        this.log.push(
          `${Math.round(u.chars_per_sec).toLocaleString()} characters a second here — about ${humanSecs(u.remaining_secs)} to go.`,
        );
        break;
      case "ended": {
        const id = this.open.job.id;
        this.open.job = { ...this.open.job, state: u.state, error: u.error ?? null };
        if (u.state === "failed") toasts.error(u.error ?? "the run failed");
        else if (u.state === "done") toasts.success(`${this.open.job.label} finished.`);
        // The row now carries the result and the timings, which the stream
        // does not: read it back rather than reconstruct it here.
        api(`/api/jobs/${id}`)
          .then(({ metrics, samples, ...job }) => {
            if (this.open?.job.id === id) this.open.job = job;
          })
          .catch(() => {});
        this.refresh();
        break;
      }
    }
  }

  /** The step whose model is on disk. */
  get bestStep() {
    const saved = this.open?.metrics.filter((m) => m.saved) ?? [];
    return saved.length ? saved[saved.length - 1].step : null;
  }

  async start(request) {
    try {
      const job = await api("/api/train", {
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
      toasts.info("Asked it to stop. It finishes the step it is on.");
    } catch (e) {
      toasts.error(e.message);
    }
  }

  async upload(name, text) {
    try {
      const made = await api(`/api/datasets?name=${encodeURIComponent(name)}`, {
        method: "POST",
        headers: { "content-type": "text/plain" },
        body: text,
      });
      toasts.success(`Uploaded ${made.name}.`);
      await this.refresh();
      return made;
    } catch (e) {
      toasts.error(e.message);
      return null;
    }
  }

  /**
   * Read a website into a dataset.
   *
   * Answers with the job, not the dataset: this takes minutes, and the page
   * that asked follows it the way the Training page follows a run.
   */
  async crawl(request) {
    try {
      const job = await api("/api/datasets/crawl", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(request),
      });
      await this.refresh();
      return job;
    } catch (e) {
      toasts.error(e.message);
      return null;
    }
  }

  /** A crawl that is still going, if there is one. */
  get crawling() {
    return this.jobs.find(
      (j) => j.kind === "crawl" && (j.state === "running" || j.state === "queued"),
    );
  }

  async removeDataset(id) {
    try {
      await api(`/api/datasets/${id}`, { method: "DELETE" });
      await this.refresh();
    } catch (e) {
      toasts.error(e.message);
    }
  }

  async check(datasetId, model) {
    return await api(`/api/datasets/${datasetId}/check?model=${encodeURIComponent(model)}`);
  }
}

export const training = new Training();

/** Seconds as something to read. Mirrors `nervus::text::human_secs`. */
export function humanSecs(secs) {
  const s = Math.max(0, Math.round(secs));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}
