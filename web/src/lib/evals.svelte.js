// Prompt suites, perplexity runs, and what they found.

import { api } from "./api.js";
import { watchJob } from "./jobwatch.js";
import { toasts } from "./toasts.svelte.js";

class Evals {
  suites = $state([]);
  runs = $state([]);
  /** The run being looked at: `{ job, cases, scores, log, progress }`. */
  open = $state(null);
  #watching = null;

  async refresh() {
    try {
      [this.suites, this.runs] = await Promise.all([
        api("/api/evals/suites"),
        api("/api/evals/runs"),
      ]);
    } catch (e) {
      toasts.error(e.message);
    }
  }

  async watch(id) {
    if (this.#watching?.id === id) return;
    this.#watching?.stop.abort();
    let run;
    try {
      run = await api(`/api/evals/runs/${id}`);
    } catch (e) {
      toasts.error(e.message);
      return;
    }
    const { cases, scores, ...job } = run;
    this.open = { job, cases, scores, log: [], progress: null };
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
      case "case": {
        const at = this.open.cases.findIndex((c) => c.variant === u.variant && c.idx === u.idx);
        if (at >= 0) this.open.cases[at] = u;
        else this.open.cases.push(u);
        break;
      }
      case "scored": {
        const at = this.open.scores.findIndex((s) => s.variant === u.variant);
        if (at >= 0) this.open.scores[at] = u;
        else this.open.scores.push(u);
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
        if (u.state === "failed") toasts.error(u.error ?? "the run failed");
        this.refresh();
        api(`/api/evals/runs/${id}`)
          .then(({ cases, scores, ...job }) => {
            if (this.open?.job.id === id) this.open = { ...this.open, job, cases, scores };
          })
          .catch(() => {});
        break;
    }
  }

  async saveSuite(suite) {
    try {
      const body = JSON.stringify({ name: suite.name, cases: suite.cases });
      const saved = suite.id
        ? await api(`/api/evals/suites/${suite.id}`, {
            method: "PATCH",
            headers: { "content-type": "application/json" },
            body,
          })
        : await api("/api/evals/suites", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body,
          });
      toasts.success(`Saved ${saved.name}.`);
      await this.refresh();
      return saved;
    } catch (e) {
      toasts.error(e.message);
      return null;
    }
  }

  async deleteSuite(id) {
    try {
      await api(`/api/evals/suites/${id}`, { method: "DELETE" });
      await this.refresh();
    } catch (e) {
      toasts.error(e.message);
    }
  }

  async run(path, request) {
    try {
      const job = await api(path, {
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
      toasts.info("Asked it to stop.");
    } catch (e) {
      toasts.error(e.message);
    }
  }
}

export const evals = new Evals();

/** The cases of a run, as one row per case and one column per variant. */
export function matrix(cases) {
  const variants = [...new Set(cases.map((c) => c.variant))];
  const rows = [...new Set(cases.map((c) => c.idx))].sort((a, b) => a - b);
  return {
    variants,
    rows: rows.map((idx) => {
      const any = cases.find((c) => c.idx === idx);
      return {
        idx,
        prompt: any?.prompt ?? "",
        expect: any?.expect ?? "",
        by: variants.map((v) => cases.find((c) => c.idx === idx && c.variant === v) ?? null),
      };
    }),
  };
}
