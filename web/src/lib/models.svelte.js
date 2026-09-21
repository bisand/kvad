// What is on this machine, and what the engine is holding.
//
// One store rather than each page fetching for itself: the Models page, the
// Chat page's header and the navbar all want the same answer, and three
// copies of it would disagree the moment one of them loaded something.

import { api, sse } from "./api.js";
import { toasts } from "./toasts.svelte.js";

class Models {
  /** The last listing, or null before the first one arrives. */
  listing = $state(null);
  loading = $state(false);
  /** Set while a load or a pull is running: `{ what, message, bytes, total }`. */
  busy = $state(null);
  error = $state(null);

  async refresh() {
    this.loading = true;
    try {
      this.listing = await api("/api/models");
      this.error = null;
    } catch (e) {
      this.error = e.message;
    } finally {
      this.loading = false;
    }
  }

  get loaded() {
    return this.listing?.loaded ?? null;
  }

  /** Every model on this machine, downloaded and trained, in one list. */
  get all() {
    if (!this.listing) return [];
    return [...this.listing.trained, ...this.listing.downloaded];
  }

  /** Watch a load or a pull, keeping `busy` up to date as it goes. */
  async #watch(what, path, body, finished) {
    if (this.busy) {
      toasts.warning(`Already ${this.busy.what}. Wait for it to finish.`);
      return false;
    }
    this.busy = { what, message: "starting", bytes: 0, total: 0 };
    let ok = false;
    try {
      await sse(path, body, {
        progress: (data) => {
          const p = JSON.parse(data);
          if (p.kind === "download") {
            this.busy = { what, message: p.file, bytes: p.bytes, total: p.total };
          } else if (p.kind === "fetched") {
            this.busy = { what, message: `fetched ${p.file}`, bytes: 0, total: 0 };
          } else {
            this.busy = { what, message: p.message, bytes: 0, total: 0 };
          }
        },
        [finished]: (data) => {
          ok = true;
          this.busy = null;
        },
        error: (data) => {
          throw new Error(JSON.parse(data).error);
        },
      });
    } catch (e) {
      toasts.error(e.message);
    } finally {
      this.busy = null;
      await this.refresh();
    }
    return ok;
  }

  async load(repo, backend) {
    const ok = await this.#watch("loading", "/api/models/load", { repo, backend }, "loaded");
    if (ok) toasts.success(`Loaded ${repo}.`);
    return ok;
  }

  /**
   * Pull a model. Since Phase 4 this is a job, not a stream: a checkpoint of
   * several gigabytes outlasts a browser tab, and a download that died
   * because somebody navigated away is a download that has to start again.
   * Closing this page now only stops the watching.
   */
  async pull(repo) {
    if (this.busy) {
      toasts.warning(`Already ${this.busy.what}. Wait for it to finish.`);
      return false;
    }
    let job;
    try {
      job = await api("/api/models/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ repo }),
      });
    } catch (e) {
      toasts.error(e.message);
      return false;
    }

    this.busy = { what: "pulling", message: "starting", bytes: 0, total: 0 };
    let ok = false;
    try {
      await sse(
        `/api/jobs/${job.id}/events`,
        undefined,
        {
          update: (data) => {
            const u = JSON.parse(data);
            if (u.kind === "download") {
              this.busy = { what: "pulling", message: u.file, bytes: u.bytes, total: u.total };
            } else if (u.kind === "status") {
              this.busy = { what: "pulling", message: u.message, bytes: 0, total: 0 };
            } else if (u.kind === "ended") {
              ok = u.state === "done";
              if (!ok && u.error) throw new Error(u.error);
            }
          },
        },
        undefined,
        "GET",
      );
    } catch (e) {
      toasts.error(e.message);
    } finally {
      this.busy = null;
      await this.refresh();
    }
    if (ok) toasts.success(`Pulled ${repo}.`);
    return ok;
  }

  async unload() {
    try {
      const { unloaded } = await api("/api/models/unload", { method: "POST" });
      toasts.info(unloaded ? `Unloaded ${unloaded}.` : "Nothing was loaded.");
    } catch (e) {
      toasts.error(e.message);
    }
    await this.refresh();
  }

  async setActive(repo) {
    try {
      await api("/api/models/active", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ repo }),
      });
      toasts.info(repo ? `${repo} is the default model.` : "No default model.");
    } catch (e) {
      toasts.error(e.message);
    }
    await this.refresh();
  }

  async remove(id) {
    try {
      await api(`/api/models?id=${encodeURIComponent(id)}`, { method: "DELETE" });
      toasts.success(`Deleted ${id}.`);
    } catch (e) {
      toasts.error(e.message);
    }
    await this.refresh();
  }

  /** One row of the quantised-weights table: this model at this precision. */
  async forgetQuantised(repo, precision) {
    const where = `repo=${encodeURIComponent(repo)}&precision=${encodeURIComponent(precision)}`;
    try {
      await api(`/api/qcache?${where}`, { method: "DELETE" });
      toasts.info(`Threw away the ${precision} weights for ${repo}.`);
    } catch (e) {
      toasts.error(e.message);
    }
    await this.refresh();
  }

  async search(query) {
    return await api(`/api/models/search?q=${encodeURIComponent(query)}`);
  }
}

export const models = new Models();

/** Bytes as something to read: "1.5 GB". Mirrors `hub::human_bytes`. */
export function humanBytes(b) {
  if (b == null) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = b;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u += 1;
  }
  return u === 0 ? `${b} B` : `${v.toFixed(1)} ${units[u]}`;
}
