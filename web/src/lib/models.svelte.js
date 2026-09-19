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

  async pull(repo) {
    const ok = await this.#watch("pulling", "/api/models/pull", { repo }, "pulled");
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

  async forgetQuantised(repo) {
    try {
      const { files } = await api(`/api/qcache?repo=${encodeURIComponent(repo)}`, {
        method: "DELETE",
      });
      toasts.info(`Threw away ${files} quantised file${files === 1 ? "" : "s"} for ${repo}.`);
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
