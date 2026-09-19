// What the server has been doing, for the two pages that show it.

import { api } from "./api.js";

class Monitoring {
  overview = $state(null);
  requests = $state(null);
  log = $state([]);
  error = $state(null);

  async refreshOverview() {
    try {
      this.overview = await api("/api/metrics");
      this.error = null;
    } catch (e) {
      this.error = e.message;
    }
  }

  async refreshDetail() {
    try {
      [this.requests, this.log] = await Promise.all([
        api("/api/metrics/requests?limit=150"),
        api("/api/metrics/log?limit=300"),
      ]);
      this.error = null;
    } catch (e) {
      this.error = e.message;
    }
  }
}

export const monitoring = new Monitoring();

/** A duration in milliseconds, at a sensible precision for its size. */
export function ms(v) {
  if (v == null) return "—";
  if (v < 1) return `${v.toFixed(2)} ms`;
  if (v < 1000) return `${v.toFixed(1)} ms`;
  return `${(v / 1000).toFixed(2)} s`;
}

export function median(values) {
  if (!values?.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}
