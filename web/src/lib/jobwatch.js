// Following a job, for the pages that are not the Training page.
//
// Evals and benchmarks are jobs like a training run is, and they are watched
// the same way: subscribe, read the history the server replays, then follow
// the live tail. The Training page grew this logic first and keeps its own
// because its updates fold into a chart; everything else needs these six
// lines and not a copy of that.

import { sse } from "./api.js";

/**
 * Watch job `id`. Returns the `AbortController` that stops watching.
 *
 * Updates that arrive before the history has finished replaying are not a
 * problem to solve here: the server sends the history first, and every kind
 * of update is keyed by something — a case by its variant and index, a
 * benchmark sample by its variant and round — so a repeat replaces rather
 * than appends.
 */
export function watchJob(id, { onUpdate, onError }) {
  const stop = new AbortController();
  sse(
    `/api/jobs/${id}/events`,
    undefined,
    { update: (data) => onUpdate(JSON.parse(data)) },
    stop.signal,
    "GET",
  ).catch((e) => {
    if (e.name !== "AbortError") onError?.(e);
  });
  return stop;
}
