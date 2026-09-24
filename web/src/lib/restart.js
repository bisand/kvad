// Waiting for the server to come back from a restart.
//
// Back when it answers after having stopped answering, or answers with an
// uptime younger than the one it had: a poll can fall either side of the
// moment it was down.

import { health } from "./api.js";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** The uptime to measure a restart against, taken before asking for it. */
export async function uptimeNow() {
  try {
    return (await health()).uptime_secs;
  } catch {
    // Measured against nothing, the first answer after a gap is enough.
    return Infinity;
  }
}

/** True when the server is back within `ms`, false when it is not. */
export async function backFromRestart(before, ms = 90_000) {
  const until = Date.now() + ms;
  let away = false;
  while (Date.now() < until) {
    await sleep(1000);
    try {
      const h = await health();
      if (away || h.uptime_secs < before) return true;
    } catch {
      away = true;
    }
  }
  return false;
}
