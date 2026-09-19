<script>
  // The real dashboard is Phase 5. What is here now is the one thing this
  // phase can honestly show: whether the browser and the server agree that
  // they are talking to each other, and what the server says it is.
  import { pageFor } from "../lib/pages.js";

  let { health, error } = $props();

  const page = pageFor("/");

  function duration(secs) {
    if (secs == null) return "—";
    if (secs < 60) return `${secs}s`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
    return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
  }
</script>

{#if error}
  <div role="alert" class="alert alert-error">
    <span>{error}</span>
  </div>
{:else if health}
  <div class="stats stats-vertical sm:stats-horizontal w-full shadow-sm">
    <div class="stat">
      <div class="stat-title">Server</div>
      <div class="stat-value text-2xl">v{health.version}</div>
      <div class="stat-desc">up {duration(health.uptime_secs)}</div>
    </div>
    <div class="stat">
      <div class="stat-title">Database</div>
      <div class="stat-value text-2xl">schema {health.schema}</div>
      <div class="stat-desc">SQLite, WAL</div>
    </div>
    <div class="stat">
      <div class="stat-title">Signed in as</div>
      <div class="stat-value text-2xl">{health.you.name}</div>
      <div class="stat-desc">{health.you.role}</div>
    </div>
    <div class="stat">
      <div class="stat-title">Web UI</div>
      <div class="stat-value text-2xl">
        {health.ui_embedded ? "embedded" : "dev"}
      </div>
      <div class="stat-desc">
        {health.ui_embedded ? "served from the binary" : "served by Vite"}
      </div>
    </div>
  </div>
{:else}
  <div class="stats stats-vertical sm:stats-horizontal w-full shadow-sm">
    {#each [0, 1, 2, 3] as i (i)}
      <div class="stat gap-2">
        <div class="skeleton h-3 w-20"></div>
        <div class="skeleton h-7 w-28"></div>
        <div class="skeleton h-3 w-24"></div>
      </div>
    {/each}
  </div>
{/if}

<div class="divider"></div>

<div class="mx-auto max-w-2xl text-center">
  <p class="opacity-70">{page.blurb}</p>
  <div class="mt-4 inline-flex items-center gap-2 text-sm opacity-60">
    <span class="badge badge-sm">Phase {page.phase}</span>
    <span>the rest of this page is not built yet</span>
  </div>
</div>
