<script>
  // What the machine is doing, in the numbers this server can measure
  // honestly about itself.
  import { monitoring, ms, median } from "../lib/monitoring.svelte.js";
  import { humanBytes } from "../lib/models.svelte.js";
  import { humanSecs } from "../lib/training.svelte.js";
  import { navigate } from "../lib/router.svelte.js";
  import Sparkline from "../lib/components/Sparkline.svelte";

  let { health, error } = $props();

  // Often enough that a load or a run is noticed, rarely enough that the
  // dashboard is not itself the busiest thing in the request log.
  const EVERY = 5000;

  $effect(() => {
    monitoring.refreshOverview();
    const timer = setInterval(() => monitoring.refreshOverview(), EVERY);
    return () => clearInterval(timer);
  });

  const m = $derived(monitoring.overview);
  const decode = $derived(median(m?.decode_per_sec));
  const ttft = $derived(median(m?.ttft_millis));
  const kvShare = $derived(m?.kv ? (m.kv.cached_bytes / m.kv.max_bytes) * 100 : 0);
</script>

{#if error}
  <div role="alert" class="alert alert-error">
    <span>{error}</span>
  </div>
{:else if !m}
  <div class="stats stats-vertical sm:stats-horizontal w-full shadow-sm">
    {#each [0, 1, 2, 3] as i (i)}
      <div class="stat gap-2">
        <div class="skeleton h-3 w-20"></div>
        <div class="skeleton h-7 w-28"></div>
        <div class="skeleton h-3 w-24"></div>
      </div>
    {/each}
  </div>
{:else}
  <div class="flex flex-col gap-6">
    <!-- The engine. -->
    <section class="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      <div class="card bg-base-100 border-base-300 border">
        <div class="card-body gap-1 p-4">
          <p class="text-xs opacity-60">Loaded</p>
          {#if m.loaded}
            <p class="truncate text-lg font-medium" title={m.loaded.repo}>
              {m.loaded.repo.split("/").at(-1)}
            </p>
            <p class="text-xs opacity-70">
              {m.loaded.backend} · {(m.loaded.params / 1e6).toFixed(1)}M parameters ·
              weights {humanBytes(m.loaded.weight_bytes)}
            </p>
          {:else}
            <p class="text-lg opacity-60">nothing</p>
            <p class="text-xs opacity-70">
              <a href="/models" onclick={(e) => navigate(e, "/models")} class="link">Load a model</a>
            </p>
          {/if}
        </div>
      </div>

      <div class="card bg-base-100 border-base-300 border">
        <div class="card-body gap-1 p-4">
          <p class="text-xs opacity-60">Decode</p>
          <p class="text-lg font-medium">
            {decode ? `${decode.toFixed(1)} tok/s` : "—"}
          </p>
          <Sparkline values={m.decode_per_sec} label="tokens per second, oldest first" />
          <p class="text-xs opacity-60">
            {m.decode_per_sec.length} generation{m.decode_per_sec.length === 1 ? "" : "s"} · median
          </p>
        </div>
      </div>

      <div class="card bg-base-100 border-base-300 border">
        <div class="card-body gap-1 p-4">
          <p class="text-xs opacity-60">Time to first token</p>
          <p class="text-lg font-medium">{ttft != null ? ms(ttft) : "—"}</p>
          <Sparkline values={m.ttft_millis} label="time to first token, oldest first" />
          <!-- Prefill, which is the wait before anything appears — a different
               complaint from a reply that arrives slowly. -->
          <p class="text-xs opacity-60">prefill, before anything appears</p>
        </div>
      </div>

      <div class="card bg-base-100 border-base-300 border">
        <div class="card-body gap-1 p-4">
          <p class="text-xs opacity-60">Queue</p>
          <p class="text-lg font-medium">{m.queue_depth}</p>
          <p class="text-xs opacity-70">
            {m.queue_depth === 0
              ? "idle"
              : `${m.queue_depth} waiting for the engine`}
          </p>
          <p class="mt-1 text-xs opacity-60">
            One model, one generation at a time. Requests queue in the order they arrive.
          </p>
        </div>
      </div>
    </section>

    <!-- Memory and disk. -->
    <section class="grid gap-4 md:grid-cols-2">
      <div class="card bg-base-100 border-base-300 border">
        <div class="card-body gap-2 p-4">
          <h2 class="text-sm font-medium opacity-60">Memory</h2>
          <div class="flex items-baseline gap-2">
            <span class="text-2xl font-medium">
              {m.resident_bytes != null ? humanBytes(m.resident_bytes) : "—"}
            </span>
            <span class="text-xs opacity-60">resident</span>
          </div>
          {#if m.kv}
            <!-- The cache is not pre-allocated: a 8k-context model would
                 reserve hundreds of megabytes for a conversation of fifty
                 tokens. So both numbers matter. -->
            <p class="text-xs opacity-70">
              KV cache {humanBytes(m.kv.cached_bytes)} of {humanBytes(m.kv.max_bytes)} at full
              context ({m.kv.cached_tokens.toLocaleString()} of
              {m.kv.n_ctx.toLocaleString()} tokens)
            </p>
            <progress class="progress" value={kvShare} max="100"></progress>
            <p class="text-xs opacity-60">
              {humanBytes(m.kv.bytes_per_token)} a token. It grows as a conversation does and
              is not reserved up front.
            </p>
          {:else}
            <p class="text-xs opacity-60">No model loaded, so no KV cache.</p>
          {/if}
        </div>
      </div>

      <div class="card bg-base-100 border-base-300 border">
        <div class="card-body gap-2 p-4">
          <h2 class="text-sm font-medium opacity-60">Disk</h2>
          <div class="flex items-baseline gap-2">
            <span class="text-2xl font-medium">{humanBytes(m.disk.total)}</span>
            <span class="text-xs opacity-60">in all</span>
          </div>
          <table class="table table-xs">
            <tbody>
              <tr>
                <td>Downloaded</td>
                <td class="text-right">{humanBytes(m.disk.downloaded)}</td>
                <td class="text-xs opacity-60">can be fetched again</td>
              </tr>
              <tr>
                <td>Trained here</td>
                <td class="text-right">{humanBytes(m.disk.trained)}</td>
                <td class="text-xs opacity-60">cannot</td>
              </tr>
              <tr>
                <td>Quantised</td>
                <td class="text-right">{humanBytes(m.disk.quantised)}</td>
                <td class="text-xs opacity-60">derived; safe to delete</td>
              </tr>
              <tr>
                <td>Datasets</td>
                <td class="text-right">{humanBytes(m.disk.datasets)}</td>
                <td></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- Requests, and anything running. -->
    <section class="grid gap-4 md:grid-cols-2">
      <div class="card bg-base-100 border-base-300 border">
        <div class="card-body gap-2 p-4">
          <h2 class="text-sm font-medium opacity-60">Requests</h2>
          <p class="text-xs opacity-70">
            {m.requests.toLocaleString()} in the buffer · median {ms(m.latency.median)} · p95
            {ms(m.latency.p95)}
            {#if m.errors > 0}
              · <span class="text-error">{m.errors} failed</span>
            {/if}
          </p>
          <a href="/monitoring" onclick={(e) => navigate(e, "/monitoring")} class="link text-xs">
            The request log and per-route latencies
          </a>
        </div>
      </div>

      <div class="card bg-base-100 border-base-300 border">
        <div class="card-body gap-2 p-4">
          <h2 class="text-sm font-medium opacity-60">Running</h2>
          {#if m.running.length === 0}
            <p class="text-sm opacity-60">Nothing.</p>
          {:else}
            {#each m.running as job (job.id)}
              <div class="flex items-center gap-2 text-sm">
                <span class="loading loading-spinner loading-xs"></span>
                <span class="grow truncate">{job.label}</span>
                <span class="badge badge-sm">{job.kind}</span>
              </div>
            {/each}
            <a href="/training" onclick={(e) => navigate(e, "/training")} class="link text-xs">
              Watch it
            </a>
          {/if}
        </div>
      </div>
    </section>

    <p class="text-center text-xs opacity-50">
      Server v{health?.version ?? "?"} · up {humanSecs(m.uptime_secs)}
    </p>
  </div>
{/if}
