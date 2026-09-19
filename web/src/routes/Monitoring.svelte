<script>
  // The request log, what each route costs, and what the server has said.
  import { monitoring, ms } from "../lib/monitoring.svelte.js";
  import Sparkline from "../lib/components/Sparkline.svelte";

  let tail;
  let following = $state(true);
  let onlyErrors = $state(false);

  $effect(() => {
    monitoring.refreshDetail();
    const timer = setInterval(() => monitoring.refreshDetail(), 4000);
    return () => clearInterval(timer);
  });

  // Follow the log down as it grows, unless the reader has scrolled back.
  $effect(() => {
    void monitoring.log.length;
    if (following && tail) tail.scrollTop = tail.scrollHeight;
  });

  const requests = $derived(
    (monitoring.requests?.recent ?? []).filter((r) => !onlyErrors || r.status >= 400),
  );
  const routes = $derived(monitoring.requests?.by_route ?? []);
  // The widest median, so the bars are relative to the slowest route rather
  // than to an arbitrary ceiling.
  const slowest = $derived(Math.max(1, ...routes.map((r) => r.latency?.median ?? r.median ?? 0)));

  function statusColour(status) {
    if (status >= 500) return "badge-error";
    if (status >= 400) return "badge-warning";
    return "badge-ghost";
  }

  function level(line) {
    if (line.includes(" ERROR ")) return "text-error";
    if (line.includes(" WARN ")) return "text-warning";
    return "";
  }
</script>

<div class="flex flex-col gap-6">
  {#if monitoring.error}
    <div role="alert" class="alert alert-error"><span>{monitoring.error}</span></div>
  {/if}

  <!-- What each route costs. -->
  <section>
    <h2 class="mb-1 text-sm font-medium opacity-60">By route</h2>
    <p class="mb-3 text-xs opacity-60">
      Grouped by the route pattern rather than the path, so <code>/api/jobs/17</code> and
      <code>/api/jobs/18</code> are one row. Percentiles are taken from the samples
      themselves — there are few enough to sort, so nothing here is an estimate.
      <code>/v1/chat/completions</code> times itself and so measures the whole
      generation; every other streaming route is timed to its first byte, which for a
      model load or a job stream is the moment before the work rather than after it.
    </p>
    {#if routes.length === 0}
      <p class="text-sm opacity-60">Nothing recorded yet.</p>
    {:else}
      <div class="overflow-x-auto">
        <table class="table table-sm">
          <thead>
            <tr>
              <th class="w-full">Route</th>
              <th class="text-right">Calls</th>
              <th class="text-right">Median</th>
              <th class="text-right">p95</th>
              <th class="text-right">Worst</th>
              <th class="text-right">Failed</th>
            </tr>
          </thead>
          <tbody>
            {#each routes as r (r.method + r.path)}
              <tr>
                <td class="max-w-0">
                  <div class="flex items-center gap-2">
                    <span class="badge badge-ghost badge-sm font-mono">{r.method}</span>
                    <span class="truncate font-mono text-xs">{r.path}</span>
                  </div>
                  <div class="bg-base-300 mt-1 h-1 w-full rounded">
                    <div
                      class="bg-primary h-1 rounded"
                      style:width="{Math.max(1, (r.median / slowest) * 100)}%"
                    ></div>
                  </div>
                </td>
                <td class="text-right text-xs whitespace-nowrap opacity-70">{r.count}</td>
                <td class="text-right text-xs whitespace-nowrap">{ms(r.median)}</td>
                <td class="text-right text-xs whitespace-nowrap opacity-70">{ms(r.p95)}</td>
                <td class="text-right text-xs whitespace-nowrap opacity-70">{ms(r.worst)}</td>
                <td class="text-right text-xs whitespace-nowrap">
                  {#if r.errors > 0}<span class="text-error">{r.errors}</span>{:else}—{/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>

  <!-- Every request, newest first. -->
  <section>
    <div class="mb-2 flex items-center gap-3">
      <h2 class="text-sm font-medium opacity-60">Recent requests</h2>
      <span class="grow"></span>
      <label class="flex cursor-pointer items-center gap-2 text-xs">
        <input type="checkbox" class="toggle toggle-xs" bind:checked={onlyErrors} />
        only failures
      </label>
    </div>
    <div class="max-h-96 overflow-y-auto">
      <table class="table table-xs table-pin-rows">
        <thead>
          <tr>
            <th>Status</th>
            <th class="w-full">Route</th>
            <th class="text-right">Took</th>
            <th class="text-right whitespace-nowrap">Generation</th>
          </tr>
        </thead>
        <tbody>
          {#each requests as r, i (r.at_millis + r.path + i)}
            <tr>
              <td><span class="badge badge-xs {statusColour(r.status)}">{r.status}</span></td>
              <td class="max-w-0 truncate font-mono text-xs">
                <span class="opacity-50">{r.method}</span>
                {r.path}
              </td>
              <td class="text-right text-xs whitespace-nowrap">{ms(r.millis)}</td>
              <td class="text-right text-xs whitespace-nowrap opacity-70">
                {#if r.generation}
                  {r.generation.generated_tokens} tok at
                  {r.generation.decode_per_sec.toFixed(0)}/s
                  {#if r.generation.cached_tokens > 0}
                    · {r.generation.cached_tokens} cached
                  {/if}
                {:else}
                  —
                {/if}
              </td>
            </tr>
          {:else}
            <tr><td colspan="4" class="py-6 text-center text-sm opacity-60">Nothing to show.</td></tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>

  <!-- What the server said. -->
  <section>
    <div class="mb-2 flex items-center gap-3">
      <h2 class="text-sm font-medium opacity-60">Server log</h2>
      <span class="grow"></span>
      <label class="flex cursor-pointer items-center gap-2 text-xs">
        <input type="checkbox" class="toggle toggle-xs" bind:checked={following} />
        follow
      </label>
    </div>
    <pre
      bind:this={tail}
      onscroll={(e) => {
        const el = e.currentTarget;
        following = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      }}
      class="bg-base-200 rounded-box max-h-72 overflow-auto p-3 text-xs leading-relaxed">{#each monitoring.log as line, i (i)}<span
          class="block {level(line)}">{line}</span>{:else}<span class="opacity-60">Nothing logged yet.</span>{/each}</pre>
    <p class="mt-1 text-xs opacity-60">
      The last few hundred lines, kept in memory and gone on restart. A log that has to
      outlive the process belongs to whatever is running it.
    </p>
  </section>
</div>
