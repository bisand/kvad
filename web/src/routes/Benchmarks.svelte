<script>
  // The measurement protocol, with a button on it.
  import { bench } from "../lib/bench.svelte.js";
  import { models } from "../lib/models.svelte.js";
  import VariantPicker from "../lib/components/VariantPicker.svelte";
  import Icon from "../lib/components/Icon.svelte";

  let variants = $state([]);
  let prompt = $state("The history of the transformer architecture begins with");
  let rounds = $state(5);
  let tokens = $state(64);

  $effect(() => {
    models.refresh();
    bench.refresh();
  });

  const open = $derived(bench.open);
  const running = $derived(open?.job?.state === "running" || open?.job?.state === "queued");
  const summary = $derived(open?.summary ?? []);
  // Bars are relative to the fastest variant in this run, because the useful
  // question is which of these is quicker and by how much.
  const fastest = $derived(Math.max(1, ...summary.map((s) => s.decode_high ?? 0)));

  async function start() {
    await bench.start({ variants, prompt, rounds, tokens });
  }

  function when(job) {
    return job.ended_at ?? job.started_at ?? job.created_at;
  }
</script>

<div class="flex flex-col gap-6">
  <section class="bg-base-200 rounded-box flex flex-col gap-3 p-4">
    <p class="text-sm opacity-70">
      Every round visits every variant once, and a run is several rounds — so a machine
      that warms up shows as a trend across the rounds rather than as a winner. The
      median and the range come from the samples, which are all kept.
    </p>
    <p class="text-xs opacity-60">
      A benchmark will not start while a training run, an eval or another benchmark is
      going: a measurement taken next to other work is a measurement of the other work.
      Each timed generation also starts from an empty KV cache, or the second round
      would report a time to first token that no first round would ever see.
    </p>

    <VariantPicker bind:value={variants} />

    <div class="flex flex-wrap items-end gap-3">
      <label class="flex flex-col gap-1 text-xs">
        <span class="opacity-70">Rounds</span>
        <input type="number" min="1" max="20" class="input input-sm w-24" bind:value={rounds} />
      </label>
      <label class="flex flex-col gap-1 text-xs">
        <span class="opacity-70">Tokens each</span>
        <input type="number" min="8" max="1024" step="8" class="input input-sm w-28" bind:value={tokens} />
      </label>
      <label class="flex grow flex-col gap-1 text-xs">
        <span class="opacity-70">Prompt</span>
        <input class="input input-sm w-full font-mono" bind:value={prompt} />
      </label>
      <button class="btn btn-primary btn-sm" onclick={start} disabled={!variants.length || running}>
        {#if running}<span class="loading loading-spinner loading-xs"></span>{/if}
        Measure
      </button>
    </div>
    {#if variants.length}
      <p class="text-xs opacity-60">
        {variants.length * rounds} generations, and a model load before each one that
        changes variant.
      </p>
    {/if}
  </section>

  {#if open}
    <section class="flex flex-col gap-3">
      <div class="flex items-center gap-3">
        <h2 class="text-sm font-medium opacity-60">{open.job.label}</h2>
        <span class="badge badge-sm" class:badge-success={open.job.state === "done"}
          class:badge-error={open.job.state === "failed"}>{open.job.state}</span>
        <span class="grow"></span>
        {#if running}
          {#if open.progress}
            <span class="text-xs opacity-70">
              {open.progress.done} of {open.progress.total}
            </span>
          {/if}
          <button class="btn btn-xs" onclick={() => bench.cancel(open.job.id)}>Stop</button>
        {/if}
      </div>

      {#if open.job.error}
        <div role="alert" class="alert alert-error"><span>{open.job.error}</span></div>
      {/if}
      {#if running && open.log.at(-1)}
        <p class="text-xs opacity-60">{open.log.at(-1)}</p>
      {/if}

      {#if summary.length}
        <div class="overflow-x-auto">
          <table class="table table-sm">
            <thead>
              <tr>
                <th class="w-full">Variant</th>
                <th class="text-right">Runs</th>
                <th class="text-right">Decode median</th>
                <th class="text-right">Range</th>
                <th class="text-right">TTFT median</th>
              </tr>
            </thead>
            <tbody>
              {#each summary as s (s.variant)}
                <tr>
                  <td class="max-w-0">
                    <div class="truncate font-mono text-xs">{s.variant}</div>
                    <div class="bg-base-300 relative mt-1 h-1.5 w-full rounded">
                      <!-- The bar is the range; the notch is the median. -->
                      <div class="bg-primary/30 absolute h-1.5 rounded"
                        style:left="{(s.decode_low / fastest) * 100}%"
                        style:width="{Math.max(1, ((s.decode_high - s.decode_low) / fastest) * 100)}%"></div>
                      <div class="bg-primary absolute h-1.5 w-1 rounded"
                        style:left="{(s.decode_median / fastest) * 100}%"></div>
                    </div>
                  </td>
                  <td class="text-right text-xs opacity-70">{s.runs}</td>
                  <td class="text-right font-mono text-xs whitespace-nowrap">
                    {s.decode_median?.toFixed(1)} tok/s
                  </td>
                  <td class="text-right font-mono text-xs whitespace-nowrap opacity-70">
                    {s.decode_low?.toFixed(1)}–{s.decode_high?.toFixed(1)}
                  </td>
                  <td class="text-right font-mono text-xs whitespace-nowrap opacity-70">
                    {s.ttft_median?.toFixed(0)} ms
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
        <p class="text-xs opacity-60">
          The bar is the whole spread of what was measured and the notch is the median.
          Two bars that overlap are two numbers that have not been told apart — which
          is the thing a single timing run cannot show you.
        </p>
      {/if}

      {#if open.samples.length}
        <details class="collapse-arrow bg-base-200 rounded-box collapse">
          <summary class="collapse-title text-sm font-medium">
            Every sample ({open.samples.length})
          </summary>
          <div class="collapse-content">
            <table class="table table-xs">
              <thead>
                <tr>
                  <th>Round</th><th class="w-full">Variant</th>
                  <th class="text-right">Decode</th><th class="text-right">Prefill</th>
                  <th class="text-right">TTFT</th>
                </tr>
              </thead>
              <tbody>
                {#each open.samples as s (s.variant + s.round)}
                  <tr>
                    <td class="opacity-50">{s.round}</td>
                    <td class="max-w-0 truncate font-mono text-xs">{s.variant}</td>
                    <td class="text-right font-mono text-xs">{s.decode_per_sec.toFixed(1)}</td>
                    <td class="text-right font-mono text-xs opacity-70">{s.prefill_per_sec.toFixed(0)}</td>
                    <td class="text-right font-mono text-xs opacity-70">{s.ttft_millis.toFixed(0)} ms</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </details>
      {/if}
    </section>
  {/if}

  <section>
    <h2 class="mb-2 text-sm font-medium opacity-60">Earlier runs</h2>
    {#if bench.runs.length === 0}
      <p class="text-sm opacity-60">Nothing measured yet.</p>
    {:else}
      <table class="table table-sm">
        <tbody>
          {#each bench.runs as r (r.id)}
            <tr class="hover:bg-base-200 cursor-pointer" onclick={() => bench.watch(r.id)}>
              <td class="w-full">
                <div class="text-sm">{r.label}</div>
                <div class="text-xs opacity-60">
                  {r.params?.rounds ?? "?"} rounds · {r.params?.tokens ?? "?"} tokens
                </div>
              </td>
              <td class="text-xs whitespace-nowrap opacity-60">{when(r)}</td>
              <td>
                <span class="badge badge-sm" class:badge-success={r.state === "done"}
                  class:badge-error={r.state === "failed"}>{r.state}</span>
              </td>
              <td><Icon path="M9 18l6-6-6-6" size={16} /></td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </section>
</div>
