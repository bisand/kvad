<script>
  import { models, humanBytes } from "../lib/models.svelte.js";
  import { auth } from "../lib/auth.svelte.js";
  import { toasts } from "../lib/toasts.svelte.js";
  import Icon from "../lib/components/Icon.svelte";

  // Every one of these is enforced on the server as well; hiding them is so
  // that a user is not offered buttons that answer 403.
  const may = $derived(auth.isAdmin);

  const TRASH = "M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6";
  const SEARCH = "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM21 21l-4.3-4.3";

  /** A parameter count, as people say it: 596M, 8.0B. */
  function params(n) {
    if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B params`;
    return `${Math.round(n / 1e6)}M params`;
  }

  /**
   * What the weights cost here at each precision, for the tooltip.
   *
   * The verdict badge says the best one that fits; this says the whole story,
   * because "too big" is much more useful when you can see by how much.
   */
  function atEachPrecision(r) {
    if (!r.memory) return "size unknown";
    return `weights here: ${humanBytes(r.memory.f32)} at f32 · ${humanBytes(
      r.memory.q8,
    )} at q8 · ${humanBytes(r.memory.q4)} at q4`;
  }

  let query = $state("");
  let results = $state(null);
  let searching = $state(false);
  let confirming = $state(null);

  // The picker chooses where the *next* load runs. Changing it does not touch
  // the model already in memory — the backend is decided when weights are
  // read, which is the same rule the TUI's `p` key follows.
  let backend = $state(null);
  const chosen = $derived(backend ?? models.listing?.backend ?? "cpu-q8");

  $effect(() => {
    models.refresh();
  });

  async function search(event) {
    event?.preventDefault();
    if (!query.trim()) return;
    searching = true;
    try {
      results = await models.search(query.trim());
      if (results.length === 0) toasts.info(`Nothing on the Hub matches “${query.trim()}”.`);
    } catch (e) {
      toasts.error(e.message);
    } finally {
      searching = false;
    }
  }

  async function confirmDelete() {
    const id = confirming;
    confirming = null;
    if (id) await models.remove(id);
  }
</script>

<div class="flex flex-col gap-6">
  <!-- What the engine is holding, and the one control that changes it. -->
  <section class="card bg-base-100 border-base-300 border">
    <div class="card-body gap-4 p-4 sm:p-6">
      <div class="flex flex-wrap items-start justify-between gap-4">
        <div class="min-w-0">
          <h2 class="text-sm font-medium opacity-60">Loaded</h2>
          {#if models.loaded}
            <p class="mt-1 truncate text-lg font-medium">{models.loaded.repo}</p>
            <p class="mt-1 text-xs opacity-70">{models.loaded.summary}</p>
            <p class="mt-1 text-xs opacity-70">
              {(models.loaded.params / 1e6).toFixed(1)}M parameters ·
              weights {humanBytes(models.loaded.weight_bytes)} ·
              {models.loaded.backend} ·
              {models.loaded.instruct ? "instruction-tuned" : "base model (completion only)"}
            </p>
          {:else}
            <p class="mt-1 text-lg opacity-60">nothing</p>
            <p class="mt-1 text-xs opacity-70">
              The engine holds one model at a time. Load one below to chat with it.
            </p>
          {/if}
        </div>

        <div class="flex items-center gap-2" class:hidden={!may}>
          <label class="text-xs opacity-60" for="backend">Next load on</label>
          <select
            id="backend"
            class="select select-sm w-36"
            bind:value={() => chosen, (v) => (backend = v)}
          >
            {#each models.listing?.backends ?? [] as choice (choice.id)}
              <option value={choice.id}>{choice.label}</option>
            {/each}
          </select>
          {#if models.loaded}
            <button class="btn btn-sm" onclick={() => models.unload()} disabled={!!models.busy}>
              Unload
            </button>
          {/if}
        </div>
      </div>

      {#if models.busy}
        <div>
          <p class="mb-1 truncate text-xs opacity-70">
            {models.busy.what}: {models.busy.message}
            {#if models.busy.total > 0}
              — {humanBytes(models.busy.bytes)} of {humanBytes(models.busy.total)}
            {/if}
          </p>
          {#if models.busy.total > 0}
            <progress class="progress" value={models.busy.bytes} max={models.busy.total}></progress>
          {:else}
            <progress class="progress"></progress>
          {/if}
        </div>
      {/if}
    </div>
  </section>

  <!-- On this machine. -->
  <section>
    <h2 class="mb-2 text-sm font-medium opacity-60">
      On this machine
      {#if models.listing}
        <span class="opacity-60">({models.all.length})</span>
      {/if}
    </h2>

    {#if !models.listing}
      <div class="flex flex-col gap-2">
        {#each [0, 1, 2] as i (i)}<div class="skeleton h-14 w-full"></div>{/each}
      </div>
    {:else if models.all.length === 0}
      <div class="bg-base-200 rounded-box p-6 text-center text-sm opacity-70">
        Nothing downloaded yet. Search the Hub below — <code>smollm</code> is a good start.
      </div>
    {:else}
      <div class="overflow-x-auto">
        <table class="table">
          <thead>
            <tr>
              <!-- `w-full` on the first column and `whitespace-nowrap` on the
                   others: the name gets whatever is left over, which on a
                   narrow screen is not much, and everything else keeps the
                   width it needs. -->
              <th class="w-full">Model</th>
              <th>Size</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {#each models.all as m (m.id)}
              <tr class="hover:bg-base-200/50">
                <td class="max-w-0 min-w-40">
                  <div class="flex items-center gap-2">
                    <span class="truncate font-medium">{m.id}</span>
                    {#if m.trained}<span class="badge badge-sm">trained here</span>{/if}
                    {#if models.listing.active === m.id}
                      <span class="badge badge-sm badge-soft">default</span>
                    {/if}
                    {#if models.loaded?.repo === m.id}
                      <span class="badge badge-sm badge-success badge-soft">loaded</span>
                    {/if}
                  </div>
                  <div class="mt-0.5 text-xs opacity-60">
                    {m.blocker ?? m.arch}
                  </div>
                </td>
                <td class="text-sm whitespace-nowrap opacity-70">{humanBytes(m.bytes)}</td>
                <td class="whitespace-nowrap">
                  <div class="flex justify-end gap-1" class:hidden={!may}>
                    <button
                      class="btn btn-sm"
                      disabled={!m.runnable || !!models.busy || models.loaded?.repo === m.id}
                      onclick={() => models.load(m.id, chosen)}
                    >
                      Load
                    </button>
                    <button
                      class="btn btn-sm btn-ghost"
                      disabled={models.listing.active === m.id}
                      onclick={() => models.setActive(m.id)}
                      title="The model `kvad run` picks with no --model"
                    >
                      Default
                    </button>
                    <button
                      class="btn btn-sm btn-ghost"
                      aria-label={`Delete ${m.id}`}
                      onclick={() => (confirming = m.id)}
                    >
                      <Icon path={TRASH} size={16} />
                    </button>
                  </div>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>

  {#if !may}
    <div role="alert" class="alert alert-soft text-sm">
      <span>
        Loading, pulling and deleting models are an administrator's to do. What is
        loaded is what you can chat with.
      </span>
    </div>
  {/if}

  <!-- The Hub. Searching it makes an outbound request on this machine's
       behalf, so it is an administrator's too. -->
  {#if may}
  <section>
    <h2 class="mb-2 text-sm font-medium opacity-60">Search the Hub</h2>
    <form class="join w-full max-w-lg" onsubmit={search}>
      <input
        class="input join-item w-full"
        placeholder="smollm, qwen2.5, gpt2…"
        bind:value={query}
        aria-label="Search the HuggingFace Hub"
      />
      <button class="btn join-item" disabled={searching || !query.trim()}>
        {#if searching}
          <span class="loading loading-spinner loading-sm"></span>
        {:else}
          <Icon path={SEARCH} size={18} />
        {/if}
        Search
      </button>
    </form>

    {#if results?.length}
      <div class="mt-4 overflow-x-auto">
        <table class="table">
          <thead>
            <tr>
              <th class="w-full">Model</th>
              <th class="text-right whitespace-nowrap">Download</th>
              <th class="whitespace-nowrap">Fits here</th>
              <th class="text-right whitespace-nowrap">Downloads</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {#each results as r (r.id)}
              <tr class="hover:bg-base-200/50">
                <td class="max-w-0 min-w-40">
                  <div class="flex items-center gap-2">
                    <span class="truncate font-medium">{r.id}</span>
                    {#if r.local}<span class="badge badge-sm badge-soft">here</span>{/if}
                    {#if r.looks_instruct}<span class="badge badge-sm">chat</span>{/if}
                  </div>
                  <!-- `blocker` is the same verdict `kvad search` prints, from
                       the Hub's own config.json — so the listing can say which
                       results are runnable before anything is downloaded. -->
                  <div class="mt-0.5 text-xs opacity-60">{r.blocker ?? r.arch}</div>
                </td>
                <td class="text-right text-sm whitespace-nowrap opacity-70">
                  {r.bytes ? humanBytes(r.bytes) : "—"}
                  {#if r.params}
                    <div class="text-xs opacity-60">{params(r.params)}</div>
                  {/if}
                </td>
                <td class="whitespace-nowrap">
                  <!-- The download size and what it costs here are different
                       numbers: loading quantises, so a 16 GB checkpoint is
                       4.5 GB of weights at q8. This column is the second one,
                       against what this machine has. -->
                  {#if !r.size_known}
                    <span class="text-xs opacity-50">unknown</span>
                  {:else if r.fits_at}
                    <div class="tooltip" data-tip={atEachPrecision(r)}>
                      <span class="badge badge-sm badge-success badge-soft">
                        {r.fits_at}
                      </span>
                    </div>
                  {:else}
                    <div class="tooltip" data-tip={atEachPrecision(r)}>
                      <span class="badge badge-sm badge-error badge-soft">too big</span>
                    </div>
                  {/if}
                </td>
                <td class="text-right text-sm whitespace-nowrap opacity-70">
                  {r.downloads.toLocaleString()}
                </td>
                <td class="text-right whitespace-nowrap">
                  <button
                    class="btn btn-sm"
                    disabled={!r.runnable || !!models.busy}
                    onclick={() => (r.local ? models.load(r.id, chosen) : models.pull(r.id))}
                  >
                    {r.local ? "Load" : "Pull"}
                  </button>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>

  {/if}

  <!-- Quantised weights: derived, and always safe to throw away. -->
  {#if may && models.listing?.qcache.length}
    <section>
      <h2 class="mb-1 text-sm font-medium opacity-60">Quantised weights</h2>
      <p class="mb-2 text-xs opacity-60">
        Weights quantised once and reused, so a load maps a file instead of recomputing.
        Deleting one costs the next load some seconds and nothing else.
      </p>
      <div class="overflow-x-auto">
        <table class="table table-sm">
          <tbody>
            {#each models.listing.qcache as q (q.repo + q.precision)}
              <tr>
                <td class="w-full max-w-0 truncate">{q.repo}</td>
                <td class="whitespace-nowrap"><span class="badge badge-sm">{q.precision}</span></td>
                <td class="text-sm whitespace-nowrap opacity-70">{humanBytes(q.bytes)}</td>
                <td class="text-right">
                  <button class="btn btn-xs btn-ghost" onclick={() => models.forgetQuantised(q.repo)}>
                    Forget
                  </button>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>
  {/if}
</div>

{#if confirming}
  <div class="modal modal-open" role="dialog">
    <div class="modal-box">
      <h3 class="text-lg font-medium">Delete {confirming}?</h3>
      <p class="py-3 text-sm opacity-70">
        {#if models.all.find((m) => m.id === confirming)?.trained}
          This model was trained here. There is no other copy of it, and nothing can
          download it again — it would have to be trained from scratch.
        {:else}
          The files go from the HuggingFace cache. It can be downloaded again.
        {/if}
      </p>
      <div class="modal-action">
        <button class="btn btn-sm" onclick={() => (confirming = null)}>Cancel</button>
        <button class="btn btn-sm btn-error" onclick={confirmDelete}>Delete</button>
      </div>
    </div>
    <button class="modal-backdrop" aria-label="Cancel" onclick={() => (confirming = null)}></button>
  </div>
{/if}
