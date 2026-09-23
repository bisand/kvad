<script>
  import { api } from "../lib/api.js";
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
   * The verdict badge says the precision a run would use; this says the whole
   * story, because a model that does not fit is much more useful when you can
   * see by how much and against what.
   */
  function atEachPrecision(r) {
    if (!r.memory) return "size unknown";
    const each = `weights here: ${humanBytes(r.memory.f32)} at f32 · ${humanBytes(
      r.memory.q8,
    )} at q8 · ${humanBytes(r.memory.q4)} at q4`;
    if (!r.streams) return each;
    const has = r.usable_memory ? humanBytes(r.usable_memory) : "what this machine has";
    return `${each} — none of them fit in ${has}, so the weights are read from disk as the model runs. That works and it is much slower: expect a fraction of the speed a model that fits would give you.`;
  }

  /** The one-line version, for a downloaded model with no size breakdown. */
  const STREAMS_TIP =
    "Too large for memory, so the weights are read from disk as the model runs. It works, and it is much slower than a model that fits.";

  const thousands = (n) => n.toLocaleString();

  /** A click on a control inside a `<summary>` toggles the row as well,
   *  because toggling is what a summary does with a click. Buttons in a row
   *  header have their own job, so they take the event and keep it. */
  const notToggle = (fn) => (e) => {
    e.preventDefault();
    e.stopPropagation();
    fn();
  };

  /** What the Hub said about a search result, once somebody opened it.
   *  `undefined` is "never asked", `null` is "asked, and it could not say".
   *
   *  Which rows are *open* is the radio inputs' business, not this file's —
   *  that is what `collapse` is for. This only remembers the answers, so
   *  re-opening a row does not ask again. */
  let hubDetail = $state({});

  async function askHub(id) {
    if (hubDetail[id] !== undefined) return;
    hubDetail = { ...hubDetail, [id]: "asking" };
    try {
      hubDetail = {
        ...hubDetail,
        [id]: await api(`/api/models/detail?repo=${encodeURIComponent(id)}`),
      };
    } catch (e) {
      // The server's reason, not a shrug of our own: it knows things worth
      // saying, like that DeepSeek publishes V3 in fp8 and this engine reads
      // bf16. Shown in the row rather than a toast, because it belongs to
      // the row and the rest of the page carries on without it.
      hubDetail = { ...hubDetail, [id]: { error: e.message } };
    }
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
      <!-- daisyUI `collapse`, one per model, so the whole row is the control
           rather than a chevron somebody has to aim at. Radio inputs and a
           shared name make it an accordion: opening one closes the last. -->
      <div class="flex flex-col gap-1">
        {#each models.all as m (m.id)}
          <!-- `<details>` rather than a radio: a radio cannot be unchecked by
               clicking it again, so a row opened that way could never be
               closed. This toggles, and without a `name` several can stay
               open at once, which is what comparing two models wants. -->
          <details class="collapse collapse-arrow rounded-box bg-base-200/40">
            <summary class="collapse-title flex flex-wrap items-center gap-x-3 gap-y-1">
              <div class="flex min-w-40 flex-1 items-center gap-2">
                <span class="truncate font-medium">{m.id}</span>
                {#if m.trained}<span class="badge badge-sm">trained here</span>{/if}
                {#if models.listing.active === m.id}
                  <span class="badge badge-sm badge-soft">default</span>
                {/if}
                {#if models.loaded?.repo === m.id}
                  <span class="badge badge-sm badge-success badge-soft">loaded</span>
                {/if}
                {#if m.streams}
                  <div class="tooltip" data-tip={STREAMS_TIP}>
                    <span class="badge badge-sm badge-warning badge-soft whitespace-nowrap">
                      disk streaming
                    </span>
                  </div>
                {/if}
              </div>
              <span class="text-sm whitespace-nowrap opacity-70">{humanBytes(m.bytes)}</span>
              <!-- The input covers the title to make it clickable, so anything
                   meant to stay clickable has to sit above it. -->
              <div class="relative z-1 flex gap-1" class:hidden={!may}>
                <button
                  class="btn btn-sm"
                  disabled={!m.runnable || !!models.busy || models.loaded?.repo === m.id}
                  onclick={notToggle(() => models.load(m.id, chosen))}
                >
                  Load
                </button>
                <button
                  class="btn btn-sm btn-ghost"
                  disabled={models.listing.active === m.id}
                  onclick={notToggle(() => models.setActive(m.id))}
                  title="The model `kvad run` picks with no --model"
                >
                  Default
                </button>
                <button
                  class="btn btn-sm btn-ghost"
                  aria-label={`Delete ${m.id}`}
                  onclick={notToggle(() => (confirming = m.id))}
                >
                  <Icon path={TRASH} size={16} />
                </button>
              </div>
              <div class="w-full text-xs opacity-60">{m.blocker ?? m.arch}</div>
            </summary>
            <div class="collapse-content text-xs">
              {#if m.detail}
                <div class="font-mono opacity-80">{m.detail.summary}</div>
                <div class="mt-2 flex flex-wrap gap-x-6 gap-y-1 opacity-70">
                  <span>{m.detail.n_layer} layers</span>
                  <span>
                    {m.detail.n_head} heads{m.detail.n_kv_head !== m.detail.n_head
                      ? ` (${m.detail.n_kv_head} KV)`
                      : ""}
                  </span>
                  <span>{thousands(m.detail.n_embd)} embedding</span>
                  <span>{thousands(m.detail.n_ctx)} context</span>
                  <span>{thousands(m.detail.vocab_size)} vocab</span>
                  {#if m.detail.params}<span>{params(m.detail.params)}</span>{/if}
                </div>
                {#if m.detail.experts}
                  <!-- The working set is not decoration: a cache holding fewer
                       experts than one pass reads evicts every one of them
                       before its next use. -->
                  <div class="mt-2 opacity-70">
                    mixture of experts — {m.detail.experts.count} per layer,
                    {m.detail.experts.per_token} chosen per token, so one token reads
                    {thousands(m.detail.experts.working_set)} of them across the model
                  </div>
                {/if}
                {#if m.detail.memory}
                  <div class="mt-2 opacity-70">
                    weights here: {humanBytes(m.detail.memory.f32)} at f32 ·
                    {humanBytes(m.detail.memory.q8)} at q8 ·
                    {humanBytes(m.detail.memory.q4)} at q4
                    {#if m.streams}— none of them fit, so they are read from disk{/if}
                  </div>
                {/if}
              {:else}
                <span class="opacity-60">
                  Nothing to show until its config can be read.
                </span>
              {/if}
            </div>
          </details>
        {/each}
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
      <!-- Same accordion as the downloaded list, with its own radio group so
           opening a search result does not close a downloaded one. -->
      <div class="mt-4 flex flex-col gap-1">
        {#each results as r (r.id)}
          <!-- As the downloaded list: a `<details>` toggles, and a radio
               would open a row that could then never be closed. `ontoggle`
               rather than `onchange`, and it fires on close too, which is
               why `askHub` checks whether it already has the answer. -->
          <details
            class="collapse collapse-arrow rounded-box bg-base-200/40"
            ontoggle={(e) => e.currentTarget.open && r.arch && askHub(r.id)}
          >
            <summary class="collapse-title flex flex-wrap items-center gap-x-3 gap-y-1">
              <div class="flex min-w-40 flex-1 items-center gap-2">
                <span class="truncate font-medium">{r.id}</span>
                {#if r.local}<span class="badge badge-sm badge-soft">here</span>{/if}
                {#if r.looks_instruct}<span class="badge badge-sm">chat</span>{/if}
              </div>
              <div class="text-right text-sm whitespace-nowrap opacity-70">
                {r.bytes ? humanBytes(r.bytes) : "—"}
                {#if r.params}<span class="text-xs opacity-60">· {params(r.params)}</span>{/if}
              </div>
              <!-- What the weights cost here, which is not the download size:
                   loading quantises, so a 16 GB checkpoint is 4.5 GB at q8. -->
              {#if !r.size_known}
                <span class="text-xs opacity-50">unknown</span>
              {:else if r.streams}
                <div class="tooltip" data-tip={atEachPrecision(r)}>
                  <span class="badge badge-sm badge-warning badge-soft whitespace-nowrap">
                    {r.fits_at} · disk
                  </span>
                </div>
              {:else if r.fits_at}
                <div class="tooltip" data-tip={atEachPrecision(r)}>
                  <span class="badge badge-sm badge-success badge-soft">{r.fits_at}</span>
                </div>
              {:else}
                <span class="text-xs opacity-50">unknown</span>
              {/if}
              <span class="text-right text-sm whitespace-nowrap opacity-70">
                {r.downloads.toLocaleString()}
              </span>
              <!-- Above the input that makes the rest of the row clickable. -->
              <div class="relative z-1">
                <button
                  class="btn btn-sm"
                  disabled={!r.runnable || !!models.busy}
                  onclick={notToggle(() =>
                    r.local ? models.load(r.id, chosen) : models.pull(r.id),
                  )}
                >
                  {r.local ? "Load" : "Pull"}
                </button>
              </div>
              <!-- The same verdict `kvad search` prints, from the Hub's own
                   config, so the listing says what is runnable before anything
                   is downloaded. -->
              <div class="w-full text-xs opacity-60">{r.blocker ?? r.arch}</div>
            </summary>
            <div class="collapse-content text-xs">
              {#if !r.arch}
                <span class="opacity-60">
                  This build has no reader for that architecture, so there is nothing
                  to describe.
                </span>
              {:else if hubDetail[r.id] === "asking"}
                  <span class="opacity-60">reading its config…</span>
                {:else if hubDetail[r.id]?.error}
                <span class="opacity-70">{hubDetail[r.id].error}</span>
              {:else if hubDetail[r.id]?.summary}
                  <div class="font-mono opacity-80">{hubDetail[r.id].summary}</div>
                  <div class="mt-2 flex flex-wrap gap-x-6 gap-y-1 opacity-70">
                    <span>{hubDetail[r.id].n_layer} layers</span>
                    <span>
                      {hubDetail[r.id].n_head} heads{hubDetail[r.id].n_kv_head !==
                      hubDetail[r.id].n_head
                        ? ` (${hubDetail[r.id].n_kv_head} KV)`
                        : ""}
                    </span>
                    <span>{thousands(hubDetail[r.id].n_embd)} embedding</span>
                    <span>{thousands(hubDetail[r.id].n_ctx)} context</span>
                    <span>{thousands(hubDetail[r.id].vocab_size)} vocab</span>
                  </div>
                  {#if hubDetail[r.id].experts}
                    <div class="mt-2 opacity-70">
                      mixture of experts — {hubDetail[r.id].experts.count} per layer,
                      {hubDetail[r.id].experts.per_token} chosen per token, so one token reads
                      {thousands(hubDetail[r.id].experts.working_set)} of them across the model
                    </div>
                  {/if}
                  {#if r.memory}
                    <div class="mt-2 opacity-70">
                      weights here: {humanBytes(r.memory.f32)} at f32 ·
                      {humanBytes(r.memory.q8)} at q8 · {humanBytes(r.memory.q4)} at q4
                      {#if r.streams}— none of them fit, so they are read from disk{/if}
                    </div>
                  {/if}
              {:else}
                <span class="opacity-60">reading its config…</span>
              {/if}
            </div>
          </details>
        {/each}
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
        Each row is a file of its own; forgetting one costs the next load at that precision
        some seconds and nothing else.
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
                  <button
                    class="btn btn-xs btn-ghost"
                    onclick={() => models.forgetQuantised(q.repo, q.precision)}
                  >
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
