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
    return `${each}, against ${has}.`;
  }

  /**
   * What running from the disk costs, for a badge's tooltip.
   *
   * Total size is not the answer to that, and badging by it gave a dense 70B
   * the same badge as an 80B mixture that runs. What a token reads is: the
   * server estimates the bytes each token pages in, and the two models that
   * were measured anchor what those numbers mean.
   */
  function diskTip(f) {
    const each = f.disk_per_token == null ? null : humanBytes(f.disk_per_token);
    if (f.crawls) {
      return f.mixture
        ? `Too large for memory, and a token reads enough of it to page in about ${each} from disk. Measured here, a model paging 2.5 GB a token ran at 1.5 tok/s.`
        : `Too large for memory, and a dense model reads every weight for every token, so each one pages in at least ${each} from disk. It runs, at a crawl.`;
    }
    if (each) {
      return `Too large for memory, but a mixture reads only a few of its experts per token: about ${each} from disk each. Measured here, a model paging 0.3 GB a token ran at 10.7 tok/s.`;
    }
    return "Too large for memory. It is a mixture, which can stream well if it is sparse enough — the Hub's summary does not say how many experts it has. Open the row to find out.";
  }

  /** The badge's words, beside the precision. */
  const diskWord = (f) => (f.crawls ? "crawls" : f.disk_per_token != null ? "streams" : "disk");

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

  async function askHub(r) {
    const id = r.id;
    if (hubDetail[id] !== undefined) return;
    hubDetail = { ...hubDetail, [id]: "asking" };
    // The parameter count rides along so the answer can carry a verdict
    // without the server asking the Hub a second time.
    const sized = r.params ? `&params=${r.params}` : "";
    try {
      hubDetail = {
        ...hubDetail,
        [id]: await api(`/api/models/detail?repo=${encodeURIComponent(id)}${sized}`),
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
  //
  // Null until somebody picks one, and a load then names no backend, so the
  // server chooses for each model. It knows what this page does not: that a
  // model is too big for the GPU, or stored in fp8, which the GPU cannot
  // read. Sending the picker's first option on every load is how
  // Qwen3-0.6B-FP8 was put on the GPU and failed there.
  let backend = $state(null);

  $effect(() => {
    models.refresh();
  });

  /** A search row's verdict: the row's own, until opening it has fetched the
   *  whole config, whose answer is exact where the Hub's summary was not. */
  function rowFit(r) {
    const f = hubDetail[r.id]?.fit;
    return f ? { ...f, mixture: r.mixture, fits_at: f.precision } : r;
  }

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

<!-- The part of an opened row that is the same for a downloaded model and a
     search result: what a mixture reads, what the weights cost here, and
     whether that fits. -->
{#snippet shape(d)}
  {#if d.experts}
    <!-- The working set is not decoration: a cache holding fewer experts than
         one pass reads evicts every one of them before its next use, and it is
         what a model over memory has to page in. -->
    <div class="mt-2 opacity-70">
      mixture of experts — {d.experts.count} per layer, {d.experts.per_token} chosen per
      token, so one token reads {thousands(d.experts.working_set)} of them across
      {d.experts.layers} layers
      {#if d.experts.working_set_bytes}
        — {humanBytes(d.experts.working_set_bytes.q4)} at q4, of a
        {humanBytes(d.experts.store_bytes.q4)} store
      {/if}
    </div>
  {/if}
  {#if d.memory}
    <div class="mt-2 opacity-70">
      weights here: {humanBytes(d.memory.f32)} at f32 · {humanBytes(d.memory.q8)} at q8 ·
      {humanBytes(d.memory.q4)} at q4
    </div>
  {/if}
  {#if d.fit}
    <div class="mt-2">
      {#if !d.fit.streams}
        <span class="opacity-70">Fits in memory here at {d.fit.precision}.</span>
      {:else if d.fit.crawls}
        <span class="text-error">
          {d.experts ? "Crawls" : "Crawls — a dense model reads every weight for every token"}:
          none of it fits, and each token pages in about {humanBytes(d.fit.disk_per_token)}
          from disk.
        </span>
      {:else if d.fit.disk_per_token != null}
        <span class="text-warning">
          Streams: none of it fits, but a token pages in only about
          {humanBytes(d.fit.disk_per_token)} from disk at {d.fit.precision}. Measured here, a
          model paging 0.3 GB a token ran at 10.7 tok/s.
        </span>
      {:else}
        <span class="opacity-70">Does not fit in memory, so it is read from disk.</span>
      {/if}
    </div>
  {/if}
  {#if d.stored_as}
    <!-- The one fact about an fp8 checkpoint nobody would guess is that it
         does not save memory here: the loader decodes it and quantises
         again at whatever precision the backend runs. -->
    <div class="mt-2 opacity-70">
      stored as {d.stored_as} — about half the download of bf16. Decoded as it loads, so
      in memory it costs what any copy of this model does at the precision it runs at.
      Checked against Qwen3-0.6B's bf16 publication, fp8 weights differ by 1.7% on
      average.
    </div>
  {/if}
{/snippet}

<div class="flex flex-col gap-6">
  <!-- What is in memory, what it costs, and the controls that change it. -->
  <section class="card bg-base-100 border-base-300 border">
    <div class="card-body gap-4 p-4 sm:p-6">
      <div class="flex flex-wrap items-start justify-between gap-4">
        <div class="min-w-0 grow">
          <h2 class="text-sm font-medium opacity-60">In memory</h2>
          {#if models.memory}
            {@const m = models.memory}
            <div class="mt-2 max-w-md">
              <progress
                class="progress"
                class:progress-warning={m.left < m.total * 0.15}
                value={m.total - m.left}
                max={m.total}
              ></progress>
              <p class="text-xs opacity-70">
                {humanBytes(m.total - m.left)} of {humanBytes(m.total)} taken ·
                {humanBytes(m.left)} left · each charged for {m.context.toLocaleString()} tokens of
                context
              </p>
            </div>
          {/if}
          {#if models.residents.length === 0}
            <p class="mt-2 text-lg opacity-60">nothing</p>
            <p class="mt-1 text-xs opacity-70">
              Load a model below to chat with it. Several can be in memory at once, as long as
              they fit; nothing is unloaded to make room for another.
            </p>
          {/if}
        </div>

        <div class="flex items-center gap-2" class:hidden={!may}>
          <label class="text-xs opacity-60" for="backend">Next load on</label>
          <select
            id="backend"
            class="select select-sm w-36"
            bind:value={() => backend ?? "", (v) => (backend = v || null)}
          >
            <option value="">best for each</option>
            {#each models.listing?.backends ?? [] as choice (choice.id)}
              <option value={choice.id}>{choice.label}</option>
            {/each}
          </select>
          {#if models.residents.length > 1}
            <button class="btn btn-sm" onclick={() => models.unload()} disabled={!!models.busy}>
              Unload all
            </button>
          {/if}
        </div>
      </div>

      {#if models.residents.length > 0}
        <ul class="flex flex-col gap-2">
          {#each models.residents as r (r.id)}
            <li class="bg-base-200/40 rounded-box flex flex-wrap items-center gap-x-4 gap-y-1 p-3">
              <div class="min-w-0 grow">
                <p class="truncate font-medium">
                  {r.repo}
                  <span class="badge badge-sm ml-1">{r.backend}</span>
                  {#if r.streams}
                    <span class="badge badge-sm badge-warning badge-soft">streams from disk</span>
                  {/if}
                </p>
                <p class="mt-1 text-xs opacity-70">
                  {r.summary} · {(r.params / 1e6).toFixed(1)}M parameters ·
                  {r.instruct ? "instruction-tuned" : "base model (completion only)"}
                </p>
                <p class="mt-1 text-xs opacity-70">
                  charged {humanBytes(r.commit)} · weights {humanBytes(r.weight_bytes)} ·
                  {r.cached_tokens.toLocaleString()} tokens cached · clients name it
                  <code>{r.id}</code>
                </p>
              </div>
              <button
                class="btn btn-sm"
                class:hidden={!may}
                onclick={() => models.unload(r.id)}
                disabled={!!models.busy}
              >
                Unload
              </button>
            </li>
          {/each}
        </ul>
      {/if}

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
                {#if models.resident(m.id)}
                  <span class="badge badge-sm badge-success badge-soft">in memory</span>
                {/if}
                {#if m.streams}
                  <div class="tooltip" data-tip={diskTip(m)}>
                    <span
                      class="badge badge-sm badge-soft whitespace-nowrap"
                      class:badge-warning={!m.crawls}
                      class:badge-error={m.crawls}
                    >
                      {diskWord(m)} from disk
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
                  disabled={!m.runnable ||
                    !!models.busy ||
                    models.residents.some(
                      (r) => r.repo === m.id && (!backend || r.id.endsWith(`@${backend}`)),
                    )}
                  onclick={notToggle(() => models.load(m.id, backend))}
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
                {@render shape(m.detail)}
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
        Loading, pulling and deleting models are an administrator's to do. What is in
        memory is what you can chat with.
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
            ontoggle={(e) => e.currentTarget.open && r.arch && askHub(r)}
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
              {#if !r.runnable}
                <!-- A precision badge is a claim about a run. This row cannot
                     have one, so it does not get to make the claim; the line
                     underneath says what is wrong. -->
                <span class="badge badge-sm badge-error badge-soft whitespace-nowrap">
                  can't run
                </span>
              {:else if !r.size_known}
                <span class="text-xs opacity-50">unknown</span>
              {:else if rowFit(r).streams}
                {@const f = rowFit(r)}
                <div class="tooltip" data-tip={`${diskTip(f)} ${atEachPrecision(r)}`}>
                  <span
                    class="badge badge-sm badge-soft whitespace-nowrap"
                    class:badge-warning={!f.crawls}
                    class:badge-error={f.crawls}
                  >
                    {f.fits_at} · {diskWord(f)}
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
                    r.local ? models.load(r.id, backend) : models.pull(r.id),
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
                  {@render shape(hubDetail[r.id])}
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
