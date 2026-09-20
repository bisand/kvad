<script>
  // The text runs are trained on, and the one question worth asking about it
  // before a run starts: which characters a model has no token for.
  import { training } from "../lib/training.svelte.js";
  import { toasts } from "../lib/toasts.svelte.js";
  import { humanBytes } from "../lib/models.svelte.js";
  import { watchJob } from "../lib/jobwatch.js";
  import { api } from "../lib/api.js";
  import Icon from "../lib/components/Icon.svelte";

  const TRASH =
    "M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6";

  let name = $state("");
  let text = $state("");
  let uploading = $state(false);
  let confirming = $state(null);
  let against = $state("");
  let checks = $state({});

  // Reading a website.
  let url = $state("");
  let crawlName = $state("");
  let named = $state(false);
  let showing = $state(false);
  let limits = $state({ same_host: false, max_pages: 400, max_mb: 16, delay_ms: 250, drop_rare: 10 });
  /** The crawl being watched: `{ job, done, total, line }`. */
  let crawl = $state(null);
  let watcher = null;

  $effect(() => {
    training.refresh();
  });

  // A crawl outlives the tab that started it, so pick up whichever one is
  // going — after a reload, or in a second window.
  $effect(() => {
    const going = training.crawling;
    if (going && crawl?.job.id !== going.id) follow(going);
  });

  $effect(() => () => watcher?.abort());

  /**
   * A name from an address: `doc.rust-lang.org/book/` becomes
   * `doc.rust-lang.org-book`. The server has the last word on what is a legal
   * name; this only has to be a good suggestion.
   */
  function nameFrom(address) {
    try {
      const u = new URL(address);
      const parts = u.pathname.split("/").filter(Boolean);
      const last = parts.pop()?.replace(/\.(html?|php|md|txt)$/i, "");
      return [u.hostname, ...parts, last]
        .filter(Boolean)
        .join("-")
        .replace(/[^A-Za-z0-9._-]+/g, "-")
        .replace(/^[.-]+/, "")
        .slice(0, 96);
    } catch {
      return "";
    }
  }

  $effect(() => {
    if (!named) crawlName = nameFrom(url);
  });

  function follow(job) {
    watcher?.abort();
    crawl = { job, done: 0, total: 0, line: "" };
    watcher = watchJob(job.id, {
      onUpdate: (u) => {
        if (!crawl) return;
        if (u.kind === "progress") crawl = { ...crawl, done: u.done, total: u.total };
        else if (u.kind === "status") crawl = { ...crawl, line: u.message };
        else if (u.kind === "ended") ended(job.id, u);
      },
      onError: (e) => toasts.error(e.message),
    });
  }

  /** The row now carries the result; the stream does not. Read it back. */
  async function ended(id, update) {
    if (update.state === "failed") toasts.error(update.error ?? "the crawl failed");
    else toasts.success("The text is in the datasets below.");
    try {
      const { metrics, samples, ...job } = await api(`/api/jobs/${id}`);
      if (crawl?.job.id === id) crawl = { ...crawl, job };
    } catch {
      // The toast said how it ended; the job list below says the rest.
    }
    await training.refresh();
  }

  async function start(event) {
    event.preventDefault();
    const job = await training.crawl({
      url: url.trim(),
      name: crawlName.trim(),
      same_host: limits.same_host,
      max_pages: Number(limits.max_pages),
      max_bytes: Math.round(Number(limits.max_mb) * 1024 * 1024),
      delay_ms: Number(limits.delay_ms),
      drop_rare: Number(limits.drop_rare),
    });
    if (job) follow(job);
  }

  async function pick(event) {
    const file = event.currentTarget.files?.[0];
    if (!file) return;
    text = await file.text();
    // A sensible default name, which can still be typed over.
    if (!name.trim()) name = file.name.replace(/\.[^.]+$/, "");
  }

  async function upload(event) {
    event.preventDefault();
    if (!text) return toasts.warning("Choose a file, or paste some text.");
    uploading = true;
    if (await training.upload(name.trim(), text)) {
      name = "";
      text = "";
      event.currentTarget.reset();
    }
    uploading = false;
  }

  /** Ask what each dataset would cost the chosen model. */
  async function checkAll(model) {
    checks = {};
    if (!model) return;
    for (const d of training.datasets) {
      try {
        checks[d.id] = await training.check(d.id, model);
      } catch {
        // A dataset whose file has gone cannot be checked; the row already
        // says so.
      }
    }
  }

  const counted = $derived(text ? [...text].length : 0);
  const distinct = $derived(text ? new Set(text).size : 0);

  // Searching one.
  let searchIn = $state(null);
  let question = $state("");
  let searching = $state(false);
  let found = $state(null);

  async function ask(event) {
    event.preventDefault();
    if (!searchIn || !question.trim()) return;
    searching = true;
    try {
      found = await api(
        `/api/datasets/${searchIn}/search?q=${encodeURIComponent(question.trim())}&k=5`,
      );
    } catch (e) {
      toasts.error(e.message);
      found = null;
    }
    searching = false;
  }

  const live = $derived(crawl?.job.state === "running" || crawl?.job.state === "queued");
  const result = $derived(crawl?.job.result ?? null);

  /** Where the crawl would be allowed to go, in the words the server uses. */
  const scope = $derived.by(() => {
    try {
      const u = new URL(url);
      if (limits.same_host) return `${u.origin}/`;
      const path = u.pathname.endsWith("/")
        ? u.pathname
        : u.pathname.slice(0, u.pathname.lastIndexOf("/") + 1);
      return `${u.origin}${path || "/"}`;
    } catch {
      return "";
    }
  });
</script>

<div class="mx-auto flex max-w-4xl flex-col gap-6">
  <section class="card bg-base-100 border-base-300 border">
    <form class="card-body gap-3 p-4" onsubmit={upload}>
      <h2 class="text-sm font-medium opacity-60">Add text</h2>
      <div class="flex flex-wrap items-end gap-2">
        <fieldset class="fieldset">
          <legend class="fieldset-legend">File</legend>
          <input type="file" class="file-input file-input-sm" accept=".txt,text/*" onchange={pick} />
        </fieldset>
        <fieldset class="fieldset grow">
          <legend class="fieldset-legend">Name</legend>
          <input class="input input-sm w-full" bind:value={name} placeholder="shakespeare" required />
        </fieldset>
        <button class="btn btn-sm" disabled={uploading || !text}>
          {#if uploading}<span class="loading loading-spinner loading-xs"></span>{/if}
          Upload
        </button>
      </div>
      {#if text}
        <p class="text-xs opacity-60">
          {counted.toLocaleString()} characters, {distinct} distinct — a model trained on this
          would have a vocabulary of {distinct}.
        </p>
      {/if}
      <p class="text-xs opacity-60">
        Plain text. A character tokeniser gives an id to every distinct character, so the
        second number above is the size of the model's embedding table and output head.
      </p>
    </form>
  </section>


  <!-- The other way to get a corpus: point it at a documentation site and
       let it read. A job, because it is minutes and hundreds of requests. -->
  <section class="card bg-base-100 border-base-300 border">
    <form class="card-body gap-3 p-4" onsubmit={start}>
      <h2 class="text-sm font-medium opacity-60">Read a website</h2>
      <div class="flex flex-wrap items-end gap-2">
        <fieldset class="fieldset grow">
          <legend class="fieldset-legend">Address</legend>
          <input
            class="input input-sm w-full"
            type="url"
            bind:value={url}
            placeholder="https://doc.rust-lang.org/book/"
            required
          />
        </fieldset>
        <fieldset class="fieldset">
          <legend class="fieldset-legend">Name</legend>
          <input
            class="input input-sm"
            bind:value={crawlName}
            oninput={() => (named = true)}
            placeholder="doc.rust-lang.org-book"
            required
          />
        </fieldset>
        <button class="btn btn-sm" disabled={live || !url.trim() || !crawlName.trim()}>
          {#if live}<span class="loading loading-spinner loading-xs"></span>{/if}
          Start
        </button>
      </div>

      <p class="text-xs opacity-60">
        {#if scope}
          Follows links under <code>{scope}</code> and nowhere else.
        {:else}
          Follows links under the address's own directory and nowhere else —
          <code>/book/</code> links into the standard library's documentation on nearly every
          page, and that is a hundred times the book.
        {/if}
        <code>robots.txt</code> is obeyed, and the text and a record of every page read are
        kept together.
      </p>

      <button
        type="button"
        class="link link-hover w-fit text-xs opacity-60"
        onclick={() => (showing = !showing)}
      >
        {showing ? "Hide" : "Show"} limits
      </button>

      {#if showing}
        <div class="flex flex-wrap items-end gap-2">
          <fieldset class="fieldset">
            <legend class="fieldset-legend">Pages</legend>
            <input class="input input-sm w-24" type="number" min="1" max="5000" bind:value={limits.max_pages} />
          </fieldset>
          <fieldset class="fieldset">
            <legend class="fieldset-legend">Megabytes</legend>
            <input class="input input-sm w-24" type="number" min="1" max="64" bind:value={limits.max_mb} />
          </fieldset>
          <fieldset class="fieldset">
            <legend class="fieldset-legend">Pause (ms)</legend>
            <input class="input input-sm w-24" type="number" min="50" max="10000" step="50" bind:value={limits.delay_ms} />
          </fieldset>
          <fieldset class="fieldset">
            <legend class="fieldset-legend">Drop characters seen under</legend>
            <input class="input input-sm w-24" type="number" min="0" max="1000" bind:value={limits.drop_rare} />
          </fieldset>
          <label class="label cursor-pointer gap-2 text-xs">
            <input type="checkbox" class="checkbox checkbox-sm" bind:checked={limits.same_host} />
            The whole host, not just this directory
          </label>
        </div>
        <p class="text-xs opacity-60">
          The web writes with three kinds of quotation mark and two kinds of dash, and a
          character tokeniser gives every one of them a row of its own. Typography is mapped
          onto ASCII, and characters seen fewer times than that are dropped — a row of an
          embedding table seen eight times is a row that never trained. ASCII is never
          dropped however rare it is, and the run says what went, so the number can be
          argued with.
        </p>
      {/if}
    </form>

    {#if crawl}
      <div class="border-base-300 flex flex-col gap-1 border-t px-4 py-3">
        {#if live}
          <progress class="progress w-full" value={crawl.done} max={Math.max(crawl.total, 1)}
          ></progress>
          <div class="flex items-center gap-2 text-xs opacity-70">
            <span class="whitespace-nowrap">{crawl.done} of about {crawl.total}</span>
            <span class="grow truncate">{crawl.line}</span>
            <button
              type="button"
              class="btn btn-xs btn-ghost"
              onclick={() => training.cancel(crawl.job.id)}
            >
              Stop
            </button>
          </div>
        {:else if crawl.job.error}
          <p class="text-error text-xs">{crawl.job.error}</p>
        {:else if result}
          <p class="text-xs opacity-70">
            <span class="font-medium">{result.dataset}</span>
            — {result.pages.toLocaleString()} pages, {result.characters.toLocaleString()}
            characters, {result.distinct} distinct{#if result.skipped}, {result.skipped} skipped{/if}{#if result.stopped}, stopped at {result.stopped}{/if}.
          </p>
          {#if result.dropped}
            <p class="text-xs opacity-60">
              {result.dropped} rare character{result.dropped === 1 ? "" : "s"} removed, and
              {result.mapped.toLocaleString()} mapped onto ASCII.
            </p>
          {/if}
          {#if result.manifest}
            <p class="text-xs opacity-50">Every page it read: <code>{result.manifest}</code></p>
          {/if}
        {/if}
      </div>
    {/if}
  </section>

  <section>
    <div class="mb-2 flex flex-wrap items-center gap-2">
      <h2 class="text-sm font-medium opacity-60">Datasets</h2>
      <span class="grow"></span>
      <!-- The check that saves an hour: `kvad train --from` refuses a text
           containing a character the model has never seen, but only after it
           has been chosen and started. -->
      <label class="text-xs opacity-60" for="against">Check against</label>
      <select
        id="against"
        class="select select-xs w-48"
        bind:value={against}
        onchange={(e) => checkAll(e.currentTarget.value)}
      >
        <option value="">no model</option>
        {#each training.options?.continuable ?? [] as m (m)}
          <option value={m}>{m}</option>
        {/each}
      </select>
    </div>

    {#if training.datasets.length === 0}
      <div class="bg-base-200 rounded-box p-6 text-center text-sm opacity-70">
        Nothing yet. <code>scripts/get-text.sh</code> in this repository fetches a corpus to
        start with.
      </div>
    {:else}
      <table class="table">
        <thead>
          <tr>
            <th class="w-full">Name</th>
            <th class="whitespace-nowrap">Characters</th>
            <th class="whitespace-nowrap">Distinct</th>
            <th class="whitespace-nowrap">Size</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {#each training.datasets as d (d.id)}
            <tr class="hover:bg-base-200/50">
              <td class="max-w-0">
                <div class="truncate font-medium">{d.name}</div>
                {#if d.source}
                  <div class="truncate text-xs opacity-50">
                    {d.source}{d.manifest ? " · manifest kept beside it" : ""}
                  </div>
                {/if}
                {#if !d.present}
                  <div class="text-error text-xs">the file is gone from disk</div>
                {:else if checks[d.id]}
                  {#if checks[d.id].unseen_count === 0}
                    <div class="text-xs opacity-60">
                      every character is one <code>{against}</code> knows
                    </div>
                  {:else}
                    <div class="text-error text-xs">
                      {checks[d.id].unseen_count} character{checks[d.id].unseen_count === 1
                        ? ""
                        : "s"}
                      <code>{against}</code> has no token for:
                      <code>{checks[d.id].unseen}</code>
                    </div>
                  {/if}
                {/if}
              </td>
              <td class="text-sm whitespace-nowrap opacity-70">{d.characters.toLocaleString()}</td>
              <td class="text-sm whitespace-nowrap opacity-70">{d.distinct}</td>
              <td class="text-sm whitespace-nowrap opacity-70">{humanBytes(d.bytes)}</td>
              <td class="text-right">
                <button
                  class="btn btn-sm btn-ghost"
                  aria-label={`Delete ${d.name}`}
                  onclick={() => (confirming = d)}
                >
                  <Icon path={TRASH} size={16} />
                </button>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </section>
</div>

<!-- Retrieval, on its own. Whether the right passage comes back and whether
     a model then reads it properly are different questions that fail for
     different reasons, and only the first one has an answer you can look at. -->
{#if training.datasets.length}
  <div class="mx-auto mt-6 flex max-w-4xl flex-col gap-6">
    <section class="card bg-base-100 border-base-300 border">
      <form class="card-body gap-3 p-4" onsubmit={ask}>
        <h2 class="text-sm font-medium opacity-60">Search a dataset</h2>
        <div class="flex flex-wrap items-end gap-2">
          <fieldset class="fieldset">
            <legend class="fieldset-legend">In</legend>
            <select class="select select-sm w-56" bind:value={searchIn}>
              <option value={null} disabled>choose one</option>
              {#each training.datasets as d (d.id)}
                <option value={d.id}>{d.name}</option>
              {/each}
            </select>
          </fieldset>
          <fieldset class="fieldset grow">
            <legend class="fieldset-legend">Question</legend>
            <input
              class="input input-sm w-full"
              bind:value={question}
              placeholder="what is on the stack"
            />
          </fieldset>
          <button class="btn btn-sm" disabled={searching || !searchIn || !question.trim()}>
            {#if searching}<span class="loading loading-spinner loading-xs"></span>{/if}
            Search
          </button>
        </div>

        {#if found}
          <p class="text-xs opacity-60">
            {found.passages.length} passage{found.passages.length === 1 ? "" : "s"} out of
            {found.chunks.toLocaleString()} chunks, over a vocabulary of
            {found.vocabulary.toLocaleString()} words.
            {#if found.passages.length === 0}Nothing matched — the words in the question are not
              in this corpus.{/if}
          </p>
          {#each found.passages as p (p.from)}
            <div class="border-base-300 rounded-box border p-3">
              <div class="flex flex-wrap items-baseline gap-2">
                <span class="badge badge-sm badge-neutral">{p.score.toFixed(2)}</span>
                <span class="text-sm font-medium">{p.heading}</span>
                <span class="text-xs opacity-40">
                  {p.from === p.to ? `chunk ${p.from}` : `chunks ${p.from}–${p.to}`}
                </span>
                {#if p.source}
                  <a class="link link-hover truncate text-xs opacity-60" href={p.source}>{p.source}</a>
                {/if}
              </div>
              <!-- Why this came back, which an embedding index could not say. -->
              <div class="mt-1 flex flex-wrap gap-1">
                {#each p.because.slice(0, 5) as [word, score] (word)}
                  <span class="badge badge-ghost badge-xs">{word} {score.toFixed(1)}</span>
                {/each}
              </div>
              <p class="mt-2 max-h-32 overflow-y-auto text-xs whitespace-pre-wrap opacity-70">
                {p.text}
              </p>
            </div>
          {/each}
        {:else}
          <p class="text-xs opacity-60">
            BM25 over the passages a dataset splits into, built when asked and kept until the file
            changes. The badges under each hit are what each word of the question contributed.
          </p>
        {/if}
      </form>
    </section>
  </div>
{/if}

{#if confirming}
  <div class="modal modal-open" role="dialog">
    <div class="modal-box">
      <h3 class="text-lg font-medium">Delete {confirming.name}?</h3>
      <p class="py-3 text-sm opacity-70">
        The text file goes. Models already trained on it are untouched — a model carries
        the tokeniser it was trained with and does not read this file again.
      </p>
      <div class="modal-action">
        <button class="btn btn-sm" onclick={() => (confirming = null)}>Cancel</button>
        <button
          class="btn btn-sm btn-error"
          onclick={async () => {
            const d = confirming;
            confirming = null;
            await training.removeDataset(d.id);
          }}
        >
          Delete
        </button>
      </div>
    </div>
    <button class="modal-backdrop" aria-label="Cancel" onclick={() => (confirming = null)}></button>
  </div>
{/if}
