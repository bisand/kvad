<script>
  // Two ways of asking whether a model is any good.
  import { evals, matrix } from "../lib/evals.svelte.js";
  import { models } from "../lib/models.svelte.js";
  import { training } from "../lib/training.svelte.js";
  import VariantPicker from "../lib/components/VariantPicker.svelte";
  import Icon from "../lib/components/Icon.svelte";

  let tab = $state("suites");
  let editing = $state(null);
  let variants = $state([]);
  let maxTokens = $state(64);
  let dataset = $state(null);
  let window_ = $state(512);

  $effect(() => {
    models.refresh();
    evals.refresh();
    training.refresh();
  });

  const open = $derived(evals.open);
  const running = $derived(open?.job?.state === "running" || open?.job?.state === "queued");
  const grid = $derived(matrix(open?.cases ?? []));

  function blank() {
    return { id: null, name: "", cases: [{ prompt: "", expect: "", match: "contains" }] };
  }

  function edit(suite) {
    editing = suite
      ? { id: suite.id, name: suite.name, cases: suite.cases.map((c) => ({ ...c })) }
      : blank();
  }

  async function save() {
    const saved = await evals.saveSuite(editing);
    if (saved) editing = null;
  }

  async function runSuite(suite) {
    if (!variants.length) return;
    await evals.run("/api/evals/run", {
      suite: suite.id,
      variants,
      max_tokens: maxTokens,
    });
    tab = "results";
  }

  async function runPerplexity() {
    if (!variants.length || !dataset) return;
    await evals.run("/api/evals/perplexity", {
      dataset: Number(dataset),
      variants,
      window: window_,
    });
    tab = "results";
  }
</script>

<div class="flex flex-col gap-6">
  <div role="tablist" class="tabs tabs-box w-fit">
    <button role="tab" class="tab" class:tab-active={tab === "suites"} onclick={() => (tab = "suites")}>
      Prompt suites
    </button>
    <button role="tab" class="tab" class:tab-active={tab === "perplexity"} onclick={() => (tab = "perplexity")}>
      Perplexity
    </button>
    <button role="tab" class="tab" class:tab-active={tab === "results"} onclick={() => (tab = "results")}>
      Runs
    </button>
  </div>

  {#if tab !== "results"}
    <section class="bg-base-200 rounded-box flex flex-col gap-3 p-4">
      <h2 class="text-sm font-medium opacity-60">Run against</h2>
      <VariantPicker bind:value={variants} />
      <p class="text-xs opacity-60">
        One variant at a time, alone: anything else in memory is unloaded first, and
        each variant is loaded, asked everything, and put away before the next. The same
        model at two precisions is the comparison this page is for.
      </p>
    </section>
  {/if}

  {#if tab === "suites"}
    <section class="flex flex-col gap-3">
      <div class="flex items-center gap-3">
        <h2 class="text-sm font-medium opacity-60">Suites</h2>
        <span class="grow"></span>
        <label class="flex items-center gap-2 text-xs">
          <span class="opacity-70">Tokens per answer</span>
          <input type="number" min="1" max="1024" class="input input-xs w-20" bind:value={maxTokens} />
        </label>
        <button class="btn btn-sm" onclick={() => edit(null)}>
          <Icon path="M12 5v14M5 12h14" size={16} /> New suite
        </button>
      </div>

      <p class="text-xs opacity-60">
        Cases run greedily, with a fixed seed, so a case that fails is a change in the
        model rather than a change in the dice. <code>contains</code> ignores case and
        surrounding space; <code>equals</code> wants the whole answer.
      </p>

      {#each evals.suites as suite (suite.id)}
        <div class="bg-base-200 rounded-box flex items-center gap-3 p-3">
          <div class="grow">
            <div class="text-sm font-medium">{suite.name}</div>
            <div class="text-xs opacity-60">{suite.cases.length} cases</div>
          </div>
          <button class="btn btn-ghost btn-xs" onclick={() => edit(suite)}>Edit</button>
          <button class="btn btn-ghost btn-xs" onclick={() => evals.deleteSuite(suite.id)}>Delete</button>
          <button class="btn btn-primary btn-xs" onclick={() => runSuite(suite)}
            disabled={!variants.length || running}>Run</button>
        </div>
      {:else}
        <p class="text-sm opacity-60">No suites yet.</p>
      {/each}

      {#if editing}
        <div class="bg-base-200 rounded-box flex flex-col gap-3 p-4">
          <input class="input input-sm w-full" placeholder="Suite name" bind:value={editing.name} />
          {#each editing.cases as c, i (i)}
            <div class="flex flex-wrap items-center gap-2">
              <input class="input input-sm grow font-mono" placeholder="Prompt" bind:value={c.prompt} />
              <select class="select select-sm w-28" bind:value={c.match}>
                <option value="contains">contains</option>
                <option value="equals">equals</option>
              </select>
              <input class="input input-sm grow font-mono" placeholder="Expected" bind:value={c.expect} />
              <button class="btn btn-ghost btn-sm btn-square"
                onclick={() => (editing.cases = editing.cases.filter((_, at) => at !== i))}
                aria-label="Remove case">
                <Icon path="M6 6l12 12M18 6L6 18" size={16} />
              </button>
            </div>
          {/each}
          <div class="flex gap-2">
            <button class="btn btn-sm"
              onclick={() => (editing.cases = [...editing.cases, { prompt: "", expect: "", match: "contains" }])}>
              Add a case
            </button>
            <span class="grow"></span>
            <button class="btn btn-ghost btn-sm" onclick={() => (editing = null)}>Cancel</button>
            <button class="btn btn-primary btn-sm" onclick={save}>Save</button>
          </div>
        </div>
      {/if}
    </section>
  {:else if tab === "perplexity"}
    <section class="flex flex-col gap-3">
      <p class="text-sm opacity-70">
        How surprised a model is by text it did not write: the exponential of its mean
        surprise per token, so 20 means it was choosing between about twenty equally
        likely tokens at each step. Lower is better. The vocabulary size is what a model
        that had learnt nothing would score.
      </p>
      <p class="text-xs opacity-60">
        Scored in windows, each starting from an empty cache, and the first token of each
        window is not graded — nothing precedes it, so there is no prediction to mark.
        That makes the number slightly pessimistic and, more importantly, comparable
        between models.
      </p>

      <div class="flex flex-wrap items-end gap-3">
        <label class="flex flex-col gap-1 text-xs">
          <span class="opacity-70">Held-out text</span>
          <select class="select select-sm min-w-56" bind:value={dataset}>
            <option value={null}>Pick a dataset…</option>
            {#each training.datasets as d (d.id)}
              <option value={d.id}>{d.name} ({d.characters.toLocaleString()} characters)</option>
            {/each}
          </select>
        </label>
        <label class="flex flex-col gap-1 text-xs">
          <span class="opacity-70">Window</span>
          <input type="number" min="2" max="8192" class="input input-sm w-24" bind:value={window_} />
        </label>
        <button class="btn btn-primary btn-sm" onclick={runPerplexity}
          disabled={!variants.length || !dataset || running}>Score</button>
      </div>
      <p class="text-xs opacity-60">
        Datasets are the ones on the Datasets page. Use text the model has not been
        trained on, or the number measures memorisation.
      </p>
    </section>
  {/if}

  {#if tab === "results"}
    <section class="flex flex-col gap-3">
      {#if open}
        <div class="flex items-center gap-3">
          <h2 class="text-sm font-medium opacity-60">{open.job.label}</h2>
          <span class="badge badge-sm" class:badge-success={open.job.state === "done"}
            class:badge-error={open.job.state === "failed"}>{open.job.state}</span>
          <span class="grow"></span>
          {#if running}
            {#if open.progress}
              <span class="text-xs opacity-70">{open.progress.done} of {open.progress.total}</span>
            {/if}
            <button class="btn btn-xs" onclick={() => evals.cancel(open.job.id)}>Stop</button>
          {/if}
        </div>
        {#if open.job.error}
          <div role="alert" class="alert alert-error"><span>{open.job.error}</span></div>
        {/if}
        {#if running && open.log.at(-1)}
          <p class="text-xs opacity-60">{open.log.at(-1)}</p>
        {/if}

        {#if open.scores.length}
          <table class="table table-sm">
            <thead>
              <tr>
                <th class="w-full">Variant</th>
                <th class="text-right">Perplexity</th>
                <th class="text-right">Bits/token</th>
                <th class="text-right">Tokens scored</th>
                <th class="text-right">Took</th>
              </tr>
            </thead>
            <tbody>
              {#each open.scores as s (s.variant)}
                <tr>
                  <td class="max-w-0 truncate font-mono text-xs">{s.variant}</td>
                  <td class="text-right font-mono">{s.perplexity.toFixed(2)}</td>
                  <td class="text-right font-mono text-xs opacity-70">{s.bits_per_token.toFixed(2)}</td>
                  <td class="text-right text-xs opacity-70">{s.scored.toLocaleString()}</td>
                  <td class="text-right text-xs opacity-70">{s.took_secs.toFixed(1)}s</td>
                </tr>
              {/each}
            </tbody>
          </table>
        {/if}

        {#if grid.rows.length}
          <div class="overflow-x-auto">
            <table class="table table-sm">
              <thead>
                <tr>
                  <th class="w-full min-w-56">Case</th>
                  {#each grid.variants as v (v)}
                    <th class="font-mono text-xs font-normal">{v}</th>
                  {/each}
                </tr>
              </thead>
              <tbody>
                {#each grid.rows as row (row.idx)}
                  <tr>
                    <td class="min-w-56">
                      <div class="font-mono text-xs">{row.prompt}</div>
                      <div class="text-xs opacity-60">wants “{row.expect}”</div>
                    </td>
                    {#each row.by as cell (cell?.variant ?? Math.random())}
                      <td>
                        {#if !cell}
                          <span class="opacity-40">—</span>
                        {:else}
                          <div class="tooltip" data-tip={cell.got}>
                            <span class="badge badge-sm"
                              class:badge-success={cell.passed}
                              class:badge-error={!cell.passed}>
                              {cell.passed ? "pass" : "fail"}
                            </span>
                          </div>
                        {/if}
                      </td>
                    {/each}
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
          <p class="text-xs opacity-60">Hover a verdict to read what the model actually said.</p>
        {/if}
      {/if}

      <h2 class="mt-4 text-sm font-medium opacity-60">Earlier runs</h2>
      {#if evals.runs.length === 0}
        <p class="text-sm opacity-60">Nothing run yet.</p>
      {:else}
        <table class="table table-sm">
          <tbody>
            {#each evals.runs as r (r.id)}
              <tr class="hover:bg-base-200 cursor-pointer" onclick={() => evals.watch(r.id)}>
                <td class="w-full">
                  <div class="text-sm">{r.label}</div>
                  <div class="text-xs opacity-60">
                    {#if r.result?.of}
                      {r.result.passed} of {r.result.of} passed
                    {:else if r.result?.scores?.length}
                      {r.result.scores.length} scored
                    {/if}
                  </div>
                </td>
                <td class="text-xs whitespace-nowrap opacity-60">
                  {r.ended_at ?? r.started_at ?? r.created_at}
                </td>
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
  {/if}
</div>
