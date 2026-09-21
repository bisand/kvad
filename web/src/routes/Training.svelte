<script>
  import { training, humanSecs } from "../lib/training.svelte.js";
  import { models } from "../lib/models.svelte.js";
  import { router, navigate } from "../lib/router.svelte.js";
  import { toasts } from "../lib/toasts.svelte.js";
  import LossChart from "../lib/components/LossChart.svelte";

  let form = $state({
    dataset: null,
    name: "",
    from: "",
    size: "small",
    steps: null,
    lr: null,
    eval_every: null,
    threads: null,
    sample: 160,
  });
  let starting = $state(false);
  // What continuing the chosen model would cost, asked before anything slow.
  let verdict = $state(null);

  $effect(() => {
    training.refresh();
    models.refresh();
  });

  // Fill the defaults in once they arrive, and follow whatever is running.
  $effect(() => {
    const o = training.options;
    if (o && form.steps === null) {
      form = { ...form, steps: o.defaults.steps, lr: o.defaults.lr, eval_every: o.defaults.eval_every, threads: o.defaults.threads };
    }
    if (o && form.dataset === null && training.datasets.length) {
      form.dataset = training.datasets[0].id;
    }
  });

  $effect(() => {
    const live = training.running;
    if (live && !training.open) training.watch(live.id);
  });

  // The unseen-character check: the answer somebody wants before committing
  // an hour, not after.
  $effect(() => {
    const { from, dataset } = form;
    verdict = null;
    if (!from || !dataset) return;
    let alive = true;
    training
      .check(dataset, from)
      .then((v) => alive && (verdict = v))
      .catch(() => {});
    return () => (alive = false);
  });

  const busy = $derived(!!training.running);
  const open = $derived(training.open);

  async function start(event) {
    event.preventDefault();
    if (!form.dataset) return toasts.warning("Upload a dataset first.");
    starting = true;
    const request = {
      dataset: form.dataset,
      size: form.size,
      steps: Number(form.steps),
      lr: Number(form.lr),
      eval_every: Number(form.eval_every),
      threads: Number(form.threads),
      sample: Number(form.sample),
    };
    if (form.from) request.from = form.from;
    if (form.name.trim()) request.name = form.name.trim();
    await training.start(request);
    starting = false;
  }

  /// Whether a run got far enough to measure anything.
  ///
  /// `measured` is the honest answer and rows written before it existed do
  /// not have one; for those, a best loss that is there at all is one that was
  /// measured.
  function measured(result) {
    return result?.measured ?? result?.best_val != null;
  }

  function stateBadge(state) {
    return (
      { running: "badge-info", done: "badge-success", failed: "badge-error", cancelled: "badge-warning" }[
        state
      ] ?? ""
    );
  }
</script>

<div class="flex flex-col gap-6">
  {#if busy}
    <!-- Measured, not guessed: five generations at each setting, with idle
         runs either side to catch drift. Training on all 18 cores of this
         machine left chat at 45–52% of idle speed; capped to 8, 67%. So the
         honest thing to say is "slower", not "queued". -->
    <div role="alert" class="alert alert-info">
      <span>
        A training run is using the cores. Replies stay possible and come at roughly half
        speed while it runs — capping the run's threads gets some of that back.
      </span>
    </div>
  {/if}

  <div class="grid gap-6 lg:grid-cols-[22rem_1fr]">
    <!-- What to run. -->
    <form class="card bg-base-100 border-base-300 h-fit border" onsubmit={start}>
      <div class="card-body gap-3 p-4">
        <h2 class="text-sm font-medium opacity-60">New run</h2>

        <fieldset class="fieldset">
          <legend class="fieldset-legend">Dataset</legend>
          <select class="select select-sm w-full" bind:value={form.dataset}>
            {#each training.datasets as d (d.id)}
              <option value={d.id} disabled={!d.present}>
                {d.name} — {d.characters.toLocaleString()} chars, {d.distinct} distinct
              </option>
            {:else}
              <option value={null}>nothing uploaded yet</option>
            {/each}
          </select>
          {#if training.datasets.length === 0}
            <p class="mt-1 text-xs opacity-60">
              <a href="/datasets" onclick={(e) => navigate(e, "/datasets")} class="link">
                Upload a text file
              </a> first.
            </p>
          {/if}
        </fieldset>

        <fieldset class="fieldset">
          <legend class="fieldset-legend">Continue a model</legend>
          <select class="select select-sm w-full" bind:value={form.from}>
            <option value="">no — train a new one</option>
            {#each training.options?.continuable ?? [] as m (m)}
              <option value={m}>{m}</option>
            {/each}
          </select>
          {#if verdict}
            {#if verdict.unseen_count === 0}
              <p class="mt-1 text-xs opacity-60">
                Every character in this text is one <code>{verdict.model}</code> already has a
                token for.
              </p>
            {:else}
              <!-- `kvad train` gives this error too, but only after reading the
                   file and only for the first offending character. -->
              <p class="text-error mt-1 text-xs">
                <code>{verdict.dataset}</code> has {verdict.unseen_count} character{verdict.unseen_count ===
                1
                  ? ""
                  : "s"}
                <code>{verdict.model}</code> has no token for:
                <code>{verdict.unseen}</code>. A vocabulary is fixed at first training — train a
                new model on both texts instead.
              </p>
            {/if}
          {/if}
        </fieldset>

        <fieldset class="fieldset">
          <legend class="fieldset-legend">Name</legend>
          <input
            class="input input-sm w-full"
            bind:value={form.name}
            placeholder={form.from ? `${form.from} (in place)` : "shakespeare"}
          />
          <p class="mt-1 text-xs opacity-60">
            {form.from
              ? "Leave empty to train it in place, keeping no copy of the old one."
              : "How you will run it afterwards: kvad run --model NAME."}
          </p>
        </fieldset>

        {#if !form.from}
          <fieldset class="fieldset">
            <legend class="fieldset-legend">Size</legend>
            <select class="select select-sm w-full" bind:value={form.size}>
              {#each training.options?.sizes ?? [] as s (s.name)}
                <option value={s.name}>{s.name} — {s.shape}</option>
              {/each}
            </select>
          </fieldset>
        {:else}
          <p class="text-xs opacity-60">
            A model's shape is fixed when it is first trained, so there is no size to pick.
          </p>
        {/if}

        <div class="grid grid-cols-2 gap-2">
          <fieldset class="fieldset">
            <legend class="fieldset-legend">Steps</legend>
            <input class="input input-sm w-full" type="number" min="1" bind:value={form.steps} />
          </fieldset>
          <fieldset class="fieldset">
            <legend class="fieldset-legend">Learning rate</legend>
            <input class="input input-sm w-full" type="number" step="0.0001" bind:value={form.lr} />
          </fieldset>
          <fieldset class="fieldset">
            <legend class="fieldset-legend">Check every</legend>
            <input class="input input-sm w-full" type="number" min="1" bind:value={form.eval_every} />
          </fieldset>
          <fieldset class="fieldset">
            <legend class="fieldset-legend">Threads of {training.options?.cores ?? "?"}</legend>
            <input
              class="input input-sm w-full"
              type="number"
              min="1"
              max={training.options?.cores ?? 64}
              bind:value={form.threads}
            />
          </fieldset>
        </div>

        <button class="btn btn-sm btn-primary" disabled={busy || starting || !form.dataset}>
          {#if starting}<span class="loading loading-spinner loading-xs"></span>{/if}
          {busy ? "A run is already going" : "Start"}
        </button>
      </div>
    </form>

    <!-- What it is doing. -->
    <div class="flex min-w-0 flex-col gap-4">
      {#if open}
        <section class="card bg-base-100 border-base-300 border">
          <div class="card-body gap-3 p-4">
            <div class="flex flex-wrap items-center gap-2">
              <h2 class="font-medium">{open.job.label}</h2>
              <span class="badge badge-sm {stateBadge(open.job.state)}">{open.job.state}</span>
              <span class="text-xs opacity-60">
                {open.job.params.size} · {open.job.params.steps.toLocaleString()} steps ·
                {open.job.params.dataset_name}
              </span>
              <span class="grow"></span>
              {#if open.job.state === "running"}
                <button class="btn btn-sm" onclick={() => training.cancel(open.job.id)}>Stop</button>
              {/if}
              {#if measured(open.job.result)}
                <button
                  class="btn btn-sm"
                  onclick={async () => {
                    await models.load(open.job.result.handle, "cpu-f32");
                    router.go("/chat");
                  }}
                >
                  Chat with this
                </button>
              {/if}
            </div>

            {#if open.metrics.length}
              <LossChart
                metrics={open.metrics}
                bestStep={training.bestStep}
                steps={open.job.params?.steps}
              />
              {#key open.metrics.length}
                <p class="text-xs opacity-60">
                  {open.metrics.length} checkpoint{open.metrics.length === 1 ? "" : "s"} ·
                  step {open.metrics.at(-1).step.toLocaleString()} of
                  {open.job.params.steps.toLocaleString()} ·
                  {Math.round(open.metrics.at(-1).chars_per_sec).toLocaleString()} chars/s ·
                  {humanSecs(open.metrics.at(-1).elapsed_secs)} elapsed
                  {#if training.bestStep !== null}
                    · best at step {training.bestStep.toLocaleString()}
                  {/if}
                </p>
              {/key}
            {:else if open.job.state === "running"}
              <div class="flex items-center gap-2 py-8 text-sm opacity-60">
                <span class="loading loading-spinner loading-sm"></span>
                Training. The first checkpoint is at step {open.job.params.eval_every}.
              </div>
            {/if}

            {#if open.job.error}
              <div role="alert" class="alert alert-error text-sm">{open.job.error}</div>
            {/if}

            {#if open.job.result}
              <p class="text-xs opacity-70">
                {#if open.job.result.improved === false}
                  <!-- A continuation that helped nothing. Reporting a kept
                       step here would name a step whose model was never
                       written; the model is the one the run started from. -->
                  Nothing beat the model it continued ({open.job.result.best_val.toFixed(3)});
                  the best checkpoint reached
                  {open.job.result.reached?.toFixed(3) ?? "nothing"}, so nothing was written ·
                {:else if measured(open.job.result)}
                  Kept step {open.job.result.best_step.toLocaleString()}: validation loss
                  {open.job.result.best_val.toFixed(3)} (the last step measured
                  {open.job.result.last_val.toFixed(3)}) ·
                {:else}
                  <!-- Stopped before the first checkpoint, so there is no loss
                       to report and no model was written. -->
                  Stopped before its first checkpoint, so nothing was measured and nothing
                  was saved ·
                {/if}
                {open.job.result.params.toLocaleString()} parameters ·
                {humanSecs(open.job.result.elapsed_secs)}
              </p>
            {/if}

            {#if training.log.length}
              <details class="text-xs">
                <summary class="cursor-pointer opacity-60">Log</summary>
                <pre class="bg-base-200 rounded-box mt-2 max-h-40 overflow-auto p-2 text-xs">{training.log.join("\n")}</pre>
              </details>
            {/if}
          </div>
        </section>

        {#if open.samples.length}
          <section>
            <h2 class="mb-2 text-sm font-medium opacity-60">What it writes</h2>
            <div class="flex flex-col gap-2">
              {#each [...open.samples].sort((a, b) => b.step - a.step) as s (s.step)}
                <div class="bg-base-200 rounded-box p-3">
                  <p class="mb-1 text-xs opacity-50">step {s.step.toLocaleString()}</p>
                  <p class="font-mono text-xs whitespace-pre-wrap">{s.text}</p>
                </div>
              {/each}
            </div>
          </section>
        {/if}
      {/if}

      <!-- Everything that has run. -->
      <section>
        <h2 class="mb-2 text-sm font-medium opacity-60">Runs</h2>
        {#if training.jobs.filter((j) => j.kind === "train").length === 0}
          <p class="text-sm opacity-60">Nothing yet.</p>
        {:else}
          <table class="table table-sm">
            <tbody>
              {#each training.jobs.filter((j) => j.kind === "train") as j (j.id)}
                <tr class="hover:bg-base-200/50 cursor-pointer" onclick={() => training.watch(j.id)}>
                  <td class="w-full">
                    <span class="font-medium">{j.label}</span>
                    <span class="ml-2 text-xs opacity-60">{j.params.dataset_name}</span>
                  </td>
                  <td class="whitespace-nowrap">
                    <span class="badge badge-sm {stateBadge(j.state)}">{j.state}</span>
                  </td>
                  <td class="text-xs whitespace-nowrap opacity-60">
                    {measured(j.result) ? j.result.best_val.toFixed(3) : "—"}
                  </td>
                  <td class="text-xs whitespace-nowrap opacity-60">{j.created_at}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        {/if}
      </section>
    </div>
  </div>
</div>
