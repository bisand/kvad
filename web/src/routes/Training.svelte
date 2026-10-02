<script>
  import { training, humanSecs } from "../lib/training.svelte.js";
  import { models, humanBytes } from "../lib/models.svelte.js";
  import { router, navigate } from "../lib/router.svelte.js";
  import { toasts } from "../lib/toasts.svelte.js";
  import LossChart from "../lib/components/LossChart.svelte";

  // Which loop the form starts: a language model from a text, or a LoRA for
  // an image model from a set of pictures. One page, because both are a run:
  // a dataset, some steps, a loss to watch, something kept at the end.
  let kind = $state("text");

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
  let lora = $state({
    dataset: null,
    model: "",
    name: "",
    from: "",
    size: null,
    rank: null,
    steps: null,
    lr: null,
    eval_every: null,
    // One prompt a line.
    samples: "",
    sample_size: "",
    sample_steps: null,
  });
  let starting = $state(false);
  // What continuing the chosen model would cost, asked before anything slow.
  let verdict = $state(null);

  $effect(() => {
    training.refresh();
    models.refresh();
  });

  const texts = $derived(training.datasets.filter((d) => d.kind !== "pictures"));
  const sets = $derived(training.datasets.filter((d) => d.kind === "pictures"));

  // Fill the defaults in once they arrive, and follow whatever is running.
  $effect(() => {
    const o = training.options;
    if (o && form.steps === null) {
      form = { ...form, steps: o.defaults.steps, lr: o.defaults.lr, eval_every: o.defaults.eval_every, threads: o.defaults.threads };
    }
    if (o && form.dataset === null && texts.length) {
      form.dataset = texts[0].id;
    }
    if (o?.lora && lora.steps === null) {
      const d = o.lora.defaults;
      lora = {
        ...lora,
        size: d.size,
        rank: d.rank,
        steps: d.steps,
        lr: d.lr,
        eval_every: d.eval_every,
        sample_steps: d.sample_steps,
        model: o.lora.models[0] ?? "",
      };
    }
    if (o && lora.dataset === null && sets.length) {
      lora.dataset = sets[0].id;
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
  const isLora = (job) => job?.params?.loop === "lora";

  // What a LoRA run is asked for, and whether the server has the room: a run
  // is charged to the memory the models are, before it starts.
  const prompts = $derived(
    lora.samples
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean),
  );
  const charge = $derived.by(() => {
    const row = training.options?.lora?.sizes.find((s) => s.size === Number(lora.size));
    return row ? (prompts.length ? row.bytes_sampling : row.bytes) : null;
  });
  const left = $derived(training.options?.lora?.left ?? null);
  const fits = $derived(charge === null || left === null || charge <= left);
  const unavailable = $derived(training.options?.lora?.unavailable ?? null);

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

  async function startLora(event) {
    event.preventDefault();
    if (!lora.dataset) return toasts.warning("Upload a set of pictures first.");
    if (prompts.length > 4) return toasts.warning("Four prompts at most: each is drawn at every measurement.");
    starting = true;
    const request = {
      loop: "lora",
      dataset: lora.dataset,
      name: lora.name.trim(),
      model: lora.model,
      size: Number(lora.size),
      rank: Number(lora.rank),
      steps: Number(lora.steps),
      lr: Number(lora.lr),
      eval_every: Number(lora.eval_every),
      samples: prompts,
      sample_steps: Number(lora.sample_steps),
    };
    if (lora.from) request.from = lora.from;
    if (lora.sample_size) request.sample_size = Number(lora.sample_size);
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

  /// A loss to read: three places for a text run's, which is around 1, and
  /// four for a diffusion run's, which is around 0.01.
  function loss(job, value) {
    return value == null ? "—" : value.toFixed(isLora(job) ? 4 : 3);
  }

  /// A LoRA's loss against the model's own, as a signed share of it.
  function against(value, base) {
    if (value == null || !base) return "";
    const share = (value / base - 1) * 100;
    return `${share > 0 ? "+" : "−"}${Math.abs(share).toFixed(1)}%`;
  }

  // A LoRA run's samples as a grid: a row a prompt, a column a measurement,
  // so that a row reads left to right as what the LoRA learned.
  const drawnAt = $derived([...new Set((open?.pictures ?? []).map((p) => p.step))].sort((a, b) => a - b));
  const drawnOf = $derived([...new Set((open?.pictures ?? []).map((p) => p.prompt))].sort((a, b) => a - b));
  function drawn(prompt, step) {
    return open.pictures.find((p) => p.prompt === prompt && p.step === step);
  }
  let looking = $state(null);

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
    {#if isLora(training.running)}
      <div role="alert" class="alert alert-info">
        <span>
          A LoRA run is on the GPU, in a process of its own. Anything else asked of the GPU
          while it runs shares it with the run.
        </span>
      </div>
    {:else}
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
  {/if}

  <div class="grid gap-6 lg:grid-cols-[22rem_1fr]">
    <!-- What to run. -->
    <div class="card bg-base-100 border-base-300 h-fit border">
      <div class="card-body gap-3 p-4">
        <h2 class="text-sm font-medium opacity-60">New run</h2>
        <div role="tablist" class="tabs tabs-box tabs-sm">
          <button role="tab" class="tab grow" class:tab-active={kind === "text"} onclick={() => (kind = "text")}>
            A language model
          </button>
          <button role="tab" class="tab grow" class:tab-active={kind === "lora"} onclick={() => (kind = "lora")}>
            A LoRA for images
          </button>
        </div>

        {#if kind === "text"}
          <form class="flex flex-col gap-3" onsubmit={start}>
            <fieldset class="fieldset">
              <legend class="fieldset-legend">Dataset</legend>
              <select class="select select-sm w-full" bind:value={form.dataset}>
                {#each texts as d (d.id)}
                  <option value={d.id} disabled={!d.present}>
                    {d.name} — {d.characters.toLocaleString()} chars, {d.distinct} distinct
                  </option>
                {:else}
                  <option value={null}>nothing uploaded yet</option>
                {/each}
              </select>
              {#if texts.length === 0}
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
          </form>
        {:else if unavailable}
          <p class="text-sm opacity-70">{unavailable}.</p>
        {:else}
          <form class="flex flex-col gap-3" onsubmit={startLora}>
            <fieldset class="fieldset">
              <legend class="fieldset-legend">Pictures</legend>
              <select class="select select-sm w-full" bind:value={lora.dataset}>
                {#each sets as d (d.id)}
                  <option value={d.id} disabled={!d.present}>{d.name} — {d.items} pictures</option>
                {:else}
                  <option value={null}>nothing uploaded yet</option>
                {/each}
              </select>
              {#if sets.length === 0}
                <p class="mt-1 text-xs opacity-60">
                  <a href="/datasets" onclick={(e) => navigate(e, "/datasets")} class="link">
                    Upload pictures and their captions
                  </a> first.
                </p>
              {/if}
            </fieldset>

            <fieldset class="fieldset">
              <legend class="fieldset-legend">For the model</legend>
              <select class="select select-sm w-full" bind:value={lora.model}>
                {#each training.options?.lora?.models ?? [] as m (m)}
                  <option value={m}>{m}</option>
                {:else}
                  <option value="">none on this machine</option>
                {/each}
              </select>
              <p class="mt-1 text-xs opacity-60">
                SDXL, or a fine-tune of it published as a repo. Its weights stay as they are; the
                LoRA is a pair of thin matrices on each of its attention layers.
              </p>
            </fieldset>

            <fieldset class="fieldset">
              <legend class="fieldset-legend">Name</legend>
              <input class="input input-sm w-full" bind:value={lora.name} placeholder="my-style" required />
            </fieldset>

            {#if training.options?.lora?.continuable.length}
              <fieldset class="fieldset">
                <legend class="fieldset-legend">Go on from</legend>
                <select class="select select-sm w-full" bind:value={lora.from}>
                  <option value="">no — start a new one</option>
                  {#each training.options.lora.continuable as l (l)}
                    <option value={l}>{l}</option>
                  {/each}
                </select>
              </fieldset>
            {/if}

            <div class="grid grid-cols-2 gap-2">
              <fieldset class="fieldset">
                <legend class="fieldset-legend">Size</legend>
                <select class="select select-sm w-full" bind:value={lora.size}>
                  {#each training.options?.lora?.sizes ?? [] as s (s.size)}
                    <option value={s.size}>{s.size} × {s.size}</option>
                  {/each}
                </select>
              </fieldset>
              <fieldset class="fieldset">
                <legend class="fieldset-legend">Rank</legend>
                <input class="input input-sm w-full" type="number" min="1" max="128" bind:value={lora.rank} />
              </fieldset>
              <fieldset class="fieldset">
                <legend class="fieldset-legend">Steps</legend>
                <input class="input input-sm w-full" type="number" min="1" bind:value={lora.steps} />
              </fieldset>
              <fieldset class="fieldset">
                <legend class="fieldset-legend">Learning rate</legend>
                <input class="input input-sm w-full" type="number" step="0.00001" bind:value={lora.lr} />
              </fieldset>
              <fieldset class="fieldset">
                <legend class="fieldset-legend">Measure every</legend>
                <input class="input input-sm w-full" type="number" min="1" bind:value={lora.eval_every} />
              </fieldset>
            </div>

            <fieldset class="fieldset">
              <legend class="fieldset-legend">Draw at each measurement</legend>
              <textarea
                class="textarea textarea-sm w-full"
                rows="2"
                bind:value={lora.samples}
                placeholder="a lighthouse, my-style"
              ></textarea>
              <p class="mt-1 text-xs opacity-60">
                One prompt a line, four at most. Each is drawn from the same seed every time, so a
                row of its pictures differs by what the LoRA learned and by nothing else.
              </p>
            </fieldset>

            {#if prompts.length}
              <div class="grid grid-cols-2 gap-2">
                <fieldset class="fieldset">
                  <legend class="fieldset-legend">Sample size</legend>
                  <select class="select select-sm w-full" bind:value={lora.sample_size}>
                    <option value="">the run's</option>
                    {#each training.options?.lora?.sizes ?? [] as s (s.size)}
                      <option value={String(s.size)}>{s.size} × {s.size}</option>
                    {/each}
                  </select>
                </fieldset>
                <fieldset class="fieldset">
                  <legend class="fieldset-legend">Sample steps</legend>
                  <input class="input input-sm w-full" type="number" min="1" max="100" bind:value={lora.sample_steps} />
                </fieldset>
              </div>
            {/if}

            {#if charge !== null && left !== null}
              <p class="text-xs {fits ? 'opacity-60' : 'text-error'}">
                This run is charged {humanBytes(charge)} of memory, as a model is;
                {humanBytes(left)} is left beside the models in memory{fits
                  ? "."
                  : ". Unload a model first."}
              </p>
            {/if}

            <button
              class="btn btn-sm btn-primary"
              disabled={busy || starting || !lora.dataset || !lora.model || !lora.name.trim() || !fits}
            >
              {#if starting}<span class="loading loading-spinner loading-xs"></span>{/if}
              {busy ? "A run is already going" : "Start"}
            </button>
          </form>
        {/if}
      </div>
    </div>

    <!-- What it is doing. -->
    <div class="flex min-w-0 flex-col gap-4">
      {#if open}
        <section class="card bg-base-100 border-base-300 border">
          <div class="card-body gap-3 p-4">
            <div class="flex flex-wrap items-center gap-2">
              <h2 class="font-medium">{open.job.label}</h2>
              <span class="badge badge-sm {stateBadge(open.job.state)}">{open.job.state}</span>
              <span class="text-xs opacity-60">
                {#if isLora(open.job)}
                  a LoRA for {open.job.params.model} · {open.job.params.size} × {open.job.params.size} ·
                {:else}
                  {open.job.params.size} ·
                {/if}
                {open.job.params.steps.toLocaleString()} steps ·
                {open.job.params.dataset_name}
              </span>
              <span class="grow"></span>
              {#if open.job.state === "running"}
                <button class="btn btn-sm" onclick={() => training.cancel(open.job.id)}>Stop</button>
              {/if}
              {#if measured(open.job.result) && !isLora(open.job)}
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

            {#if isLora(open.job)}
              {#if open.measures.length}
                <!-- The validation loss alone. A step's training loss says
                     mostly which noise level it drew; this is the same
                     pictures at the same levels with the same noise every
                     time, so two points differ by what was learned. -->
                <LossChart
                  metrics={open.measures}
                  bestStep={training.bestStep}
                  steps={open.job.params?.steps}
                  train={false}
                />
                {#key `${open.measures.length}/${open.stepped?.step}`}
                  <p class="text-xs opacity-60">
                    {open.measures.length} measurement{open.measures.length === 1 ? "" : "s"} of the
                    validation loss · the model's own, before any step, is
                    {loss(open.job, open.measures[0].val_loss)}
                    {#if open.job.state === "running" && open.stepped}
                      · step {open.stepped.step.toLocaleString()} of
                      {open.stepped.steps.toLocaleString()}, {open.stepped.secs.toFixed(1)} s a step
                    {/if}
                    · {humanSecs(open.measures.at(-1).elapsed_secs)} elapsed
                    {#if training.bestStep !== null}
                      · best at step {training.bestStep.toLocaleString()}
                    {/if}
                  </p>
                {/key}
              {:else if open.job.state === "running"}
                <div class="flex items-center gap-2 py-8 text-sm opacity-60">
                  <span class="loading loading-spinner loading-sm"></span>
                  Reading the pictures and loading the model. The first measurement is of the model
                  as it is, before any step.
                </div>
              {/if}
            {:else if open.metrics.length}
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

            {#if open.job.result && isLora(open.job)}
              {@const r = open.job.result}
              <p class="text-xs opacity-70">
                {#if !measured(r)}
                  Stopped before its first step, so nothing was trained and nothing was saved ·
                {:else}
                  Kept step {r.best_step.toLocaleString()}: validation loss {loss(open.job, r.best_val)},
                  {against(r.best_val, r.base_val)} of the model's own {loss(open.job, r.base_val)}
                  {#if !r.improved}
                    — no better than the model without the LoRA, on pictures it did not train on
                  {/if}
                  · {r.pictures} pictures trained on{r.held_out ? `, ${r.held_out} held out` : ""} ·
                {/if}
                {(r.params / 1e6).toFixed(1)} M numbers on {r.layers} layers ·
                {humanSecs(r.elapsed_secs)}
              </p>
              {#if r.handle}
                <p class="text-xs opacity-70">
                  The LoRA: <code class="break-all">{r.handle}</code>{#if r.last}; and the last step's,
                    <code class="break-all">{r.last}</code>{/if}
                </p>
              {/if}
            {:else if open.job.result}
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

        {#if open.pictures.length}
          <section>
            <h2 class="mb-2 text-sm font-medium opacity-60">What it draws</h2>
            <p class="mb-2 text-xs opacity-60">
              A row a prompt, a column a measurement. Step 0 is the model without the LoRA, and
              every picture in a row starts from the same noise.
            </p>
            <div class="flex flex-col gap-4">
              {#each drawnOf as prompt (prompt)}
                <div>
                  <p class="mb-1 text-xs opacity-70">{open.job.params.samples?.[prompt] ?? `prompt ${prompt + 1}`}</p>
                  <div class="flex gap-2 overflow-x-auto pb-1">
                    {#each drawnAt as step (step)}
                      {@const p = drawn(prompt, step)}
                      {#if p}
                        <button
                          class="shrink-0 cursor-zoom-in text-left"
                          onclick={() => (looking = { ...p, text: open.job.params.samples?.[prompt] })}
                        >
                          <img
                            class="rounded-box bg-base-200 size-40 object-cover"
                            src={p.url}
                            alt={`step ${step}`}
                            loading="lazy"
                          />
                          <span class="text-xs opacity-50">
                            step {step.toLocaleString()}{step === training.bestStep ? " · kept" : ""}
                          </span>
                        </button>
                      {/if}
                    {/each}
                  </div>
                </div>
              {/each}
            </div>
          </section>
        {/if}

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
                    {#if isLora(j)}<span class="badge badge-ghost badge-sm ml-1">LoRA</span>{/if}
                    <span class="ml-2 text-xs opacity-60">{j.params.dataset_name}</span>
                  </td>
                  <td class="whitespace-nowrap">
                    <span class="badge badge-sm {stateBadge(j.state)}">{j.state}</span>
                  </td>
                  <td class="text-xs whitespace-nowrap opacity-60">
                    {measured(j.result) ? loss(j, j.result.best_val) : "—"}
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

{#if looking}
  <div class="modal modal-open" role="dialog">
    <div class="modal-box max-w-3xl">
      <img class="rounded-box w-full" src={looking.url} alt={looking.text ?? "a sample"} />
      <p class="mt-2 text-xs opacity-70">
        {looking.text ?? ""} · after {looking.step.toLocaleString()} step{looking.step === 1 ? "" : "s"}
      </p>
    </div>
    <button class="modal-backdrop" aria-label="Close" onclick={() => (looking = null)}></button>
  </div>
{/if}
