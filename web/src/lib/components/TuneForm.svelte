<script>
  // What a LoRA run is asked for: the pictures, the model they adapt, and
  // how long and how large. Everything the server would refuse is said here
  // first where it can be: a picture with no caption, a size that does not
  // fit in what memory is left.
  import { training, humanSecs } from "../training.svelte.js";
  import { navigate } from "../router.svelte.js";
  import { toasts } from "../toasts.svelte.js";

  let { busy = false } = $props();

  let form = $state({
    dataset: null,
    model: null,
    name: "",
    size: null,
    steps: null,
    lr: null,
    eval_every: null,
    rank: null,
    caption: "",
    samples: "",
    sample_size: null,
  });
  let starting = $state(false);

  const o = $derived(training.tune);
  const sets = $derived(training.datasetsOf("pictures"));
  const chosen = $derived(sets.find((d) => d.id === form.dataset) ?? null);

  // The defaults, once the server has said them, and the first of each list.
  $effect(() => {
    if (o && form.size === null) {
      const d = o.defaults;
      form = { ...form, size: d.size, steps: d.steps, lr: d.lr, eval_every: d.eval_every, rank: d.rank, sample_size: d.sample_size };
    }
    if (o && form.model === null && o.models.length) {
      form.model = o.models.includes(o.base) ? o.base : o.models[0];
    }
    if (form.dataset === null && sets.length) form.dataset = sets[0].id;
  });

  const prompts = $derived(
    form.samples
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean),
  );
  const side = $derived(o?.sides.find((s) => s.side === Number(form.size)) ?? null);
  /** Bytes set aside for the run as it is asked for. */
  const takes = $derived(side ? (prompts.length ? side.takes_sampling : side.takes) : 0);
  const fits = $derived(!o || takes <= o.left);
  const gb = (bytes) => `${(bytes / 1e9).toFixed(1)} GB`;

  // Seconds a step and a 20-step sample took on an M5 Pro, by pixels a side
  // (docs/tune.md): enough to tell ten minutes from two hours beforehand.
  const STEP = { 512: 2.1, 768: 4.6, 1024: 8.4 };
  const SAMPLE = { 512: 16, 768: 39, 1024: 72 };
  const estimate = $derived.by(() => {
    const steps = Number(form.steps);
    const every = Number(form.eval_every);
    if (!(steps > 0) || !(every > 0) || !STEP[form.size]) return null;
    const measurements = Math.floor(steps / every) + 1;
    return steps * STEP[form.size] + measurements * prompts.length * (SAMPLE[form.sample_size] ?? 0);
  });

  const needsCaption = $derived(!!chosen?.uncaptioned && !form.caption.trim());
  const ready = $derived(
    !!o && !o.unavailable && !!chosen && chosen.pictures > 0 && !!form.model && !!form.name.trim() && !needsCaption && fits && prompts.length <= (o?.max_samples ?? 4),
  );

  async function start(event) {
    event.preventDefault();
    if (!ready) return toasts.warning("The run is not ready to start; see the notes in the form.");
    starting = true;
    const request = {
      dataset: form.dataset,
      model: form.model,
      name: form.name.trim(),
      size: Number(form.size),
      steps: Number(form.steps),
      lr: Number(form.lr),
      eval_every: Math.min(Number(form.eval_every), Number(form.steps)),
      rank: Number(form.rank),
      samples: prompts,
      sample_size: Math.min(Number(form.sample_size), Number(form.size)),
    };
    if (form.caption.trim()) request.caption = form.caption.trim();
    await training.startTune(request);
    starting = false;
  }
</script>

<form class="flex flex-col gap-3" onsubmit={start}>
  {#if o === null}
    <p class="text-xs opacity-60">This server does not train LoRAs.</p>
  {:else if o.unavailable}
    <div role="alert" class="alert alert-warning text-xs">{o.unavailable}</div>
  {/if}

  <fieldset class="fieldset">
    <legend class="fieldset-legend">Pictures</legend>
    <select class="select select-sm w-full" bind:value={form.dataset}>
      {#each sets as d (d.id)}
        <option value={d.id} disabled={!d.present || d.pictures === 0}>
          {d.name} — {d.pictures} picture{d.pictures === 1 ? "" : "s"}
        </option>
      {:else}
        <option value={null}>none uploaded yet</option>
      {/each}
    </select>
    {#if sets.length === 0}
      <p class="mt-1 text-xs opacity-60">
        <a href="/datasets" onclick={(e) => navigate(e, "/datasets")} class="link">Add pictures</a>
        first: ten to forty of one subject or one style, each with a caption.
      </p>
    {:else if chosen && chosen.pictures < 5}
      <p class="mt-1 text-xs opacity-60">
        With fewer than five, none is held out: the loss is then measured on pictures the run
        trains on, and says how well those are fitted, not how any other would be.
      </p>
    {/if}
  </fieldset>

  {#if chosen?.uncaptioned}
    <fieldset class="fieldset">
      <legend class="fieldset-legend">Caption for the {chosen.uncaptioned} without one</legend>
      <input class="input input-sm w-full {needsCaption ? 'input-warning' : ''}" bind:value={form.caption} placeholder="a photo of sks dog" />
      <p class="mt-1 text-xs opacity-60">
        A picture trained with no caption teaches the model that the subject is what it draws
        when told nothing. Give those one here, or
        <a href="/datasets" onclick={(e) => navigate(e, "/datasets")} class="link">write each its own</a>.
      </p>
    </fieldset>
  {/if}

  <fieldset class="fieldset">
    <legend class="fieldset-legend">For the model</legend>
    <select class="select select-sm w-full" bind:value={form.model}>
      {#each o?.models ?? [] as m (m)}
        <option value={m}>{m}</option>
      {:else}
        <option value={null}>no SDXL model on this machine</option>
      {/each}
    </select>
    {#if o && o.models.length === 0}
      <p class="mt-1 text-xs opacity-60">
        A LoRA is trained for SDXL or a fine-tune of it. Pull <code>{o.base}</code> on the
        <a href="/models" onclick={(e) => navigate(e, "/models")} class="link">Models page</a>.
      </p>
    {/if}
  </fieldset>

  <fieldset class="fieldset">
    <legend class="fieldset-legend">Name</legend>
    <input class="input input-sm w-full" bind:value={form.name} placeholder="my-dog" pattern="[A-Za-z0-9._\-]+" />
    <p class="mt-1 text-xs opacity-60">One word. It is then a LoRA to choose on the Images page.</p>
  </fieldset>

  <div class="grid grid-cols-2 gap-2">
    <fieldset class="fieldset">
      <legend class="fieldset-legend">Size</legend>
      <select class="select select-sm w-full" bind:value={form.size}>
        {#each o?.sides ?? [] as s (s.side)}
          <option value={s.side}>{s.side}×{s.side}</option>
        {/each}
      </select>
    </fieldset>
    <fieldset class="fieldset">
      <legend class="fieldset-legend">Steps</legend>
      <input class="input input-sm w-full" type="number" min="1" bind:value={form.steps} />
    </fieldset>
    <fieldset class="fieldset">
      <legend class="fieldset-legend">Learning rate</legend>
      <input class="input input-sm w-full" type="number" step="0.00001" min="0" bind:value={form.lr} />
    </fieldset>
    <fieldset class="fieldset">
      <legend class="fieldset-legend">Measure every</legend>
      <input class="input input-sm w-full" type="number" min="1" bind:value={form.eval_every} />
    </fieldset>
    <fieldset class="fieldset">
      <legend class="fieldset-legend">Rank</legend>
      <input class="input input-sm w-full" type="number" min="1" max="128" bind:value={form.rank} />
    </fieldset>
    <fieldset class="fieldset">
      <legend class="fieldset-legend">Samples at</legend>
      <select class="select select-sm w-full" bind:value={form.sample_size} disabled={!prompts.length}>
        {#each (o?.sides ?? []).filter((s) => s.side <= Number(form.size)) as s (s.side)}
          <option value={s.side}>{s.side}×{s.side}</option>
        {/each}
      </select>
    </fieldset>
  </div>

  <fieldset class="fieldset">
    <legend class="fieldset-legend">Prompts to draw as it learns</legend>
    <textarea
      class="textarea textarea-sm w-full text-xs {prompts.length > (o?.max_samples ?? 4) ? 'textarea-warning' : ''}"
      rows="2"
      bind:value={form.samples}
      placeholder="one a line, up to {o?.max_samples ?? 4}: a photo of sks dog on a beach"
    ></textarea>
    <p class="mt-1 text-xs opacity-60">
      Each is drawn before the first step and at every measurement, from the same seed, so a
      row of them differs only by what the LoRA has learned.
    </p>
  </fieldset>

  {#if side}
    <p class="text-xs {fits ? 'opacity-60' : 'text-error'}">
      {gb(takes)} is set aside for the run{#if !fits}, and {gb(o.left)} is left: unload a model
        on the <a href="/models" onclick={(e) => navigate(e, "/models")} class="link">Models page</a>,
        or choose a smaller size{/if}.
      {#if estimate}About {humanSecs(estimate)} on an M5 Pro.{/if}
    </p>
  {/if}

  <button class="btn btn-sm btn-primary" disabled={busy || starting || !ready}>
    {#if starting}<span class="loading loading-spinner loading-xs"></span>{/if}
    {busy ? "A run is already going" : "Start"}
  </button>
</form>
