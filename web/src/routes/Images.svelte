<script>
  // Text to image: a prompt, the knobs a denoiser has, and every picture kept.
  import { images, SIZES } from "../lib/images.svelte.js";
  import { models } from "../lib/models.svelte.js";
  import { navigate } from "../lib/router.svelte.js";
  import Icon from "../lib/components/Icon.svelte";

  const WARN = "M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z";
  const TRASH = "M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6";
  const REUSE = "M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5";
  const DOWNLOAD = "M12 3v12M7 10l5 5 5-5M5 21h14";

  $effect(() => {
    models.refresh();
    images.refresh();
  });

  const im = $derived(images);
  const d = $derived(images.defaults);
  const p = $derived(images.progress);
  const elapsed = $derived(p && p.step ? (Date.now() - p.started) / 1000 : 0);

  let open = $state(null);

  function pickSize(event) {
    const s = SIZES[Number(event.currentTarget.value)];
    if (!s) return;
    im.width = s.width;
    im.height = s.height;
  }
</script>

<div class="flex flex-col gap-6">
  {#if models.listing && im.choices.length === 0}
    <div role="alert" class="alert">
      <Icon path={WARN} />
      <span>
        There is no image model on this machine. Pull one on the
        <a href="/models" class="link" onclick={(e) => navigate(e, "/models")}>Models page</a> —
        <code>stabilityai/stable-diffusion-xl-base-1.0</code> is about 7 GB.
      </span>
    </div>
  {/if}

  <div class="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
    <!-- What to make. -->
    <section class="flex flex-col gap-3">
      <label class="flex flex-col gap-1">
        <span class="text-sm opacity-70">Model</span>
        <select class="select w-full" bind:value={() => im.chosen?.id, (id) => (im.model = id)} disabled={im.running}>
          {#each im.choices as c (c.id)}
            <option value={c.id}>{c.label}</option>
          {/each}
        </select>
      </label>

      <label class="flex flex-col gap-1">
        <span class="text-sm opacity-70">Prompt</span>
        <textarea class="textarea h-28 w-full" bind:value={im.prompt} placeholder="What to draw…"></textarea>
      </label>

      <label class="flex flex-col gap-1">
        <span class="text-sm opacity-70">Negative prompt</span>
        <input class="input w-full" bind:value={im.negative} placeholder="What to steer away from (optional)" />
      </label>

      <div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <label class="flex flex-col gap-1">
          <span class="text-sm opacity-70">Width</span>
          <input class="input" type="number" step="8" min="256" max="2048" bind:value={im.width} placeholder={d?.width ?? "default"} />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-sm opacity-70">Height</span>
          <input class="input" type="number" step="8" min="256" max="2048" bind:value={im.height} placeholder={d?.height ?? "default"} />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-sm opacity-70">Steps</span>
          <input class="input" type="number" min="1" max="200" bind:value={im.steps} placeholder={d?.steps ?? "default"} />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-sm opacity-70">Guidance</span>
          <input class="input" type="number" step="0.5" min="0" max="30" bind:value={im.guidance} placeholder={d?.guidance ?? "default"} />
        </label>
      </div>

      <div class="flex flex-wrap items-center gap-3">
        <select class="select select-sm w-auto" aria-label="Size preset" onchange={pickSize}>
          <option value="">Size preset…</option>
          {#each SIZES as s, i (s.label)}
            <option value={i}>{s.label} · {s.width}×{s.height}</option>
          {/each}
        </select>
        <label class="label text-sm">
          <input type="checkbox" class="checkbox checkbox-sm" bind:checked={im.fixSeed} />
          Fixed seed
        </label>
        <input class="input input-sm w-36" type="number" min="0" bind:value={im.seed} placeholder="random" disabled={!im.fixSeed} />
      </div>
      <p class="text-xs opacity-60">
        Blank fields are the model's own defaults{#if !d}, which it reports once it is loaded{/if}.
        Guidance is how hard each step is pushed towards the prompt and away from the
        negative one; every step runs the model twice while it is above 1. The same seed
        and settings make the same picture.
      </p>

      <div class="flex items-center gap-2">
        <button class="btn btn-primary" onclick={() => im.make()} disabled={im.running || !im.chosen || !im.prompt.trim()}>
          {#if im.running}<span class="loading loading-spinner loading-xs"></span>{/if}
          Generate
        </button>
        {#if im.running}
          <button class="btn" onclick={() => im.stop()}>Stop</button>
        {/if}
      </div>
    </section>

    <!-- What is being made, or what was made last. -->
    <section class="bg-base-200 rounded-box flex min-h-80 flex-col items-center justify-center gap-3 p-4">
      {#if p?.loading}
        <span class="loading loading-spinner"></span>
        <p class="text-sm opacity-70">
          Loading {im.chosen?.id}{#if models.busy} — {models.busy.message}{/if}
        </p>
      {:else if p}
        {#if p.preview}
          <img src={p.preview} alt="A rough preview of the picture so far" class="aspect-auto w-full max-w-lg rounded" />
        {:else}
          <span class="loading loading-spinner"></span>
        {/if}
        <progress class="progress w-full max-w-lg" value={p.step} max={p.total || 1}></progress>
        <p class="text-sm opacity-70">
          {#if p.step === 0}
            Waiting for the engine — the prompt is being encoded, or another request is ahead of this one.
          {:else}
            Step {p.step} of {p.total} · {(elapsed / p.step).toFixed(1)} s a step
          {/if}
        </p>
        <p class="text-xs opacity-50">
          The preview is the latent mixed straight into colour, at an eighth of the size: blurry
          and slightly off, and free. The real picture is decoded at the end.
        </p>
      {:else if im.last}
        <img src={im.last.url} alt={im.prompt} class="w-full max-w-2xl rounded" />
        <p class="text-xs opacity-70">
          {im.last.width}×{im.last.height} · {im.last.steps} steps · guidance {im.last.guidance} ·
          seed {im.last.seed} · denoised in {im.last.denoise_secs.toFixed(1)} s
          ({(im.last.denoise_secs / im.last.steps).toFixed(2)} s a step), decoded in
          {im.last.decode_secs.toFixed(1)} s
        </p>
      {:else}
        <p class="text-sm opacity-60">Pictures appear here as they are made.</p>
      {/if}
    </section>
  </div>

  <!-- Everything kept. -->
  <section class="flex flex-col gap-3">
    <h2 class="text-lg font-semibold">Gallery</h2>
    {#if im.gallery.length === 0}
      <p class="text-sm opacity-60">Nothing yet. Every picture made here, or by any client on this account, is kept.</p>
    {:else}
      <div class="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
        {#each im.gallery as g (g.id)}
          <figure class="card bg-base-200 overflow-hidden">
            <button type="button" onclick={() => (open = g)} aria-label="Show larger">
              <img src={g.url} alt={g.prompt} loading="lazy" class="aspect-square w-full object-cover" />
            </button>
            <figcaption class="flex flex-col gap-1 p-3">
              <p class="line-clamp-2 text-sm" title={g.prompt}>{g.prompt}</p>
              <p class="text-xs opacity-60">{g.width}×{g.height} · {g.steps} steps · seed {g.seed}</p>
              <div class="flex gap-1">
                <button class="btn btn-ghost btn-xs" onclick={() => im.reuse(g)} title="Put these settings back in the form">
                  <Icon path={REUSE} size={14} /> Reuse
                </button>
                <a class="btn btn-ghost btn-xs" href={g.url} download={`kvad-${g.id}.png`} title="Download the PNG">
                  <Icon path={DOWNLOAD} size={14} />
                </a>
                <span class="grow"></span>
                <button class="btn btn-ghost btn-xs" onclick={() => im.remove(g.id)} title="Delete">
                  <Icon path={TRASH} size={14} />
                </button>
              </div>
            </figcaption>
          </figure>
        {/each}
      </div>
    {/if}
  </section>
</div>

{#if open}
  <dialog class="modal modal-open" onclose={() => (open = null)}>
    <div class="modal-box max-w-5xl">
      <img src={open.url} alt={open.prompt} class="w-full rounded" />
      <p class="mt-3 text-sm">{open.prompt}</p>
      {#if open.negative_prompt}<p class="text-xs opacity-60">not: {open.negative_prompt}</p>{/if}
      <p class="text-xs opacity-60">
        {open.model} · {open.backend} · {open.width}×{open.height} · {open.steps} steps · guidance
        {open.guidance} · seed {open.seed} · {open.secs.toFixed(1)} s · {open.created_at}
      </p>
    </div>
    <form method="dialog" class="modal-backdrop">
      <button onclick={() => (open = null)}>close</button>
    </form>
  </dialog>
{/if}
