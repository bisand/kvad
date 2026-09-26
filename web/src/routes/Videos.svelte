<script>
  // Text to video: a prompt, a size and a length, and every video kept.
  import { videos, SIZES, live } from "../lib/videos.svelte.js";
  import { models } from "../lib/models.svelte.js";
  import { navigate } from "../lib/router.svelte.js";
  import Icon from "../lib/components/Icon.svelte";

  const WARN = "M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z";
  const TRASH = "M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6";
  const REUSE = "M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5";
  const DOWNLOAD = "M12 3v12M7 10l5 5 5-5M5 21h14";

  $effect(() => {
    models.refresh();
    return videos.watch();
  });

  const v = $derived(videos);
  const d = $derived(videos.defaults);
  /** What the right-hand panel shows: the oldest video still being made,
   *  since that is the one the engine is on, or else the newest done. */
  const current = $derived(
    v.running.at(-1) ?? v.gallery.find((g) => g.status === "completed") ?? null,
  );

  /** `stage 2 · 61% · about 40 s left`, from what the server last said. */
  function where(g) {
    const k = g.kvad;
    if (g.status === "queued") return "Queued — waiting for the engine";
    const parts = [k.phase ?? "starting", `${Math.floor(k.progress * 100)}%`];
    // From the share done so far: rough early on, and not shown until
    // there is something to go on.
    if (k.started_at && k.progress > 0.05) {
      const spent = Date.now() / 1000 - k.started_at;
      parts.push(`about ${Math.round((spent * (1 - k.progress)) / k.progress)} s left`);
    }
    return parts.join(" · ");
  }

  function took(g) {
    const k = g.kvad;
    return (k.encode_secs + k.denoise_secs + k.decode_secs).toFixed(0);
  }

  function pickSize(event) {
    const s = SIZES[Number(event.currentTarget.value)];
    if (!s) return;
    v.width = s.width;
    v.height = s.height;
  }
</script>

<div class="flex flex-col gap-6">
  {#if models.listing && v.choices.length === 0}
    <div role="alert" class="alert">
      <Icon path={WARN} />
      <span>
        There is no video model on this machine. Pull one on the
        <a href="/models" class="link" onclick={(e) => navigate(e, "/models")}>Models page</a> —
        <code>Lightricks/LTX-2.5</code> reads about 66 GB of its repo.
      </span>
    </div>
  {/if}

  <div class="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
    <!-- What to make. -->
    <section class="flex flex-col gap-3">
      <label class="flex flex-col gap-1">
        <span class="text-sm opacity-70">Model</span>
        <select class="select w-full" bind:value={() => v.chosen?.id, (id) => (v.model = id)} disabled={v.starting}>
          {#each v.choices as c (c.id)}
            <option value={c.id}>{c.label}</option>
          {/each}
        </select>
      </label>

      <label class="flex flex-col gap-1">
        <span class="text-sm opacity-70">Prompt</span>
        <textarea
          class="textarea h-28 w-full"
          bind:value={v.prompt}
          placeholder="What happens, what it looks like, and what it sounds like…"
        ></textarea>
      </label>

      <div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <label class="flex flex-col gap-1">
          <span class="text-sm opacity-70">Width</span>
          <input class="input" type="number" step={d?.multiple ?? 64} min="64" bind:value={v.width} placeholder={d?.width ?? "default"} />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-sm opacity-70">Height</span>
          <input class="input" type="number" step={d?.multiple ?? 64} min="64" bind:value={v.height} placeholder={d?.height ?? "default"} />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-sm opacity-70">Seconds</span>
          <input
            class="input"
            type="number"
            step="0.5"
            min="0.5"
            bind:value={v.seconds}
            placeholder={d ? (d.frames / d.fps).toFixed(1) : "default"}
          />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-sm opacity-70">Frames a second</span>
          <input class="input" type="number" min="1" max="120" bind:value={v.fps} placeholder={d?.fps ?? "default"} />
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
          <input type="checkbox" class="checkbox checkbox-sm" bind:checked={v.sound} />
          Sound
        </label>
        <label class="label text-sm">
          <input type="checkbox" class="checkbox checkbox-sm" bind:checked={v.fixSeed} />
          Fixed seed
        </label>
        <input class="input input-sm w-36" type="number" min="0" bind:value={v.seed} placeholder="random" disabled={!v.fixSeed} />
      </div>
      {#if v.tooBig}
        <p class="text-warning text-sm">
          That is more than this model makes here: at most {d.max_frames} frames, and
          {(d.max_volume / 1e6).toFixed(0)} million pixels × frames, which is what fits in this
          machine's memory and has been measured.
        </p>
      {/if}
      <p class="text-xs opacity-60">
        Blank fields are the model's own defaults{#if !d}, which it reports once it is loaded{/if}.
        A length becomes the nearest number of frames the model can make{#if v.frames()}:
          {v.frames()} frames{/if}. The same seed and settings make the same video.
      </p>
      <p class="text-xs opacity-60">
        A video holds the GPU for its whole length, and everything else asked of this model waits
        behind it: on an M5 Pro, about a minute and a quarter for 5 s at 768×512, and six minutes
        at 1536×1024. It is the server's job once it is asked for; closing this page does not stop it.
      </p>

      <div class="flex items-center gap-2">
        <button
          class="btn btn-primary"
          onclick={() => v.make()}
          disabled={v.starting || !v.chosen || !v.prompt.trim() || v.tooBig}
        >
          {#if v.starting}<span class="loading loading-spinner loading-xs"></span>{/if}
          Generate
        </button>
        {#if v.starting && models.busy}
          <span class="text-sm opacity-70">Loading {v.chosen?.id} — {models.busy.message}</span>
        {/if}
      </div>
    </section>

    <!-- What is being made, or what was made last. -->
    <section class="bg-base-200 rounded-box flex min-h-80 flex-col items-center justify-center gap-3 p-4">
      {#if current && live(current)}
        {#if current.status === "queued"}
          <span class="loading loading-spinner"></span>
        {/if}
        <progress class="progress w-full max-w-lg" value={current.kvad.progress} max="1"></progress>
        <p class="text-sm opacity-70">{where(current)}</p>
        <p class="line-clamp-2 max-w-lg text-center text-xs opacity-60">{current.prompt}</p>
        <p class="text-xs opacity-50">
          The text path encodes the prompt, the DiT makes the clip at half size and again at full
          size, and the decoders make the frames and the sound. Progress is weighted by what each
          part takes.
        </p>
        <button class="btn btn-sm" onclick={() => v.remove(current)}>Stop and delete</button>
      {:else if current}
        {#key current.id}
          <!-- svelte-ignore a11y_media_has_caption -->
          <video
            src={current.kvad.url}
            poster={current.kvad.thumbnail_url}
            controls
            preload="metadata"
            class="w-full max-w-2xl rounded"
          ></video>
        {/key}
        <p class="text-xs opacity-70">
          {current.size} · {current.seconds} s at {current.kvad.fps} fps · seed {current.kvad.seed}
          · made in {took(current)} s: text {current.kvad.encode_secs.toFixed(1)} s, denoise
          {current.kvad.denoise_secs.toFixed(1)} s, decode {current.kvad.decode_secs.toFixed(1)} s
        </p>
      {:else}
        <p class="text-sm opacity-60">Videos appear here as they are made.</p>
      {/if}
    </section>
  </div>

  <!-- Everything kept. -->
  <section class="flex flex-col gap-3">
    <h2 class="text-lg font-semibold">Gallery</h2>
    {#if v.gallery.length === 0}
      <p class="text-sm opacity-60">Nothing yet. Every video made here, or by any client on this account, is kept.</p>
    {:else}
      <div class="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {#each v.gallery as g (g.id)}
          <figure class="card bg-base-200 overflow-hidden">
            {#if g.status === "completed"}
              <!-- svelte-ignore a11y_media_has_caption -->
              <video src={g.kvad.url} poster={g.kvad.thumbnail_url} controls preload="none" class="aspect-video w-full bg-black object-contain"></video>
            {:else if live(g)}
              <div class="flex aspect-video w-full flex-col items-center justify-center gap-2 p-4">
                <progress class="progress w-3/4" value={g.kvad.progress} max="1"></progress>
                <p class="text-xs opacity-70">{where(g)}</p>
              </div>
            {:else}
              <div class="flex aspect-video w-full items-center justify-center p-4">
                <p class="text-error text-xs">{g.error?.message ?? g.status}</p>
              </div>
            {/if}
            <figcaption class="flex flex-col gap-1 p-3">
              <p class="line-clamp-2 text-sm" title={g.prompt}>{g.prompt}</p>
              <p class="text-xs opacity-60">
                {g.size} · {g.seconds} s · seed {g.kvad.seed}{#if !g.kvad.audio} · no sound{/if}
              </p>
              <div class="flex gap-1">
                <button class="btn btn-ghost btn-xs" onclick={() => v.reuse(g)} title="Put these settings back in the form">
                  <Icon path={REUSE} size={14} /> Reuse
                </button>
                {#if g.status === "completed"}
                  <a class="btn btn-ghost btn-xs" href={g.kvad.url} download={`kvad-${g.id}.mp4`} title="Download the MP4">
                    <Icon path={DOWNLOAD} size={14} />
                  </a>
                {/if}
                <span class="grow"></span>
                <button class="btn btn-ghost btn-xs" onclick={() => v.remove(g)} title={live(g) ? "Stop and delete" : "Delete"}>
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
