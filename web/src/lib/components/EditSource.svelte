<script>
  // The picture an image is made from, where it is made from one: choose
  // it, say how far it may change, and paint where, if only part should.
  //
  // The mask is painted here because the other way to make one is another
  // program. What is painted is kept as strokes, in the picture's own
  // pixels, and drawn twice: on the canvas over the picture, to see, and on
  // one nobody sees at the picture's size, white on black, which is the
  // mask that is sent.
  import Icon from "./Icon.svelte";

  /** `form`: the Images page's store. `edits`: whether the chosen model
   *  makes an image from a picture, or null before it is loaded and says. */
  let { form, edits = null, busy = false } = $props();

  const CLOSE = "M6 6l12 12M18 6 6 18";

  let canvas = $state(null);
  let natural = $state(null);
  /** Each stroke a brush width and its points, all in the picture's pixels. */
  let strokes = $state([]);
  let brush = $state(12);
  let painting = false;

  function pick(event) {
    const file = event.currentTarget.files?.[0];
    if (file) form.startFrom(file);
    event.currentTarget.value = "";
  }

  // A new picture has no mask, and one that came with a mask (a reused
  // edit) keeps it until something is painted over it.
  $effect(() => {
    void form.source?.url;
    strokes = [];
    natural = null;
  });

  function loaded(event) {
    natural = { width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight };
  }

  function draw(ctx, colour, list) {
    ctx.strokeStyle = colour;
    ctx.fillStyle = colour;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const s of list) {
      ctx.lineWidth = s.width;
      ctx.beginPath();
      s.points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      // One point is a dot, which a line of no length does not draw.
      if (s.points.length === 1) ctx.arc(s.points[0][0], s.points[0][1], s.width / 2, 0, Math.PI * 2);
      s.points.length === 1 ? ctx.fill() : ctx.stroke();
    }
  }

  // What is seen: the strokes, over the picture.
  $effect(() => {
    if (!canvas || !natural) return;
    canvas.width = natural.width;
    canvas.height = natural.height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, natural.width, natural.height);
    draw(ctx, "rgba(255, 60, 90, 0.55)", strokes);
  });

  /** The mask that is sent: white where it was painted, black elsewhere. */
  function mask() {
    if (!natural || strokes.length === 0) return null;
    const hidden = document.createElement("canvas");
    hidden.width = natural.width;
    hidden.height = natural.height;
    const ctx = hidden.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, natural.width, natural.height);
    draw(ctx, "#fff", strokes);
    return hidden.toDataURL("image/png");
  }

  function at(event) {
    const box = canvas.getBoundingClientRect();
    return [((event.clientX - box.left) / box.width) * natural.width, ((event.clientY - box.top) / box.height) * natural.height];
  }

  function down(event) {
    if (busy || !natural) return;
    painting = true;
    canvas.setPointerCapture(event.pointerId);
    // The brush is a share of the picture's width, so that it paints as
    // wide on a photograph as it looks on the page.
    strokes = [...strokes, { width: (brush / 100) * natural.width, points: [at(event)] }];
  }

  function move(event) {
    if (!painting) return;
    const last = strokes[strokes.length - 1];
    strokes = [...strokes.slice(0, -1), { ...last, points: [...last.points, at(event)] }];
  }

  function up() {
    if (!painting) return;
    painting = false;
    form.mask = mask();
  }

  function clear() {
    strokes = [];
    form.mask = null;
  }

  /** The strength in the box: what was typed, or the default as its hint. */
  const fallback = $derived(form.mask ? 1 : 0.75);
</script>

<div class="flex flex-col gap-2">
  <div class="flex items-center gap-2">
    <span class="text-sm opacity-70">Start from a picture</span>
    <span class="grow"></span>
    {#if form.source}
      <button class="btn btn-ghost btn-xs" onclick={() => form.clearSource()} disabled={busy}>
        <Icon path={CLOSE} size={14} /> Remove
      </button>
    {/if}
  </div>

  {#if !form.source}
    <input type="file" class="file-input file-input-sm w-full" accept="image/*" onchange={pick} disabled={busy} />
    <p class="text-xs opacity-60">
      Optional. With one, the picture is noised part of the way and drawn over from there: the
      prompt says what it should become.
    </p>
  {:else}
    {#if edits === false}
      <p class="text-warning text-xs">This model draws from a prompt alone. SDXL and SD 1.5 make an image from a picture.</p>
    {/if}
    <div class="relative w-full overflow-hidden rounded">
      <img src={form.source.url} alt={form.source.name} class="block w-full" onload={loaded} />
      <canvas
        bind:this={canvas}
        class="absolute inset-0 size-full touch-none {busy ? '' : 'cursor-crosshair'}"
        aria-label="Paint where the picture may change"
        onpointerdown={down}
        onpointermove={move}
        onpointerup={up}
        onpointercancel={up}
      ></canvas>
      {#if form.mask && strokes.length === 0}
        <!-- A mask that came with a reused edit, shown as it was sent. -->
        <img src={form.mask} alt="The mask" class="pointer-events-none absolute inset-0 size-full opacity-40 mix-blend-screen" />
      {/if}
    </div>
    <div class="flex flex-wrap items-center gap-3">
      <label class="flex items-center gap-2 text-sm">
        <span class="opacity-70">Strength</span>
        <input
          class="input input-sm w-24"
          type="number"
          min="0.05"
          max="1"
          step="0.05"
          bind:value={form.strength}
          placeholder={String(fallback)}
          disabled={busy}
        />
      </label>
      <label class="flex items-center gap-2 text-sm">
        <span class="opacity-70">Brush</span>
        <input class="range range-xs w-24" type="range" min="2" max="40" bind:value={brush} disabled={busy} />
      </label>
      {#if form.mask}
        <button class="btn btn-ghost btn-xs" onclick={clear} disabled={busy}>Clear the mask</button>
      {/if}
    </div>
    <p class="text-xs opacity-60">
      Strength is how far the picture is taken towards noise before it is drawn again: near 0
      it comes back as it was, at 1 nothing of it is left.
      {#if form.mask}
        Only what is painted is drawn anew; the rest comes back as it is.
      {:else}
        Paint on the picture to change only that part of it.
      {/if}
    </p>
  {/if}
</div>
