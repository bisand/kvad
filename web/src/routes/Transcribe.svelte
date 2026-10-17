<script>
  // Speech to text: a recording in, what is said in it out, a segment at a
  // time as the Whisper hears it.
  import { transcribe, LANGUAGES, clock } from "../lib/transcribe.svelte.js";
  import { models, humanBytes } from "../lib/models.svelte.js";
  import { navigate } from "../lib/router.svelte.js";
  import Icon from "../lib/components/Icon.svelte";

  const WARN = "M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z";
  const FILE = "M9 18V5l12-2v13M9 18a3 3 0 1 1-6 0 3 3 0 0 1 6 0zM21 16a3 3 0 1 1-6 0 3 3 0 0 1 6 0z";
  const COPY = "M8 8h12v12H8zM4 16V4h12";
  const DOWNLOAD = "M12 3v12M7 10l5 5 5-5M5 21h14";

  $effect(() => {
    models.refresh();
  });

  // A clock for the time taken, which the events only move once a window.
  let now = $state(Date.now());
  $effect(() => {
    const tick = setInterval(() => (now = Date.now()), 500);
    return () => clearInterval(tick);
  });

  const t = $derived(transcribe);
  let dragging = $state(false);

  function dropped(e) {
    e.preventDefault();
    dragging = false;
    if (!t.running) t.choose(e.dataTransfer?.files?.[0]);
  }

  /** `1:02:03`, or `2:03` under an hour: where a segment starts. */
  function at(seconds) {
    const s = Math.floor(seconds);
    const mm = String(Math.floor(s / 60) % 60);
    const ss = String(s % 60).padStart(2, "0");
    return s >= 3600 ? `${Math.floor(s / 3600)}:${mm.padStart(2, "0")}:${ss}` : `${mm}:${ss}`;
  }

  const took = $derived(t.startedAt ? ((t.finishedAt ?? now) - t.startedAt) / 1000 : 0);
  const where = $derived(
    t.phase === "loading"
      ? `Loading ${t.chosen?.id}${models.busy ? ` — ${models.busy.message}` : ""}`
      : t.phase === "sending"
        ? "Sending the recording, and reading its sound"
        : t.total
          ? `${at(t.heard)} of ${at(t.total)} heard · ${took.toFixed(0)} s`
          : `Listening to the first 30 seconds · ${took.toFixed(0)} s`,
  );
  const languageName = (code) => LANGUAGES.find((l) => l.code === code)?.name ?? code;
  const saved = $derived(t.segments.length > 0 && !t.running);
</script>

<div class="flex flex-col gap-6">
  {#if models.listing && t.choices.length === 0}
    <div role="alert" class="alert">
      <Icon path={WARN} />
      <span>
        There is no Whisper on this machine. Pull one on the
        <a href="/models" class="link" onclick={(e) => navigate(e, "/models")}>Models page</a> —
        <code>openai/whisper-tiny</code> is 150 MB, and <code>NbAiLab/nb-whisper-small</code> knows Norwegian.
      </span>
    </div>
  {/if}

  <div class="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
    <!-- What to listen to. -->
    <section class="flex flex-col gap-3">
      <label class="flex flex-col gap-1">
        <span class="text-sm opacity-70">Model</span>
        <select class="select w-full" bind:value={() => t.chosen?.id, (id) => (t.model = id)} disabled={t.running}>
          {#each t.choices as c (c.id)}
            <option value={c.id}>{c.label}</option>
          {/each}
        </select>
      </label>

      <label
        class="rounded-box border-base-content/20 flex cursor-pointer flex-col items-center gap-2 border-2 border-dashed p-6 text-center transition-colors"
        class:bg-base-200={dragging}
        class:border-primary={dragging}
        ondragover={(e) => {
          e.preventDefault();
          dragging = true;
        }}
        ondragleave={() => (dragging = false)}
        ondrop={dropped}
      >
        <Icon path={FILE} size={28} class="opacity-60" />
        {#if t.file}
          <span class="font-medium break-all">{t.file.name}</span>
          <span class="text-xs opacity-60">{humanBytes(t.file.size)} · drop another or click to change it</span>
        {:else}
          <span>Drop a recording here, or click to choose one</span>
          <span class="text-xs opacity-60">WAV, MP3, M4A, FLAC, OGG, or a video's sound — up to 200 MiB</span>
        {/if}
        <input
          type="file"
          accept="audio/*,video/*"
          class="hidden"
          disabled={t.running}
          onchange={(e) => {
            t.choose(e.currentTarget.files?.[0]);
            e.currentTarget.value = "";
          }}
        />
      </label>

      <div class="grid grid-cols-2 gap-3">
        <label class="flex flex-col gap-1">
          <span class="text-sm opacity-70">Language spoken</span>
          <select class="select w-full" bind:value={t.language} disabled={t.running}>
            <option value="">Model's own</option>
            <option value="auto">Detect it</option>
            {#each LANGUAGES as l (l.code)}
              <option value={l.code}>{l.name}</option>
            {/each}
          </select>
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-sm opacity-70">Write it down</span>
          <select class="select w-full" bind:value={t.task} disabled={t.running}>
            <option value="transcribe">As spoken</option>
            <option value="translate">In English</option>
          </select>
        </label>
      </div>

      <label class="flex flex-col gap-1">
        <span class="text-sm opacity-70">Prompt</span>
        <textarea
          class="textarea h-16 w-full"
          bind:value={t.prompt}
          disabled={t.running}
          placeholder="Optional: names and spellings to expect — Grieg, Kygo, Bodø"
        ></textarea>
      </label>

      <label class="label text-sm">
        <input type="checkbox" class="checkbox checkbox-sm" bind:checked={t.timestamps} disabled={t.running} />
        Timestamps
      </label>
      <p class="text-xs opacity-60">
        With timestamps the Whisper times each sentence, and the transcript can be saved as
        subtitles; without, each 30-second window is one piece. The model's own language is the
        one its checkpoint names, or else detected from the first window. The prompt is read as
        what was said just before the recording, so its spellings carry over. Nothing is kept on
        the server.
      </p>

      <div class="flex items-center gap-2">
        <button class="btn btn-primary" onclick={() => t.start()} disabled={t.running || !t.chosen || !t.file}>
          {#if t.running}<span class="loading loading-spinner loading-xs"></span>{/if}
          {t.task === "translate" ? "Translate" : "Transcribe"}
        </button>
        {#if t.running}
          <button class="btn" onclick={() => t.cancel()}>Cancel</button>
        {/if}
      </div>
    </section>

    <!-- What was heard. -->
    <section class="bg-base-200 rounded-box flex min-h-80 flex-col gap-3 p-4">
      {#if t.running}
        <progress class="progress w-full" value={t.total ? t.heard / t.total : undefined} max="1"></progress>
        <p class="text-sm opacity-70">{where}</p>
      {:else if t.error}
        <div role="alert" class="alert alert-error alert-soft">
          <Icon path={WARN} />
          <span>{t.error}</span>
        </div>
      {:else if t.stopped}
        <p class="text-sm opacity-70">
          Cancelled at {at(t.heard)}{#if t.total}{" "}of {at(t.total)}{/if}. What was heard by then is below.
        </p>
      {:else if t.done}
        <p class="text-xs opacity-70">
          {at(t.done.duration)} of {languageName(t.done.language)}{#if t.asked?.task === "translate"}, in English{/if}
          · heard in {took.toFixed(1)} s, {(t.done.duration / Math.max(took, 0.1)).toFixed(0)}× as fast as it plays
        </p>
      {/if}

      {#if t.segments.length}
        <div class="flex max-h-[32rem] flex-col gap-1 overflow-y-auto">
          {#each t.segments as s, i (i)}
            <p class="flex gap-3 text-sm">
              <span class="shrink-0 font-mono text-xs leading-5 opacity-50" title="{clock(s.start, '.')} → {clock(s.end, '.')}">{at(s.start)}</span>
              <span>{s.text.trim()}</span>
            </p>
          {/each}
        </div>
      {:else if !t.running && !t.error}
        <p class="m-auto text-sm opacity-60">What is said appears here, a sentence at a time, as it is heard.</p>
      {/if}

      {#if saved}
        <div class="mt-auto flex flex-wrap items-center gap-1 pt-2">
          <button class="btn btn-sm" onclick={() => t.copy()}>
            <Icon path={COPY} size={14} /> Copy text
          </button>
          <button class="btn btn-sm" onclick={() => t.download("text")} title="The text, as response_format=text writes it">
            <Icon path={DOWNLOAD} size={14} /> .txt
          </button>
          <button
            class="btn btn-sm"
            onclick={() => t.download("srt")}
            disabled={!t.asked?.timestamps}
            title={t.asked?.timestamps ? "Subtitles, SubRip" : "Subtitles need timestamps: turn them on and transcribe again"}
          >
            <Icon path={DOWNLOAD} size={14} /> .srt
          </button>
          <button
            class="btn btn-sm"
            onclick={() => t.download("vtt")}
            disabled={!t.asked?.timestamps}
            title={t.asked?.timestamps ? "Subtitles, WebVTT" : "Subtitles need timestamps: turn them on and transcribe again"}
          >
            <Icon path={DOWNLOAD} size={14} /> .vtt
          </button>
        </div>
      {/if}
    </section>
  </div>
</div>
