<script>
  // The model, without the conversation around it.
  import { playground, confidence, percent } from "../lib/playground.svelte.js";
  import { models } from "../lib/models.svelte.js";
  import { bench } from "../lib/bench.svelte.js";
  import VariantPicker from "../lib/components/VariantPicker.svelte";
  import Icon from "../lib/components/Icon.svelte";

  let tab = $state("complete");
  let inspecting = $state("The cat sat on the mat, and it was 2026.");
  let selected = $state(null);
  let variants = $state([]);
  let comparePrompt = $state("The history of the transformer architecture begins with");

  $effect(() => {
    models.refresh();
    bench.refresh();
  });

  const loaded = $derived(models.loaded);
  const p = $derived(playground);

  // The compare view is a one-round benchmark that keeps what each variant
  // wrote — the same machinery the Benchmarks page uses, asked a different
  // question. See `compare.rs`.
  const comparison = $derived(
    bench.open?.job?.params?.keep_text ? bench.open : null,
  );
  const byVariant = $derived.by(() => {
    const samples = comparison?.samples ?? [];
    return [...new Set(samples.map((s) => s.variant))].map((variant) => ({
      variant,
      sample: samples.find((s) => s.variant === variant),
    }));
  });

  async function compare() {
    if (!variants.length) return;
    await bench.start({
      variants,
      prompt: comparePrompt,
      rounds: 1,
      tokens: 64,
      keep_text: true,
    });
  }

  function tokenColour(t) {
    const chosen = t.top?.find((c) => c.chosen);
    return chosen ? confidence(chosen.prob) : "";
  }

  // A distinct colour per token, so the inspector shows where the cuts are.
  const STRIPE = ["bg-primary/15", "bg-secondary/15", "bg-accent/15", "bg-info/15"];
</script>

<div class="flex flex-col gap-6">
  {#if !loaded}
    <div role="alert" class="alert">
      <Icon path="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
      <span>No model is loaded. Load one from the Models page — the playground talks to
        the model picked on the Chat page, or the one used last.</span>
    </div>
  {/if}

  <div role="tablist" class="tabs tabs-box w-fit">
    <button role="tab" class="tab" class:tab-active={tab === "complete"} onclick={() => (tab = "complete")}>
      Completion
    </button>
    <button role="tab" class="tab" class:tab-active={tab === "tokens"} onclick={() => (tab = "tokens")}>
      Tokeniser
    </button>
    <button role="tab" class="tab" class:tab-active={tab === "compare"} onclick={() => (tab = "compare")}>
      Side by side
    </button>
  </div>

  {#if tab === "complete"}
    <!-- Raw completion, with the choice behind every token. -->
    <div class="grid gap-6 lg:grid-cols-[2fr_1fr]">
      <section class="flex flex-col gap-3">
        <textarea
          class="textarea textarea-bordered h-32 w-full font-mono text-sm"
          bind:value={p.prompt}
          placeholder="Text for the model to continue…"
        ></textarea>

        <div class="flex flex-wrap items-center gap-2">
          <button class="btn btn-primary btn-sm" onclick={() => p.complete()} disabled={p.running || !loaded}>
            {#if p.running}<span class="loading loading-spinner loading-xs"></span>{/if}
            Continue
          </button>
          {#if p.running}
            <button class="btn btn-sm" onclick={() => p.stop()}>Stop</button>
          {/if}
          <span class="grow"></span>
          {#if p.stats}
            <span class="text-xs opacity-70">
              {p.stats.generated_tokens} tokens at {p.stats.decode_per_sec.toFixed(1)}/s ·
              {p.stats.prompt_tokens} in the prompt ·
              {(p.stats.prefill_secs * 1000).toFixed(0)} ms to the first
            </span>
          {/if}
        </div>

        <div class="bg-base-200 rounded-box min-h-40 p-4 font-mono text-sm leading-7 whitespace-pre-wrap">
          <span class="opacity-50">{p.prompt}</span
          >{#each p.tokens as t, i (i)}<button
              type="button"
              class="hover:bg-base-300 rounded {tokenColour(t)}"
              class:bg-base-300={selected === i}
              onclick={() => (selected = selected === i ? null : i)}
              disabled={!t.top}>{t.text}</button
            >{/each}{#if p.running}<span class="animate-pulse">▍</span>{/if}
        </div>

        {#if p.settings.explain > 0}
          <p class="text-xs opacity-60">
            Colour is how sure the model was about the token it picked: green above 60%,
            blue above 25%, amber above 5%, red below. Click any token to see what else
            it was choosing between.
          </p>
        {/if}
      </section>

      <aside class="flex flex-col gap-4">
        <section class="bg-base-200 rounded-box p-4">
          <h2 class="mb-3 text-sm font-medium opacity-60">Sampler</h2>
          <div class="flex flex-col gap-3 text-sm">
            <label class="flex flex-col gap-1">
              <span class="flex justify-between"><span>Temperature</span>
                <span class="font-mono opacity-70">{p.settings.temperature.toFixed(2)}</span></span>
              <input type="range" min="0" max="2" step="0.05" class="range range-xs"
                bind:value={p.settings.temperature} />
            </label>
            <label class="flex flex-col gap-1">
              <span class="flex justify-between"><span>Top-k</span>
                <span class="font-mono opacity-70">{p.settings.top_k || "off"}</span></span>
              <input type="range" min="0" max="200" step="1" class="range range-xs"
                bind:value={p.settings.top_k} />
            </label>
            <label class="flex flex-col gap-1">
              <span class="flex justify-between"><span>Top-p</span>
                <span class="font-mono opacity-70">{p.settings.top_p.toFixed(2)}</span></span>
              <input type="range" min="0" max="1" step="0.01" class="range range-xs"
                bind:value={p.settings.top_p} />
            </label>
            <label class="flex flex-col gap-1">
              <span class="flex justify-between"><span>Tokens</span>
                <span class="font-mono opacity-70">{p.settings.max_tokens}</span></span>
              <input type="range" min="8" max="512" step="8" class="range range-xs"
                bind:value={p.settings.max_tokens} />
            </label>

            <label class="flex cursor-pointer items-center justify-between gap-2">
              <span>Fix the seed</span>
              <input type="checkbox" class="toggle toggle-sm" bind:checked={p.settings.fixSeed} />
            </label>
            {#if p.settings.fixSeed}
              <input type="number" class="input input-sm w-full" bind:value={p.settings.seed} />
            {/if}
            <p class="text-xs opacity-60">
              With the seed fixed, the same prompt gives the same text, so a change in
              the output is a change you made. Without it, every run draws from where
              the last one left off.
            </p>

            <label class="flex flex-col gap-1">
              <span class="flex justify-between"><span>Show candidates</span>
                <span class="font-mono opacity-70">{p.settings.explain || "off"}</span></span>
              <input type="range" min="0" max="20" step="1" class="range range-xs"
                bind:value={p.settings.explain} />
            </label>
          </div>
        </section>

        {#if selected != null && p.tokens[selected]?.top}
          {@const t = p.tokens[selected]}
          <section class="bg-base-200 rounded-box p-4">
            <h2 class="mb-1 text-sm font-medium opacity-60">
              Instead of <code class="bg-base-300 rounded px-1">{t.text}</code>
            </h2>
            <p class="mb-3 text-xs opacity-60">
              The model's own probabilities, over the whole vocabulary. Dimmed rows are
              ones top-k and top-p had already removed, so they could not have been
              picked however the dice fell.
            </p>
            <table class="table table-xs">
              <tbody>
                {#each t.top as c (c.id)}
                  <tr class:opacity-40={!c.kept}>
                    <td class="font-mono">
                      {#if c.chosen}<span class="text-success">▸</span>{/if}
                      <span class="whitespace-pre">{c.text === "" ? "␀" : c.text}</span>
                    </td>
                    <td class="text-right font-mono text-xs opacity-70">{percent(c.prob)}</td>
                    <td class="w-20">
                      <div class="bg-base-300 h-1.5 rounded">
                        <div class="bg-primary h-1.5 rounded" style:width="{Math.max(2, c.prob * 100)}%"></div>
                      </div>
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </section>
        {/if}
      </aside>
    </div>
  {:else if tab === "tokens"}
    <!-- What the tokeniser actually did. -->
    <section class="flex flex-col gap-3">
      <textarea
        class="textarea textarea-bordered h-28 w-full font-mono text-sm"
        bind:value={inspecting}
      ></textarea>
      <div class="flex items-center gap-3">
        <button class="btn btn-primary btn-sm" onclick={() => p.tokenize(inspecting)} disabled={!loaded}>
          Split it
        </button>
        {#if p.split}
          <span class="text-sm opacity-70">
            {p.split.count} tokens from {p.split.characters} characters —
            {(p.split.characters / Math.max(1, p.split.count)).toFixed(1)} characters each
          </span>
        {/if}
      </div>

      {#if p.split}
        <div class="bg-base-200 rounded-box flex flex-wrap gap-0.5 p-4 font-mono text-sm">
          {#each p.split.tokens as t, i (i)}
            <span
              class="rounded px-0.5 {STRIPE[i % STRIPE.length]}"
              title="id {t.id} · {t.token}">{t.piece === "" ? "␀" : t.piece}</span>
          {/each}
        </div>
        <p class="text-xs opacity-60">
          The pieces above are cut from your text; the table below also gives the
          vocabulary entry, which is where byte-level BPE's <code>Ġ</code> for a leading
          space lives. A token is not a word and not a character, and this is where the
          difference stops being abstract: numbers, accents and emoji all cost more than
          they look.
        </p>
        <div class="max-h-96 overflow-y-auto">
          <table class="table table-xs table-pin-rows">
            <thead>
              <tr><th>#</th><th>Id</th><th>Vocabulary entry</th><th class="w-full">Covers</th></tr>
            </thead>
            <tbody>
              {#each p.split.tokens as t, i (i)}
                <tr>
                  <td class="opacity-50">{i}</td>
                  <td class="font-mono">{t.id}</td>
                  <td class="font-mono whitespace-pre">{t.token}</td>
                  <td class="font-mono whitespace-pre">{t.piece}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </section>
  {:else}
    <!-- The same prompt, every variant, one after another. -->
    <section class="flex flex-col gap-4">
      <div class="bg-base-200 rounded-box flex flex-col gap-3 p-4">
        <p class="text-xs opacity-60">
          A model measured beside another is measured on the memory they share, so "side
          by side" happens one after another: each variant is loaded alone — anything
          else in memory is unloaded first — given the same prompt with the same seed,
          and unloaded when the next one's turn comes. Expect it to take as long as the
          loads do.
        </p>
        <textarea class="textarea textarea-bordered h-20 w-full font-mono text-sm"
          bind:value={comparePrompt}></textarea>
        <VariantPicker bind:value={variants} />
        <div class="flex items-center gap-3">
          <button class="btn btn-primary btn-sm" onclick={compare}
            disabled={!variants.length || bench.open?.job?.state === "running"}>
            Run them
          </button>
          {#if bench.open?.job?.state === "running"}
            <button class="btn btn-sm" onclick={() => bench.cancel(bench.open.job.id)}>Stop</button>
            <span class="text-xs opacity-70">
              {bench.open.log.at(-1) ?? "starting"}
            </span>
          {/if}
        </div>
      </div>

      {#if comparison}
        <div class="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {#each byVariant as { variant, sample } (variant)}
            <div class="bg-base-200 rounded-box flex flex-col gap-2 p-4">
              <h3 class="font-mono text-xs opacity-70">{variant}</h3>
              {#if sample}
                <p class="text-xs opacity-60">
                  {sample.decode_per_sec.toFixed(1)} tok/s ·
                  {sample.ttft_millis.toFixed(0)} ms to the first
                </p>
                <p class="font-mono text-sm whitespace-pre-wrap">{sample.text}</p>
              {:else}
                <span class="loading loading-dots loading-sm"></span>
              {/if}
            </div>
          {/each}
        </div>
      {/if}
    </section>
  {/if}
</div>
