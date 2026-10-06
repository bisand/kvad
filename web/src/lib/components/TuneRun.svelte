<script>
  // A LoRA run being followed, or read back: its validation loss, what it
  // drew of each prompt at each measurement, and how it ended.
  import { training, humanSecs } from "../training.svelte.js";
  import { images } from "../images.svelte.js";
  import { router } from "../router.svelte.js";
  import LossChart from "./LossChart.svelte";

  let { open, stateBadge } = $props();

  const job = $derived(open.job);
  const p = $derived(job.params);
  const result = $derived(job.result);
  const live = $derived(job.state === "running" || job.state === "queued");

  /** The steps a picture was drawn at, in order: the columns of the grid. */
  const steps = $derived([...new Set(open.pictures.map((x) => x.step))].sort((a, b) => a - b));
  const drawn = (prompt, step) => open.pictures.find((x) => x.prompt === prompt && x.step === step);
  /** The model's own loss, before the first step: what a LoRA has to beat. */
  const base = $derived(open.metrics.find((m) => m.step === 0)?.val_loss ?? result?.base_val ?? null);
  const against = (loss) => (base ? `${(((loss - base) / base) * 100).toFixed(1)}%` : "");

  /** Go and draw with it: the model it is for, the LoRA, a prompt it knows. */
  function draw() {
    images.model = result.model;
    images.loras = [{ name: result.lora, scale: 1 }];
    if (p.samples?.[0]) images.prompt = p.samples[0];
    router.go("/images");
  }
</script>

<section class="card bg-base-100 border-base-300 border">
  <div class="card-body gap-3 p-4">
    <div class="flex flex-wrap items-center gap-2">
      <h2 class="font-medium">{job.label}</h2>
      <span class="badge badge-sm badge-soft">LoRA</span>
      <span class="badge badge-sm {stateBadge(job.state)}">{job.state}</span>
      <span class="text-xs opacity-60">
        {p.model} · {p.size}×{p.size} · {p.steps.toLocaleString()} steps · {p.pictures} pictures
        from {p.dataset_name}
      </span>
      <span class="grow"></span>
      {#if job.state === "running"}
        <button class="btn btn-sm" onclick={() => training.cancel(job.id)}>Stop</button>
      {/if}
      {#if result?.lora}
        <button class="btn btn-sm" onclick={draw}>Draw with it</button>
      {/if}
    </div>

    {#if live}
      <progress class="progress w-full" value={open.progress?.done ?? 0} max={open.progress?.total ?? p.steps}></progress>
      <p class="text-xs opacity-60">
        {#if open.progress}
          step {open.progress.done.toLocaleString()} of {open.progress.total.toLocaleString()}
        {:else}
          Reading the pictures and loading the model; the first step follows.
        {/if}
        {#if training.log.length}· {training.log.at(-1)}{/if}
      </p>
    {/if}

    {#if open.metrics.length}
      <LossChart metrics={open.metrics} bestStep={training.bestStep} steps={p.steps} validationOnly />
      {#key open.metrics.length}
        <p class="text-xs opacity-60">
          Validation loss {open.metrics.at(-1).val_loss.toFixed(4)} at step
          {open.metrics.at(-1).step.toLocaleString()}{#if base && open.metrics.at(-1).step > 0}, {against(open.metrics.at(-1).val_loss)}
            against the model's own {base.toFixed(4)}{/if}
          · {humanSecs(open.metrics.at(-1).elapsed_secs)} elapsed
          {#if training.bestStep !== null}· best at step {training.bestStep.toLocaleString()}{/if}
        </p>
      {/key}
      <p class="text-xs opacity-50">
        The loss on pictures held out of training, at the same noise levels with the same noise
        each time. The training loss is not drawn: it says which noise levels the last steps
        happened to draw, and little else.
      </p>
    {/if}

    {#if job.error}
      <div role="alert" class="alert alert-error text-sm whitespace-pre-wrap">{job.error}</div>
    {/if}

    {#if result}
      <p class="text-xs opacity-70">
        {#if !result.measured}
          Stopped before a step was kept, so no LoRA was written ·
        {:else if result.improved}
          Kept step {result.best_step.toLocaleString()}: validation loss
          {result.best_val.toFixed(4)}, {against(result.best_val)} against the model's own ·
        {:else}
          Kept step {result.best_step.toLocaleString()}, at {result.best_val.toFixed(4)}: no better
          than the model without the LoRA ({result.base_val.toFixed(4)}), so on pictures it did
          not train on it learned nothing that carries over ·
        {/if}
        {(result.trained / 1e6).toFixed(1)} M numbers on {result.layers} layers ·
        {result.held_out} of {result.pictures + result.held_out} pictures held out ·
        {humanSecs(result.elapsed_secs)}
      </p>
      {#if result.lora}
        <p class="text-xs opacity-50"><code>{result.lora}</code></p>
      {/if}
    {/if}

    {#if training.log.length}
      <details class="text-xs">
        <summary class="cursor-pointer opacity-60">Log</summary>
        <pre class="bg-base-200 rounded-box mt-2 max-h-40 overflow-auto p-2 text-xs">{training.log.join("\n")}</pre>
      </details>
    {/if}
  </div>
</section>

{#if steps.length}
  <section>
    <h2 class="mb-2 text-sm font-medium opacity-60">What it draws</h2>
    <div class="flex flex-col gap-4">
      {#each p.samples as prompt, i (i)}
        <div>
          <p class="mb-1 text-xs opacity-70">{prompt}</p>
          <div class="flex gap-2 overflow-x-auto pb-1">
            {#each steps as step (step)}
              {@const pic = drawn(i, step)}
              <figure class="w-40 shrink-0">
                {#if pic}
                  <a href={`/api/jobs/${job.id}/pictures/${pic.file}`} target="_blank" rel="noreferrer">
                    <img
                      class="rounded-field aspect-square w-full object-cover"
                      src={`/api/jobs/${job.id}/pictures/${pic.file}`}
                      alt={`${prompt}, after ${step} steps`}
                      loading="lazy"
                    />
                  </a>
                {:else}
                  <div class="bg-base-200 rounded-field aspect-square w-full"></div>
                {/if}
                <figcaption class="mt-1 text-center text-xs opacity-50">
                  {step === 0 ? "the model's own" : `step ${step.toLocaleString()}`}{step === training.bestStep ? " · kept" : ""}
                </figcaption>
              </figure>
            {/each}
          </div>
        </div>
      {/each}
    </div>
  </section>
{/if}
