<script>
  // The text runs are trained on, and the one question worth asking about it
  // before a run starts: which characters a model has no token for.
  import { training } from "../lib/training.svelte.js";
  import { toasts } from "../lib/toasts.svelte.js";
  import { humanBytes } from "../lib/models.svelte.js";
  import Icon from "../lib/components/Icon.svelte";

  const TRASH =
    "M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6";

  let name = $state("");
  let text = $state("");
  let uploading = $state(false);
  let confirming = $state(null);
  let against = $state("");
  let checks = $state({});

  $effect(() => {
    training.refresh();
  });

  async function pick(event) {
    const file = event.currentTarget.files?.[0];
    if (!file) return;
    text = await file.text();
    // A sensible default name, which can still be typed over.
    if (!name.trim()) name = file.name.replace(/\.[^.]+$/, "");
  }

  async function upload(event) {
    event.preventDefault();
    if (!text) return toasts.warning("Choose a file, or paste some text.");
    uploading = true;
    if (await training.upload(name.trim(), text)) {
      name = "";
      text = "";
      event.currentTarget.reset();
    }
    uploading = false;
  }

  /** Ask what each dataset would cost the chosen model. */
  async function checkAll(model) {
    checks = {};
    if (!model) return;
    for (const d of training.datasets) {
      try {
        checks[d.id] = await training.check(d.id, model);
      } catch {
        // A dataset whose file has gone cannot be checked; the row already
        // says so.
      }
    }
  }

  const counted = $derived(text ? [...text].length : 0);
  const distinct = $derived(text ? new Set(text).size : 0);
</script>

<div class="mx-auto flex max-w-4xl flex-col gap-6">
  <section class="card bg-base-100 border-base-300 border">
    <form class="card-body gap-3 p-4" onsubmit={upload}>
      <h2 class="text-sm font-medium opacity-60">Add text</h2>
      <div class="flex flex-wrap items-end gap-2">
        <fieldset class="fieldset">
          <legend class="fieldset-legend">File</legend>
          <input type="file" class="file-input file-input-sm" accept=".txt,text/*" onchange={pick} />
        </fieldset>
        <fieldset class="fieldset grow">
          <legend class="fieldset-legend">Name</legend>
          <input class="input input-sm w-full" bind:value={name} placeholder="shakespeare" required />
        </fieldset>
        <button class="btn btn-sm" disabled={uploading || !text}>
          {#if uploading}<span class="loading loading-spinner loading-xs"></span>{/if}
          Upload
        </button>
      </div>
      {#if text}
        <p class="text-xs opacity-60">
          {counted.toLocaleString()} characters, {distinct} distinct — a model trained on this
          would have a vocabulary of {distinct}.
        </p>
      {/if}
      <p class="text-xs opacity-60">
        Plain text. A character tokeniser gives an id to every distinct character, so the
        second number above is the size of the model's embedding table and output head.
      </p>
    </form>
  </section>

  <section>
    <div class="mb-2 flex flex-wrap items-center gap-2">
      <h2 class="text-sm font-medium opacity-60">Datasets</h2>
      <span class="grow"></span>
      <!-- The check that saves an hour: `kvad train --from` refuses a text
           containing a character the model has never seen, but only after it
           has been chosen and started. -->
      <label class="text-xs opacity-60" for="against">Check against</label>
      <select
        id="against"
        class="select select-xs w-48"
        bind:value={against}
        onchange={(e) => checkAll(e.currentTarget.value)}
      >
        <option value="">no model</option>
        {#each training.options?.continuable ?? [] as m (m)}
          <option value={m}>{m}</option>
        {/each}
      </select>
    </div>

    {#if training.datasets.length === 0}
      <div class="bg-base-200 rounded-box p-6 text-center text-sm opacity-70">
        Nothing yet. <code>scripts/get-text.sh</code> in this repository fetches a corpus to
        start with.
      </div>
    {:else}
      <table class="table">
        <thead>
          <tr>
            <th class="w-full">Name</th>
            <th class="whitespace-nowrap">Characters</th>
            <th class="whitespace-nowrap">Distinct</th>
            <th class="whitespace-nowrap">Size</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {#each training.datasets as d (d.id)}
            <tr class="hover:bg-base-200/50">
              <td class="max-w-0">
                <div class="truncate font-medium">{d.name}</div>
                {#if !d.present}
                  <div class="text-error text-xs">the file is gone from disk</div>
                {:else if checks[d.id]}
                  {#if checks[d.id].unseen_count === 0}
                    <div class="text-xs opacity-60">
                      every character is one <code>{against}</code> knows
                    </div>
                  {:else}
                    <div class="text-error text-xs">
                      {checks[d.id].unseen_count} character{checks[d.id].unseen_count === 1
                        ? ""
                        : "s"}
                      <code>{against}</code> has no token for:
                      <code>{checks[d.id].unseen}</code>
                    </div>
                  {/if}
                {/if}
              </td>
              <td class="text-sm whitespace-nowrap opacity-70">{d.characters.toLocaleString()}</td>
              <td class="text-sm whitespace-nowrap opacity-70">{d.distinct}</td>
              <td class="text-sm whitespace-nowrap opacity-70">{humanBytes(d.bytes)}</td>
              <td class="text-right">
                <button
                  class="btn btn-sm btn-ghost"
                  aria-label={`Delete ${d.name}`}
                  onclick={() => (confirming = d)}
                >
                  <Icon path={TRASH} size={16} />
                </button>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </section>
</div>

{#if confirming}
  <div class="modal modal-open" role="dialog">
    <div class="modal-box">
      <h3 class="text-lg font-medium">Delete {confirming.name}?</h3>
      <p class="py-3 text-sm opacity-70">
        The text file goes. Models already trained on it are untouched — a model carries
        the tokeniser it was trained with and does not read this file again.
      </p>
      <div class="modal-action">
        <button class="btn btn-sm" onclick={() => (confirming = null)}>Cancel</button>
        <button
          class="btn btn-sm btn-error"
          onclick={async () => {
            const d = confirming;
            confirming = null;
            await training.removeDataset(d.id);
          }}
        >
          Delete
        </button>
      </div>
    </div>
    <button class="modal-backdrop" aria-label="Cancel" onclick={() => (confirming = null)}></button>
  </div>
{/if}
