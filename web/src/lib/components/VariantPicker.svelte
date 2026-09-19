<script>
  // Which models, at which precisions, a comparison is between.
  //
  // A variant is a model and a backend, and the whole page's meaning rests on
  // being able to pick the *same* model twice at two precisions — so the
  // control is "add a row", not a multi-select over models.
  import { models } from "../models.svelte.js";
  import Icon from "./Icon.svelte";

  let { value = $bindable([]), max = 4 } = $props();

  const available = $derived(models.all.filter((m) => m.runnable));
  const backends = $derived(models.listing?.backends ?? []);

  function add() {
    if (value.length >= max) return;
    const model = available[0]?.id ?? "";
    const backend = backends[0]?.id ?? "cpu-q8";
    value = [...value, { model, backend }];
  }

  function remove(i) {
    value = value.filter((_, at) => at !== i);
  }

  // The same model at the same precision twice is a comparison of nothing,
  // and the server refuses it. Say so here rather than on submit.
  const duplicate = $derived.by(() => {
    const seen = new Set();
    for (const v of value) {
      const key = `${v.model}@${v.backend}`;
      if (seen.has(key)) return key;
      seen.add(key);
    }
    return null;
  });
</script>

<div class="flex flex-col gap-2">
  {#each value as v, i (i)}
    <div class="flex flex-wrap items-center gap-2">
      <select class="select select-sm min-w-48 grow" bind:value={v.model}>
        {#each available as m (m.id)}
          <option value={m.id}>{m.id}</option>
        {/each}
      </select>
      <select class="select select-sm w-32" bind:value={v.backend}>
        {#each backends as b (b.id)}
          <option value={b.id}>{b.label}</option>
        {/each}
      </select>
      <button class="btn btn-ghost btn-sm btn-square" onclick={() => remove(i)} aria-label="Remove">
        <Icon path="M6 6l12 12M18 6L6 18" size={16} />
      </button>
    </div>
  {:else}
    <p class="text-sm opacity-60">Nothing to compare yet.</p>
  {/each}

  <div class="flex items-center gap-3">
    <button class="btn btn-sm" onclick={add} disabled={value.length >= max || !available.length}>
      <Icon path="M12 5v14M5 12h14" size={16} />
      Add a model
    </button>
    {#if duplicate}
      <span class="text-warning text-xs">{duplicate} is in this twice.</span>
    {/if}
  </div>

  {#if !available.length}
    <p class="text-xs opacity-60">
      No runnable models on this machine yet — download or train one first.
    </p>
  {/if}
</div>
