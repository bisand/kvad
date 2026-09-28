<script>
  // The LoRAs a request applies: add one, choose which, its strength, take
  // it off. For the Images and Videos pages, whose stores keep `loras` and
  // the LoRAs on this machine for the chosen model (`loraChoices`).
  import { models } from "../models.svelte.js";
  import { navigate } from "../router.svelte.js";
  import Icon from "./Icon.svelte";

  /** `form`: the page's store. `takes`: whether the chosen model takes
   *  LoRAs, or null before it is loaded and says. `busy`: a request running. */
  let { form, takes = null, busy = false } = $props();

  const PLUS = "M12 5v14M5 12h14";
  const CLOSE = "M6 6l12 12M18 6 6 18";

  // A LoRA by its file first, which is what tells a repo's several apart,
  // and its repo after: `Qwen-Image-Lightning-8steps-V2.0-bf16 · lightx2v/…`.
  function label(name) {
    const at = name.lastIndexOf(":");
    if (at < 0 || !name.slice(0, at).includes("/")) return name;
    return `${name.slice(at + 1).replace(/\.safetensors$/i, "")} · ${name.slice(0, at)}`;
  }
</script>

<div class="flex flex-col gap-2">
  <div class="flex items-center gap-2">
    <span class="text-sm opacity-70">LoRAs</span>
    <span class="grow"></span>
    <button
      class="btn btn-ghost btn-xs"
      onclick={() => form.addLora()}
      disabled={busy || takes === false || form.loras.length >= Math.min(4, form.loraChoices.length)}
    >
      <Icon path={PLUS} size={14} /> Add
    </button>
  </div>
  {#if takes === false}
    <p class="text-xs opacity-60">This model takes no LoRAs; so far Qwen-Image, FLUX, SDXL, SD 1.5 and LTX-2.5 do.</p>
  {:else if form.loraChoices.length === 0 && models.all.some((m) => m.lora)}
    <p class="text-xs opacity-60">None on this machine is for this model, as far as their layers' names say.</p>
  {:else if form.loraChoices.length === 0}
    <p class="text-xs opacity-60">
      None on this machine. Pull one on the
      <a href="/models" class="link" onclick={(e) => navigate(e, "/models")}>Models page</a>, as
      <code>repo:file.safetensors</code> or a repo whose only LoRA it is.
    </p>
  {/if}
  {#each form.loras as l, i (l.name)}
    <div class="flex items-center gap-2">
      <select
        class="select select-sm min-w-0 grow"
        aria-label="LoRA"
        value={l.name}
        onchange={(e) => form.setLora(i, { name: e.currentTarget.value })}
        disabled={busy}
      >
        {#each form.loraChoices as c (c)}
          <option value={c} disabled={c !== l.name && form.loras.some((o) => o.name === c)}>{label(c)}</option>
        {/each}
      </select>
      <input
        class="input input-sm w-20"
        type="number"
        step="0.1"
        min="-10"
        max="10"
        aria-label="Strength"
        title="Strength: 1 as it was trained"
        value={l.scale}
        oninput={(e) => form.setLora(i, { scale: e.currentTarget.value })}
        disabled={busy}
      />
      <button class="btn btn-ghost btn-xs" onclick={() => form.removeLora(i)} disabled={busy} aria-label="Remove">
        <Icon path={CLOSE} size={14} />
      </button>
    </div>
  {/each}
</div>
