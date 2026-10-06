<script>
  // What is in a dataset of pictures: each one, and the caption a run would
  // read beside it, which can be written or changed here.
  import { training } from "../training.svelte.js";
  import { toasts } from "../toasts.svelte.js";
  import Icon from "./Icon.svelte";

  let { dataset } = $props();

  const TRASH =
    "M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6";

  let pictures = $state(null);
  let failed = $state(null);
  /** Captions being written and not saved yet, by file. */
  let drafts = $state({});

  async function load() {
    try {
      pictures = await training.pictures(dataset.id);
      failed = null;
    } catch (e) {
      failed = e.message;
    }
  }

  // Read again whenever the dataset's own counts change: an upload added to it.
  $effect(() => {
    void dataset.id;
    void dataset.pictures;
    load();
  });

  async function save(p) {
    const text = (drafts[p.file] ?? "").trim();
    if (text === (p.caption ?? "")) return;
    if (!text) return toasts.warning("A caption says what is in the picture; an empty one is not kept.");
    try {
      await training.caption(dataset.id, p.file, text);
      pictures = pictures.map((q) => (q.file === p.file ? { ...q, caption: text } : q));
      const { [p.file]: _, ...rest } = drafts;
      drafts = rest;
      training.refresh();
    } catch (e) {
      toasts.error(e.message);
    }
  }

  async function remove(p) {
    try {
      await training.removePicture(dataset.id, p.file);
      pictures = pictures.filter((q) => q.file !== p.file);
      training.refresh();
    } catch (e) {
      toasts.error(e.message);
    }
  }
</script>

{#if failed}
  <p class="text-error text-xs">{failed}</p>
{:else if pictures === null}
  <span class="loading loading-spinner loading-sm opacity-60"></span>
{:else if pictures.length === 0}
  <p class="text-xs opacity-60">No pictures in it yet. Add some above, under the same name.</p>
{:else}
  <div class="grid gap-3 sm:grid-cols-2">
    {#each pictures as p (p.file)}
      <div class="bg-base-200/40 rounded-box flex gap-3 p-2">
        <img
          class="rounded-field size-24 shrink-0 object-cover"
          src={`/api/datasets/${dataset.id}/files/${encodeURIComponent(p.file)}`}
          alt={p.caption ?? p.file}
          loading="lazy"
        />
        <div class="flex min-w-0 grow flex-col gap-1">
          <div class="flex items-center gap-1">
            <span class="truncate text-xs opacity-60">{p.file}</span>
            <span class="grow"></span>
            <button class="btn btn-ghost btn-xs" aria-label={`Remove ${p.file}`} onclick={() => remove(p)}>
              <Icon path={TRASH} size={14} />
            </button>
          </div>
          <textarea
            class="textarea textarea-sm w-full grow text-xs {p.caption ? '' : 'textarea-warning'}"
            rows="2"
            aria-label={`Caption of ${p.file}`}
            placeholder="no caption yet: what is in this picture?"
            value={drafts[p.file] ?? p.caption ?? ""}
            oninput={(e) => (drafts = { ...drafts, [p.file]: e.currentTarget.value })}
            onblur={() => save(p)}
          ></textarea>
        </div>
      </div>
    {/each}
  </div>
  <p class="mt-2 text-xs opacity-60">
    A caption is kept when you leave its box. Say what is in the picture, and name the subject
    or the style with a word of its own (<code>sks dog</code>, <code>in kvadbox style</code>):
    that word is what a prompt then asks the LoRA for.
  </p>
{/if}
