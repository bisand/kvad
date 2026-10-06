<script>
  // Pictures to train a LoRA on: choose them, and the captions beside them,
  // and they go into a dataset a file at a time.
  //
  // A caption is the `.txt` of a picture's name, which is how kohya's
  // scripts and diffusers keep them and how `kvad-gpu tune` reads them, so a
  // folder somebody already has uploads as it is. Captions can also be
  // written here afterwards, in the dataset's row.
  import { training } from "../training.svelte.js";
  import { toasts } from "../toasts.svelte.js";
  import { humanBytes } from "../models.svelte.js";

  /** Called with the dataset once an upload has put something into it. */
  let { onadded = null } = $props();

  const PICTURE = /\.(jpe?g|png|webp|bmp)$/i;
  const CAPTION = /\.txt$/i;

  let name = $state("");
  let chosen = $state([]);
  let ignored = $state(0);
  let uploading = $state(null);
  let refused = $state([]);

  /**
   * A file's name as the server will take it: letters, digits, spaces and
   * `-`, `_`, `.`. A camera's and a browser's names are freer than that, and
   * a picture and its caption are renamed the same way, so they stay a pair.
   */
  function safe(file) {
    const dot = file.lastIndexOf(".");
    const stem = file
      .slice(0, dot)
      .normalize("NFKD")
      .replace(/[^A-Za-z0-9 _.-]+/g, "_")
      .replace(/^[.\s]+/, "")
      .slice(0, 100);
    return `${stem || "picture"}${file.slice(dot).toLowerCase()}`;
  }

  function pick(event) {
    const all = [...(event.currentTarget.files ?? [])];
    const kept = all.filter((f) => PICTURE.test(f.name) || CAPTION.test(f.name));
    ignored = all.length - kept.length;
    chosen = kept.map((f) => ({ name: safe(f.name), body: f, picture: PICTURE.test(f.name) }));
    refused = [];
    // A folder's name is the likeliest name for what was in it.
    const folder = all[0]?.webkitRelativePath?.split("/")[0];
    if (!name.trim() && folder) name = folder.replace(/[^A-Za-z0-9 _.-]+/g, "-").slice(0, 96);
  }

  const pictures = $derived(chosen.filter((f) => f.picture));
  const stems = $derived(new Set(chosen.filter((f) => !f.picture).map((f) => f.name.replace(CAPTION, ""))));
  const bare = $derived(pictures.filter((f) => !stems.has(f.name.replace(PICTURE, ""))).length);
  const bytes = $derived(pictures.reduce((sum, f) => sum + f.body.size, 0));

  async function upload(event) {
    event.preventDefault();
    if (!pictures.length && !chosen.length) return toasts.warning("Choose some pictures first.");
    const form = event.currentTarget;
    uploading = { done: 0, total: chosen.length };
    // Pictures first: a caption whose picture was refused is then the one
    // file out of place, and not the other way about.
    const ordered = [...chosen].sort((a, b) => Number(b.picture) - Number(a.picture));
    const went = await training.uploadPictures(name.trim(), ordered, (done, total) => (uploading = { done, total }));
    uploading = null;
    if (!went) return;
    refused = went.refused;
    const arrived = chosen.length - refused.length;
    if (arrived) toasts.success(`${arrived} file${arrived === 1 ? "" : "s"} added to ${went.dataset.name}.`);
    if (!refused.length) {
      chosen = [];
      name = "";
      form.reset();
    }
    onadded?.(went.dataset);
  }
</script>

<section class="card bg-base-100 border-base-300 border">
  <form class="card-body gap-3 p-4" onsubmit={upload}>
    <h2 class="text-sm font-medium opacity-60">Add pictures</h2>
    <div class="flex flex-wrap items-end gap-2">
      <fieldset class="fieldset">
        <legend class="fieldset-legend">Pictures, and their captions</legend>
        <input
          type="file"
          class="file-input file-input-sm"
          accept="image/jpeg,image/png,image/webp,image/bmp,.txt,text/plain"
          multiple
          onchange={pick}
        />
      </fieldset>
      <fieldset class="fieldset grow">
        <legend class="fieldset-legend">Name</legend>
        <input class="input input-sm w-full" bind:value={name} placeholder="my-dog" required />
      </fieldset>
      <button class="btn btn-sm" disabled={!!uploading || !chosen.length || !name.trim()}>
        {#if uploading}<span class="loading loading-spinner loading-xs"></span>{/if}
        Upload
      </button>
    </div>

    {#if uploading}
      <progress class="progress w-full" value={uploading.done} max={uploading.total}></progress>
      <p class="text-xs opacity-60">{uploading.done} of {uploading.total}</p>
    {:else if chosen.length}
      <p class="text-xs opacity-60">
        {pictures.length} picture{pictures.length === 1 ? "" : "s"}, {humanBytes(bytes)}{#if bare}
          — {bare} with no caption beside {bare === 1 ? "it" : "them"}, which can be written after{/if}.
        {#if ignored}{ignored} other file{ignored === 1 ? " is" : "s are"} left out.{/if}
      </p>
    {/if}

    {#if refused.length}
      <div role="alert" class="alert alert-warning flex-col items-start gap-1 text-xs">
        <span class="font-medium">{refused.length} not added:</span>
        {#each refused as why (why)}
          <span>{why}</span>
        {/each}
      </div>
    {/if}

    <p class="text-xs opacity-60">
      JPEG, PNG, WebP or BMP, for training a LoRA. A picture's caption is a <code>.txt</code>
      with the same name: <code>one.jpg</code> and <code>one.txt</code>. Choosing the same
      name again adds to what is there. Ten to forty pictures of one subject or one style is
      a usual set.
    </p>
  </form>
</section>
