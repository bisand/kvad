<script>
  // The OpenAPI document, rendered.
  //
  // Our own renderer rather than Swagger UI or Scalar from a CDN: this binary
  // is meant to be one file you can run on a machine with no internet, and a
  // docs page that fetches 400 KB of JavaScript from somewhere else is not
  // that. The document is ours and it is small, so showing it is a list.
  import { api } from "../lib/api.js";
  import Icon from "../lib/components/Icon.svelte";

  let doc = $state(null);
  let error = $state(null);
  let filter = $state("");
  let open = $state(new Set());

  $effect(() => {
    api("/api/openapi.json")
      .then((d) => (doc = d))
      .catch((e) => (error = e.message));
  });

  /** Every operation, flattened out of the document's `paths`. */
  const operations = $derived.by(() => {
    if (!doc) return [];
    const out = [];
    for (const [path, methods] of Object.entries(doc.paths)) {
      for (const [method, op] of Object.entries(methods)) {
        out.push({ path, method, ...op, key: `${method} ${path}` });
      }
    }
    return out;
  });

  const matching = $derived(
    operations.filter((o) => {
      const q = filter.trim().toLowerCase();
      if (!q) return true;
      return (
        o.path.toLowerCase().includes(q) ||
        o.summary?.toLowerCase().includes(q) ||
        o.tags?.[0]?.toLowerCase().includes(q)
      );
    }),
  );

  // Grouped in the document's own tag order, which is the order the sidebar
  // visits the pages.
  const groups = $derived.by(() => {
    const order = (doc?.tags ?? []).map((t) => t.name);
    return order
      .map((name) => ({
        name,
        about: doc.tags.find((t) => t.name === name)?.description ?? "",
        operations: matching
          .filter((o) => o.tags?.[0] === name)
          .sort(
            (a, b) =>
              a.path.localeCompare(b.path) ||
              ORDER.indexOf(a.method) - ORDER.indexOf(b.method),
          ),
      }))
      .filter((g) => g.operations.length);
  });

  function toggle(key) {
    const next = new Set(open);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    open = next;
  }

  // The document writes prose with `backticks` in it, because that is how a
  // person writes about a field name. Splitting on them is the whole of the
  // Markdown this page needs; a Markdown library for one rule would be more
  // bytes than the page.
  function parts(text) {
    return (text ?? "").split("`").map((piece, i) => ({ code: i % 2 === 1, piece }));
  }

  // get, post, patch, delete — what a reader expects, rather than what
  // sorting the JSON's keys happens to give.
  const ORDER = ["get", "post", "put", "patch", "delete"];

  const COLOUR = {
    get: "badge-ghost",
    post: "badge-success",
    patch: "badge-warning",
    delete: "badge-error",
    put: "badge-warning",
  };

  /** Whether an operation needs no credential at all. */
  function isOpen(op) {
    return (op.security ?? []).some((s) => Object.keys(s).length === 0);
  }
</script>

<div class="flex flex-col gap-6">
  {#if error}
    <div role="alert" class="alert alert-error"><span>{error}</span></div>
  {:else if !doc}
    <span class="loading loading-spinner"></span>
  {:else}
    <section class="bg-base-200 rounded-box flex flex-col gap-3 p-4">
      <div class="flex flex-wrap items-center gap-3">
        <h2 class="text-sm font-medium">
          {doc.info.title}
          <span class="opacity-60">v{doc.info.version}</span>
        </h2>
        <span class="grow"></span>
        <a class="btn btn-sm" href="/api/openapi.json" download="kvad-openapi.json">
          <Icon path="M12 3v12M7 10l5 5 5-5M5 21h14" size={16} />
          openapi.json
        </a>
      </div>
      <div class="text-sm whitespace-pre-line opacity-70">
        {#each parts(doc.info.description) as { code, piece }, i (i)}
          {#if code}<code class="bg-base-300 rounded px-1 text-xs">{piece}</code>{:else}{piece}{/if}
        {/each}
      </div>
    </section>

    <label class="input input-sm w-full max-w-md">
      <Icon path="M21 21l-4.3-4.3M11 19a8 8 0 1 1 0-16 8 8 0 0 1 0 16z" size={16} />
      <input placeholder="Filter by path, summary or area…" bind:value={filter} />
    </label>

    {#each groups as group (group.name)}
      <section class="flex flex-col gap-2">
        <h2 class="text-sm font-medium opacity-60">{group.name}</h2>
        <p class="text-xs opacity-60">{group.about}</p>
        <div class="flex flex-col gap-1">
          {#each group.operations as op (op.key)}
            <div class="bg-base-200 rounded-box overflow-hidden">
              <button
                class="hover:bg-base-300 flex w-full items-center gap-3 p-2 text-left"
                onclick={() => toggle(op.key)}
              >
                <span class="badge badge-sm w-16 font-mono {COLOUR[op.method]}">
                  {op.method}
                </span>
                <code class="text-xs">{op.path}</code>
                <span class="grow"></span>
                <span class="hidden text-xs opacity-60 sm:inline">{op.summary}</span>
                {#if isOpen(op)}
                  <span class="badge badge-xs badge-warning">open</span>
                {/if}
              </button>

              {#if open.has(op.key)}
                <div class="flex flex-col gap-3 border-t border-base-300 p-3 text-sm">
                  <div class="whitespace-pre-line opacity-80">
                    {#each parts(op.description) as { code, piece }, i (i)}
                      {#if code}<code
                          class="bg-base-300 rounded px-1 text-xs">{piece}</code
                        >{:else}{piece}{/if}
                    {/each}
                  </div>

                  {#if op.parameters?.length}
                    <div>
                      <h3 class="mb-1 text-xs font-medium opacity-60">Parameters</h3>
                      <table class="table table-xs">
                        <tbody>
                          {#each op.parameters as p (p.in + p.name)}
                            <tr>
                              <td class="font-mono whitespace-nowrap">{p.name}</td>
                              <td class="text-xs opacity-60">{p.in}</td>
                              <td class="text-xs opacity-60">
                                {p.required ? "required" : "optional"}
                              </td>
                              <td class="text-xs">
                                {#each parts(p.description) as { code, piece }, i (i)}
                                  {#if code}<code class="text-xs">{piece}</code
                                    >{:else}{piece}{/if}
                                {/each}
                              </td>
                            </tr>
                          {/each}
                        </tbody>
                      </table>
                    </div>
                  {/if}

                  {#if op.requestBody}
                    <div>
                      <h3 class="mb-1 text-xs font-medium opacity-60">
                        Body · {Object.keys(op.requestBody.content)[0]}
                      </h3>
                      <p class="text-xs opacity-80">
                        {#each parts(op.requestBody.description) as { code, piece }, i (i)}
                          {#if code}<code
                              class="bg-base-300 rounded px-1">{piece}</code
                            >{:else}{piece}{/if}
                        {/each}
                      </p>
                    </div>
                  {/if}

                  <div>
                    <h3 class="mb-1 text-xs font-medium opacity-60">Answers</h3>
                    <p class="text-xs opacity-80">
                      {Object.keys(op.responses["200"].content)[0]}
                    </p>
                  </div>

                  <div class="text-xs opacity-60">
                    <code>curl -H "Authorization: Bearer kvad_…" {op.method !== "get"
                      ? `-X ${op.method.toUpperCase()} `
                      : ""}{window.location.origin}{op.path}</code>
                  </div>
                </div>
              {/if}
            </div>
          {/each}
        </div>
      </section>
    {:else}
      <p class="text-sm opacity-60">Nothing matches “{filter}”.</p>
    {/each}
  {/if}
</div>
