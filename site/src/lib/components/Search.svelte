<script>
  // Search over the documentation. The index is /search.json, fetched and
  // built the first time this opens; after that a query is answered in the
  // page, with nothing sent anywhere.
  import { goto } from "$app/navigation";
  import { tick, untrack } from "svelte";

  let { open = $bindable(false) } = $props();

  let dialog = $state();
  let input = $state();
  let query = $state("");
  let index = $state(null);
  let failed = $state(false);
  let active = $state(0);

  async function build() {
    try {
      const [{ default: MiniSearch }, entries] = await Promise.all([
        import("minisearch"),
        fetch("/search.json").then((r) => r.json()),
      ]);
      const mini = new MiniSearch({
        fields: ["heading", "page", "text"],
        storeFields: ["url", "page", "group", "heading", "text"],
        searchOptions: { boost: { heading: 4, page: 3 }, prefix: true, fuzzy: 0.15 },
      });
      mini.addAll(entries);
      index = mini;
    } catch {
      failed = true;
    }
  }

  $effect(() => {
    if (open) {
      if (!dialog.open) dialog.showModal();
      // Untracked, so the index arriving does not run this again and select
      // what has been typed in the meantime.
      untrack(() => {
        if (!index && !failed) build();
      });
      tick().then(() => input?.select());
    } else if (dialog?.open) {
      dialog.close();
    }
  });

  const results = $derived(index && query.trim() ? index.search(query).slice(0, 12) : []);

  $effect(() => {
    results;
    active = 0;
  });

  // The words around the first match, so a result says why it is one.
  function snippet(r) {
    const text = r.text;
    const terms = Object.keys(r.match);
    const lower = text.toLowerCase();
    let at = -1;
    for (const t of terms) {
      const i = lower.indexOf(t);
      if (i >= 0 && (at < 0 || i < at)) at = i;
    }
    const start = Math.max(0, at - 60);
    const cut = text.slice(start, start + 170);
    return { cut: (start ? "…" : "") + cut + (start + 170 < text.length ? "…" : ""), terms };
  }

  function parts(text, terms) {
    if (!terms.length) return [{ text, hit: false }];
    const re = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
    return text.split(re).map((p, i) => ({ text: p, hit: i % 2 === 1 }));
  }

  function go(r) {
    open = false;
    goto(r.url);
  }

  function keys(e) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      active = Math.min(active + 1, results.length - 1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      active = Math.max(active - 1, 0);
    } else if (e.key === "Enter" && results[active]) {
      e.preventDefault();
      go(results[active]);
    }
    tick().then(() => dialog.querySelector(".hit.on")?.scrollIntoView({ block: "nearest" }));
  }
</script>

<svelte:window
  onkeydown={(e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      open = !open;
    } else if (e.key === "/" && !open && !/^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName ?? "")) {
      e.preventDefault();
      open = true;
    }
  }}
/>

<dialog
  bind:this={dialog}
  onclose={() => (open = false)}
  onclick={(e) => e.target === dialog && (open = false)}
  aria-label="Search the documentation"
>
  <div class="box">
    <div class="field">
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true">
        <circle cx="11" cy="11" r="6.5" /><path d="m16 16 4.5 4.5" />
      </svg>
      <input
        bind:this={input}
        bind:value={query}
        onkeydown={keys}
        type="search"
        placeholder="Search the documentation"
        autocomplete="off"
        spellcheck="false"
        aria-controls="search-results"
      />
      <kbd>esc</kbd>
    </div>

    <div class="results" id="search-results" role="listbox">
      {#if failed}
        <p class="note">The search index could not be loaded.</p>
      {:else if !query.trim()}
        <p class="note">Try <button type="button" onclick={() => (query = "install")}>install</button>,
          <button type="button" onclick={() => (query = "lora")}>lora</button> or
          <button type="button" onclick={() => (query = "tool calls")}>tool calls</button>.</p>
      {:else if !index}
        <p class="note">Loading…</p>
      {:else if !results.length}
        <p class="note">Nothing found for “{query}”.</p>
      {:else}
        {#each results as r, i (r.id)}
          {@const s = snippet(r)}
          <a
            class="hit"
            class:on={i === active}
            href={r.url}
            role="option"
            aria-selected={i === active}
            onmousemove={() => (active = i)}
            onclick={(e) => {
              e.preventDefault();
              go(r);
            }}
          >
            <span class="where">{r.group} <span aria-hidden="true">/</span> {r.page}</span>
            <span class="what">{r.heading || r.page}</span>
            <span class="why">{#each parts(s.cut, s.terms) as p, j (j)}{#if p.hit}<mark>{p.text}</mark>{:else}{p.text}{/if}{/each}</span>
          </a>
        {/each}
      {/if}
    </div>
  </div>
</dialog>

<style>
  dialog {
    width: min(640px, calc(100vw - 32px));
    max-height: min(560px, calc(100vh - 120px));
    margin: 12vh auto auto;
    padding: 0;
    border: 1px solid var(--line-strong);
    border-radius: 12px;
    background: var(--surface);
    color: var(--text);
    box-shadow: var(--shadow);
    overflow: hidden;
  }
  dialog[open] {
    display: flex;
    animation: rise 0.18s var(--ease);
  }
  dialog::backdrop {
    background: rgb(10 12 18 / 0.55);
    backdrop-filter: blur(2px);
  }
  @keyframes rise {
    from {
      opacity: 0;
      transform: translateY(8px) scale(0.99);
    }
  }
  .box {
    display: flex;
    flex-direction: column;
    width: 100%;
    min-height: 0;
  }
  .field {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 16px;
    border-bottom: 1px solid var(--line);
    color: var(--muted);
  }
  input {
    flex: 1;
    height: 54px;
    border: 0;
    outline: 0;
    background: none;
    color: var(--text);
    font: inherit;
  }
  input::-webkit-search-cancel-button {
    display: none;
  }
  kbd {
    padding: 3px 6px;
    border: 1px solid var(--line);
    border-radius: 5px;
    font-size: 11px;
    color: var(--faint);
  }
  .results {
    overflow-y: auto;
    padding: 8px;
  }
  .note {
    padding: 18px 12px;
    color: var(--muted);
    font-size: 15px;
  }
  .note button {
    color: var(--accent);
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  .hit {
    display: grid;
    gap: 2px;
    padding: 10px 12px;
    border-radius: 8px;
    text-decoration: none;
  }
  .hit.on {
    background: var(--raised);
  }
  .where {
    font: 400 11.5px/1.4 var(--mono);
    color: var(--faint);
  }
  .what {
    font-weight: 500;
  }
  .why {
    font-size: 14px;
    line-height: 1.5;
    color: var(--muted);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  mark {
    background: var(--accent-soft);
    color: var(--text);
    border-radius: 2px;
    padding: 0 1px;
  }
</style>
