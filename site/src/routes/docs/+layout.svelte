<script>
  import { page } from "$app/state";
  import { NAV, href } from "#lib/docs-nav.js";

  let { children } = $props();
  let open = $state(false);

  const path = $derived(page.url.pathname);
  const current = $derived(NAV.flatMap((g) => g.pages).find((p) => href(p.slug) === path));

  $effect(() => {
    path;
    open = false;
  });
</script>

<div class="wrap docs">
  <nav class:open aria-label="Documentation">
    <button class="menu" type="button" aria-expanded={open} onclick={() => (open = !open)}>
      <span>{current?.title ?? "Documentation"}</span>
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="m6 9 6 6 6-6" /></svg>
    </button>
    <div class="list">
      {#each NAV as group (group.title)}
        <section>
          <h2>{group.title}</h2>
          {#each group.pages as p (p.slug)}
            <a href={href(p.slug)} aria-current={href(p.slug) === path ? "page" : undefined}>{p.title}</a>
          {/each}
        </section>
      {/each}
    </div>
  </nav>

  {@render children()}
</div>

<style>
  .docs {
    display: grid;
    grid-template-columns: 216px minmax(0, 1fr) 208px;
    gap: 56px;
    padding-top: 44px;
    align-items: start;
  }
  nav {
    position: sticky;
    top: calc(var(--header) + 36px);
    max-height: calc(100vh - var(--header) - 60px);
    overflow-y: auto;
    padding-bottom: 20px;
    scrollbar-width: thin;
  }
  .menu {
    display: none;
  }
  section {
    display: grid;
    gap: 1px;
    margin-bottom: 26px;
  }
  h2 {
    font: 500 11.5px/1 var(--mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--faint);
    margin-bottom: 9px;
  }
  a {
    padding: 5px 10px;
    margin-left: -10px;
    border-radius: 6px;
    font-size: 14.5px;
    color: var(--muted);
    text-decoration: none;
  }
  a:hover {
    color: var(--text);
  }
  a[aria-current] {
    color: var(--text);
    background: var(--raised);
    font-weight: 500;
  }

  @media (max-width: 1180px) {
    .docs {
      grid-template-columns: 216px minmax(0, 1fr);
      gap: 44px;
    }
  }
  @media (max-width: 820px) {
    .docs {
      grid-template-columns: minmax(0, 1fr);
      gap: 24px;
      padding-top: 16px;
    }
    nav {
      position: static;
      max-height: none;
      padding: 0;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--surface);
    }
    .menu {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      padding: 12px 14px;
      font-weight: 500;
      font-size: 15px;
    }
    .menu svg {
      transition: transform 0.2s var(--ease);
    }
    .open .menu svg {
      transform: rotate(180deg);
    }
    .list {
      display: none;
      padding: 6px 14px 4px 24px;
      border-top: 1px solid var(--line);
    }
    .open .list {
      display: block;
    }
    section {
      margin: 14px 0;
    }
  }
</style>
