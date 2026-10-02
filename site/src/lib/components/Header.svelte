<script>
  import { page } from "$app/state";
  import { REPO } from "#lib/site.js";
  import Wordmark from "./Wordmark.svelte";
  import ThemeToggle from "./ThemeToggle.svelte";

  let { onsearch } = $props();

  const LINKS = [
    { href: "/#features", label: "Features" },
    { href: "/apple-silicon/", label: "Apple silicon" },
    { href: "/docs/", label: "Docs" },
    { href: "/docs/internals/", label: "How it works" },
  ];

  let menu = $state(false);
  const path = $derived(page.url.pathname);

  function current(href) {
    if (href.startsWith("/#")) return false;
    if (href === "/docs/") return path.startsWith("/docs/") && !path.startsWith("/docs/internals/");
    return path.startsWith(href);
  }

  // Leaving a page closes the menu that led there.
  $effect(() => {
    path;
    menu = false;
  });
</script>

<header class:open={menu}>
  <div class="wrap bar">
    <a class="home group" href="/" aria-label="Kvad, home">
      <Wordmark height={22} />
    </a>

    <nav aria-label="Main">
      {#each LINKS as l (l.href)}
        <a href={l.href} aria-current={current(l.href) ? "page" : undefined}>{l.label}</a>
      {/each}
    </nav>

    <div class="tools">
      <button class="search" type="button" onclick={onsearch}>
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true">
          <circle cx="11" cy="11" r="6.5" /><path d="m16 16 4.5 4.5" />
        </svg>
        <span>Search</span>
        <kbd>⌘K</kbd>
      </button>
      <ThemeToggle />
      <a class="icon" href={REPO} aria-label="Kvad on GitHub" rel="noopener">
        <svg viewBox="0 0 24 24" width="19" height="19" fill="currentColor" aria-hidden="true">
          <path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.7c-2.78.6-3.37-1.34-3.37-1.34-.46-1.15-1.11-1.46-1.11-1.46-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.9 1.52 2.34 1.08 2.91.83.09-.65.35-1.09.63-1.34-2.22-.25-4.56-1.11-4.56-4.94 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.64 0 0 .84-.27 2.75 1.02a9.6 9.6 0 0 1 5 0c1.91-1.29 2.75-1.02 2.75-1.02.55 1.37.2 2.39.1 2.64.64.7 1.03 1.59 1.03 2.68 0 3.84-2.34 4.68-4.57 4.93.36.31.68.92.68 1.85v2.74c0 .27.18.58.69.48A10 10 0 0 0 12 2z" />
        </svg>
      </a>
      <button class="burger" type="button" aria-label="Menu" aria-expanded={menu} onclick={() => (menu = !menu)}>
        <span></span><span></span>
      </button>
    </div>
  </div>
</header>

<style>
  header {
    position: sticky;
    top: 0;
    z-index: 20;
    background: color-mix(in srgb, var(--bg) 86%, transparent);
    backdrop-filter: saturate(1.4) blur(14px);
    border-bottom: 1px solid var(--line);
  }
  .bar {
    display: flex;
    align-items: center;
    gap: 28px;
    height: var(--header);
  }
  .home {
    display: flex;
    align-items: center;
    padding: 6px 0;
    text-decoration: none;
  }
  nav {
    display: flex;
    gap: 4px;
    margin-right: auto;
  }
  nav a {
    padding: 7px 11px;
    border-radius: 7px;
    font-size: 15px;
    color: var(--muted);
    text-decoration: none;
  }
  nav a:hover,
  nav a[aria-current] {
    color: var(--text);
  }
  nav a[aria-current] {
    background: var(--raised);
  }
  .tools {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .search {
    display: flex;
    align-items: center;
    gap: 9px;
    height: 36px;
    min-width: 190px;
    margin-right: 6px;
    padding: 0 9px 0 11px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--surface);
    color: var(--muted);
    font-size: 14px;
    transition: border-color 0.15s;
  }
  .search:hover {
    border-color: var(--line-strong);
  }
  .search span {
    margin-right: auto;
  }
  kbd {
    padding: 2px 5px;
    border: 1px solid var(--line);
    border-radius: 4px;
    font-size: 11px;
    color: var(--faint);
  }
  .icon {
    display: grid;
    place-items: center;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    color: var(--muted);
  }
  .icon:hover {
    color: var(--text);
    background: var(--raised);
  }
  .burger {
    display: none;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 6px;
  }
  .burger span {
    width: 18px;
    height: 1.5px;
    background: currentColor;
    transition: transform 0.2s var(--ease);
  }
  .open .burger span:first-child {
    transform: translateY(3.75px) rotate(45deg);
  }
  .open .burger span:last-child {
    transform: translateY(-3.75px) rotate(-45deg);
  }

  @media (max-width: 860px) {
    .bar {
      gap: 12px;
    }
    .tools {
      margin-left: auto;
    }
    .search {
      min-width: 0;
      width: 36px;
      padding: 0;
      justify-content: center;
      border-color: transparent;
      background: none;
      margin: 0;
    }
    .search span,
    .search kbd {
      display: none;
    }
    .burger {
      display: flex;
    }
    nav {
      display: none;
      position: absolute;
      top: var(--header);
      left: 0;
      right: 0;
      flex-direction: column;
      gap: 0;
      padding: 8px var(--gutter) 16px;
      background: var(--bg);
      border-bottom: 1px solid var(--line);
    }
    .open nav {
      display: flex;
    }
    nav a {
      padding: 12px 0;
      font-size: 17px;
      border-radius: 0;
      border-bottom: 1px solid var(--line);
    }
    nav a[aria-current] {
      background: none;
    }
  }
</style>
