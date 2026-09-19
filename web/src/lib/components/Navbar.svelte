<script>
  import Icon from "./Icon.svelte";
  import ThemeMenu from "./ThemeMenu.svelte";

  let { drawerId, title, health, error } = $props();

  const MENU = "M4 6h16M4 12h16M4 18h16";
</script>

<header class="navbar bg-base-100 border-base-300 sticky top-0 z-10 min-h-16 border-b px-2 sm:px-4">
  <div class="navbar-start gap-2">
    <label for={drawerId} class="btn btn-ghost btn-square drawer-button lg:hidden" aria-label="Menu">
      <Icon path={MENU} size={22} />
    </label>
    <h1 class="text-base font-medium">{title}</h1>
  </div>

  <div class="navbar-end gap-2">
    <!-- What the server says about itself, where it is visible without
         opening a console: which build, and whether the UI in the browser came
         out of that build or out of Vite. The error wins over the last good
         answer — a green light beside numbers from two minutes ago is worse
         than no light at all. -->
    <span class="hidden items-center gap-2 text-xs opacity-60 sm:flex" title={error ?? undefined}>
      {#if error}
        <span class="status status-error" aria-hidden="true"></span>
        not responding
      {:else if health}
        <span class="status status-success" aria-hidden="true"></span>
        v{health.version} · schema {health.schema} · {health.you.name}
      {:else}
        <span class="status" aria-hidden="true"></span>
        connecting
      {/if}
    </span>
    <ThemeMenu />
  </div>
</header>
