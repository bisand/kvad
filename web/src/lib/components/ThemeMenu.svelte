<script>
  import { theme, THEMES } from "../theme.svelte.js";
  import Icon from "./Icon.svelte";

  const ICONS = {
    system: "M3 5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2zM8 21h8M12 16v5",
    light:
      "M12 4V2M12 22v-2M6.3 6.3 4.9 4.9M19.1 19.1l-1.4-1.4M4 12H2M22 12h-2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0z",
    dim: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
  };

  let open = $state(false);
  let root;

  function choose(id) {
    theme.set(id);
    open = false;
  }

  // Closed by a click somewhere else or by Escape, and by nothing else. An
  // earlier version closed on `mouseleave`, which meant the menu disappeared
  // whenever the pointer wandered — including on the way to the item being
  // aimed at.
  $effect(() => {
    if (!open) return;
    const outside = (e) => {
      if (!root.contains(e.target)) open = false;
    };
    const escape = (e) => {
      if (e.key === "Escape") open = false;
    };
    // `pointerdown`, not `click`: a press that starts outside should dismiss
    // the menu even if the pointer is released somewhere else.
    document.addEventListener("pointerdown", outside, true);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside, true);
      document.removeEventListener("keydown", escape);
    };
  });
</script>

<div class="dropdown dropdown-end" class:dropdown-open={open} bind:this={root}>
  <button
    class="btn btn-ghost btn-sm btn-square"
    aria-label="Theme"
    aria-haspopup="menu"
    aria-expanded={open}
    onclick={() => (open = !open)}
  >
    <Icon path={ICONS[theme.current]} />
  </button>
  {#if open}
    <ul class="dropdown-content menu bg-base-100 rounded-box z-30 mt-2 w-40 p-2 shadow" role="menu">
      {#each THEMES as option (option.id)}
        <li>
          <button
            role="menuitemradio"
            aria-checked={theme.current === option.id}
            class={theme.current === option.id ? "menu-active" : ""}
            onclick={() => choose(option.id)}
          >
            <Icon path={ICONS[option.id]} size={16} />
            {option.label}
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</div>
