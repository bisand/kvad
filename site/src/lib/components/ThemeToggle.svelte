<script>
  // Dark or light. With nothing stored the page follows the system; a click
  // stores the other one and the page keeps it.
  function flip() {
    const root = document.documentElement;
    const now =
      root.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const next = now === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      localStorage.setItem("kvad-theme", next);
    } catch {}
  }
</script>

<button class="toggle" type="button" onclick={flip} aria-label="Switch between dark and light">
  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true">
    <circle cx="12" cy="12" r="8.5" />
    <path d="M12 3.5v17" />
    <path d="M12 3.5a8.5 8.5 0 0 1 0 17z" fill="currentColor" stroke="none" />
  </svg>
</button>

<style>
  .toggle {
    display: grid;
    place-items: center;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    color: var(--muted);
    transition:
      color 0.15s,
      background 0.15s;
  }
  .toggle:hover {
    color: var(--text);
    background: var(--raised);
  }
  svg {
    transition: transform 0.4s var(--ease);
  }
  :global(:root[data-theme="light"]) svg {
    transform: rotate(180deg);
  }
</style>
