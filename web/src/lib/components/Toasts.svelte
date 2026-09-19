<script>
  import { toasts } from "../toasts.svelte.js";

  // daisyUI's alert colours, by kind. A map rather than string interpolation
  // so that Tailwind's scanner can see every class that might be used.
  const COLOUR = {
    info: "alert-info",
    success: "alert-success",
    warning: "alert-warning",
    error: "alert-error",
  };
</script>

<!-- `aria-live` so a screen reader hears what a sighted user sees flash by. -->
<div class="toast toast-end z-50" aria-live="polite" aria-atomic="false">
  {#each toasts.items as toast (toast.id)}
    <div role="alert" class="alert {COLOUR[toast.kind]} max-w-md">
      <span class="grow text-sm">{toast.message}</span>
      <button
        class="btn btn-ghost btn-xs btn-circle"
        aria-label="Dismiss"
        onclick={() => toasts.dismiss(toast.id)}
      >
        ✕
      </button>
    </div>
  {/each}
</div>
