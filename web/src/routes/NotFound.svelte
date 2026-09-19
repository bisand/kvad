<script>
  import { router, navigate } from "../lib/router.svelte.js";
</script>

<div class="mx-auto max-w-2xl py-12 text-center">
  <h2 class="text-xl font-medium">No such page</h2>
  <p class="mt-2 opacity-70">
    <code class="text-sm">{router.path}</code> is not one of this server's pages.
  </p>
  <a href="/" onclick={(e) => navigate(e, "/")} class="btn btn-sm mt-6">Back to the dashboard</a>
</div>
