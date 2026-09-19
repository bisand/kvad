<script>
  // A page that is planned and not written. It says which phase brings it,
  // because "coming soon" is a promise and a phase number is a fact anyone
  // can check against docs/ui-plan.md.
  import Icon from "./Icon.svelte";

  let { page } = $props();
</script>

<div class="mx-auto max-w-2xl py-12 text-center">
  <div class="bg-base-200 mx-auto mb-5 grid size-14 place-items-center rounded-box opacity-70">
    <Icon path={page.icon} size={28} />
  </div>
  <h2 class="text-xl font-medium">{page.label}</h2>
  <p class="mt-2 opacity-70">{page.blurb}</p>
  <div class="mt-6 inline-flex items-center gap-2 text-sm opacity-60">
    <span class="badge badge-sm">Phase {page.phase}</span>
    <span>not built yet — see <code class="text-xs">docs/ui-plan.md</code></span>
  </div>
</div>
