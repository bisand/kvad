<script>
  // One <svg> for every icon in the app. `path` comes from `pages.js` or
  // from the caller; `size` is in pixels because these sit next to text and
  // have to line up with it.
  let { path, size = 20, class: klass = "" } = $props();
</script>

<svg
  xmlns="http://www.w3.org/2000/svg"
  width={size}
  height={size}
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width="1.5"
  stroke-linecap="round"
  stroke-linejoin="round"
  class={klass}
  aria-hidden="true"
>
  <path d={path} />
</svg>
