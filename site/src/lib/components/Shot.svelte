<script>
  // A screenshot of the web UI in a browser frame. There are two files for
  // each, one per theme, made by scripts/shots.js: `name` is the part of the
  // file name before `-dark.webp` and `-light.webp`.
  let { name, alt, width = 1440, height = 900, address = "127.0.0.1:5823", eager = false } = $props();
</script>

<figure class="shot">
  <div class="bar" aria-hidden="true">
    <i></i><i></i><i></i>
    <span>{address}</span>
  </div>
  {#each ["dark", "light"] as theme (theme)}
    <img
      class="only-{theme}"
      src="/shots/{name}-{theme}.webp"
      {alt}
      {width}
      {height}
      loading={eager ? "eager" : "lazy"}
      decoding="async"
    />
  {/each}
</figure>

<style>
  .shot {
    margin: 0;
    border-radius: 10px;
    border: 1px solid var(--line-strong);
    background: var(--surface);
    overflow: hidden;
    box-shadow: var(--shadow);
  }
  .bar {
    display: flex;
    align-items: center;
    gap: 7px;
    height: 34px;
    padding: 0 13px;
    border-bottom: 1px solid var(--line);
  }
  .bar i {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--line-strong);
  }
  .bar span {
    margin: 0 auto;
    padding-right: 44px;
    font: 400 12px/1 var(--mono);
    color: var(--faint);
  }
  img {
    width: 100%;
    height: auto;
  }
</style>
