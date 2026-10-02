<script>
  import { page } from "$app/state";
</script>

<svelte:head><title>Not found · Kvad</title></svelte:head>

<section class="wrap">
  <p class="label">{page.status}</p>
  <h1>There is no page here.</h1>
  <p>The address may have moved. <a href="/docs/">The documentation</a> has a search, and <a href="/">the front page</a> has the rest.</p>
</section>

<style>
  section {
    padding-block: 120px 60px;
    display: grid;
    gap: 18px;
  }
  h1 {
    font-size: clamp(2.2rem, 5vw, 3.4rem);
  }
  p:last-child {
    color: var(--muted);
  }
  a {
    color: var(--accent);
  }
</style>
