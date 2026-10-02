<script>
  import "../app.css";
  import Header from "#lib/components/Header.svelte";
  import Footer from "#lib/components/Footer.svelte";
  import Search from "#lib/components/Search.svelte";

  let { children } = $props();
  let searching = $state(false);
</script>

<a class="skip" href="#main">Skip to the content</a>
<Header onsearch={() => (searching = true)} />
<main id="main">
  {@render children()}
</main>
<Footer />
<Search bind:open={searching} />

<style>
  .skip {
    position: absolute;
    left: 12px;
    top: -60px;
    z-index: 30;
    padding: 10px 14px;
    border-radius: 8px;
    background: var(--accent);
    color: var(--on-accent);
    transition: top 0.15s;
  }
  .skip:focus {
    top: 12px;
  }
</style>
