<script>
  // What every page says about itself to a search engine, to a link preview
  // and to a crawler: one address, a title and a description, and, where a
  // page has any, its structured data and its Markdown source.
  import { SITE } from "#lib/site.js";

  let { title, description = "", path, type = "website", markdown = "", data = null } = $props();

  const url = $derived(SITE + path);
  // `<` escaped, so no text in the data can close the script element.
  const json = $derived(data ? JSON.stringify(data).replace(/</g, "\\u003c") : "");
</script>

<svelte:head>
  <title>{title}</title>
  {#if description}<meta name="description" content={description} />{/if}
  <link rel="canonical" href={url} />
  {#if markdown}<link rel="alternate" type="text/markdown" href={SITE + markdown} />{/if}
  <meta property="og:site_name" content="Kvad" />
  <meta property="og:type" content={type} />
  <meta property="og:title" content={title} />
  {#if description}<meta property="og:description" content={description} />{/if}
  <meta property="og:url" content={url} />
  <meta property="og:image" content="{SITE}/og.png" />
  <meta name="twitter:card" content="summary_large_image" />
  {#if json}{@html `<script type="application/ld+json">${json}</` + `script>`}{/if}
</svelte:head>
