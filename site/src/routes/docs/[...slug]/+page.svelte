<script>
  import { codeCopy } from "#lib/copy.js";
  import { href } from "#lib/docs-nav.js";
  import { SITE } from "#lib/site.js";
  import Meta from "#lib/components/Meta.svelte";

  let { data } = $props();

  // For search engines: where the page sits, and what it is.
  const structured = $derived([
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "Documentation", item: `${SITE}/docs/` },
        { "@type": "ListItem", position: 2, name: data.group },
        { "@type": "ListItem", position: 3, name: data.title, item: SITE + href(data.slug) },
      ],
    },
    {
      "@context": "https://schema.org",
      "@type": "TechArticle",
      headline: data.title,
      description: data.description || data.summary || undefined,
      url: SITE + href(data.slug),
      dateModified: data.changed || undefined,
      author: { "@type": "Person", name: "André Biseth" },
      isPartOf: { "@type": "WebSite", name: "Kvad", url: SITE },
    },
  ]);

  // Which heading the reader is under, for the list on the right.
  let here = $state("");

  $effect(() => {
    data.slug;
    here = "";
    const heads = [...document.querySelectorAll(".prose :is(h2, h3)[id]")];
    if (!heads.length) return;
    const mark = () => {
      let cur = "";
      for (const h of heads) {
        if (h.getBoundingClientRect().top <= innerHeight * 0.3) cur = h.id;
        else break;
      }
      here = cur;
    };
    mark();
    addEventListener("scroll", mark, { passive: true });
    return () => removeEventListener("scroll", mark);
  });
</script>

<Meta
  title="{data.title} · Kvad documentation"
  description={data.description || data.summary}
  path={href(data.slug)}
  type="article"
  markdown={data.markdown}
  data={structured}
/>

<article>
  <header>
    <p class="label">{data.group}</p>
    <h1 id={data.titleId || undefined}>{data.title}</h1>
    {#if data.description}<p class="lede">{data.description}</p>{/if}
  </header>

  {#key data.slug}
    <div class="prose" use:codeCopy>{@html data.html}</div>
  {/key}

  <footer>
    <div class="pager">
      {#if data.prev}
        <a class="prev" href={href(data.prev.slug)}><span>Previous</span>{data.prev.title}</a>
      {/if}
      {#if data.next}
        <a class="next" href={href(data.next.slug)}><span>Next</span>{data.next.title}</a>
      {/if}
    </div>
    <p class="edit">
      <a href={data.edit} rel="noopener">Edit this page on GitHub</a>
      <a href={data.markdown}>This page as Markdown</a>
    </p>
  </footer>
</article>

{#if data.toc.length > 1}
  <aside aria-label="On this page">
    <h2>On this page</h2>
    <ul>
      {#each data.toc as t (t.id)}
        <li class:sub={t.depth === 3}>
          <a href="#{t.id}" class:on={here === t.id}>{t.text}</a>
        </li>
      {/each}
    </ul>
  </aside>
{/if}

<style>
  article {
    min-width: 0;
    max-width: 760px;
    width: 100%;
  }
  header {
    display: grid;
    gap: 14px;
    margin-bottom: 36px;
  }
  h1 {
    font-size: clamp(2.1rem, 4.6vw, 2.9rem);
  }
  .lede {
    font-size: 1.18rem;
    line-height: 1.55;
    color: var(--muted);
    max-width: 60ch;
  }
  footer {
    margin-top: 72px;
    display: grid;
    gap: 22px;
  }
  .pager {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }
  .pager a {
    display: grid;
    gap: 4px;
    padding: 15px 18px;
    border: 1px solid var(--line);
    border-radius: 10px;
    text-decoration: none;
    font-weight: 500;
    transition: border-color 0.15s;
  }
  .pager a:hover {
    border-color: var(--accent);
  }
  .pager span {
    font: 400 11.5px/1 var(--mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--faint);
  }
  .next {
    grid-column: 2;
    text-align: right;
  }
  .edit {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 24px;
    font-size: 14px;
    color: var(--muted);
  }

  aside {
    position: sticky;
    top: calc(var(--header) + 36px);
    align-self: start;
    max-height: calc(100vh - var(--header) - 60px);
    overflow-y: auto;
    font-size: 13.5px;
  }
  aside h2 {
    font: 500 11.5px/1 var(--mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--faint);
    margin-bottom: 12px;
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    border-left: 1px solid var(--line);
  }
  li a {
    display: block;
    padding: 4px 0 4px 14px;
    margin-left: -1px;
    border-left: 1px solid transparent;
    color: var(--muted);
    text-decoration: none;
    line-height: 1.4;
    transition:
      color 0.15s,
      border-color 0.15s;
  }
  li.sub a {
    padding-left: 26px;
  }
  li a:hover {
    color: var(--text);
  }
  li a.on {
    color: var(--accent);
    border-left-color: var(--accent);
  }
  @media (max-width: 1180px) {
    aside {
      display: none;
    }
  }
</style>
