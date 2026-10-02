# kvad.eu

The marketing pages and the documentation, as a static site: SvelteKit with
the static adapter, built into `site/build` and published to GitHub Pages by
`.github/workflows/site.yml`.

```bash
cd site
npm ci
npm run dev        # http://localhost:5173
npm run build      # site/build
```

## Where the content is

| What | Where |
|---|---|
| The front page and the Apple silicon page | `src/routes/+page.svelte`, `src/routes/apple-silicon/` |
| The guide | `content/docs/*.md`, one file a page |
| The table of contents | `src/lib/docs-nav.js` |
| "How it works" | the repository's own `docs/*.md`, rendered as they are |
| `kvad.eu/install.sh` | the repository's `install.sh`, copied by `scripts/sync.js` |

A new guide page is a Markdown file and a line in `docs-nav.js`. The search
index, the sitemap and the previous/next links all come from that list.

Nothing on the pages is made up. The numbers are from `docs/` or from a run
that is quoted; the terminal output in `src/lib/captures/` was copied from a
terminal; the code on the front page is read out of `crates/llm/src/tensor.rs`
at build time.

## Screenshots

`static/shots/` holds the web UI in both themes. They are taken from a
running server, so take them from one whose data you are happy to publish: a
second `kvad-serve` with an empty data directory, and a Hub cache holding
only the models you want listed.

```bash
KVAD_DATA_DIR=/tmp/kvad-demo kvad-serve --bind 127.0.0.1:5899 &
# load a model, hold a conversation, run a benchmark and a training run
node scripts/shots.js http://127.0.0.1:5899
```

It drives the Chrome that is installed, through Playwright.

## The domain

`static/CNAME` is `kvad.eu`. At the registrar, the apex needs GitHub Pages'
four `A` records (185.199.108.153 to 185.199.111.153) and `www` a `CNAME` to
`bisand.github.io`. In the repository's settings, Pages' source is "GitHub
Actions".

## Search engines

After each deploy the workflow sends every address in the sitemap to
IndexNow, which Bing and the engines that share its index read. The key is
the 32-character file name in `static/`, and is public by design. Google does
not read IndexNow: it finds the pages from the sitemap, submitted once in
Search Console.

`/llms.txt` lists every documentation page as Markdown, for a language
model, and each page is served that way at `/docs/<page>.md`.
