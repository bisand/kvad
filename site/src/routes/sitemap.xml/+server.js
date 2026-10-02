import { PAGES, href } from "#lib/docs-nav.js";
import { changed, changedPage } from "#lib/server/docs.js";
import { SITE } from "#lib/site.js";

export const prerender = true;

// Each address with the day its source last changed, from git. The workflow
// checks out the whole history for this; without it there is no date, and
// an address with no date is still an address.
export function GET() {
  const urls = [
    ["/", changed("src/routes/+page.svelte")],
    ["/apple-silicon/", changed("src/routes/apple-silicon/+page.svelte")],
    ...PAGES.map((p) => [href(p.slug), changedPage(p.slug)]),
  ];
  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map(([u, day]) => `  <url><loc>${SITE}${u}</loc>${day ? `<lastmod>${day}</lastmod>` : ""}</url>`).join("\n")}
</urlset>
`;
  return new Response(body, { headers: { "content-type": "application/xml" } });
}
