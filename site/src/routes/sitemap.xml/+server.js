import { PAGES, href } from "#lib/docs-nav.js";
import { SITE } from "#lib/site.js";

export const prerender = true;

export function GET() {
  const urls = ["/", "/apple-silicon/", ...PAGES.map((p) => href(p.slug))];
  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map((u) => `  <url><loc>${SITE}${u}</loc></url>`).join("\n")}
</urlset>
`;
  return new Response(body, { headers: { "content-type": "application/xml" } });
}
