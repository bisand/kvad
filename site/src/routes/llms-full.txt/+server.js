// /llms-full.txt: the guide in one file. The long-form pages under "How it
// works" are left out and linked instead: they are five times the guide.

import { NAV, href } from "#lib/docs-nav.js";
import { markdown, mdPath } from "#lib/server/docs.js";
import { SITE } from "#lib/site.js";

export const prerender = true;

export function GET() {
  const guide = NAV.flatMap((g) => g.pages).filter((p) => !p.source);
  const long = NAV.flatMap((g) => g.pages).filter((p) => p.source);
  const parts = guide.map((p) => `<!-- ${SITE}${href(p.slug)} -->\n\n${markdown(p.slug)}`);
  const more = long.map((p) => `- [${p.title}](${SITE}${mdPath(p.slug)})`).join("\n");
  const body = `${parts.join("\n\n---\n\n")}\n\n---\n\n# How it works, in full\n\n${more}\n`;
  return new Response(body, { headers: { "content-type": "text/plain; charset=utf-8" } });
}
