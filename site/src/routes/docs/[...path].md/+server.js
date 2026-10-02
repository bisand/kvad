// Every documentation page as Markdown: /docs/install.md beside
// /docs/install/. For a model or a script, which reads this more surely
// than it reads the page. /llms.txt is the list of them.

import { error } from "@sveltejs/kit";
import { PAGES } from "#lib/docs-nav.js";
import { markdown } from "#lib/server/docs.js";

export const prerender = true;

export const entries = () => PAGES.map((p) => ({ path: p.slug || "index" }));

export function GET({ params }) {
  const text = markdown(params.path === "index" ? "" : params.path);
  if (text === null) error(404, "No such page");
  return new Response(text, { headers: { "content-type": "text/markdown; charset=utf-8" } });
}
