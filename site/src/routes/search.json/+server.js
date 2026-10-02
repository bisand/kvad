// The search index: one entry for each section of each documentation page.
// Rendered once at build time and fetched the first time the search opens.

import { json } from "@sveltejs/kit";
import { PAGES, href } from "#lib/docs-nav.js";
import { load } from "#lib/server/docs.js";

export const prerender = true;

export function GET() {
  const entries = [];
  for (const page of PAGES) {
    const doc = load(page.slug);
    for (const s of doc.sections) {
      entries.push({
        id: entries.length,
        url: href(page.slug) + (s.id ? `#${s.id}` : ""),
        page: doc.title,
        group: page.group,
        heading: s.heading,
        text: s.text,
      });
    }
  }
  return json(entries);
}
