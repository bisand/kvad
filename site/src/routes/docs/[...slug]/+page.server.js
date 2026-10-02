import { error } from "@sveltejs/kit";
import { PAGES } from "#lib/docs-nav.js";
import { load as read, mdPath, changedPage } from "#lib/server/docs.js";

export const entries = () => PAGES.map((p) => ({ slug: p.slug }));

export function load({ params }) {
  const slug = params.slug.replace(/\/$/, "");
  const doc = read(slug);
  if (!doc) error(404, "No such page");

  const i = PAGES.findIndex((p) => p.slug === slug);
  const { sections, ...page } = doc;
  return {
    ...page,
    markdown: mdPath(slug),
    changed: changedPage(slug),
    group: PAGES[i].group, prev: PAGES[i - 1] ?? null, next: PAGES[i + 1] ?? null,
  };
}
