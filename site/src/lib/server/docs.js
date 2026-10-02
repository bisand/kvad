// Reads a documentation page from wherever it lives and renders it.

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { PAGES } from "#lib/docs-nav.js";
import { BLOB } from "#lib/site.js";
import { frontmatter, render } from "./md.js";

const GUIDE = resolve("content/docs");
const REPO_DOCS = resolve("../docs");

export function load(slug) {
  const page = PAGES.find((p) => p.slug === slug);
  if (!page) return null;

  if (page.source) {
    const src = readFileSync(resolve(REPO_DOCS, page.source), "utf8");
    const out = render(src, page.source);
    return {
      ...out,
      slug,
      nav: page.title,
      title: out.title.replace(/^Crate \d: /, "").replace(/`/g, ""),
      description: "",
      edit: `${BLOB}/docs/${page.source}`,
    };
  }

  const file = `${slug.replaceAll("/", "-") || "index"}.md`;
  const { meta, body } = frontmatter(readFileSync(resolve(GUIDE, file), "utf8"));
  const out = render(body);
  return {
    ...out,
    slug,
    nav: page.title,
    title: meta.title || out.title || page.title,
    description: meta.description || "",
    edit: `${BLOB}/site/content/docs/${file}`,
  };
}
