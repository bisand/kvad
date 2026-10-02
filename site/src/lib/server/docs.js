// Reads a documentation page from wherever it lives and renders it.

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { PAGES } from "#lib/docs-nav.js";
import { BLOB, SITE } from "#lib/site.js";
import { frontmatter, render, rewrite } from "./md.js";

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
      // For search engines and link previews only; the page does not print it.
      summary: page.summary ?? "",
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

function file(page) {
  return page.source
    ? resolve(REPO_DOCS, page.source)
    : resolve(GUIDE, `${page.slug.replaceAll("/", "-") || "index"}.md`);
}

/** The path of a page's Markdown on the site: `/docs/install.md`. */
export const mdPath = (slug) => `/docs/${slug || "index"}.md`;

/**
 * A page as Markdown, for a reader that would rather have that than HTML.
 * The same text the page is rendered from, under its title, with every link
 * made absolute so it means the same thing wherever the file ends up.
 */
export function markdown(slug) {
  const page = PAGES.find((p) => p.slug === slug);
  if (!page) return null;
  const doc = load(slug);
  let { body } = frontmatter(readFileSync(file(page), "utf8"));
  body = body
    .replace(/^\[← Back to the README\]\([^)]*\)\s*/m, "")
    .replace(/^# .*\n+/, "")
    .replace(/(\]\()([^)\s]+)(\))/g, (_, a, url, b) => {
      const to = rewrite(url, page.source);
      return a + (to.startsWith("/") ? SITE + to : to) + b;
    });
  const lede = doc.description ? `> ${doc.description}\n\n` : "";
  return `# ${doc.title}\n\n${lede}${body.trim()}\n`;
}

/**
 * The day a file last changed, from git, as YYYY-MM-DD. Empty when git
 * cannot say: outside a checkout, or in one with no history.
 */
export function changed(path) {
  try {
    return execFileSync("git", ["log", "-1", "--format=%cs", "--", path], { encoding: "utf8" }).trim();
  } catch {
    return "";
  }
}

export const changedPage = (slug) => {
  const page = PAGES.find((p) => p.slug === slug);
  return page ? changed(file(page)) : "";
};
