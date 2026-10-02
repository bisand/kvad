// Markdown to HTML, at build time.
//
// Two kinds of file come through here: the guide under content/docs, and the
// repository's own docs/*.md. The second kind was written to be read on
// GitHub, so its links are relative to docs/ and its anchors are GitHub's.
// `rewrite` turns those into this site's addresses, and `slug` makes the
// same anchors GitHub does so a `#fragment` written there lands here too.

import { Marked } from "marked";
import hljs from "highlight.js/lib/common";
import { BLOB } from "#lib/site.js";
import { PAGES, href } from "#lib/docs-nav.js";

const BY_SOURCE = new Map(PAGES.filter((p) => p.source).map((p) => [p.source, p.slug]));
BY_SOURCE.set("install.md", "install");

const esc = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

// GitHub's rule: lower case, drop everything but letters, digits, spaces,
// hyphens and underscores, then a hyphen per space. Two spaces give two.
export function slug(text) {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s_-]/gu, "")
    .replace(/\s/g, "-");
}

export function rewrite(url, source) {
  if (!source || /^([a-z]+:|\/|#)/i.test(url)) return url;
  const [path, frag] = url.split("#");
  const tail = frag ? `#${frag}` : "";
  if (path.startsWith("../")) return `${BLOB}/${path.slice(3)}${tail}`;
  if (BY_SOURCE.has(path)) return href(BY_SOURCE.get(path)) + tail;
  if (/\.(png|gif|jpe?g|svg|webp)$/i.test(path)) return `/internals/${path}`;
  return `${BLOB}/docs/${path}${tail}`;
}

const plain = (tokens) =>
  tokens.map((t) => (t.tokens ? plain(t.tokens) : (t.text ?? t.raw ?? ""))).join("");

export function frontmatter(src) {
  const m = /^---\n([\s\S]*?)\n---\n/.exec(src);
  if (!m) return { meta: {}, body: src };
  const meta = {};
  for (const line of m[1].split("\n")) {
    const i = line.indexOf(":");
    if (i > 0) meta[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return { meta, body: src.slice(m[0].length) };
}

/**
 * @param {string} src     the Markdown
 * @param {string} [source] the file's name under docs/, for a file from there
 */
export function render(src, source) {
  const toc = [];
  const sections = [{ id: "", heading: "", text: "" }];
  const seen = new Map();
  let title = "";
  let titleId = "";

  const marked = new Marked({ gfm: true });
  marked.use({
    // In document order, before anything is rendered: headings get their
    // ids here, so the text after one is filed under it for the search.
    walkTokens(t) {
      if (t.type === "link" || t.type === "image") t.href = rewrite(t.href, source);
      if (t.type === "heading" && t.depth > 1) {
        const text = plain(t.tokens);
        let id = slug(text);
        const n = seen.get(id) ?? 0;
        seen.set(id, n + 1);
        if (n) id += `-${n}`;
        t.id = id;
        if (t.depth <= 3) toc.push({ id, text, depth: t.depth });
        if (t.depth === 2) sections.push({ id, heading: text, text: "" });
      }
      const block = t.type === "paragraph" || t.type === "code" || (t.type === "text" && t.tokens);
      if (block) sections.at(-1).text += " " + t.text;
    },
    renderer: {
      heading({ tokens, depth, id }) {
        // The page draws its own h1. It keeps the id this one would have
        // had, because other files link to it.
        if (depth === 1) {
          if (!title) {
            title = plain(tokens);
            titleId = slug(title);
          }
          return "";
        }
        const inner = this.parser.parseInline(tokens);
        return `<h${depth} id="${id}"><a class="anchor" href="#${id}" aria-label="Link to this section">#</a>${inner}</h${depth}>\n`;
      },
      code({ text, lang }) {
        const name = (lang || "").split(/\s/)[0];
        const known = name && hljs.getLanguage(name);
        const body = known ? hljs.highlight(text, { language: name }).value : esc(text);
        return `<div class="code" data-lang="${esc(name)}"><pre><code>${body}</code></pre></div>\n`;
      },
      link({ href, title, tokens }) {
        const out = /^https?:/.test(href) ? ' rel="noopener"' : "";
        const t = title ? ` title="${esc(title)}"` : "";
        return `<a href="${esc(href)}"${t}${out}>${this.parser.parseInline(tokens)}</a>`;
      },
      image({ href, title, text }) {
        return `<img src="${esc(href)}" alt="${esc(text)}" loading="lazy"${title ? ` title="${esc(title)}"` : ""}>`;
      },
    },
  });

  // The line every file under docs/ opens with, which means nothing here.
  const body = src.replace(/^\[← Back to the README\]\([^)]*\)\s*/m, "");
  const html = marked
    .parse(body)
    .replace(/<table>/g, '<div class="table"><table>')
    .replace(/<\/table>/g, "</table></div>");

  for (const s of sections) s.text = s.text.replace(/\s+/g, " ").trim();
  return { html, toc, title, titleId, sections: sections.filter((s) => s.text) };
}
