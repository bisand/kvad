// /llms.txt: what this site is and where its pages are, as Markdown, for a
// language model. The convention is llmstxt.org's: a title, a summary in a
// quote, then lists of links, each to a page's Markdown.

import { NAV } from "#lib/docs-nav.js";
import { load, mdPath } from "#lib/server/docs.js";
import { REPO, SITE, INSTALL } from "#lib/site.js";

export const prerender = true;

export function GET() {
  const groups = NAV.map((g) => {
    const lines = g.pages.map((p) => {
      const doc = load(p.slug);
      const about = doc.description || doc.summary;
      return `- [${doc.title}](${SITE}${mdPath(p.slug)})${about ? `: ${about}` : ""}`;
    });
    return `## ${g.title}\n\n${lines.join("\n")}`;
  });

  const body = `# Kvad

> Kvad is a local inference engine written in Rust. It runs language, image and video models on Apple silicon and, on the CPU, on Linux, with a command line, a terminal app, a web UI and an OpenAI-compatible HTTP API. It has Metal kernels of its own for the M5's GPU, and every kernel is in a repository meant to be read.

Install it with \`${INSTALL}\`. The source is at ${REPO}, under the MIT licence. It is developed and measured on an M5 Pro; the numbers in these pages are from that machine unless they say otherwise. It runs one generation at a time: there is no continuous batching, and no CUDA build.

- [The front page](${SITE}/)
- [Kvad on Apple silicon](${SITE}/apple-silicon/): what the M5 kernels are worth, and what they are not
- [Every page below in one file](${SITE}/llms-full.txt)

${groups.join("\n\n")}
`;
  return new Response(body, { headers: { "content-type": "text/plain; charset=utf-8" } });
}
