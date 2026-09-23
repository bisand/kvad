// Markdown, for what a model says.
//
// Instruction-tuned models answer in Markdown whether or not anybody asked
// them to, and shown as plain text that is literal `**bold**`, `###` before
// every heading, and code as a wall between two rows of backticks.
//
// The output is untrusted. A model can be talked into writing
// `<img src=x onerror=…>` as easily as anything else, and rendering means
// `{@html}`, which hands it the page — so everything `marked` produces goes
// through DOMPurify before it reaches the DOM, and there is no path around it.

import { Marked } from "marked";
import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/core";

// The core build and a chosen few, rather than all ~190 languages: the bundle
// ships inside the server binary. These are what a model is asked about here.
import bash from "highlight.js/lib/languages/bash";
import c from "highlight.js/lib/languages/c";
import cpp from "highlight.js/lib/languages/cpp";
import css from "highlight.js/lib/languages/css";
import diff from "highlight.js/lib/languages/diff";
import go from "highlight.js/lib/languages/go";
import ini from "highlight.js/lib/languages/ini";
import java from "highlight.js/lib/languages/java";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import python from "highlight.js/lib/languages/python";
import rust from "highlight.js/lib/languages/rust";
import sql from "highlight.js/lib/languages/sql";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";

for (const [name, lang] of Object.entries({
  bash, c, cpp, css, diff, go, ini, java, javascript, json, markdown, python,
  rust, sql, typescript, xml, yaml,
})) {
  hljs.registerLanguage(name, lang);
}
// What models write after the fence, mapped to what is registered.
hljs.registerAliases(["sh", "shell", "zsh", "console"], { languageName: "bash" });
hljs.registerAliases(["toml"], { languageName: "ini" });
hljs.registerAliases(["html", "svg"], { languageName: "xml" });
hljs.registerAliases(["js", "jsx", "svelte"], { languageName: "javascript" });
hljs.registerAliases(["ts", "tsx"], { languageName: "typescript" });
hljs.registerAliases(["py"], { languageName: "python" });
hljs.registerAliases(["rs"], { languageName: "rust" });
hljs.registerAliases(["yml"], { languageName: "yaml" });

const escape = (s) =>
  s.replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);

const marked = new Marked({
  gfm: true,
  // A single newline is a line break. Models write one where they mean one,
  // and CommonMark's reading — the same paragraph — runs their lists of
  // short lines together.
  breaks: true,
  renderer: {
    // HTML the model wrote is shown, not run. Sanitising alone would still
    // let it through with its `class`es, and this page's classes include
    // `fixed inset-0` — enough to lay a convincing panel over the app. With
    // raw HTML escaped, every tag and class on the page came from this file.
    html({ text }) {
      return escape(text);
    },
    // An image becomes a link to it. Loading whatever URL a model writes
    // would tell that host who is reading, and when; a link is only
    // followed by somebody who chose to.
    image({ href, text }) {
      return `<a href="${escape(href ?? "")}">image: ${escape(text || href || "")}</a>`;
    },
    // A fenced block gets a bar with its language and a copy button. The
    // button is plain markup with a data attribute: it cannot carry a handler
    // through the sanitiser, and should not, so `Chat.svelte` listens for it
    // on the bubble instead.
    code({ text, lang }) {
      const name = (lang || "").trim().split(/\s+/)[0].toLowerCase();
      const known = name && hljs.getLanguage(name);
      // Unlabelled code stays plain rather than guessed at: auto-detection is
      // the slow part of the library, it runs on every streamed token, and it
      // guesses wrong often enough to colour prose as if it were Perl.
      const body = known ? hljs.highlight(text, { language: name }).value : escape(text);
      return (
        `<div class="md-code">` +
        `<div class="md-code-bar"><span>${escape(name || "text")}</span>` +
        `<button type="button" class="btn btn-ghost btn-xs" data-copy>Copy</button></div>` +
        `<pre><code class="hljs">${body}</code></pre></div>`
      );
    },
  },
});

// Every link leaves the app rather than replacing it, and takes nothing with
// it. Set after sanitising, so the sanitiser never has to be told to allow a
// `target` from the model — the model's own attributes are gone by then.
DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A" && node.hasAttribute("href")) {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer");
  }
});

/** A model's reply as sanitised HTML. Safe to hand to `{@html}`. */
export function render(src) {
  // Mid-stream, the reply is whatever has arrived. An unclosed fence renders
  // as code to the end — which it is, so far — and a lone `*` stays a `*`.
  return DOMPurify.sanitize(marked.parse(src ?? ""), {
    // The copy button's hook, and nothing else from the `data-` family.
    ALLOWED_ATTR: ["href", "title", "class", "data-copy", "type", "align", "start"],
    FORBID_TAGS: ["style", "form", "input", "textarea", "select", "iframe", "object", "embed"],
  });
}

/**
 * The bubble's click handler: copies a code block when its button is pressed.
 *
 * One listener on the bubble rather than one per button, because the buttons
 * are rebuilt from HTML on every streamed token.
 */
export async function copyFromBlock(event) {
  const button = event.target.closest?.("[data-copy]");
  if (!button) return;
  const code = button.closest(".md-code")?.querySelector("code");
  if (!code) return;
  try {
    await navigator.clipboard.writeText(code.textContent);
    button.textContent = "Copied";
  } catch {
    button.textContent = "Copy failed";
  }
  setTimeout(() => (button.textContent = "Copy"), 1500);
}
