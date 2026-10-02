// The front page quotes the engine. It reads the function out of the source
// at build time, so what the page shows is what the repository holds.

import { readFileSync } from "node:fs";
import hljs from "highlight.js/lib/common";

function excerpt() {
  try {
    const src = readFileSync("../crates/llm/src/tensor.rs", "utf8");
    const from = src.indexOf("/// RMSNorm:");
    const to = src.indexOf("\n}\n", src.indexOf("pub fn rms_norm"));
    if (from < 0 || to < 0) return "";
    return hljs.highlight(src.slice(from, to + 2), { language: "rust" }).value;
  } catch {
    return "";
  }
}

function version() {
  try {
    return /^version\s*=\s*"([^"]+)"/m.exec(readFileSync("../Cargo.toml", "utf8"))[1];
  } catch {
    return "";
  }
}

export const load = () => ({ excerpt: excerpt(), version: version() });
