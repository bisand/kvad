// Copies what the site serves from the rest of the repository into static/,
// before a build or a dev server starts. Both destinations are ignored by
// git: the repository's copy is the one that is edited.
//
//   install.sh        so `curl https://kvad.eu/install.sh | sh` is this file
//   docs/digits, …    the pictures the long-form docs show

import { cpSync, mkdirSync, rmSync } from "node:fs";

mkdirSync("static/internals", { recursive: true });
cpSync("../install.sh", "static/install.sh");
for (const dir of ["digits", "video"]) {
  rmSync(`static/internals/${dir}`, { recursive: true, force: true });
  cpSync(`../docs/${dir}`, `static/internals/${dir}`, { recursive: true });
}
