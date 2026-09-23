---
name: web-ui
description: kvad's web UI — Svelte 5 and daisyUI 5 in web/, embedded into kvad-serve. Read before editing anything under web/src, adding an HTTP route, or trying to see a UI change in a browser.
---

# The web UI

`web/` is a Svelte 5 app built by Vite. `crates/serve` bakes `web/dist` into
the binary with `rust-embed`, so a release `kvad-serve` carries the whole UI
and needs nothing beside it.

## Cargo never runs npm

There is no `build.rs` calling Node, deliberately: `cargo test` has to work on
a machine that has never installed npm. So a change under `web/src` is
invisible until `web/dist` is rebuilt **and the Rust binary is rebuilt around
it**. Four steps, and skipping the third is why a change "did not take":

```bash
cd web && npm run build                       # web/dist
cargo build --release -p kvad-serve           # embeds web/dist
cp target/release/kvad-serve ~/.local/bin/    # if running from there
launchctl kickstart -k gui/$(id -u)/net.kvad.serve
```

The service is a launchd agent (`net.kvad.serve`, plist in
`~/Library/LaunchAgents`). `kill` it and launchd restarts the *old* binary;
`kickstart -k` is the one that picks up a new one.

To check a change really landed, grep the built bundle for a string literal
from your markup — identifiers are minified, string literals are not:

```bash
grep -c "some new sentence" web/dist/assets/*.js
```

## daisyUI

daisyUI 5 on Tailwind 4, loaded from `web/src/app.css` with `@plugin "daisyui"`.
There is no `tailwind.config.js`. Prefer daisyUI class names over hand-rolled
Tailwind, and prefer the default variant — `btn`, not `btn-primary` — unless a
colour is carrying meaning. Read the component's own docs before using it; the
daisyUI skill has one file per component.

### Disclosure rows

`collapse` is the component, and it takes a `details`/`summary` as readily as a
div. Use that form:

```svelte
<details class="collapse collapse-arrow rounded-box bg-base-200/40">
  <summary class="collapse-title">…</summary>
  <div class="collapse-content">…</div>
</details>
```

Two things learned the hard way:

- **A radio cannot be unchecked by clicking it.** `collapse` built on radio
  inputs is a one-way door: a row opens and can never be closed, only swapped
  for another. `<details>` toggles. Without a `name` several stay open at once,
  which is what comparing two rows wants.
- **A click on a button inside `<summary>` also toggles the row**, because that
  is what a summary does with a click. Wrap the handler — `Models.svelte` has
  `notToggle` — to take the event and keep it. If you build the `<input>` form
  of `collapse` instead, the input covers the title and controls need
  `relative z-1` to stay above it.

`collapse-title` is one element, so a `<table>` row cannot be one. The model
lists were tables and had to stop being tables for this to work.

## Conventions

- `Icon.svelte` draws every icon: `<Icon path={CONST} size={16} />`, with the
  path as a local `const`.
- `api()` in `lib/api.js` wraps every call. Its errors are whole sentences
  meant to be shown as they are — do not prefix them. The server's own
  messages are worth surfacing: `Spec::from_config` explains that DeepSeek
  publishes V3 in fp8, which is more use than "could not read it".
- Svelte 5 runes. `$state` holds plain objects; replace them rather than
  mutating, or the update does not land.

## Adding an HTTP route

`crates/serve/src/openapi.rs` has an `ENDPOINTS` table, and a test asserts it
matches the router exactly. A new `.route(...)` without an `Endpoint` entry
fails `the_document_and_the_router_describe_the_same_server`. `Access` is
`Anyone`, `SignedIn` or `Admin`.
