// Vite builds the UI into `dist`, which `kvad-serve` embeds with rust-embed.
//
// Two things here are load-bearing:
//
//   * `base: "./"` — nothing, so assets are referenced from the root. The
//     server is the root; there is no sub-path deployment to support.
//   * the dev proxy — in development this dev server owns the page and axum
//     owns `/api` and `/v1`. Proxying them keeps the browser on one origin,
//     so cookies and the `Origin` check that Phase 3 adds behave in
//     development exactly as they do in production.
import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";

// Where axum is listening. Matches `server.bind`'s default in kvad.toml.
const API = process.env.KVAD_API ?? "http://127.0.0.1:5823";

// `dist` has to exist even when empty, or `rust-embed` fails to compile and
// `cargo test` needs npm after all. `.gitkeep` is what makes git carry an
// otherwise-ignored directory, and `emptyOutDir` deletes it along with
// everything else, so write it back.
const keepDist = {
  name: "kvad-keep-dist",
  closeBundle() {
    writeFileSync(resolve(import.meta.dirname, "dist/.gitkeep"), "");
  },
};

export default defineConfig({
  plugins: [svelte(), tailwindcss(), keepDist],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: API, changeOrigin: false },
      "/v1": { target: API, changeOrigin: false },
    },
  },
  build: {
    // The server caches everything under `assets/` forever and `index.html`
    // never; see `assets.rs`. Vite fingerprints these names, which is what
    // makes that safe.
    outDir: "dist",
    emptyOutDir: true,
  },
});
