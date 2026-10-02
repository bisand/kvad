import { sveltekit } from "@sveltejs/kit/vite";
import adapter from "@sveltejs/adapter-static";
import { defineConfig } from "vite";

// A static site: every page is rendered at build time into site/build, and
// GitHub Pages serves the files. 404.html is what Pages shows for a path
// that has no file.
export default defineConfig({
  plugins: [
    sveltekit({
      adapter: adapter({ pages: "build", assets: "build", fallback: "404.html" }),
      // The long-form docs link to each other's headings. One that has been
      // renamed should be reported, and should not stop the site building.
      prerender: { handleMissingId: "warn" },
    }),
  ],
  // The pages read ../docs and ../install.sh, which are outside this folder.
  server: { fs: { allow: [".."] } },
});
