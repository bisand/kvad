// The documentation's table of contents, in reading order.
//
// A guide page is `content/docs/<slug>.md`. An internals page is one of the
// repository's own `docs/*.md`, rendered as it is: `source` names the file.
// The sidebar, the previous/next links and the search index are all this list.

export const NAV = [
  {
    title: "Get started",
    pages: [
      { slug: "", title: "Introduction" },
      { slug: "install", title: "Install" },
      { slug: "quickstart", title: "Quick start" },
      { slug: "platforms", title: "Platforms" },
    ],
  },
  {
    title: "Use",
    pages: [
      { slug: "models", title: "Models and backends" },
      { slug: "chat", title: "Chat and completion" },
      { slug: "web-ui", title: "The web UI" },
      { slug: "terminal-app", title: "The terminal app" },
      { slug: "images", title: "Images" },
      { slug: "video", title: "Video" },
      { slug: "lora", title: "LoRAs" },
      { slug: "training", title: "Train your own" },
    ],
  },
  {
    title: "Serve",
    pages: [
      { slug: "server", title: "The server" },
      { slug: "api", title: "OpenAI-compatible API" },
      { slug: "agents", title: "Tool calls and agents" },
      { slug: "authentication", title: "Authentication" },
      { slug: "configuration", title: "Configuration" },
      { slug: "service", title: "Background service" },
    ],
  },
  {
    title: "Reference",
    pages: [
      { slug: "cli", title: "Command line" },
      { slug: "benchmarks", title: "Benchmarks and evals" },
      { slug: "troubleshooting", title: "Troubleshooting" },
    ],
  },
  {
    title: "How it works",
    pages: [
      { slug: "internals", title: "Reading the engine" },
      { slug: "internals/nervus", title: "1. nervus", source: "nervus.md" },
      { slug: "internals/engine", title: "2. kvad", source: "engine.md" },
      { slug: "internals/gpu", title: "3. kvad-gpu", source: "gpu.md" },
      { slug: "internals/tui", title: "4. kvad-tui", source: "tui.md" },
      { slug: "internals/serve", title: "5. kvad-serve", source: "serve.md" },
      { slug: "internals/tune", title: "Training a LoRA", source: "tune.md" },
      { slug: "internals/roadmap", title: "Roadmap", source: "roadmap.md" },
    ],
  },
];

export const PAGES = NAV.flatMap((g) => g.pages.map((p) => ({ ...p, group: g.title })));

export const href = (slug) => (slug ? `/docs/${slug}/` : "/docs/");
