// The documentation's table of contents, in reading order.
//
// A guide page is `content/docs/<slug>.md`. An internals page is one of the
// repository's own `docs/*.md`, rendered as it is: `source` names the file,
// and `summary` describes it to a search engine, since the file has no
// description of its own.
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
      { slug: "internals/nervus", title: "1. nervus", source: "nervus.md",
        summary: "A neural network and backpropagation from scratch in Rust with no dependencies, then a GPT trained on a text file: gradient checks, checkpoints, and where the time went.",
      },
      { slug: "internals/engine", title: "2. kvad", source: "engine.md",
        summary: "Transformer inference written by hand: six architectures as plugins, block-wise quantisation, the integer and float kernels, batched prefill and the KV cache.",
      },
      { slug: "internals/gpu", title: "3. kvad-gpu", source: "gpu.md",
        summary: "The same forward passes on Metal: numbers against the CPU, quantised weights on the GPU, and the image models, written out rather than imported.",
      },
      { slug: "internals/tui", title: "4. kvad-tui", source: "tui.md",
        summary: "The terminal app: three threads, a KV cache kept across chat turns, and six backends on one key.",
      },
      { slug: "internals/serve", title: "5. kvad-serve", source: "serve.md",
        summary: "The HTTP server and web UI: reasoning output, tool calls, model residency, the default backend, and benchmarks that refuse to mislead.",
      },
      { slug: "internals/tune", title: "Training a LoRA", source: "tune.md",
        summary: "How Kvad trains an SDXL LoRA on a folder of pictures: what a step does, why validation loss picks the step that is kept, and what a run costs.",
      },
      { slug: "internals/roadmap", title: "Roadmap", source: "roadmap.md",
        summary: "What has to be built before Kvad is a serving engine, in dependency order: a paged KV cache, continuous batching, then the hardware.",
      },
    ],
  },
];

export const PAGES = NAV.flatMap((g) => g.pages.map((p) => ({ ...p, group: g.title })));

export const href = (slug) => (slug ? `/docs/${slug}/` : "/docs/");
