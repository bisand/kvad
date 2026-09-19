// Every page the UI will have, and which phase brings it.
//
// The whole list is here from the start, and the ones that are not built say
// so. A sidebar that grew an item per release would hide the shape of the
// thing; this way the plan in `docs/ui-plan.md` and the navigation are the
// same document, and a page that lies about being ready is impossible.
//
// `phase` is null for a page that is built, and the phase number for one that
// is not; `Unbuilt.svelte` renders the second kind.
//
// `icon` is an SVG path drawn at 24×24 with a 1.5 stroke — see `Icon.svelte`.
// Inline rather than an icon package: ten icons is not worth a dependency.

export const PAGES = [
  {
    path: "/",
    label: "Dashboard",
    phase: null,
    blurb: "Throughput, queue depth, memory and disk, and whatever is running.",
    icon: "M3 13h6v8H3zM3 3h6v7H3zM13 3h8v5h-8zM13 11h8v10h-8z",
  },
  {
    path: "/models",
    label: "Models",
    phase: null,
    blurb: "What is on this machine, what is on the Hub, and which one is loaded.",
    icon: "M12 3 3 7.5 12 12l9-4.5zM3 12l9 4.5 9-4.5M3 16.5 12 21l9-4.5",
  },
  {
    path: "/chat",
    label: "Chat",
    phase: null,
    blurb: "Talk to the loaded model, with the numbers for every reply.",
    icon: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z",
  },
  {
    path: "/playground",
    label: "Playground",
    phase: null,
    blurb: "Raw completion, two models side by side, and what the tokeniser saw.",
    icon: "M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.8-3.8a6 6 0 0 1-7.9 7.9l-6.9 6.9a2.1 2.1 0 0 1-3-3l6.9-6.9a6 6 0 0 1 7.9-7.9z",
  },
  {
    path: "/training",
    label: "Training",
    phase: null,
    blurb: "Start a run, watch the loss curve, read what it writes, stop it.",
    icon: "M3 17l6-6 4 4 8-8M21 7v5h-5",
  },
  {
    path: "/datasets",
    label: "Datasets",
    phase: null,
    blurb: "The text files runs are trained on, and what is in them.",
    icon: "M4 7c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 7v10c0 1.7 3.6 3 8 3s8-1.3 8-3V7M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  },
  {
    path: "/evals",
    label: "Evals",
    phase: null,
    blurb: "Perplexity on held-out text, and prompt suites as regression tests.",
    icon: "M9 11l3 3L22 4M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11",
  },
  {
    path: "/benchmarks",
    label: "Benchmarks",
    phase: null,
    blurb: "Interleaved A/B, five runs, median and range — the protocol, as a button.",
    icon: "M12 20v-6M6 20v-4M18 20V8M3 20h18",
  },
  {
    path: "/monitoring",
    label: "Monitoring",
    phase: null,
    blurb: "Requests, latencies, errors, and the log as it happens.",
    icon: "M22 12h-4l-3 9L9 3l-3 9H2",
  },
  {
    path: "/settings",
    label: "Settings",
    phase: null,
    blurb: "Authentication, API keys, thread counts, sampling defaults.",
    icon: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 9 19.4a1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 4.6 9a1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z",
  },
];

/** The page a path belongs to, or `undefined` for a path we do not serve. */
export function pageFor(path) {
  return PAGES.find((p) => p.path === path);
}
