<script>
  // A QR code as one SVG path. Always dark on white, whatever the theme:
  // a camera reads the modules' contrast, and some readers will not take a
  // light-on-dark code at all. Four modules of quiet zone, as the standard
  // asks.
  import { encode } from "../qr.js";

  // `size` is roughly how wide to draw it. Each module gets a whole number of
  // pixels, at least three: a pairing link is version 19 or so, 101 modules
  // across with the margin, and fractional modules blur into each other on
  // the screen a camera is reading.
  let { text, size = 300 } = $props();

  const QUIET = 4;
  let modules = $derived(encode(text));
  let span = $derived(modules.length + QUIET * 2);
  let width = $derived(span * Math.max(3, Math.floor(size / span)));
  let path = $derived(
    modules
      .flatMap((row, y) => row.map((dark, x) => (dark ? `M${x + QUIET} ${y + QUIET}h1v1h-1z` : "")))
      .join(""),
  );
</script>

<svg
  xmlns="http://www.w3.org/2000/svg"
  {width}
  height={width}
  viewBox={`0 0 ${span} ${span}`}
  shape-rendering="crispEdges"
  role="img"
  aria-label="QR code of the pairing link"
  class="rounded-box"
>
  <rect width={span} height={span} fill="#fff" />
  <path d={path} fill="#000" />
</svg>
