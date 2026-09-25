<script>
  // The wordmark, docs/brand/kvad-wordmark.svg, and the same word in the
  // younger futhark: ᚴᚢᛅᛏ. There is no v or d rune, so ᚢ stands for the v and
  // ᛏ for the d. Over the link that holds it (an ancestor with the `group`
  // class), the letters sink into a blur and the runes come up out of one, a
  // glyph at a time, the way a skald recites.
  //
  // Each glyph is its own <svg>, stacked on the others, because a CSS filter
  // on an element inside an SVG is not honoured everywhere and one on an
  // <svg> box is. All of them share the wordmark's viewBox, so a glyph is
  // drawn where it stands in the word and the stack needs no positioning.
  //
  // One pen throughout: 4.5 units, round ends, on a centre-line grid of
  // ascender 4, x-height 20 and baseline 50. A glyph is a list of
  // [path, stroke width, opacity].
  const LATIN = [
    [["M4 4 V50 M21 20 L4 38 M10 32.5 L21.5 50", 4.5, 1]],
    [
      ["M48.5 29.5 C46.2 22.1 41.7 13.1 34.9 8.5 M48.5 29.5 C50.8 22.1 55.3 13.1 62.1 8.5", 2.2, 0.45],
      ["M48.5 39 C45.7 30 40.2 19 32 13.5 M48.5 39 C51.3 30 56.8 19 65 13.5", 2.2, 0.7],
      ["M48.5 50 C45.3 39.5 38.8 26.5 29 20 M48.5 50 C51.7 39.5 58.2 26.5 68 20", 4.5, 1],
    ],
    [["M100 20 V50 M100 35 A15 15 0 1 1 70 35 A15 15 0 1 1 100 35", 4.5, 1]],
    [["M142.5 4 V50 M142.5 35 A15 15 0 1 1 112.5 35 A15 15 0 1 1 142.5 35", 4.5, 1]],
  ];
  // Each rune stands where its letter was, full height like the k and the d.
  const RUNES = [
    [["M6 4 V50 M6 25 L21 5", 4.5, 1]],
    [["M38 50 V4 L59 15 V50", 4.5, 1]],
    [["M85 4 V50 M76 35 L94 20", 4.5, 1]],
    [["M128 4 V50 M116 17 L128 4 L140 17", 4.5, 1]],
  ];

  // In pixels. The viewBox is 149 by 55 and the baseline is at 50, so the
  // bottom 5/55 hangs below the baseline; the negative margin puts the
  // baseline where a flex row aligning baselines expects it.
  let { height = 24 } = $props();
</script>

<span
  class="wordmark"
  style="height: {height}px; width: {(height * 149) / 55}px; margin-bottom: {(-height * 5) / 55}px"
  aria-hidden="true"
>
  {#each [["latin", LATIN], ["rune", RUNES]] as [kind, glyphs] (kind)}
    {#each glyphs as glyph, i (i)}
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="-1 0 149 55"
        class="glyph {kind}"
        style="--i: {i}"
        fill="none"
        stroke="currentColor"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        {#each glyph as [d, width, opacity] (d)}
          <path {d} stroke-width={width} {opacity} />
        {/each}
      </svg>
    {/each}
  {/each}
</span>

<style>
  .wordmark {
    position: relative;
    display: inline-block;
    flex-shrink: 0;
  }

  .glyph {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    overflow: visible;
    /* Leaving the link: the runes go back quickly, all at once. */
    transition:
      opacity 350ms ease-in,
      filter 350ms ease-in;
  }

  .rune {
    opacity: 0;
    filter: blur(5px);
  }

  /* Over the link: each letter goes, and then its rune arrives, slower than
     the letter left and 140ms after the glyph before it. */
  :global(.group:hover) .latin,
  :global(.group:focus-visible) .latin {
    opacity: 0;
    filter: blur(3px);
    transition:
      opacity 500ms ease-out calc(var(--i) * 140ms),
      filter 500ms ease-out calc(var(--i) * 140ms);
  }

  :global(.group:hover) .rune,
  :global(.group:focus-visible) .rune {
    opacity: 1;
    filter: blur(0);
    transition:
      opacity 1100ms cubic-bezier(0.2, 0.6, 0.2, 1) calc(150ms + var(--i) * 140ms),
      filter 1100ms cubic-bezier(0.2, 0.6, 0.2, 1) calc(150ms + var(--i) * 140ms);
  }

  /* No mist and no stagger: a plain cross-fade. */
  @media (prefers-reduced-motion: reduce) {
    .glyph,
    :global(.group:hover) .glyph,
    :global(.group:focus-visible) .glyph {
      filter: none;
      transition: opacity 200ms linear;
    }
  }
</style>
