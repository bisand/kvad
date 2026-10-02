<script>
  // Training and validation loss against step.
  //
  // uPlot rather than a charting framework: it is 45 KB, it redraws a growing
  // series without rebuilding it, and a run of 2000 steps evaluating every 250
  // is eight points that have to look right rather than eight thousand that
  // have to be fast. The reason it earns its place is the *live* part — a
  // point arriving every few seconds for an hour.
  //
  // Colours come from daisyUI's CSS variables, read at draw time, so the chart
  // follows the theme instead of having its own opinion about dark mode.
  import uPlot from "uplot";
  import "uplot/dist/uPlot.min.css";

  // `train: false` leaves the training loss out, for a run whose training
  // loss says mostly which noise level a step drew (a LoRA for a diffusion
  // model): the line worth watching there is the validation loss alone.
  let { metrics = [], bestStep = null, steps = null, train = true } = $props();

  let host;
  let chart = null;

  function colour(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  /** [steps, train, val], or [steps, val] — uPlot wants columns, not rows. */
  function columns(rows) {
    const ordered = [...rows].sort((a, b) => a.step - b.step);
    return [
      ordered.map((m) => m.step),
      ...(train ? [ordered.map((m) => m.train_loss)] : []),
      ordered.map((m) => m.val_loss),
    ];
  }

  /**
   * Tick labels with as many decimals as the ticks are apart. uPlot's own
   * stop at three, which is right for a text run's loss, around 1, and
   * labels every tick of a diffusion run's, around 0.007, the same.
   */
  function ticks(u, values) {
    const gap = values.length > 1 ? Math.abs(values[1] - values[0]) : 1;
    const places = Math.min(6, Math.max(0, Math.ceil(-Math.log10(gap) - 1e-9)));
    return values.map((v) => v.toFixed(places));
  }

  function build() {
    if (!host) return;
    chart?.destroy();
    const ink = colour("--color-base-content", "#666");
    const line = `color-mix(in oklch, ${ink} 18%, transparent)`;
    chart = new uPlot(
      {
        width: host.clientWidth || 600,
        height: 240,
        padding: [12, 12, 0, 0],
        legend: { live: true },
        cursor: { y: false },
        scales: {
          x: {
            time: false,
            // The axis is the whole run, not the part of it that has
            // happened yet. Letting uPlot fit the data meant the axis
            // rescaled every time a checkpoint landed — 0–200 at step 100,
            // 100–200 at step 200 — so the curve kept changing shape while
            // somebody watched it, which is the one thing a live chart is
            // for. Read at draw time rather than baked in, so a rebuild is
            // not needed when a different run is opened.
            range: (u, lo, hi) => (steps > 0 ? [0, steps] : [lo, hi]),
          },
        },
        axes: [
          { stroke: ink, grid: { stroke: line }, ticks: { stroke: line }, label: "step" },
          { stroke: ink, grid: { stroke: line }, ticks: { stroke: line }, label: "loss", values: ticks, size: 64 },
        ],
        series: [
          { label: "step" },
          ...(train ? [{ label: "train", stroke: colour("--color-primary", "#4f46e5"), width: 2 }] : []),
          {
            label: "validation",
            value: (u, v) => (v == null ? "--" : v.toPrecision(4)),
            stroke: colour("--color-secondary", "#0891b2"),
            width: 2,
            // The step whose model is on disk, marked where it happened.
            points: {
              show: true,
              size: (u, i) => (u.data[0][i] === bestStep ? 9 : 4),
            },
          },
        ],
      },
      columns(metrics),
      host,
    );
  }

  $effect(() => {
    // Rebuilt when the theme changes, because the colours were read once.
    void metrics.length;
    void bestStep;
    // Which lines there are is decided when the chart is built.
    if (chart && chart.series.length !== (train ? 3 : 2)) build();
    else if (!chart) build();
    else chart.setData(columns(metrics));
  });

  $effect(() => {
    const resize = () => chart && host && chart.setSize({ width: host.clientWidth, height: 240 });
    const themed = new MutationObserver(build);
    window.addEventListener("resize", resize);
    themed.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      window.removeEventListener("resize", resize);
      themed.disconnect();
      chart?.destroy();
      chart = null;
    };
  });
</script>

<div class="w-full" bind:this={host}></div>

<style>
  /* uPlot draws its own legend and axis labels; these follow the theme. */
  :global(.u-legend),
  :global(.u-label) {
    color: var(--color-base-content);
    font-size: 0.75rem;
  }
  :global(.u-legend) {
    opacity: 0.8;
  }
</style>
