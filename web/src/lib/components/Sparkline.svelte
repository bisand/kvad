<script>
  // A line, small enough to sit inside a number.
  //
  // Inline SVG rather than a charting library: there are no axes, no legend
  // and no interaction, and uPlot is already loaded for the one chart that
  // needs those. The whole job is to say whether the number has been going up
  // or down, which twenty points and a path do.
  let { values = [], height = 32, label = "" } = $props();

  const points = $derived(values.slice(-60));
  const lo = $derived(points.length ? Math.min(...points) : 0);
  const hi = $derived(points.length ? Math.max(...points) : 1);

  // A flat series would divide by zero and, drawn at full scale, would look
  // like noise; draw it down the middle instead.
  const span = $derived(hi - lo || 1);
  const path = $derived(
    points
      .map((v, i) => {
        const x = points.length === 1 ? 50 : (i / (points.length - 1)) * 100;
        const y = points.length === 1 || hi === lo ? 50 : 100 - ((v - lo) / span) * 100;
        return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" "),
  );
</script>

{#if points.length}
  <svg
    viewBox="0 0 100 100"
    preserveAspectRatio="none"
    style:height="{height}px"
    class="text-primary w-full"
    role="img"
    aria-label={label || `${points.length} samples, ${lo.toFixed(1)} to ${hi.toFixed(1)}`}
  >
    <path d={path} fill="none" stroke="currentColor" stroke-width="2" vector-effect="non-scaling-stroke" />
  </svg>
{:else}
  <div style:height="{height}px" class="grid place-items-center text-xs opacity-40">no samples yet</div>
{/if}
