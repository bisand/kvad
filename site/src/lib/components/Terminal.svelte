<script>
  // A terminal window holding real output. `lines` is text; a line that
  // starts with "$ " is a command and is drawn as one. `html` is for a
  // capture that carries its own colours (the terminal app's screens).
  let { title = "", lines = "", html = "", label = "Terminal output", small = false } = $props();

  // A command, a line that continues one after a backslash, or output.
  const rows = $derived.by(() => {
    let cont = false;
    return lines
      .replace(/\n$/, "")
      .split("\n")
      .map((text) => {
        const cmd = text.startsWith("$ ");
        const kind = cmd ? "cmd" : cont ? "more" : "out";
        cont = (cmd || cont) && text.endsWith("\\");
        return { text, kind };
      });
  });
</script>

<figure class="term" class:small aria-label={label}>
  <div class="bar" aria-hidden="true">
    <i></i><i></i><i></i>
    <span>{title}</span>
  </div>
  {#if html}
    <pre class="screen">{@html html}</pre>
  {:else}
    <pre>{#each rows as row, i (i)}{#if row.kind === "cmd"}<span class="cmd"><span class="p">$</span> {row.text.slice(2)}</span>{:else}<span class={row.kind === "more" ? "cmd" : "out"}>{row.text}</span>{/if}{"\n"}{/each}</pre>
  {/if}
</figure>

<style>
  /* A terminal is dark in either theme, as a terminal is. */
  .term {
    margin: 0;
    border-radius: 10px;
    border: 1px solid rgb(239 227 200 / 0.14);
    background: #12151f;
    color: #ebe5d6;
    overflow: hidden;
    box-shadow: var(--shadow);
  }
  .bar {
    display: flex;
    align-items: center;
    gap: 7px;
    height: 34px;
    padding: 0 13px;
    border-bottom: 1px solid rgb(239 227 200 / 0.1);
    background: #1b1f2e;
  }
  .bar i {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: rgb(239 227 200 / 0.16);
  }
  .bar span {
    margin-left: 8px;
    font: 400 12px/1 var(--mono);
    color: #8d887d;
  }
  pre {
    margin: 0;
    padding: 16px 18px 18px;
    overflow-x: auto;
    font: 400 12.75px/1.62 var(--mono);
    tab-size: 4;
  }
  .small pre {
    font-size: 11.5px;
  }
  /* Menlo first: the app draws its frames with box characters, which IBM
     Plex Mono does not have, and a fallback glyph of another width breaks
     the lines. */
  .screen {
    font-family: Menlo, "SF Mono", Consolas, "DejaVu Sans Mono", monospace;
    font-size: 12px;
    line-height: 1.25;
    padding: 12px 14px;
  }
  .cmd {
    color: #ebe5d6;
    font-weight: 500;
  }
  .p {
    color: #f0b35a;
    user-select: none;
  }
  .out {
    color: #aaa496;
  }
</style>
