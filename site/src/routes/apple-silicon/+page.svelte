<script>
  import { REPO } from "#lib/site.js";
  import { reveal } from "#lib/reveal.js";
  import Install from "#lib/components/Install.svelte";
  import Meta from "#lib/components/Meta.svelte";
  import Terminal from "#lib/components/Terminal.svelte";
  import { BENCH } from "#lib/captures/cli.js";

  const MEMORY = [
    ["Qwen2.5-0.5B", "q8", "0.6 GB"],
    ["Qwen2.5-1.5B", "q8", "1.6 GB"],
    ["Stable Diffusion 1.5", "f16", "1.9 GB"],
    ["SDXL", "f16", "6.4 GB"],
    ["Qwen3-14B", "q8", "14.6 GB"],
    ["DeepSeek-V2-Lite, 15.7 B parameters", "q8", "17.7 GB"],
    ["FLUX.1-schnell", "q8", "18.3 GB"],
    ["Qwen-Image, 20 B parameters", "q8", "29.2 GB"],
  ];
</script>

<Meta
  title="Kvad on Apple silicon · kernels for the M5's GPU"
  description="What Kvad does with an M5: Metal kernels for the GPU's matrix units, what they are worth for prefill, images and video, what they are not worth for decode, and how much model fits in unified memory."
  path="/apple-silicon/"
/>

<section class="wrap hero">
  <p class="label">Apple silicon</p>
  <h1>Built on a MacBook Pro, for the chip inside it.</h1>
  <p class="lede">
    Kvad is developed and measured on an M5 Pro. Its GPU engine is written for Metal and nothing
    else, and it has kernels of its own for the part of the M5's GPU that general-purpose shaders
    never touch.
  </p>
</section>

<section class="wrap split" use:reveal>
  <div class="say">
    <h2>The matrix units</h2>
    <p>
      The M5's GPU carries hardware for matrix products. An ordinary Metal compute shader does not
      use it: it is reached only through Metal 4's tensor operations. The framework Kvad's GPU
      engine is built on stops at about 7.5 TFLOP/s on this machine whatever it is given, because
      its shaders are the ordinary kind.
    </p>
    <p>
      So Kvad has its own. One kernel takes dense half-precision weights. Another takes quantised
      weights, unpacks each slab on the GPU, and hands it to the same hardware, which matters
      because the models worth running locally are the quantised ones.
    </p>
    <p>
      They are compiled when a model loads and used when the GPU reports the Apple10 family. On an
      M1 to M4 the same model takes the standard path, with nothing to configure.
    </p>
  </div>
  <div class="table">
    <table>
      <thead>
        <tr><th>Per matrix product</th><th>Against the stock kernel</th></tr>
      </thead>
      <tbody>
        <tr><td>Dense, 96 rows or more</td><td>3.0–3.9×</td></tr>
        <tr><td>Dense, 8 rows</td><td>1.2–1.6×</td></tr>
        <tr><td>Quantised, 8-bit</td><td>2.3–2.7×</td></tr>
        <tr><td>GGUF k-quants (Q4_K, Q5_K, Q6_K)</td><td>75–90% of the 8-bit rate</td></tr>
        <tr><td>One row, as in decode</td><td>0.87–0.95×, so not used</td></tr>
      </tbody>
    </table>
    <p class="note">
      Timed in alternating rounds against the stock kernel, because this machine drifts by up to a
      third between runs. <a href="{REPO}/issues/52" rel="noopener">The probe and its results</a>
    </p>
  </div>
</section>

<section class="band" use:reveal>
  <div class="wrap">
    <header class="head">
      <h2>What that is worth end to end</h2>
      <p>
        A model is more than its matrix products, so the whole-step numbers are smaller than the
        kernel's. These are the ones that matter.
      </p>
    </header>
    <div class="table">
      <table>
        <thead>
          <tr><th>On an M5 Pro</th><th>Before</th><th>After</th><th></th></tr>
        </thead>
        <tbody>
          <tr><td>Prefill, Qwen2.5-1.5B at bf16, 2,079 tokens</td><td>0.99 s</td><td>0.39 s</td><td>2.5×</td></tr>
          <tr><td>FLUX.1-schnell at q8, a step at 1024²</td><td>12.9–13.7 s</td><td>8.7–9.2 s</td><td>1.5×</td></tr>
          <tr><td>SDXL at f16, a step at 1024²</td><td>3.94 s</td><td>3.12 s</td><td>1.26×</td></tr>
          <tr><td>Qwen-Image with a LoRA applied, a step</td><td colspan="2">4.5% slower than without one</td><td></td></tr>
        </tbody>
      </table>
    </div>
    <p class="note">
      SDXL gains least because most of its time is convolution, which these kernels do not touch.
    </p>
  </div>
</section>

<section class="wrap split wide" use:reveal>
  <div class="say">
    <h2>What did not get faster</h2>
    <p>
      Decoding. Writing a reply is one token at a time, and each token reads every weight in the
      model once. That is bound by how fast memory can be read, not by arithmetic, and no matrix
      unit changes it. On one row the new kernel is slower than the stock one, so Kvad does not use
      it there.
    </p>
    <p>
      What helps decode is fewer bytes per weight and fewer trips to the GPU. Quantising to 4 bits
      does the first. Fusing a layer's small operations into three Metal kernels does the second,
      and took a 1.5 B model's layer from 25 kernel launches to 10: 115 to 127.5 tokens a second at
      q8.
    </p>
    <p>
      The benchmark on the right is from the day this page was written. The GPU at q4 decodes 4.4
      times as fast as the CPU at q8, and 1.6 times as fast as the GPU at q8, for the same model.
    </p>
  </div>
  <Terminal title="zsh" lines={BENCH} small label="A benchmark of Qwen2.5-1.5B on three backends" />
</section>

<section class="wrap split" use:reveal>
  <div class="say">
    <h2>Unified memory is the budget</h2>
    <p>
      On a Mac the GPU and the CPU share one pool, so the question is never how much VRAM a card
      has. It is how much of the machine a model may take. By default Kvad lets models use three
      quarters of it and leaves the rest to you.
    </p>
    <p>
      A model is charged for its weights and for a full context of KV cache before it is admitted,
      so one that loads can always finish. One that does not fit is refused, by name, rather than
      loaded into swap.
    </p>
    <p>
      The table is what each model holds once loaded, measured. Add the KV cache for a language
      model: Qwen3-14B is charged 5 GB for 32,768 tokens of it.
    </p>
  </div>
  <div class="table">
    <table>
      <thead>
        <tr><th>Model</th><th>Precision</th><th>In memory</th></tr>
      </thead>
      <tbody>
        {#each MEMORY as [model, precision, size] (model)}
          <tr><td>{model}</td><td>{precision}</td><td>{size}</td></tr>
        {/each}
      </tbody>
    </table>
  </div>
</section>

<section class="wrap split" use:reveal>
  <div class="say">
    <h2>The CPU is not an afterthought</h2>
    <p>
      Half of Kvad is a CPU engine with no framework under it: block-wise 8-bit and 4-bit weights,
      an integer dot product on Arm's <code>i8mm</code> instructions, a hand-written 4-bit decode
      kernel, and a tiled float matrix product.
    </p>
    <p>
      On an M5 Pro it decodes Qwen2.5-0.5B at 112 tokens a second, and DeepSeek-V2-Lite, 15.7
      billion parameters, at 23. It is also what a Linux machine gets.
    </p>
    <a class="more" href="/docs/internals/engine/">How the CPU engine works <span aria-hidden="true">→</span></a>
  </div>
  <div class="table">
    <table>
      <thead>
        <tr><th>Decode on the CPU, q8</th><th>Tokens a second</th></tr>
      </thead>
      <tbody>
        <tr><td>SmolLM2-135M</td><td>194</td></tr>
        <tr><td>GPT-2 medium</td><td>128</td></tr>
        <tr><td>Qwen2.5-0.5B</td><td>112–117</td></tr>
        <tr><td>Qwen2.5-1.5B</td><td>48.5</td></tr>
        <tr><td>DeepSeek-V2-Lite, 15.7 B parameters</td><td>23–27</td></tr>
      </tbody>
    </table>
    <p class="note">An M5 Pro, 18 cores. The ranges are separate measurements on different days.</p>
  </div>
</section>

<section class="wrap chips" use:reveal>
  <h2>Which Mac</h2>
  <div class="table">
    <table>
      <thead>
        <tr><th>Chip</th><th>What you get</th></tr>
      </thead>
      <tbody>
        <tr><td>M5 family</td><td>Everything on this page. This is what Kvad is measured on.</td></tr>
        <tr><td>M1 to M4</td><td>The same models and features through the standard Metal kernels. Slower prefill and image steps; decode is much the same.</td></tr>
        <tr><td>Intel</td><td>No build.</td></tr>
      </tbody>
    </table>
  </div>
  <p class="note">
    Every number here is from one machine, an M5 Pro with 48 GB in a MacBook Pro. Kvad has not been
    measured on an M5 Max, a Mac Studio or a Mac mini, and this page will say so when it has. If
    you run it on one, the benchmark page will give you numbers worth
    <a href="{REPO}/issues" rel="noopener">sending in</a>.
  </p>
</section>

<section class="wrap end" use:reveal>
  <h2>Try it on yours.</h2>
  <Install />
  <p><a href="/docs/platforms/">Platform details</a> · <a href="/docs/benchmarks/">Run the benchmark yourself</a></p>
</section>

<style>
  section {
    padding-block: clamp(44px, 7vw, 88px);
  }
  .hero {
    display: grid;
    gap: 22px;
    padding-top: clamp(48px, 8vw, 96px);
    max-width: 900px;
    margin-left: max(0px, calc((100% - var(--page) - 2 * var(--gutter)) / 2));
  }
  h1 {
    font-size: clamp(2.4rem, 5.6vw, 4rem);
    line-height: 1.04;
    letter-spacing: -0.02em;
  }
  h2 {
    font-size: clamp(1.7rem, 3vw, 2.2rem);
  }
  .lede {
    font-size: clamp(1.1rem, 1.6vw, 1.26rem);
    line-height: 1.55;
    color: var(--muted);
    max-width: 62ch;
  }
  .split {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: clamp(28px, 5vw, 72px);
    align-items: start;
  }
  .split.wide {
    grid-template-columns: minmax(0, 4fr) minmax(0, 7fr);
  }
  .say {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 16px;
  }
  .say p,
  .head p {
    color: var(--muted);
  }
  .say code {
    padding: 0.1em 0.35em;
    border-radius: 4px;
    background: var(--raised);
  }
  .head {
    display: grid;
    gap: 14px;
    max-width: 720px;
    margin-bottom: 32px;
  }
  .band {
    background: var(--surface);
    border-block: 1px solid var(--line);
  }
  .table {
    min-width: 0;
    overflow-x: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 15px;
  }
  th,
  td {
    text-align: right;
    padding: 13px 0 13px 20px;
    border-bottom: 1px solid var(--line);
    font-variant-numeric: tabular-nums;
  }
  th:first-child,
  td:first-child {
    text-align: left;
    padding-left: 0;
  }
  th {
    font: 500 12px/1.3 var(--mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--faint);
    border-bottom-color: var(--line-strong);
  }
  .band td:nth-child(2) {
    color: var(--muted);
  }
  .band td:last-child {
    font-family: var(--mono);
    color: var(--accent);
  }
  .chips td:last-child {
    text-align: left;
    color: var(--muted);
  }
  .chips th:last-child {
    text-align: left;
  }
  .chips h2 {
    margin-bottom: 24px;
  }
  .note {
    margin-top: 14px;
    font-size: 13.5px;
    line-height: 1.55;
    color: var(--faint);
    max-width: 72ch;
  }
  .note a,
  .end a {
    color: var(--muted);
  }
  .more {
    width: fit-content;
    color: var(--accent);
    font-weight: 500;
    text-decoration: none;
  }
  .more span {
    display: inline-block;
    transition: transform 0.2s var(--ease);
  }
  .more:hover span {
    transform: translateX(4px);
  }
  .end {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 20px;
    justify-items: start;
    padding-bottom: 0;
  }
  .end p {
    font-size: 15px;
    color: var(--faint);
  }
  @media (max-width: 860px) {
    .split,
    .split.wide {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>
