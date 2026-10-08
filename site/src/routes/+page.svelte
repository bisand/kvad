<script>
  import { REPO, SITE } from "#lib/site.js";
  import { reveal } from "#lib/reveal.js";
  import Install from "#lib/components/Install.svelte";
  import Meta from "#lib/components/Meta.svelte";
  import Shot from "#lib/components/Shot.svelte";
  import Terminal from "#lib/components/Terminal.svelte";
  import { RUN, API } from "#lib/captures/cli.js";
  import tuiChat from "#lib/captures/tui-chat.html?raw";

  let { data } = $props();

  const structured = $derived([
    {
      "@context": "https://schema.org",
      "@type": "SoftwareApplication",
      name: "Kvad",
      description:
        "A local inference engine in Rust for language, image and video models, with Metal kernels for Apple silicon, a web UI, a terminal app and an OpenAI-compatible API.",
      url: SITE,
      applicationCategory: "DeveloperApplication",
      operatingSystem: "macOS, Linux",
      softwareVersion: data.version || undefined,
      license: "https://opensource.org/license/mit",
      codeRepository: REPO,
      downloadUrl: `${REPO}/releases`,
      programmingLanguage: "Rust",
      offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
      author: { "@type": "Person", name: "André Biseth" },
    },
    { "@context": "https://schema.org", "@type": "WebSite", name: "Kvad", url: SITE },
  ]);

  // The four ways in. One is shown at a time, and they cross-fade.
  const WAYS = [
    {
      id: "web",
      name: "Web UI",
      title: "A web UI that is part of the binary",
      text: "Twelve pages over the same server the API runs on: chat, a playground, images, videos, training, evals, benchmarks and monitoring. Nothing to install beside it, and it is there when the service starts.",
      link: ["/docs/web-ui/", "The web UI"],
    },
    {
      id: "tui",
      name: "Terminal app",
      title: "A terminal app for when you are already there",
      text: "Search Hugging Face, download, load and chat without leaving the terminal. One key cycles a model through six backends, which is the quickest way to feel what quantisation costs and what the GPU buys.",
      link: ["/docs/terminal-app/", "The terminal app"],
    },
    {
      id: "cli",
      name: "Command line",
      title: "A command line that knows about the server",
      text: "Every route the API has is a command. With the service running, the command line sends its work there instead of loading a second copy of the weights, and the first line of output says which happened.",
      link: ["/docs/cli/", "Command reference"],
    },
    {
      id: "api",
      name: "API",
      title: "OpenAI's API, on your own address",
      text: "Chat completions with streaming, reasoning and tool calls, plus images and video, in the shapes existing SDKs already send. Point a client at port 5823 and change nothing else.",
      link: ["/docs/api/", "API guide"],
    },
  ];
  let way = $state("web");

  const GALLERY = [
    ["lighthouse", "A lighthouse on a rocky coast at dusk"],
    ["fox", "A red fox in fresh snow among birch trees"],
    ["church", "A wooden stave church in a green valley, in watercolour"],
    ["book", "An open book on a desk by a window, lit by a candle, in oils"],
  ];

  const CRATES = [
    ["nervus", "A neural network and backpropagation, from scratch. No dependencies.", "/docs/internals/nervus/"],
    ["kvad", "Transformer inference by hand: six architectures, quantisation, the kernels.", "/docs/internals/engine/"],
    ["kvad-gpu", "The same forward passes on Metal, and the image and video models.", "/docs/internals/gpu/"],
    ["kvad-tui", "The terminal app.", "/docs/internals/tui/"],
    ["kvad-serve", "The server and the web UI.", "/docs/internals/serve/"],
  ];
</script>

<Meta
  title="Kvad · Local AI for Apple silicon, written from the arithmetic up"
  description="Kvad runs language, image and video models on your Mac from one install: a web UI, a terminal app, a command line and an OpenAI-compatible API, with its own kernels for the M5's GPU."
  path="/"
  data={structured}
/>

<section class="hero wrap">
  <div class="pitch">
    <p class="label">An LLM engine in Rust</p>
    <h1>Local AI for Apple silicon, written from the arithmetic up.</h1>
    <p class="lede">
      Kvad runs language, image and video models on your own Mac, from one install. It has its own
      kernels for the M5's GPU, an OpenAI-compatible API, and a codebase you can read from the first
      matrix multiply to the last.
    </p>
    <Install />
    <p class="meta">
      {#if data.version}<span>v{data.version}</span>{/if}
      <span>macOS on Apple silicon, Linux on the CPU</span>
      <span>MIT</span>
    </p>
    <div class="actions">
      <a class="btn primary" href="/docs/quickstart/">Quick start</a>
      <a class="btn" href={REPO} rel="noopener">Read the source</a>
    </div>
  </div>

  <div class="stage">
    <Shot name="chat" alt="The chat page of Kvad's web UI: Qwen3-14B answering a question, with the decode rate and cached tokens under each reply" width={2240} height={1400} eager />
  </div>
</section>

<section class="wrap numbers" use:reveal aria-label="Measurements">
  <dl>
    <div>
      <dt>215 <small>tok/s</small></dt>
      <dd>Qwen2.5-1.5B decoding on Metal at q4. Median of five interleaved rounds.</dd>
    </div>
    <div>
      <dt>2.5<small>×</small></dt>
      <dd>Faster prefill on the M5's matrix units: 2,079 tokens in 0.39 s, down from 0.99 s.</dd>
    </div>
    <div>
      <dt>14.6 <small>GB</small></dt>
      <dd>Qwen3-14B in memory at q8. Quantised once, then mapped from disk on every load.</dd>
    </div>
    <div>
      <dt>92 <small>s</small></dt>
      <dd>An SDXL picture at 1024², 30 steps, including the decode.</dd>
    </div>
  </dl>
  <p>Measured on an M5 Pro with 48 GB. <a href="/apple-silicon/">How, and what did not get faster</a></p>
</section>

<section class="wrap ways" id="features" use:reveal>
  <header class="head">
    <p class="label">Four ways in</p>
    <h2>One engine behind a browser, a terminal, a shell and an API.</h2>
  </header>

  <div class="tabs" role="tablist" aria-label="Interfaces">
    {#each WAYS as w (w.id)}
      <button role="tab" type="button" aria-selected={way === w.id} aria-controls="way-{w.id}" id="tab-{w.id}" onclick={() => (way = w.id)}>
        {w.name}
      </button>
    {/each}
  </div>

  <div class="panels">
    {#each WAYS as w (w.id)}
      <div class="panel" class:on={way === w.id} role="tabpanel" id="way-{w.id}" aria-labelledby="tab-{w.id}" inert={way !== w.id}>
        <div class="say">
          <h3>{w.title}</h3>
          <p>{w.text}</p>
          <a class="more" href={w.link[0]}>{w.link[1]} <span aria-hidden="true">→</span></a>
        </div>
        <div class="show">
          {#if w.id === "web"}
            <Shot name="dashboard" alt="The dashboard: the loaded model, decode rate, time to first token, queue depth, memory and disk" width={2240} height={1400} />
          {:else if w.id === "tui"}
            <Terminal title="kvad-tui" html={tuiChat} label="The terminal app's chat screen, with a reply from Qwen2.5-1.5B at 129 tokens a second" />
          {:else if w.id === "cli"}
            <Terminal title="zsh" lines={RUN} label="kvad search, load and ps in a terminal" />
          {:else}
            <Terminal title="zsh" lines={API} label="A chat completion request with curl" />
            <ul class="facts">
              <li><code>/v1/chat/completions</code><span>streaming, <code>reasoning_content</code>, tool calls</span></li>
              <li><code>/v1/images/generations</code><span>with steps, seed, LoRAs and a preview per step</span></li>
              <li><code>/v1/images/edits</code><span>from a picture, with a mask or without</span></li>
              <li><code>/v1/videos</code><span>a job you start, follow and fetch</span></li>
              <li><code>/v1/models</code><span>says which models can take tools</span></li>
            </ul>
          {/if}
        </div>
      </div>
    {/each}
  </div>
</section>

<section class="wrap runs" use:reveal>
  <header class="head">
    <p class="label">What it runs</p>
    <h2>Words, pictures and video, from the weights people already publish.</h2>
  </header>

  <div class="kinds">
    <article>
      <h3>Language</h3>
      <p>
        Six architectures, loaded straight from Hugging Face safetensors: the Llama family (Llama,
        Mistral, Qwen 2 to 3, SmolLM2), DeepSeek V2 and V3, Qwen3.5 and Qwen3-Next, and GPT-2.
      </p>
      <p>On the CPU or on Metal, at f32, bf16, q8 or q4, with a KV cache that survives between turns.</p>
      <a class="more" href="/docs/models/">Models and backends <span aria-hidden="true">→</span></a>
    </article>
    <article>
      <h3>Images</h3>
      <p>
        Stable Diffusion 1.5 and SDXL with their fine-tunes, FLUX.1-schnell, FLUX.1-dev and
        Qwen-Image, including the community's GGUF files. From a prompt, or from a picture on SDXL
        and SD 1.5.
      </p>
      <p>LoRAs are applied per request, to a model that stays loaded. An SDXL LoRA can be trained on your own pictures.</p>
      <a class="more" href="/docs/images/">Images <span aria-hidden="true">→</span></a>
    </article>
    <article>
      <h3>Video</h3>
      <p>
        LTX-2.5, with sound, from a prompt or from a picture, up to 120 frames a second, and up to
        1536×1024 with its pipeline's last pass run in tiles.
      </p>
      <p>A video is a job on the server: start it, close the page, and fetch it when it is done.</p>
      <a class="more" href="/docs/video/">Video <span aria-hidden="true">→</span></a>
    </article>
  </div>

  <figure class="gallery">
    <div>
      {#each GALLERY as [name, alt] (name)}
        <img src="/gallery/{name}.webp" {alt} width="880" height="880" loading="lazy" decoding="async" />
      {/each}
    </div>
    <figcaption>
      Made by Kvad for this page: SDXL at 1024², 30 steps, about a minute and a half each on an M5
      Pro. The prompts and seeds are in <a href="/docs/images/">the image guide</a>.
    </figcaption>
  </figure>

  <figure class="clip">
    <!-- svelte-ignore a11y_media_has_caption -->
    <video controls playsinline preload="none" poster="/gallery/fox-in-snow.webp" width="1536" height="1024" aria-label="A fox trotting through fresh snow towards the camera, snow falling">
      <source src="/gallery/fox-in-snow.mp4" type="video/mp4" />
    </video>
    <figcaption>
      Made by Kvad: LTX-2.5 at 1536×1024, 3 seconds at 48 frames a second, with its sound. 18
      minutes to denoise and 2 to decode on an M5 Pro, holding 27 GB at its peak. The prompt and
      seed are in <a href="/docs/video/#large-sizes">the video guide</a>.
    </figcaption>
  </figure>
</section>

<section class="wrap rows">
  <div class="row" use:reveal>
    <div class="say">
      <p class="label">Playground</p>
      <h2>See what the model was choosing between.</h2>
      <p>
        Generation computes a probability for every token in the vocabulary at every step, and then
        throws all but one away. The playground keeps them. Each token is coloured by how sure the
        model was, and a click shows what else was in the running.
      </p>
      <p>Beside it: the tokeniser, and two models answering the same prompt.</p>
    </div>
    <Shot name="playground" alt="The playground: a completion with each token coloured by the model's confidence, beside the sampler's settings" width={2240} height={1400} />
  </div>

  <div class="row flip" use:reveal>
    <div class="say">
      <p class="label">Benchmarks</p>
      <h2>Numbers you can reproduce, with a button.</h2>
      <p>
        Three of Kvad's own published numbers were once wrong, each from timing a single run. So
        the server has a benchmark built in, and it is strict: rounds interleave the variants, every
        sample is kept, the median is shown with its range, and it refuses to measure a machine that
        is busy with something else.
      </p>
      <a class="more" href="/docs/benchmarks/">Benchmarks and evals <span aria-hidden="true">→</span></a>
    </div>
    <Shot name="benchmarks" alt="The benchmarks page: Qwen2.5-1.5B on three backends, with the median decode rate and the range of five rounds for each" width={2240} height={1400} />
  </div>

  <div class="row" use:reveal>
    <div class="say">
      <p class="label">Memory</p>
      <h2>Several models, one honest budget.</h2>
      <p>
        Each loaded model is charged for its weights and a full context of KV cache, so a model that
        was admitted can always finish its conversation. A load that does not fit is refused with the
        name of what is in the way. Nothing is ever unloaded behind your back.
      </p>
      <a class="more" href="/docs/models/#what-is-in-memory">What is in memory <span aria-hidden="true">→</span></a>
    </div>
    <Shot name="models" alt="The models page: Qwen3-14B in memory, charged 19.6 of 36 GB, above the list of models on the machine" width={2240} height={1400} />
  </div>

  <div class="row flip" use:reveal>
    <div class="say">
      <p class="label">Training</p>
      <h2>Train something small and watch it learn.</h2>
      <p>
        A character-level GPT on any text file, by a training loop written out in the repository
        with no framework under it. The loss curve is drawn as it falls, with what the model writes
        at each checkpoint beside it.
      </p>
      <a class="more" href="/docs/training/">Train your own <span aria-hidden="true">→</span></a>
    </div>
    <Shot name="training" alt="The training page: a finished run with its training and validation loss curves, and the text the model wrote at each checkpoint" width={2240} height={1400} />
  </div>
</section>

<section class="silicon" use:reveal>
  <div class="wrap">
    <header class="head">
      <p class="label">Apple silicon</p>
      <h2>The M5's GPU has matrix units. Most software never reaches them.</h2>
      <p>
        Ordinary Metal shaders do not use them. Kvad has kernels that do, for dense and for
        quantised weights, and falls back to the standard path on earlier chips without being told.
      </p>
    </header>
    <div class="table">
      <table>
        <thead>
          <tr><th>On an M5 Pro</th><th>Standard Metal</th><th>Kvad's kernels</th><th></th></tr>
        </thead>
        <tbody>
          <tr><td>Prefill, Qwen2.5-1.5B, 2,079 tokens</td><td>0.99 s</td><td>0.39 s</td><td>2.5×</td></tr>
          <tr><td>FLUX.1-schnell, a step at 1024², q8</td><td>12.9–13.7 s</td><td>8.7–9.2 s</td><td>1.5×</td></tr>
          <tr><td>SDXL, a step at 1024², f16</td><td>3.90–3.98 s</td><td>3.09–3.15 s</td><td>1.26×</td></tr>
          <tr><td>Decode, Qwen2.5-1.5B, q8, a short context</td><td>114–117 tok/s</td><td>126–129 tok/s</td><td>1.1×</td></tr>
        </tbody>
      </table>
    </div>
    <a class="more" href="/apple-silicon/">Kvad on Apple silicon <span aria-hidden="true">→</span></a>
  </div>
</section>

<section class="wrap read" use:reveal>
  <div class="say">
    <p class="label">How it works</p>
    <h2>An engine you can read.</h2>
    <p>
      Kvad has two jobs. One is to be worth running. The other is to explain itself: every matrix
      multiply, every derivative and every attention head is code in the repository, with the
      reasoning for each constant next to it.
    </p>
    <ol class="crates">
      {#each CRATES as [name, what, href], i (name)}
        <li>
          <a {href}><span class="n">{i + 1}</span><code>{name}</code><span class="w">{what}</span></a>
        </li>
      {/each}
    </ol>
    <a class="more" href="/docs/internals/">Reading the engine <span aria-hidden="true">→</span></a>
  </div>
  {#if data.excerpt}
    <figure class="source">
      <div class="code"><pre><code>{@html data.excerpt}</code></pre></div>
      <figcaption>
        <a href="{REPO}/blob/master/crates/llm/src/tensor.rs" rel="noopener"><code>crates/llm/src/tensor.rs</code></a>,
        as it is in the repository.
      </figcaption>
    </figure>
  {/if}
</section>

<section class="wrap notyet" use:reveal>
  <div class="say">
    <p class="label">Not yet</p>
    <h2>What Kvad does not do.</h2>
    <p>You should know before you install it, not after.</p>
  </div>
  <ul>
    <li>
      <strong>One generation at a time.</strong> Requests queue. There is no continuous batching
      and no paged KV cache, so it serves one person well and a team badly.
    </li>
    <li>
      <strong>The GPU is Metal only.</strong> Linux runs language models on the CPU. There is no
      CUDA build, and no images or video off a Mac.
    </li>
    <li>
      <strong>Tool calls cannot be forced.</strong> <code>tool_choice</code> is <code>auto</code> or
      <code>none</code>, and only the Hermes-style format that Qwen uses is parsed.
    </li>
    <li>
      <strong>No embeddings endpoint.</strong> Chat, images and video are the three that exist.
    </li>
  </ul>
  <a class="more" href="/docs/internals/roadmap/">The roadmap <span aria-hidden="true">→</span></a>
</section>

<section class="wrap end" use:reveal>
  <h2>Install it, and ask it something.</h2>
  <Install />
  <p>
    Then <code>kvad run --prompt "Why is the sky blue?"</code>, or open
    <code>http://127.0.0.1:5823</code>. <a href="/docs/install/">Other ways to install</a>
  </p>
</section>

<style>
  section {
    padding-block: clamp(48px, 7vw, 84px);
  }
  h2 {
    font-size: clamp(1.9rem, 3.6vw, 2.75rem);
  }
  h3 {
    font-size: 1.45rem;
  }
  .head {
    display: grid;
    gap: 16px;
    max-width: 780px;
    margin-bottom: 44px;
  }
  .head > p:not(.label),
  .say > p:not(.label) {
    color: var(--muted);
  }
  .say {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 16px;
    align-content: start;
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

  /* Hero */
  .hero {
    padding-block: clamp(48px, 8vw, 96px) 0;
  }
  .pitch {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 22px;
    max-width: 800px;
  }
  h1 {
    font-size: clamp(2.6rem, 6.4vw, 4.6rem);
    line-height: 1.02;
    letter-spacing: -0.022em;
  }
  .lede {
    font-size: clamp(1.1rem, 1.6vw, 1.28rem);
    line-height: 1.55;
    color: var(--muted);
    max-width: 62ch;
  }
  .meta {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 0;
    font: 400 12.5px/1.5 var(--mono);
    color: var(--faint);
  }
  .meta span + span::before {
    content: "·";
    margin: 0 0.7em;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 6px;
  }
  .stage {
    margin-top: clamp(40px, 6vw, 72px);
  }

  /* Numbers */
  .numbers {
    padding-block: clamp(48px, 7vw, 80px);
  }
  dl {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    margin: 0;
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
  }
  dl > div {
    padding: 28px 24px 28px 0;
  }
  dl > div + div {
    padding-left: 24px;
    border-left: 1px solid var(--line);
  }
  dt {
    font: 500 clamp(2.2rem, 4vw, 3.1rem) / 1 var(--serif);
    letter-spacing: -0.02em;
    font-variant-numeric: lining-nums;
  }
  dt small {
    font: 400 0.36em/1 var(--mono);
    letter-spacing: 0;
    color: var(--accent);
    margin-left: 0.15em;
  }
  dd {
    margin: 12px 0 0;
    font-size: 14.5px;
    line-height: 1.5;
    color: var(--muted);
  }
  .numbers > p {
    margin-top: 16px;
    font: 400 12.5px/1.5 var(--mono);
    color: var(--faint);
  }
  .numbers > p a {
    color: var(--muted);
  }

  /* Four ways in */
  .tabs {
    display: flex;
    gap: 4px;
    border-bottom: 1px solid var(--line);
    overflow-x: auto;
    scrollbar-width: none;
  }
  .tabs button {
    position: relative;
    padding: 12px 16px;
    white-space: nowrap;
    font-weight: 500;
    font-size: 15px;
    color: var(--muted);
    transition: color 0.15s;
  }
  .tabs button:first-child {
    padding-left: 0;
  }
  .tabs button:hover {
    color: var(--text);
  }
  .tabs button::after {
    content: "";
    position: absolute;
    left: 16px;
    right: 16px;
    bottom: -1px;
    height: 2px;
    background: var(--accent);
    transform: scaleX(0);
    transition: transform 0.25s var(--ease);
  }
  .tabs button:first-child::after {
    left: 0;
  }
  .tabs button[aria-selected="true"] {
    color: var(--text);
  }
  .tabs button[aria-selected="true"]::after {
    transform: scaleX(1);
  }
  /* The panels share one grid cell, so the section is as tall as the
     tallest and nothing below it moves when the tab changes. */
  .panels {
    display: grid;
    margin-top: 36px;
  }
  .panel {
    grid-area: 1 / 1;
    display: grid;
    grid-template-columns: minmax(0, 5fr) minmax(0, 9fr);
    gap: clamp(28px, 5vw, 64px);
    align-items: start;
    align-content: start;
    opacity: 0;
    visibility: hidden;
    transform: translateY(6px);
    transition:
      opacity 0.3s var(--ease),
      transform 0.3s var(--ease),
      visibility 0s 0.3s;
  }
  .panel.on {
    opacity: 1;
    visibility: visible;
    transform: none;
    transition-delay: 0s;
  }
  .show {
    display: grid;
    gap: 20px;
    min-width: 0;
  }
  .facts {
    list-style: none;
    margin: 0;
    padding: 0;
    border-top: 1px solid var(--line);
  }
  .facts li {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 20px;
    justify-content: space-between;
    padding: 11px 0;
    border-bottom: 1px solid var(--line);
    font-size: 14.5px;
  }
  .facts li > code {
    color: var(--accent);
  }
  .facts span {
    color: var(--muted);
  }

  /* What it runs */
  .kinds {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    border-top: 1px solid var(--line);
  }
  .kinds article {
    display: grid;
    gap: 12px;
    align-content: start;
    padding: 28px 28px 8px 0;
  }
  .kinds article + article {
    padding-left: 28px;
    border-left: 1px solid var(--line);
  }
  .kinds p {
    font-size: 15.5px;
    color: var(--muted);
  }
  .gallery {
    margin: 48px 0 0;
  }
  .gallery div {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
  }
  .gallery img {
    width: 100%;
    height: auto;
    aspect-ratio: 1;
    object-fit: cover;
    border-radius: 8px;
  }
  .clip {
    margin: 32px 0 0;
  }
  .clip video {
    width: 100%;
    height: auto;
    border-radius: 8px;
    background: var(--line);
  }
  figcaption {
    margin-top: 14px;
    font-size: 13.5px;
    color: var(--faint);
  }
  figcaption a {
    color: var(--muted);
  }

  /* Alternating rows */
  .rows {
    display: grid;
    gap: clamp(64px, 10vw, 128px);
  }
  .row {
    display: grid;
    grid-template-columns: minmax(0, 5fr) minmax(0, 8fr);
    gap: clamp(28px, 5vw, 72px);
    align-items: center;
  }
  .row.flip .say {
    order: 2;
  }
  .row.flip {
    grid-template-columns: minmax(0, 8fr) minmax(0, 5fr);
  }

  /* Apple silicon band */
  .silicon {
    background: var(--surface);
    border-block: 1px solid var(--line);
  }
  .silicon .table {
    overflow-x: auto;
    margin-bottom: 28px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 15.5px;
  }
  th,
  td {
    text-align: right;
    padding: 14px 0 14px 24px;
    border-bottom: 1px solid var(--line);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }
  th:first-child,
  td:first-child {
    text-align: left;
    padding-left: 0;
    white-space: normal;
    min-width: 220px;
  }
  th {
    font: 500 12px/1.3 var(--mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--faint);
  }
  td:nth-child(2) {
    color: var(--muted);
  }
  td:last-child {
    font-family: var(--mono);
    color: var(--accent);
  }

  /* Read the engine */
  .read {
    display: grid;
    grid-template-columns: minmax(0, 5fr) minmax(0, 7fr);
    gap: clamp(32px, 4vw, 56px);
    align-items: start;
  }
  .crates {
    list-style: none;
    margin: 8px 0;
    padding: 0;
    border-top: 1px solid var(--line);
  }
  .crates a {
    display: grid;
    grid-template-columns: 24px 96px minmax(0, 1fr);
    gap: 12px;
    align-items: baseline;
    padding: 12px 0;
    border-bottom: 1px solid var(--line);
    text-decoration: none;
    transition: padding 0.2s var(--ease);
  }
  .crates a:hover {
    padding-left: 6px;
  }
  .crates .n {
    font: 400 12px/1 var(--mono);
    color: var(--faint);
  }
  .crates code {
    color: var(--accent);
  }
  .crates .w {
    font-size: 14.5px;
    color: var(--muted);
  }
  .source {
    margin: 0;
    min-width: 0;
  }
  .source pre {
    font-size: 12px;
    line-height: 1.65;
  }

  /* Not yet */
  .notyet {
    display: grid;
    grid-template-columns: minmax(0, 4fr) minmax(0, 7fr);
    gap: 24px clamp(32px, 5vw, 72px);
  }
  .notyet ul {
    list-style: none;
    margin: 0;
    padding: 0;
    border-top: 1px solid var(--line);
  }
  .notyet li {
    padding: 16px 0;
    border-bottom: 1px solid var(--line);
    color: var(--muted);
  }
  .notyet li strong {
    color: var(--text);
    font-weight: 600;
  }
  .notyet .more {
    grid-column: 2;
  }

  /* The end */
  .end {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 22px;
    justify-items: start;
    padding-bottom: 0;
  }
  .end p {
    color: var(--muted);
    font-size: 15.5px;
  }
  .end a {
    color: var(--accent);
  }
  :is(.end, .notyet, .facts) code {
    padding: 0.1em 0.35em;
    border-radius: 4px;
    background: var(--raised);
  }

  @media (max-width: 900px) {
    dl {
      grid-template-columns: 1fr 1fr;
    }
    dl > div:nth-child(3) {
      padding-left: 0;
      border-left: 0;
    }
    dl > div:nth-child(n + 3) {
      border-top: 1px solid var(--line);
    }
    .panel,
    .row,
    .row.flip,
    .read,
    .notyet {
      grid-template-columns: minmax(0, 1fr);
    }
    .row.flip .say {
      order: 0;
    }
    .notyet .more {
      grid-column: 1;
    }
    .kinds {
      grid-template-columns: 1fr;
    }
    .kinds article,
    .kinds article + article {
      padding: 24px 0;
      border-left: 0;
    }
    .kinds article + article {
      border-top: 1px solid var(--line);
    }
    .gallery div {
      grid-template-columns: 1fr 1fr;
    }
  }
  @media (max-width: 520px) {
    dl {
      grid-template-columns: 1fr;
    }
    dl > div,
    dl > div + div {
      padding: 22px 0;
      border-left: 0;
    }
    dl > div + div {
      border-top: 1px solid var(--line);
    }
    .crates a {
      grid-template-columns: 20px minmax(0, 1fr);
    }
    .crates .w {
      grid-column: 2;
    }
  }
</style>
