<script>
  import { REPO } from "#lib/site.js";
</script>

<footer>
  <div class="wrap grid">
    <div class="about">
      <svg viewBox="-1 0 149 55" height="26" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="4.5" role="img" aria-label="Kvad, in the younger futhark">
        <path d="M6 4 V50 M6 25 L21 5" />
        <path d="M38 50 V4 L59 15 V50" />
        <path d="M85 4 V50 M76 35 L94 20" />
        <path d="M128 4 V50 M116 17 L128 4 L140 17" />
      </svg>
      <p>
        <em>Kvad</em> is Old Norse for a composed, recited poem: what a skald performs from memory,
        one line at a time.
      </p>
    </div>

    <nav aria-label="Product">
      <h2>Kvad</h2>
      <a href="/#features">Features</a>
      <a href="/apple-silicon/">Apple silicon</a>
      <a href="{REPO}/releases" rel="noopener">Releases</a>
      <a href="/docs/internals/roadmap/">Roadmap</a>
    </nav>
    <nav aria-label="Documentation">
      <h2>Documentation</h2>
      <a href="/docs/install/">Install</a>
      <a href="/docs/quickstart/">Quick start</a>
      <a href="/docs/api/">API</a>
      <a href="/docs/cli/">Command line</a>
    </nav>
    <nav aria-label="Project">
      <h2>Project</h2>
      <a href={REPO} rel="noopener">GitHub</a>
      <a href="{REPO}/issues" rel="noopener">Issues</a>
      <a href="/docs/internals/">How it works</a>
      <a href="{REPO}/blob/master/LICENSE" rel="noopener">MIT licence</a>
    </nav>
  </div>
  <div class="wrap">
    <div class="fine">
      <span>Written in Rust. Measured on an M5 Pro.</span>
      <span>© André Biseth</span>
    </div>
  </div>
</footer>

<style>
  footer {
    margin-top: 120px;
    border-top: 1px solid var(--line);
    padding: 56px 0 32px;
    font-size: 15px;
  }
  .grid {
    display: grid;
    grid-template-columns: minmax(0, 1.6fr) repeat(3, minmax(0, 1fr));
    gap: 40px;
  }
  .about svg {
    color: var(--accent);
    margin-bottom: 16px;
  }
  .about p {
    max-width: 34ch;
    color: var(--muted);
  }
  nav {
    display: grid;
    align-content: start;
    gap: 9px;
  }
  h2 {
    font: 500 12px/1 var(--mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--faint);
    margin-bottom: 6px;
  }
  nav a {
    color: var(--muted);
    text-decoration: none;
    width: fit-content;
  }
  nav a:hover {
    color: var(--text);
  }
  .fine {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 56px;
    padding-top: 20px;
    border-top: 1px solid var(--line);
    font: 400 12.5px/1.5 var(--mono);
    color: var(--faint);
  }
  @media (max-width: 760px) {
    .grid {
      grid-template-columns: 1fr 1fr;
    }
    .about {
      grid-column: 1 / -1;
    }
  }
</style>
