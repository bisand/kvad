<script>
  // The install command, as a thing to copy. Clicking anywhere on it copies.
  import { copy } from "#lib/copy.js";
  import { INSTALL } from "#lib/site.js";

  let { command = INSTALL } = $props();
  let copied = $state(false);

  async function go() {
    copied = await copy(command);
    setTimeout(() => (copied = false), 1600);
  }
</script>

<button class="install" type="button" onclick={go} aria-label="Copy the install command">
  <span class="prompt" aria-hidden="true">$</span>
  <code>{command}</code>
  <span class="state" aria-live="polite">{copied ? "Copied" : "Copy"}</span>
</button>

<style>
  .install {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    max-width: 560px;
    height: 52px;
    padding: 0 8px 0 16px;
    border: 1px solid var(--line-strong);
    border-radius: 10px;
    background: var(--surface);
    text-align: left;
    transition: border-color 0.15s;
  }
  .install:hover {
    border-color: var(--accent);
  }
  .prompt {
    font-family: var(--mono);
    color: var(--accent);
    user-select: none;
  }
  code {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 14px;
  }
  .state {
    flex-shrink: 0;
    min-width: 62px;
    padding: 8px 10px;
    border-radius: 6px;
    background: var(--raised);
    font: 500 12px/1 var(--mono);
    text-align: center;
    color: var(--muted);
    transition: color 0.15s;
  }
  .install:hover .state {
    color: var(--text);
  }
</style>
