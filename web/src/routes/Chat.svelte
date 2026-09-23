<script>
  import { chat, describeStats, DEFAULTS } from "../lib/chat.svelte.js";
  import { models } from "../lib/models.svelte.js";
  import { api } from "../lib/api.js";
  import { navigate } from "../lib/router.svelte.js";
  import Icon from "../lib/components/Icon.svelte";
  import { render, copyFromBlock } from "../lib/markdown.js";

  const PLUS = "M12 5v14M5 12h14";
  const TRASH = "M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6";
  const SEND = "M22 2 11 13M22 2l-7 20-4-9-9-4z";
  const STOP = "M6 6h12v12H6z";
  const DOWNLOAD = "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3";

  let draft = $state("");
  /// Corpora to answer from, and the one chosen. Empty for anyone who is not
  /// an administrator, because reading a dataset is an administrator's to do
  /// — so the picker simply does not appear rather than appearing and failing.
  let datasets = $state([]);
  let groundIn = $state(null);
  let showSettings = $state(false);
  let systemDraft = $state("");
  let transcript;

  $effect(() => {
    models.refresh();
    api("/api/datasets")
      .then((list) => (datasets = list.filter((d) => d.present)))
      .catch(() => {});
    // Opening the newest conversation rather than an empty pane: arriving
    // here almost always means carrying on with the last thing. Everything
    // after the `await` is untracked, so reading `chat.list` here does not
    // make this effect re-run itself.
    (async () => {
      await chat.refresh();
      if (!chat.current && chat.list.length) await chat.open(chat.list[0].id);
    })();
  });

  // Follow the reply down as it grows, but only while the reader is already
  // at the bottom — yanking the view while somebody is reading back is worse
  // than making them scroll.
  let pinned = $state(true);
  $effect(() => {
    // Touch what should re-run this.
    void chat.streaming;
    void chat.thinking;
    void chat.messages.length;
    if (pinned && transcript) transcript.scrollTop = transcript.scrollHeight;
  });

  function onScroll() {
    if (!transcript) return;
    const slack = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight;
    pinned = slack < 40;
  }

  function submit(event) {
    event.preventDefault();
    const text = draft;
    draft = "";
    pinned = true;
    // By id, `repo@backend`, so that a model in memory on two backends is
    // answered by the one that was picked rather than by whichever was used
    // last.
    chat.send(text, models.loaded?.id ?? models.loaded?.repo, groundIn);
  }

  function onKeydown(event) {
    // Enter sends, shift-enter makes a line. What every chat box does.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!chat.sending && draft.trim()) submit(event);
    }
  }

  async function openSettings() {
    systemDraft = chat.current?.system ?? "";
    showSettings = true;
  }

  function exportConversation() {
    const lines = chat.messages.map((m) => `## ${m.role}\n\n${m.content}`);
    const head = [`# ${chat.current.title}`];
    if (chat.current.system) head.push(`> system: ${chat.current.system}`);
    const blob = new Blob([[...head, ...lines].join("\n\n")], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${chat.current.title.replace(/[^\w.-]+/g, "-").slice(0, 60)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<div class="flex h-[calc(100vh-10rem)] gap-4">
  <!-- Conversations. Hidden on a narrow screen, where the transcript is all
       there is room for. -->
  <aside class="hidden w-56 shrink-0 flex-col gap-2 md:flex">
    <button class="btn btn-sm" onclick={() => chat.create()}>
      <Icon path={PLUS} size={16} /> New
    </button>
    <ul class="menu menu-sm w-full grow gap-0.5 overflow-y-auto p-0">
      {#each chat.list as c (c.id)}
        <li>
          <button
            class={chat.current?.id === c.id ? "menu-active" : ""}
            onclick={() => chat.open(c.id)}
          >
            <span class="grow truncate text-left">{c.title}</span>
            <span class="text-xs opacity-50">{c.messages}</span>
          </button>
        </li>
      {:else}
        <li class="px-2 py-4 text-xs opacity-60">No conversations yet.</li>
      {/each}
    </ul>
  </aside>

  <section class="border-base-300 bg-base-100 flex min-w-0 grow flex-col rounded-box border">
    <!-- Which model will answer, and the controls for how. -->
    <header class="border-base-300 flex items-center gap-2 border-b px-4 py-2">
      <div class="min-w-0 grow">
        {#if models.residents.length > 1}
          <select
            class="select select-sm select-ghost -ml-3 max-w-full font-medium"
            aria-label="Which model answers"
            bind:value={() => models.loaded?.id, (id) => (models.picked = id)}
          >
            {#each models.residents as r (r.id)}
              <option value={r.id}>{r.repo} · {r.backend}</option>
            {/each}
          </select>
          {#if !models.loaded?.instruct}
            <p class="text-xs opacity-60">base model — it continues text rather than answering</p>
          {/if}
        {:else if models.loaded}
          <p class="truncate text-sm font-medium">{models.loaded.repo}</p>
          <p class="text-xs opacity-60">
            {models.loaded.backend}
            {#if !models.loaded.instruct}
              · base model — it continues text rather than answering
            {/if}
          </p>
        {:else}
          <p class="text-sm opacity-70">No model loaded</p>
          <p class="text-xs opacity-60">
            <a href="/models" onclick={(e) => navigate(e, "/models")} class="link">
              Pick one on the Models page
            </a>
          </p>
        {/if}
      </div>
      {#if chat.current}
        <button class="btn btn-ghost btn-sm" onclick={exportConversation} aria-label="Export">
          <Icon path={DOWNLOAD} size={16} />
        </button>
        <button
          class="btn btn-ghost btn-sm"
          onclick={() => chat.remove(chat.current.id)}
          aria-label="Delete conversation"
        >
          <Icon path={TRASH} size={16} />
        </button>
      {/if}
      <button class="btn btn-ghost btn-sm" onclick={openSettings}>Settings</button>
    </header>

    <!-- The transcript. -->
    <div class="grow overflow-y-auto p-4" bind:this={transcript} onscroll={onScroll}>
      {#if chat.current?.system}
        <div class="alert alert-soft mb-4 text-xs">
          <span><span class="font-medium">System:</span> {chat.current.system}</span>
        </div>
      {/if}

      {#each chat.messages as m (m.id)}
        <div class="chat {m.role === 'user' ? 'chat-end' : 'chat-start'}">
          {#if m.role === "assistant"}
            <!-- Rendered, because the model wrote Markdown. `render`
                 sanitises; see lib/markdown.js. -->
            <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
            <div class="chat-bubble md" onclick={copyFromBlock}>{@html render(m.content)}</div>
          {:else}
            <!-- Typed, not generated: the literal text is the honest one. -->
            <div class="chat-bubble whitespace-pre-wrap">{m.content}</div>
          {/if}
          {#if describeStats(m.stats)}
            <div class="chat-footer mt-1 text-xs opacity-50">{describeStats(m.stats)}</div>
          {/if}
        </div>
      {/each}

      {#if chat.streaming !== null}
        <div class="chat chat-start">
          <!-- A reasoning model's working, collapsed. It is worth being able
               to read — it is where a model catches itself following the
               wrong passage — but it is not the answer, and it is often
               longer. Open while there is nothing else to show, closed once
               the answer starts. -->
          {#if chat.thinking}
            <details class="chat-header mb-1 w-full" open={chat.streaming === ""}>
              <summary class="cursor-pointer text-xs opacity-50">
                Thinking{chat.streaming === "" ? "…" : ""}
              </summary>
              <div
                class="border-base-300 bg-base-200 rounded-box mt-1 max-h-48 overflow-y-auto border p-2 text-xs whitespace-pre-wrap opacity-70"
              >
                {chat.thinking}
              </div>
            </details>
          {/if}
          <!-- Rendered as it arrives, so the formatting settles token by
               token rather than snapping into place at the end. -->
          {#if chat.streaming === ""}
            <div class="chat-bubble">
              <span class="loading loading-dots loading-sm align-middle"></span>
            </div>
          {:else}
            <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
            <div class="chat-bubble md" onclick={copyFromBlock}>{@html render(chat.streaming)}</div>
          {/if}
        </div>
      {/if}

      {#if chat.messages.length === 0 && chat.streaming === null}
        <p class="py-12 text-center text-sm opacity-50">
          {chat.current ? "Say something." : "Start a conversation."}
        </p>
      {/if}
    </div>

    <!-- The box. -->
    <form class="border-base-300 flex items-end gap-2 border-t p-3" onsubmit={submit}>
      {#if datasets.length}
        <!-- Not training: the model learns nothing. The server searches this
             corpus with the question and puts the passages it finds in front
             of the model, which then reads them. -->
        <select
          class="select select-sm w-40 shrink-0"
          bind:value={groundIn}
          title="Answer from a dataset"
        >
          <option value={null}>no dataset</option>
          {#each datasets as d (d.id)}
            <option value={d.id}>{d.name}</option>
          {/each}
        </select>
      {/if}
      <textarea
        class="textarea min-h-12 w-full grow resize-none"
        rows="1"
        placeholder={models.loaded ? "Ask something…" : "Load a model first"}
        bind:value={draft}
        onkeydown={onKeydown}
        disabled={!models.loaded}
      ></textarea>
      {#if chat.sending}
        <button type="button" class="btn btn-square" onclick={() => chat.stop()} aria-label="Stop">
          <Icon path={STOP} size={16} />
        </button>
      {:else}
        <button class="btn btn-square" disabled={!models.loaded || !draft.trim()} aria-label="Send">
          <Icon path={SEND} size={18} />
        </button>
      {/if}
    </form>
  </section>
</div>

{#if showSettings}
  <div class="modal modal-open" role="dialog">
    <div class="modal-box">
      <h3 class="text-lg font-medium">Conversation settings</h3>

      <fieldset class="fieldset mt-3">
        <legend class="fieldset-legend">System prompt</legend>
        <textarea
          class="textarea w-full"
          rows="3"
          placeholder="Be brief."
          bind:value={systemDraft}
          disabled={!chat.current}
        ></textarea>
        <!-- Not daisyUI's `label`, which does not wrap: this is two lines. -->
        <p class="mt-1 text-xs opacity-60">
          Sent as the first turn, every turn. Changing it invalidates the KV cache for
          this conversation, so the next reply prefills the whole thing again.
        </p>
      </fieldset>

      <fieldset class="fieldset">
        <legend class="fieldset-legend">Sampling</legend>
        <div class="grid grid-cols-2 gap-3">
          <label class="text-xs">
            Temperature <span class="opacity-60">{chat.sampler.temperature.toFixed(2)}</span>
            <input
              type="range"
              class="range range-xs"
              min="0"
              max="2"
              step="0.05"
              bind:value={chat.sampler.temperature}
            />
            <span class="opacity-60">0 is greedy: the same answer every time.</span>
          </label>
          <label class="text-xs">
            Top-p <span class="opacity-60">{chat.sampler.top_p.toFixed(2)}</span>
            <input
              type="range"
              class="range range-xs"
              min="0"
              max="1"
              step="0.01"
              bind:value={chat.sampler.top_p}
            />
            <span class="opacity-60">The smallest set of tokens worth this much probability.</span>
          </label>
          <label class="text-xs">
            Top-k
            <input type="number" class="input input-sm w-full" min="0" bind:value={chat.sampler.top_k} />
          </label>
          <label class="text-xs">
            Max tokens
            <input
              type="number"
              class="input input-sm w-full"
              min="1"
              max="8192"
              bind:value={chat.sampler.max_tokens}
            />
          </label>
        </div>
      </fieldset>

      <div class="modal-action">
        <button
          class="btn btn-sm btn-ghost"
          onclick={() => (chat.sampler = { ...DEFAULTS })}
        >
          Reset sampling
        </button>
        <button class="btn btn-sm" onclick={() => (showSettings = false)}>Close</button>
        <button
          class="btn btn-sm btn-primary"
          disabled={!chat.current}
          onclick={async () => {
            await chat.setSystem(systemDraft.trim());
            showSettings = false;
          }}
        >
          Save
        </button>
      </div>
    </div>
    <button class="modal-backdrop" aria-label="Close" onclick={() => (showSettings = false)}></button>
  </div>
{/if}
