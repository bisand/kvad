<script>
  // Where the data is, and moving it somewhere else.
  //
  // `[data] dir` is not a field here, because changing it alone would start
  // the server empty in the new place. The server moves the data instead and
  // writes the setting as its last step, then restarts into it. Whatever was
  // copied rather than renamed stays where it was until somebody deletes it
  // from here.
  import { api } from "../api.js";
  import { humanBytes } from "../models.svelte.js";
  import { backFromRestart, uptimeNow } from "../restart.js";
  import { toasts } from "../toasts.svelte.js";

  let info = $state(null);
  let problem = $state(null);
  // The move dialog: what was typed, and what the server says it would do.
  let asking = $state(false);
  let target = $state("");
  let plan = $state(null);
  let planProblem = $state(null);
  let checking = $state(false);
  let starting = $state(false);
  // After a move: waiting for the server to come back, or why it did not.
  let restart = $state(null);
  let confirmDelete = $state(false);
  let deleting = $state(false);
  let polling = null;

  $effect(() => {
    load();
    return () => clearTimeout(polling);
  });

  async function load() {
    try {
      info = await api("/api/data");
      problem = null;
    } catch (e) {
      problem = e.message;
    }
    watch();
  }

  const stage = $derived(info?.moving?.stage ?? null);
  const underWay = $derived(stage === "copying" || stage === "switching" || stage === "restarting");

  // While a move goes on, ask how it is going, and when it restarts the
  // server, wait for it to come back.
  function watch() {
    clearTimeout(polling);
    if (stage === "restarting") {
      waitForRestart();
    } else if (underWay) {
      polling = setTimeout(load, 1000);
    }
  }

  let before = Infinity;
  async function waitForRestart() {
    if (restart === "waiting") return;
    restart = "waiting";
    if (await backFromRestart(before)) {
      restart = null;
      await load();
      toasts.success("Moved. The server is running from the new place.");
    } else {
      restart =
        "The server has not come back after 90 seconds. `kvad service logs` on its machine says why.";
    }
  }

  function open() {
    target = "";
    plan = null;
    planProblem = null;
    asking = true;
  }

  // What moving there would do, asked as the path is typed.
  let typing = null;
  function typed() {
    clearTimeout(typing);
    plan = null;
    planProblem = null;
    if (!target.trim()) return;
    typing = setTimeout(check, 400);
  }

  async function check() {
    const asked = target.trim();
    checking = true;
    try {
      const got = await api(`/api/data/plan?to=${encodeURIComponent(asked)}`);
      if (target.trim() === asked) plan = got;
    } catch (e) {
      if (target.trim() === asked) planProblem = e.message;
    } finally {
      checking = false;
    }
  }

  async function start() {
    starting = true;
    before = await uptimeNow();
    try {
      await api("/api/data/move", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ to: plan.to }),
      });
      asking = false;
      await load();
    } catch (e) {
      toasts.error(e.message);
    } finally {
      starting = false;
    }
  }

  async function cancel() {
    try {
      await api("/api/data/move", { method: "DELETE" });
      toasts.info("Cancelling. What was copied is being removed.");
    } catch (e) {
      toasts.error(e.message);
    }
  }

  async function removeOld() {
    deleting = true;
    try {
      const got = await api("/api/data/previous", { method: "DELETE" });
      toasts.success(`Deleted the old copy, ${humanBytes(got.bytes)}.`);
      confirmDelete = false;
      await load();
    } catch (e) {
      toasts.error(e.message);
    } finally {
      deleting = false;
    }
  }

  const KIND = { data: "", database: "database", hub: "Hugging Face models" };

  // What the copies moves left behind come to, or null when there are none.
  const oldBytes = $derived(
    info?.previous?.length ? info.previous.reduce((n, p) => n + p.bytes, 0) : null,
  );
</script>

<section>
  <h2 class="mb-1 text-sm font-medium opacity-60">Data</h2>
  <p class="mb-3 text-xs opacity-60">
    The database, generated images, datasets and trained models{info?.hub.follows
      ? ", and the models pulled from Hugging Face"
      : ""}. Moving them writes <code>[data] dir</code> to kvad.toml and restarts the server
    into the new place.
  </p>

  {#if problem}
    <div role="alert" class="alert alert-error alert-soft mb-3"><span>{problem}</span></div>
  {/if}

  {#if restart === "waiting"}
    <div role="alert" class="alert alert-soft mb-3">
      <span class="loading loading-spinner loading-sm"></span>
      <span>Restarting into the new place. The page picks up when the server answers again.</span>
    </div>
  {:else if restart}
    <div role="alert" class="alert alert-error alert-soft mb-3"><span>{restart}</span></div>
  {/if}

  {#if info?.moving && underWay && stage !== "restarting"}
    {@const m = info.moving}
    <div class="rounded-box bg-base-200/40 mb-3 p-3 text-sm">
      <div class="mb-2 flex items-center gap-2">
        <span class="loading loading-spinner loading-xs"></span>
        <span>
          {#if stage === "copying"}
            Copying to <code>{m.to}</code>{m.current ? `: ${m.current}` : ""}
          {:else}
            Switching over: copying the database and what changed meanwhile
          {/if}
        </span>
        {#if stage === "copying"}
          <button class="btn btn-ghost btn-xs ml-auto" onclick={cancel}>Cancel</button>
        {/if}
      </div>
      {#if m.total_bytes}
        <progress class="progress w-full" value={m.done_bytes} max={m.total_bytes}></progress>
        <p class="mt-1 text-xs opacity-60">
          {humanBytes(m.done_bytes)} of {humanBytes(m.total_bytes)}. The server goes on serving
          until the switch, which takes a moment.
        </p>
      {/if}
    </div>
  {:else if stage === "failed"}
    <div role="alert" class="alert alert-error alert-soft mb-3">
      <span>
        The move to <code>{info.moving.to}</code> failed: {info.moving.error} Nothing was switched;
        the data is where it was.
      </span>
    </div>
  {:else if stage === "cancelled"}
    <div role="alert" class="alert alert-soft mb-3">
      <span>The move to <code>{info.moving.to}</code> was cancelled; the data is where it was.</span>
    </div>
  {/if}

  {#if oldBytes !== null}
    <div role="alert" class="alert alert-info alert-soft mb-3">
      <span>
        The data was moved from <code>{info.previous.at(-1).from}</code>{info.previous.length > 1
          ? ` (and ${info.previous.length - 1} place${info.previous.length > 2 ? "s" : ""} before that)`
          : ""}. What was copied rather than moved is still there: {humanBytes(oldBytes)}.
      </span>
      <button class="btn btn-sm whitespace-nowrap" onclick={() => (confirmDelete = true)}>
        Delete the old copy
      </button>
    </div>
  {/if}

  {#if info}
    <dl class="mb-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
      <dt class="opacity-60">Data</dt>
      <dd class="break-all font-mono text-xs">
        {info.data_dir}
        {#if !info.chosen}<span class="font-sans opacity-60"> (the default)</span>{/if}
      </dd>
      <dt class="opacity-60">Hugging Face</dt>
      <dd class="break-all font-mono text-xs">
        {info.hub.path}
        {#if info.hub.env}
          <span class="block font-sans opacity-60">
            Set by {info.hub.env} where the server is started, so it stays put when the data moves.
          </span>
        {/if}
      </dd>
      <dt class="opacity-60">Database</dt>
      <dd class="break-all font-mono text-xs">
        {info.database}
        {#if !info.database_moves}
          <span class="block font-sans opacity-60">
            Named in kvad.toml or with --db, so it stays put when the data moves.
          </span>
        {/if}
      </dd>
    </dl>

    <details class="collapse collapse-arrow rounded-box bg-base-200/40 mb-3">
      <summary class="collapse-title text-sm">
        What there is: {humanBytes(info.total_bytes)}
      </summary>
      <div class="collapse-content">
        <ul class="text-sm">
          {#each info.items as item (item.from)}
            <li class="flex justify-between gap-4">
              <span class="truncate">
                {item.name}
                {#if KIND[item.kind]}<span class="opacity-60"> · {KIND[item.kind]}</span>{/if}
              </span>
              <span class="shrink-0 tabular-nums opacity-70">{humanBytes(item.bytes)}</span>
            </li>
          {/each}
        </ul>
      </div>
    </details>

    {#if info.blocked}
      <p class="text-xs opacity-60">Cannot be moved from here: {info.blocked}.</p>
    {:else}
      <button class="btn btn-sm" disabled={underWay || restart === "waiting"} onclick={open}>
        Move the data…
      </button>
    {/if}
  {/if}
</section>

{#if asking}
  <div class="modal modal-open" role="dialog">
    <div class="modal-box max-w-2xl">
      <h3 class="text-lg font-medium">Move the data</h3>
      <fieldset class="fieldset">
        <legend class="fieldset-legend">To</legend>
        <input
          class="input input-sm w-full font-mono"
          placeholder="/Volumes/Models/kvad or ~/kvad-data"
          bind:value={target}
          oninput={typed}
        />
        <p class="label">An empty or new directory. It is made if it is not there.</p>
      </fieldset>

      {#if checking}
        <p class="py-2 text-sm opacity-60"><span class="loading loading-dots loading-xs"></span></p>
      {:else if planProblem}
        <div role="alert" class="alert alert-warning alert-soft my-2"><span>{planProblem}</span></div>
      {:else if plan}
        <ul class="my-2 text-sm">
          {#each plan.steps as s (s.from)}
            <li class="flex justify-between gap-4">
              <span class="truncate">
                {s.name}
                {#if KIND[s.kind]}<span class="opacity-60"> · {KIND[s.kind]}</span>{/if}
              </span>
              <span class="shrink-0 tabular-nums opacity-70">
                {humanBytes(s.bytes)} · {s.rename ? "moved" : "copied"}
              </span>
            </li>
          {/each}
        </ul>
        <p class="text-sm opacity-70">
          {#if plan.crosses_disk}
            {humanBytes(plan.copy_bytes)} is copied to another disk{plan.free_bytes
              ? `, which has ${humanBytes(plan.free_bytes)} free`
              : ""}, while the server goes on serving.
          {:else if plan.steps.some((s) => s.kind === "database")}
            Everything is on the same disk and is renamed, which is instant. Only the database
            is copied.
          {:else}
            Everything is on the same disk and is renamed, which is instant.
          {/if}
          What is copied stays where it was until you delete it. Then the server restarts into
          the new place: models in memory are unloaded, and a download or training run that is
          going is interrupted.
        </p>
        {#if plan.leaves_shared_cache}
          <div role="alert" class="alert alert-soft mt-2 text-sm">
            <span>
              The Hugging Face models leave <code>~/.cache/huggingface/hub</code>, where Python's
              Hugging Face tools look for them too. Those tools will download again what they
              need. The Hugging Face token stays where it is.
            </span>
          </div>
        {/if}
      {/if}

      <div class="modal-action">
        <button class="btn btn-sm" onclick={() => (asking = false)}>Cancel</button>
        <button class="btn btn-sm btn-warning" disabled={!plan || starting} onclick={start}>
          {#if starting}<span class="loading loading-spinner loading-xs"></span>{/if}
          Move and restart
        </button>
      </div>
    </div>
    <button class="modal-backdrop" aria-label="Cancel" onclick={() => (asking = false)}></button>
  </div>
{/if}

{#if confirmDelete && oldBytes !== null}
  <div class="modal modal-open" role="dialog">
    <div class="modal-box">
      <h3 class="text-lg font-medium">Delete the old copy?</h3>
      <p class="py-3 text-sm opacity-70">
        This deletes {humanBytes(oldBytes)} for good:
      </p>
      <ul class="mb-2 font-mono text-xs">
        {#each info.previous.flatMap((p) => p.paths) as p (p)}<li class="break-all">{p}</li>{/each}
      </ul>
      <p class="text-sm opacity-70">The server is using the data in <code>{info.data_dir}</code>.</p>
      <div class="modal-action">
        <button class="btn btn-sm" onclick={() => (confirmDelete = false)}>Keep it</button>
        <button class="btn btn-sm btn-error" disabled={deleting} onclick={removeOld}>
          {#if deleting}<span class="loading loading-spinner loading-xs"></span>{/if}
          Delete
        </button>
      </div>
    </div>
    <button class="modal-backdrop" aria-label="Keep it" onclick={() => (confirmDelete = false)}></button>
  </div>
{/if}
