<script>
  // `[server]` from kvad.toml, and the restart that puts a change in force.
  //
  // The server reads the file once, as it starts, so a save changes the file
  // and not the server. Where the two differ the page says so and offers the
  // restart, rather than leaving someone to wonder why a setting they saved
  // is not doing anything.
  import { api, health } from "../api.js";
  import { toasts } from "../toasts.svelte.js";

  let info = $state(null);
  let problem = $state(null);
  let form = $state(null);
  let saving = $state(false);
  let asking = $state(false);
  // What a restart is doing: null, "waiting", or a sentence saying it failed.
  let restart = $state(null);
  let residents = $state([]);
  // Where the server went when a restart moved it to another address.
  let movedTo = $state(null);

  const NAMES = {
    autoload: "load at start",
    load_on_request: "load on request",
    memory_gb: "memory",
    context: "context",
    bind: "address",
  };

  $effect(() => {
    load();
  });

  async function load() {
    try {
      const got = await api("/api/settings");
      info = got;
      problem = got.problem;
      form = got.saved ? { ...got.saved } : null;
    } catch (e) {
      problem = e.message;
    }
  }

  // What the form would save. An emptied number field is the default, which
  // the server writes by taking the key out of the file.
  function values(f) {
    const number = (v) => (v === null || v === undefined || v === "" ? null : Number(v));
    const context = number(f.context);
    return {
      autoload: f.autoload,
      load_on_request: f.load_on_request,
      memory_gb: number(f.memory_gb),
      context: context === null ? null : Math.round(context),
      bind: f.bind.trim(),
    };
  }

  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  let dirty = $derived(info?.saved && form ? !same(values(form), info.saved) : false);

  // Saved and not yet in force. The address is left out when `--bind` is
  // given: the file's address is then never the one in force, and a restart
  // would not change that.
  let pending = $derived.by(() => {
    if (!info?.saved) return [];
    return Object.keys(NAMES).filter(
      (k) => !(k === "bind" && info.bind_flag) && !same(info.saved[k], info.running[k]),
    );
  });

  async function save(event) {
    event.preventDefault();
    saving = true;
    try {
      info = await api("/api/settings", {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(values(form)),
      });
      form = { ...info.saved };
      toasts.success("Saved to kvad.toml.");
    } catch (e) {
      toasts.error(e.message);
    } finally {
      saving = false;
    }
  }

  async function ask() {
    try {
      residents = (await health()).residents ?? [];
    } catch {
      residents = [];
    }
    asking = true;
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // A link to the new address, from this page's point of view: a server told
  // to listen everywhere is still reached by the name in the address bar.
  function linkTo(bind) {
    const port = bind.slice(bind.lastIndexOf(":") + 1);
    const host = bind.slice(0, bind.lastIndexOf(":"));
    const anywhere = host === "0.0.0.0" || host === "[::]";
    return `${location.protocol}//${anywhere ? location.hostname : host}:${port}/settings`;
  }

  async function restartNow() {
    asking = false;
    const moving =
      info.saved && !info.bind_flag && info.saved.bind !== info.running.bind ? info.saved.bind : null;
    let before = Infinity;
    try {
      before = (await health()).uptime_secs;
    } catch {
      // Measured against nothing, the first answer after a gap is enough.
    }
    try {
      await api("/api/restart", { method: "POST" });
    } catch (e) {
      toasts.error(e.message);
      return;
    }
    if (moving) {
      movedTo = linkTo(moving);
      return;
    }

    restart = "waiting";
    // Back when it answers after having stopped answering, or answers with
    // an uptime younger than the one it had: a poll can fall either side of
    // the moment it was down.
    const until = Date.now() + 90_000;
    let away = false;
    while (Date.now() < until) {
      await sleep(1000);
      try {
        const h = await health();
        if (away || h.uptime_secs < before) {
          restart = null;
          await load();
          toasts.success("Restarted. The settings in kvad.toml are in force.");
          return;
        }
      } catch {
        away = true;
      }
    }
    restart =
      "The server has not come back after 90 seconds. `kvad service logs` on its machine says why.";
  }
</script>

<section>
  <h2 class="mb-1 text-sm font-medium opacity-60">Server</h2>
  <p class="mb-3 text-xs opacity-60">
    <code>[server]</code> in <code>{info?.path ?? "kvad.toml"}</code>{info && !info.exists
      ? ", which does not exist yet; saving makes it"
      : ""}. The server reads the file as it starts, so a change is in force after a restart.
  </p>

  {#if problem}
    <div role="alert" class="alert alert-error alert-soft mb-3">
      <span>{problem} Fix the file by hand; this page will not write over it.</span>
    </div>
  {/if}

  {#if movedTo}
    <div role="alert" class="alert alert-info alert-soft mb-3">
      <span>
        The server is starting again at a new address, so this page will not hear from it
        here. <a class="link" href={movedTo}>Go to {movedTo}</a>
      </span>
    </div>
  {:else if restart === "waiting"}
    <div role="alert" class="alert alert-soft mb-3">
      <span class="loading loading-spinner loading-sm"></span>
      <span>Restarting. The page picks up where it was when the server answers again.</span>
    </div>
  {:else if restart}
    <div role="alert" class="alert alert-error alert-soft mb-3">
      <span>{restart}</span>
    </div>
  {:else if pending.length}
    <div role="alert" class="alert alert-warning alert-soft mb-3">
      <span>
        Saved, and not in force until a restart:
        {pending.map((k) => NAMES[k]).join(", ")}.
      </span>
      {#if info.can_restart}
        <button class="btn btn-sm" onclick={ask}>Restart now</button>
      {/if}
    </div>
  {/if}

  {#if form}
    <form class="flex flex-col gap-3" onsubmit={save}>
      <label class="flex items-start gap-3">
        <input type="checkbox" class="toggle toggle-sm mt-0.5" bind:checked={form.autoload} />
        <span>
          <span class="text-sm">Load the active model when the service starts</span>
          <span class="block text-xs opacity-60">
            Off, nothing is in memory after a restart until a request names a model.
          </span>
        </span>
      </label>

      <label class="flex items-start gap-3">
        <input type="checkbox" class="toggle toggle-sm mt-0.5" bind:checked={form.load_on_request} />
        <span>
          <span class="text-sm">Load a model a request names</span>
          <span class="block text-xs opacity-60">
            When it fits beside what is already in memory. Nothing is unloaded to make room.
          </span>
        </span>
      </label>

      <div class="flex flex-wrap gap-x-4">
        <fieldset class="fieldset">
          <legend class="fieldset-legend">Memory for models, GB</legend>
          <input
            class="input input-sm w-44"
            type="number"
            min="0.1"
            step="any"
            placeholder={info.defaults.memory_gb
              ? `${info.defaults.memory_gb.toFixed(1)} (all usable)`
              : "all usable"}
            bind:value={form.memory_gb}
          />
          <p class="label">Empty: this machine's usable memory.</p>
        </fieldset>

        <fieldset class="fieldset">
          <legend class="fieldset-legend">Context charged per model, tokens</legend>
          <input
            class="input input-sm w-44"
            type="number"
            min="1"
            step="1"
            placeholder={`${info.defaults.context}`}
            bind:value={form.context}
          />
          <p class="label">Empty: {info.defaults.context}.</p>
        </fieldset>

        <fieldset class="fieldset">
          <legend class="fieldset-legend">Listen address</legend>
          <input class="input input-sm w-52 font-mono" bind:value={form.bind} required />
          <p class="label">
            {#if info.bind_flag}
              The service is started with <code>--bind {info.bind_flag}</code>, which wins.
            {:else}
              Anything but loopback needs an auth mode.
            {/if}
          </p>
        </fieldset>
      </div>

      <div class="flex items-center gap-2">
        <button class="btn btn-sm" disabled={!dirty || saving}>
          {#if saving}<span class="loading loading-spinner loading-xs"></span>{/if}
          Save
        </button>
        {#if dirty}
          <button type="button" class="btn btn-sm btn-ghost" onclick={() => (form = { ...info.saved })}>
            Discard
          </button>
        {/if}
        {#if info.can_restart && !pending.length && restart === null && !movedTo}
          <button type="button" class="btn btn-sm btn-ghost ml-auto" onclick={ask}>
            Restart the server
          </button>
        {/if}
      </div>
    </form>
  {/if}

  {#if info}
    <details class="collapse collapse-arrow rounded-box bg-base-200/40 mt-4">
      <summary class="collapse-title text-sm">Set elsewhere</summary>
      <div class="collapse-content text-sm">
        <p class="mb-2 text-xs opacity-60">
          Shown, not changed here: the auth mode decides who may use this page, and a
          different data directory or database is a different server rather than a setting of
          this one. Edit <code>kvad.toml</code> for these.
        </p>
        <dl class="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
          <dt class="opacity-60">Auth</dt>
          <dd><code>{info.fixed.auth}</code></dd>
          <dt class="opacity-60">Data</dt>
          <dd class="break-all font-mono text-xs">{info.fixed.data_dir}</dd>
          <dt class="opacity-60">Database</dt>
          <dd class="break-all font-mono text-xs">{info.fixed.database}</dd>
        </dl>
      </div>
    </details>
  {/if}
</section>

{#if asking}
  <div class="modal modal-open" role="dialog">
    <div class="modal-box">
      <h3 class="text-lg font-medium">Restart the server?</h3>
      <p class="py-3 text-sm opacity-70">
        It stops taking requests, gives those under way five seconds to finish, and starts
        again with what kvad.toml says.
        {#if residents.length}
          {residents.length === 1 ? "The model" : `The ${residents.length} models`} in memory
          ({residents.map((r) => r.id).join(", ")}) {residents.length === 1 ? "is" : "are"}
          unloaded.
        {:else}
          Nothing is in memory.
        {/if}
        A download or training run that is going is interrupted, and marked failed.
      </p>
      <div class="modal-action">
        <button class="btn btn-sm" onclick={() => (asking = false)}>Cancel</button>
        <button class="btn btn-sm btn-warning" onclick={restartNow}>Restart</button>
      </div>
    </div>
    <button class="modal-backdrop" aria-label="Cancel" onclick={() => (asking = false)}></button>
  </div>
{/if}
