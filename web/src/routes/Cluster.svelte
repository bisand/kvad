<script>
  // One model across several machines: this server's part in it, the models
  // that are spread out now and where each token's time goes in them, and
  // who else on the network says they are a worker.
  import { api } from "../lib/api.js";
  import { humanBytes } from "../lib/models.svelte.js";

  let cluster = $state(null);
  let found = $state(null);
  let error = $state("");
  let looking = $state(false);

  // The page looks every few seconds without asking the network, which
  // takes three; who is on the network is asked once, and again on request.
  async function refresh() {
    try {
      cluster = await api("/api/cluster?find=false");
      error = "";
    } catch (e) {
      error = e.message;
    }
  }

  async function look() {
    looking = true;
    try {
      found = (await api("/api/cluster")).found ?? [];
    } catch (e) {
      error = e.message;
    } finally {
      looking = false;
    }
  }

  $effect(() => {
    refresh();
    look();
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  });

  const each = (total, runs) => (runs > 0 && total != null ? total / runs : null);
  const fixed = (v, digits = 2) => (v == null ? "—" : v.toFixed(digits));

  // A stage's share of a token: what this server saw, and for a stage on
  // another machine how much of that the machine says was its own layers.
  // What is left of a round trip is the hop.
  //
  // A worker is asked for its own time now and then, not every token, so
  // the two halves come from the moment it was last asked: the round trips
  // this server had seen by then, and what the worker said of them.
  function token(stage) {
    const whole = each(stage.one_ms, stage.one_runs);
    if (stage.here) return { whole, inside: whole, hop: null };
    const asked = each(stage.asked_one_ms, stage.asked_one_runs);
    const inside = each(stage.inside_one_ms, stage.asked_one_runs);
    return { whole, inside, hop: asked != null && inside != null ? Math.max(0, asked - inside) : null };
  }

  const reading = (stage) => (stage.many_ms > 0 ? (stage.many_tokens / stage.many_ms) * 1000 : null);
</script>

<div class="flex flex-col gap-6">
  {#if error}
    <div role="alert" class="alert alert-error"><span>{error}</span></div>
  {/if}

  <section>
    <h2 class="mb-1 text-sm font-medium opacity-60">This server</h2>
    <p class="mb-3 text-xs opacity-60">
      A model too large for one machine can have its layers divided between several. What
      passes from one to the next is the model's working state for each token, a few
      kilobytes. Set in <code>kvad.toml</code>, under <code>[cluster]</code> and
      <code>[worker]</code>.
    </p>
    {#if cluster}
      <ul class="flex flex-col gap-1 text-sm">
        <li>
          <span class="opacity-60">Build</span>
          <span class="font-mono text-xs">{cluster.build}</span>
          <span class="text-xs opacity-60">— every machine of a cluster has to be this one</span>
        </li>
        <li>
          {#if cluster.worker}
            Holds layers for other machines as
            <span class="font-mono text-xs">{cluster.worker.name}</span>
            (<span class="font-mono text-xs">{cluster.worker.id}</span>) on port {cluster.worker.port}{#if !cluster.worker.advertised}, not advertised: name it by its address{/if}.
          {:else}
            <span class="opacity-60">Holds no layers for other machines.</span>
          {/if}
        </li>
        <li>
          {#if cluster.workers.length > 0}
            Spreads a model too large for it over
            {#each cluster.workers as worker, i (worker)}{#if i > 0}, {/if}<span class="font-mono text-xs">{worker}</span>{/each},
            placed for <span class="font-mono text-xs">{cluster.place}</span>{#if cluster.split}, split
              <span class="font-mono text-xs">{cluster.split}</span> by hand{/if}.
          {:else}
            <span class="opacity-60">Spreads its own models over nobody.</span>
          {/if}
        </li>
        {#if !cluster.has_key}
          <li class="text-warning">
            No cluster key on this machine. A worker makes one the first time it starts, and
            every machine needs the same file.
          </li>
        {/if}
      </ul>
    {:else if !error}
      <span class="loading loading-dots loading-sm"></span>
    {/if}
  </section>

  <section>
    <h2 class="mb-1 text-sm font-medium opacity-60">Models across machines</h2>
    <p class="mb-3 text-xs opacity-60">
      A token goes through the stages one after another, so what it costs is their sum. For
      a stage on another machine the bar is in two parts: the time that machine says it was
      inside its own layers, and what is left of the round trip, which is the link. A prompt
      is read by every stage at once, a chunk each, and is as fast as the slowest.
    </p>
    {#if cluster && cluster.models.length === 0}
      <p class="text-sm opacity-60">
        None. A model that fits on this machine is loaded whole and is not listed here.
      </p>
    {/if}
    {#each cluster?.models ?? [] as model (model.repo + model.precision)}
      {@const tokens = model.stages.map(token)}
      {@const sum = tokens.reduce((total, t) => total + (t.whole ?? 0), 0)}
      <div class="rounded-box bg-base-200/40 mb-3 p-3">
        <div class="mb-2 flex flex-wrap items-baseline gap-2">
          <span class="font-mono text-sm">{model.repo}</span>
          <span class="badge badge-ghost badge-sm">{model.precision}</span>
          <span class="grow"></span>
          <span class="text-xs opacity-60">
            {#if sum > 0}{fixed(sum)} ms a token in its stages, {fixed(1000 / sum, 1)} tok/s{:else}nothing written yet{/if}
          </span>
        </div>
        <div class="overflow-x-auto">
          <table class="table table-sm">
            <thead>
              <tr>
                <th>Layers</th>
                <th>Machine</th>
                <th class="w-full">A token</th>
                <th class="text-right">In its layers</th>
                <th class="text-right">The hop</th>
                <th class="text-right whitespace-nowrap">Reads a prompt</th>
                <th class="text-right">Holds</th>
              </tr>
            </thead>
            <tbody>
              {#each model.stages as stage, i (stage.layers[0])}
                {@const t = tokens[i]}
                <tr>
                  <td class="font-mono text-xs whitespace-nowrap">{stage.layers[0]}–{stage.layers[1] - 1}</td>
                  <td class="text-xs whitespace-nowrap">
                    {stage.machine}
                    {#if stage.here}
                      <span class="badge badge-ghost badge-xs">here</span>
                    {:else if stage.link}
                      <div class="opacity-60">{stage.link}</div>
                    {/if}
                  </td>
                  <td class="max-w-0">
                    <div class="text-xs">{fixed(t.whole)} ms</div>
                    <div class="bg-base-300 mt-1 flex h-1.5 w-full overflow-hidden rounded">
                      {#if sum > 0 && t.whole != null}
                        <div class="bg-primary h-1.5" style:width="{((t.inside ?? t.whole) / sum) * 100}%"></div>
                        {#if t.hop != null}
                          <div class="bg-warning h-1.5" style:width="{(t.hop / sum) * 100}%"></div>
                        {/if}
                      {/if}
                    </div>
                  </td>
                  <td class="text-right text-xs whitespace-nowrap">{fixed(t.inside)} ms</td>
                  <td class="text-right text-xs whitespace-nowrap">
                    {#if stage.here}—{:else}{fixed(t.hop)} ms{/if}
                  </td>
                  <td class="text-right text-xs whitespace-nowrap opacity-70">
                    {#if reading(stage) != null}{fixed(reading(stage), 0)} tok/s{:else}—{/if}
                  </td>
                  <td class="text-right text-xs whitespace-nowrap opacity-70">{humanBytes(stage.bytes)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
        <p class="mt-2 text-xs opacity-60">
          <span class="bg-primary mr-1 inline-block h-2 w-2 rounded-sm"></span>layers
          <span class="bg-warning mr-1 ml-3 inline-block h-2 w-2 rounded-sm"></span>the link.
          Averages since the model was loaded; a worker is asked for its own time every couple
          of seconds while the model is in use.
        </p>
      </div>
    {/each}
  </section>

  <section>
    <div class="mb-1 flex items-center gap-3">
      <h2 class="text-sm font-medium opacity-60">Workers on the network</h2>
      <span class="grow"></span>
      <button class="btn btn-xs" onclick={look} disabled={looking}>
        {#if looking}<span class="loading loading-spinner loading-xs"></span>{/if}
        Look again
      </button>
    </div>
    <p class="mb-3 text-xs opacity-60">
      Every machine that says it is a worker, whether or not this server uses it. Asking
      takes a few seconds: an answer to multicast is waited for.
    </p>
    {#if found == null}
      <span class="loading loading-dots loading-sm"></span>
    {:else if found.length === 0}
      <p class="text-sm opacity-60">
        Nobody. On another machine, <code>kvad worker</code> or <code>[worker] enabled = true</code>
        makes one.
      </p>
    {:else}
      <div class="overflow-x-auto">
        <table class="table table-sm">
          <thead>
            <tr>
              <th>Worker</th>
              <th>Id</th>
              <th>Build</th>
              <th class="w-full">Reached at</th>
            </tr>
          </thead>
          <tbody>
            {#each found as worker (worker.id + worker.name)}
              <tr>
                <td class="text-sm whitespace-nowrap">
                  {worker.name}
                  {#if worker.this_machine}<span class="badge badge-ghost badge-xs">this server</span>{/if}
                  {#if worker.named}<span class="badge badge-ghost badge-xs">named in [cluster]</span>{/if}
                </td>
                <td class="font-mono text-xs">{worker.id}</td>
                <td class="font-mono text-xs whitespace-nowrap">
                  {worker.build}
                  {#if !worker.same_build}
                    <span class="badge badge-warning badge-xs">another build: would be refused</span>
                  {/if}
                </td>
                <td class="font-mono text-xs opacity-70">
                  {#each worker.addresses as address (address)}<div>{address}</div>{/each}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>
</div>
