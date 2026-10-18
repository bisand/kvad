<script>
  // Devices paired through the tunnel: pairing a new one, and the list.
  // The web half of `kvad remote pair` and `kvad devices` — the same routes,
  // the same single-use, ten-minute link.
  import { api } from "../api.js";
  import { auth } from "../auth.svelte.js";
  import { toasts } from "../toasts.svelte.js";
  import Icon from "./Icon.svelte";
  import QrCode from "./QrCode.svelte";

  const COPY = "M8 8V5a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-3M5 8h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2z";

  // `null` until the server has said whether the tunnel is on.
  let remote = $state(null);
  let devices = $state([]);
  let asAdmin = $state(false);
  // The open link: what the server answered, and when it stops working.
  let pairing = $state(null);
  let now = $state(Date.now());
  let confirming = $state(null);

  // An administrator sees every account's devices, with their owners.
  const listPath = () => (auth.isAdmin ? "/api/devices?all=true" : "/api/devices");

  $effect(() => {
    load();
  });

  async function load() {
    try {
      [remote, devices] = await Promise.all([api("/api/remote"), api(listPath())]);
    } catch (e) {
      toasts.error(e.message);
    }
  }

  let left = $derived(pairing ? Math.max(0, Math.ceil((pairing.until - now) / 1000)) : 0);
  let clock = $derived(`${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`);

  // While a link is open, tick the countdown and watch for the device it
  // pairs: the slot is good once, so the first new device is that one, and
  // the link is spent the moment it appears.
  $effect(() => {
    if (!pairing) return;
    const known = pairing.known;
    const tick = setInterval(() => (now = Date.now()), 1000);
    const watch = setInterval(async () => {
      if (Date.now() >= pairing.until) return;
      try {
        const listed = await api(listPath());
        const fresh = listed.find((d) => !known.has(d.id));
        devices = listed;
        if (fresh) {
          pairing = null;
          toasts.success(`Paired ${fresh.name}.`);
        }
      } catch {
        // The next tick tries again; the countdown carries on regardless.
      }
    }, 2000);
    return () => {
      clearInterval(tick);
      clearInterval(watch);
    };
  });

  async function pair() {
    try {
      const made = await api("/api/remote/pair", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ admin: auth.isAdmin && asAdmin }),
      });
      now = Date.now();
      pairing = { ...made, until: now + made.expires_in * 1000, known: new Set(devices.map((d) => d.id)) };
    } catch (e) {
      toasts.error(e.message);
    }
  }

  async function unpair(d) {
    confirming = null;
    try {
      await api(`/api/devices/${d.id}`, { method: "DELETE" });
      toasts.success(`Unpaired ${d.name}.`);
    } catch (e) {
      toasts.error(e.message);
    }
    await load();
  }

  async function copy(text) {
    try {
      await navigator.clipboard.writeText(text);
      toasts.success("Copied.");
    } catch {
      toasts.warning("The browser would not let this page use the clipboard.");
    }
  }
</script>

<section>
  <h2 class="mb-1 text-sm font-medium opacity-60">Devices</h2>
  <p class="mb-3 text-xs opacity-60">
    Phones and other machines that reach this server through its encrypted tunnel, from
    anywhere. Each has a key that works only from that device, and acts as
    <span class="font-medium">user</span> unless it was paired as an administrator.
    Unpairing revokes the key and closes its connection at once.
  </p>

  {#if remote && !remote.enabled}
    <p class="mb-3 text-sm">
      The tunnel is off. A <code>[remote]</code> section in <code>kvad.toml</code> turns it on:
      see “From anywhere, through the tunnel” in <code>docs/serve.md</code>.
    </p>
  {:else if remote && pairing}
    <div class="card card-border bg-base-200/40 mb-3">
      <div class="card-body flex-col gap-4 p-4 sm:flex-row">
        {#if left > 0}
          <div class="self-center sm:self-start"><QrCode text={pairing.link} /></div>
          <div class="flex min-w-0 flex-1 flex-col gap-2 text-sm">
            <p>
              Scan this with Kvad on the phone, or on another machine run
              <code>kvad remote join</code> with the link. The device will act as
              <span class="badge badge-sm">{pairing.role}</span> on
              <span class="font-mono text-xs" title={pairing.server}>{pairing.server.slice(0, 17)}…</span>.
            </p>
            <div class="join w-full">
              <input
                class="input input-sm join-item w-full font-mono text-xs"
                readonly
                value={pairing.link}
                aria-label="Pairing link"
              />
              <button class="btn btn-sm join-item" onclick={() => copy(pairing.link)} aria-label="Copy the link">
                <Icon path={COPY} size={14} />
              </button>
            </div>
            <div role="alert" class="alert alert-warning alert-soft text-xs">
              <span>
                Good once, for <span class="font-mono font-medium">{clock}</span> more. Whoever
                has this link in that time can pair, so keep it on this screen.
              </span>
            </div>
            <div>
              <button class="btn btn-sm" onclick={() => (pairing = null)}>Close</button>
            </div>
          </div>
        {:else}
          <div class="flex flex-1 flex-wrap items-center gap-2 text-sm">
            <span class="flex-1">The link expired before a device used it.</span>
            <button class="btn btn-sm" onclick={pair}>New link</button>
            <button class="btn btn-sm btn-ghost" onclick={() => (pairing = null)}>Close</button>
          </div>
        {/if}
      </div>
    </div>
  {:else if remote}
    <div class="mb-3 flex flex-wrap items-center gap-3">
      <button class="btn btn-sm" onclick={pair}>Pair a device</button>
      {#if auth.isAdmin}
        <label class="label text-sm">
          <input type="checkbox" class="checkbox checkbox-sm" bind:checked={asAdmin} />
          as an administrator
        </label>
      {/if}
    </div>
  {/if}

  {#if devices.length}
    <table class="table table-sm">
      <tbody>
        {#each devices as d (d.id)}
          <tr>
            <td class="w-full">
              <div class="flex flex-wrap items-center gap-1">
                <span class="font-medium">{d.name}</span>
                <span class="badge badge-sm">{d.admin ? "admin" : "user"}</span>
                {#if auth.isAdmin && d.user !== auth.who?.id}
                  <span class="text-xs opacity-60">of {d.owner}</span>
                {/if}
              </div>
              <div class="flex flex-wrap gap-x-3 text-xs opacity-60">
                <span class="font-mono" title={d.node}>{d.node.slice(0, 10)}…</span>
                <span>paired {d.paired_at}</span>
                <span>{d.last_seen ? `seen ${d.last_seen}` : "never seen"}</span>
              </div>
            </td>
            <td class="text-right">
              <button class="btn btn-xs btn-ghost" onclick={() => (confirming = d)}>Unpair</button>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  {:else if remote?.enabled}
    <p class="text-sm opacity-60">No devices.</p>
  {/if}
</section>

{#if confirming}
  <div class="modal modal-open" role="dialog">
    <div class="modal-box">
      <h3 class="text-lg font-medium">Unpair {confirming.name}?</h3>
      <p class="py-3 text-sm opacity-70">
        Its key stops working and its connection closes now. To use it again it has to be
        paired again, with a new link.
      </p>
      <div class="modal-action">
        <button class="btn btn-sm" onclick={() => (confirming = null)}>Cancel</button>
        <button class="btn btn-sm btn-error" onclick={() => unpair(confirming)}>Unpair</button>
      </div>
    </div>
    <button class="modal-backdrop" aria-label="Cancel" onclick={() => (confirming = null)}></button>
  </div>
{/if}
