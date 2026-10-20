<script>
  // Devices paired through the tunnel: the list, and for an administrator
  // pairing a new one. The web half of `kvad devices` and `kvad remote pair`:
  // the same routes and the same single-use, ten-minute link.
  //
  // The link is as good as a password while it lives, so it is held in this
  // component's state and nowhere else: not in a URL, the title, storage or
  // the console. The server draws its QR code, so no encoder ships here.
  // Leaving the page closes the slot, as Cancel does.
  import { onDestroy } from "svelte";
  import { api } from "../api.js";
  import { auth } from "../auth.svelte.js";
  import { toasts } from "../toasts.svelte.js";
  import Icon from "./Icon.svelte";

  // Called when a device comes or goes, since each has an API key that
  // Settings lists.
  let { onchange = () => {} } = $props();

  const COPY =
    "M8 8V5a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-3M5 8h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2z";

  // `null` until the server has said whether the tunnel is on.
  let remote = $state(null);
  let devices = $state([]);
  let allowAdmin = $state(false);
  let opening = $state(false);
  // The open slot: what the server answered, when it stops working, and
  // which devices there were before it.
  let pairing = $state(null);
  // How the last slot ended, when it ended without a device.
  let ended = $state("");
  let now = $state(Date.now());
  let confirming = $state(null);

  // An administrator sees every account's devices, with their owners.
  const listPath = () => (auth.isAdmin ? "/api/devices?all=true" : "/api/devices");

  $effect(() => {
    load();
  });

  async function load() {
    try {
      remote = await api("/api/remote");
      if (remote.enabled) devices = await api(listPath());
    } catch (e) {
      toasts.error(e.message);
    }
  }

  let left = $derived(pairing ? Math.max(0, Math.ceil((pairing.until - now) / 1000)) : 0);
  let clock = $derived(`${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`);

  // A whole number of pixels to a module, at least three: a pairing link is
  // a version 19 code or so, about 101 modules with its margin, and
  // fractional modules blur into each other on a screen a camera reads.
  let qrWidth = $derived.by(() => {
    const span = Number(pairing?.qr.match(/viewBox="0 0 (\d+)/)?.[1] ?? 0);
    return span ? span * Math.max(3, Math.floor(280 / span)) : 0;
  });

  // While a slot is open, tick the countdown and ask whether it still is.
  // Once it is not, the link is spent one way or another: by the device it
  // paired, by a wrong try, or by time. The code goes, and the list says
  // which.
  $effect(() => {
    if (!pairing) return;
    const { slot, known } = pairing;
    const tick = setInterval(() => (now = Date.now()), 1000);
    const watch = setInterval(async () => {
      let open = Date.now() < pairing.until;
      try {
        if (open) open = (await api(`/api/remote/pair/${slot}`)).open;
      } catch {
        // The next tick asks again; the countdown carries on regardless.
        return;
      }
      if (open || pairing?.slot !== slot) return;
      const expired = Date.now() >= pairing.until;
      pairing = null;
      // The server takes the slot a moment before it writes the device, so
      // a list that has not caught up is asked once more.
      let fresh = null;
      for (const wait of [0, 1500]) {
        await new Promise((done) => setTimeout(done, wait));
        try {
          devices = await api(listPath());
        } catch (e) {
          toasts.error(e.message);
        }
        fresh = devices.find((d) => !known.has(d.id));
        if (fresh || expired) break;
      }
      if (fresh) {
        ended = "";
        onchange();
        toasts.success(`Paired ${fresh.name}.`);
      } else {
        ended = expired
          ? "The link expired before a device used it."
          : "The link was used without pairing a device, so it no longer works.";
      }
    }, 2000);
    return () => {
      clearInterval(tick);
      clearInterval(watch);
    };
  });

  async function pair() {
    opening = true;
    ended = "";
    try {
      const made = await api("/api/remote/pair", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ admin: allowAdmin, qr: true }),
      });
      now = Date.now();
      pairing = {
        ...made,
        until: now + made.expires_in * 1000,
        known: new Set(devices.map((d) => d.id)),
      };
    } catch (e) {
      toasts.error(e.message);
    } finally {
      opening = false;
    }
  }

  // Closes the slot on the server, so the link stops working now rather
  // than at the end of its ten minutes. `keepalive` lets it finish when the
  // page is going away.
  function close(keepalive = false) {
    const slot = pairing?.slot;
    pairing = null;
    if (!slot) return Promise.resolve();
    return fetch(`/api/remote/pair/${slot}`, { method: "DELETE", keepalive }).catch(() => {});
  }

  async function cancel() {
    await close();
    ended = "";
    toasts.info("Cancelled: that link no longer works.");
  }

  onDestroy(() => close(true));
  $effect(() => {
    const leaving = () => close(true);
    addEventListener("pagehide", leaving);
    return () => removeEventListener("pagehide", leaving);
  });

  async function unpair(d) {
    confirming = null;
    try {
      await api(`/api/devices/${d.id}`, { method: "DELETE" });
      toasts.success(`Unpaired ${d.name}.`);
      onchange();
    } catch (e) {
      toasts.error(e.message);
    }
    await load();
  }

  async function copy(text) {
    try {
      await navigator.clipboard.writeText(text);
      toasts.success("Copied. Paste it only where it is meant to go.");
    } catch {
      toasts.warning("The browser would not let this page use the clipboard.");
    }
  }
</script>

{#if remote?.enabled}
  <section>
    <h2 class="mb-1 text-sm font-medium opacity-60">Devices</h2>
    <p class="mb-3 text-xs opacity-60">
      Phones and other machines that reach this server through its encrypted tunnel, from
      anywhere. Each has a key that works only from that device, and acts as
      <span class="font-medium">user</span> unless it was paired with admin allowed.
      Unpairing revokes the key and closes its connection at once.
    </p>

    {#if pairing}
      <div class="card card-border bg-base-200/40 mb-3">
        <div class="card-body flex-col gap-4 p-4 sm:flex-row">
          <div
            class="rounded-box self-center overflow-hidden sm:self-start [&>svg]:block [&>svg]:size-full"
            style:width="{qrWidth}px"
            style:height="{qrWidth}px"
            role="img"
            aria-label="QR code of the pairing link"
          >
            <!-- Drawn by kvad-serve from the link: only path data, no text. -->
            {@html pairing.qr}
          </div>
          <div class="flex min-w-0 flex-1 flex-col gap-2 text-sm">
            <div role="alert" class="alert alert-warning alert-soft text-xs">
              <span>
                This code is as good as a password until it is used or expires, in
                <span class="font-mono font-medium">{clock}</span>. Anyone who scans it or has
                the link can pair a device to your account, so show it only to the device you
                mean to pair.
              </span>
            </div>
            <p>
              Scan it with Kvad on the phone, or on another machine run
              <code>kvad remote join</code> with the link. The device will act as
              <span class="badge badge-sm">{pairing.role}</span>.
            </p>
            <div class="flex items-start gap-2">
              <p
                class="bg-base-100 rounded-field min-w-0 flex-1 p-2 font-mono text-xs break-all select-all"
                aria-label="Pairing link"
              >
                {pairing.link}
              </p>
              <button class="btn btn-sm" onclick={() => copy(pairing.link)}>
                <Icon path={COPY} size={14} /> Copy
              </button>
            </div>
            <div>
              <button class="btn btn-sm" onclick={cancel}>Cancel</button>
            </div>
          </div>
        </div>
      </div>
    {:else if !auth.isAdmin}
      <p class="mb-3 text-xs opacity-60">An administrator pairs new devices.</p>
    {:else}
      {#if ended}
        <p class="mb-2 text-sm">{ended}</p>
      {/if}
      <div class="mb-3 flex flex-wrap items-center gap-3">
        <button class="btn btn-sm" onclick={pair} disabled={opening}>Pair a device</button>
        <label class="label text-sm">
          <input type="checkbox" class="checkbox checkbox-sm" bind:checked={allowAdmin} />
          allow admin
        </label>
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
    {:else}
      <p class="text-sm opacity-60">No devices.</p>
    {/if}
  </section>
{/if}

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
