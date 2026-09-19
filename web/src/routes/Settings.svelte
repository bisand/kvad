<script>
  // The auth half of Settings. Thread counts, sampling defaults and the theme
  // are other phases' business; what is here is who may use this server and
  // what they may sign in with.
  import { api } from "../lib/api.js";
  import { auth } from "../lib/auth.svelte.js";
  import { toasts } from "../lib/toasts.svelte.js";
  import Icon from "../lib/components/Icon.svelte";

  const TRASH =
    "M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6";
  const COPY = "M8 8V5a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-3M5 8h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2z";

  let users = $state([]);
  let keys = $state([]);
  let sessions = $state([]);
  // A key is shown once, here, and then never again by anybody.
  let freshKey = $state(null);
  let newKeyName = $state("");
  let newUser = $state({ name: "", password: "", role: "user" });
  let passwords = $state({ current: "", next: "" });
  let confirming = $state(null);

  $effect(() => {
    load();
  });

  async function load() {
    if (!auth.hasAccounts) return;
    try {
      [keys, sessions] = await Promise.all([api("/api/keys"), api("/api/sessions")]);
      users = auth.isAdmin ? await api("/api/users") : [];
    } catch (e) {
      toasts.error(e.message);
    }
  }

  async function act(what, run) {
    try {
      await run();
      await load();
      return true;
    } catch (e) {
      toasts.error(`${what}: ${e.message}`);
      return false;
    }
  }

  const post = (path, body) =>
    api(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });

  async function createKey(event) {
    event.preventDefault();
    const name = newKeyName.trim();
    if (!name) return;
    await act("Could not make the key", async () => {
      const made = await post("/api/keys", { name });
      freshKey = made.token;
      newKeyName = "";
    });
  }

  async function createUser(event) {
    event.preventDefault();
    const made = { ...newUser, name: newUser.name.trim() };
    if (await act("Could not create the account", () => post("/api/users", made))) {
      newUser = { name: "", password: "", role: "user" };
      toasts.success(`Created ${made.name}.`);
    }
  }

  async function changePassword(event) {
    event.preventDefault();
    const ok = await act("Could not change the password", () =>
      post("/api/auth/password", { current: passwords.current, new: passwords.next }),
    );
    passwords = { current: "", next: "" };
    if (ok) {
      // Changing a password ends every session, this one included. Saying so
      // and then doing it is better than a page that quietly 401s.
      toasts.info("Password changed. Every session ended, including this one.");
      setTimeout(() => auth.signOut(), 1200);
    }
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

<div class="mx-auto flex max-w-3xl flex-col gap-8">
  {#if !auth.hasAccounts}
    <div role="alert" class="alert alert-soft">
      <span>
        This server runs with <code>auth.mode = "none"</code>: everyone who can reach it
        is an administrator, which is why it will only bind loopback. Set a mode in
        <code>kvad.toml</code> to have accounts, sessions and API keys.
      </span>
    </div>
  {:else}
    <!-- You. -->
    <section>
      <h2 class="mb-1 text-sm font-medium opacity-60">Your account</h2>
      <p class="mb-3 text-sm">
        Signed in as <span class="font-medium">{auth.who?.name}</span>
        <span class="badge badge-sm ml-1">{auth.who?.role}</span>
      </p>

      <form class="flex flex-wrap items-end gap-2" onsubmit={changePassword}>
        <fieldset class="fieldset">
          <legend class="fieldset-legend">Current password</legend>
          <input
            class="input input-sm"
            type="password"
            autocomplete="current-password"
            bind:value={passwords.current}
            required
          />
        </fieldset>
        <fieldset class="fieldset">
          <legend class="fieldset-legend">New password</legend>
          <input
            class="input input-sm"
            type="password"
            autocomplete="new-password"
            bind:value={passwords.next}
            required
          />
        </fieldset>
        <button class="btn btn-sm">Change</button>
      </form>
    </section>

    <!-- Keys. -->
    <section>
      <h2 class="mb-1 text-sm font-medium opacity-60">API keys</h2>
      <p class="mb-3 text-xs opacity-60">
        Bearer tokens for <code>/v1</code> and everything else, carrying your role. A key
        is shown once when it is made; the server keeps only its hash, so a lost key is
        revoked and replaced rather than looked up.
      </p>

      {#if freshKey}
        <div role="alert" class="alert alert-success mb-3 flex-col items-start gap-2 text-sm">
          <span>Copy this now — it is not shown again.</span>
          <div class="join w-full">
            <input class="input input-sm join-item w-full font-mono text-xs" readonly value={freshKey} />
            <button class="btn btn-sm join-item" onclick={() => copy(freshKey)} aria-label="Copy">
              <Icon path={COPY} size={14} />
            </button>
            <button class="btn btn-sm join-item" onclick={() => (freshKey = null)}>Done</button>
          </div>
        </div>
      {/if}

      <form class="join mb-3" onsubmit={createKey}>
        <input
          class="input input-sm join-item"
          placeholder="what it is for"
          bind:value={newKeyName}
          aria-label="Key name"
        />
        <button class="btn btn-sm join-item" disabled={!newKeyName.trim()}>New key</button>
      </form>

      {#if keys.length}
        <table class="table table-sm">
          <tbody>
            {#each keys as k (k.id)}
              <tr>
                <td class="w-full">{k.name}</td>
                <td class="font-mono text-xs whitespace-nowrap opacity-60">{k.prefix}…</td>
                <td class="text-xs whitespace-nowrap opacity-60">
                  {k.last_used_at ? `used ${k.last_used_at}` : "never used"}
                </td>
                <td class="text-right">
                  <button
                    class="btn btn-xs btn-ghost"
                    onclick={() =>
                      act("Could not revoke", () => api(`/api/keys/${k.id}`, { method: "DELETE" }))}
                  >
                    Revoke
                  </button>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      {:else}
        <p class="text-sm opacity-60">No keys.</p>
      {/if}
    </section>

    <!-- Sessions. -->
    <section>
      <h2 class="mb-1 text-sm font-medium opacity-60">Signed-in browsers</h2>
      <p class="mb-3 text-xs opacity-60">
        Sessions live on the server, so revoking one takes effect at once rather than
        whenever a token would have expired.
      </p>
      <table class="table table-sm">
        <tbody>
          {#each sessions as s (s.token_hash)}
            <tr>
              <td class="w-full max-w-0 truncate text-xs">{s.user_agent ?? "unknown"}</td>
              <td class="text-xs whitespace-nowrap opacity-60">until {s.expires_at}</td>
              <td class="text-right">
                <button
                  class="btn btn-xs btn-ghost"
                  onclick={() =>
                    act("Could not revoke", () =>
                      api(`/api/sessions/${s.token_hash}`, { method: "DELETE" }),
                    )}
                >
                  Revoke
                </button>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </section>

    <!-- Everybody, for an administrator. -->
    {#if auth.isAdmin}
      <section>
        <h2 class="mb-1 text-sm font-medium opacity-60">Accounts</h2>
        <p class="mb-3 text-xs opacity-60">
          An <span class="font-medium">admin</span> can change the machine — pull models,
          load them, manage accounts. A <span class="font-medium">user</span> can talk to
          what is already loaded, and sees only their own conversations.
        </p>

        <table class="table table-sm mb-3">
          <tbody>
            {#each users as u (u.id)}
              <tr>
                <td class="w-full">
                  {u.name}
                  {#if u.id === auth.who?.id}<span class="badge badge-sm badge-soft ml-1">you</span>{/if}
                </td>
                <td class="whitespace-nowrap">
                  <select
                    class="select select-xs w-24"
                    value={u.role}
                    disabled={u.id === auth.who?.id}
                    onchange={(e) =>
                      act("Could not change the role", () =>
                        api(`/api/users/${u.id}`, {
                          method: "PATCH",
                          headers: { "content-type": "application/json" },
                          body: JSON.stringify({ role: e.currentTarget.value }),
                        }),
                      )}
                  >
                    <option value="admin">admin</option>
                    <option value="user">user</option>
                  </select>
                </td>
                <td class="text-right">
                  <button
                    class="btn btn-xs btn-ghost"
                    disabled={u.id === auth.who?.id}
                    aria-label={`Delete ${u.name}`}
                    onclick={() => (confirming = u)}
                  >
                    <Icon path={TRASH} size={14} />
                  </button>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>

        <form class="flex flex-wrap items-end gap-2" onsubmit={createUser}>
          <fieldset class="fieldset">
            <legend class="fieldset-legend">Name</legend>
            <input class="input input-sm" bind:value={newUser.name} required />
          </fieldset>
          <fieldset class="fieldset">
            <legend class="fieldset-legend">Password</legend>
            <input
              class="input input-sm"
              type="password"
              autocomplete="new-password"
              bind:value={newUser.password}
              required
            />
          </fieldset>
          <fieldset class="fieldset">
            <legend class="fieldset-legend">Role</legend>
            <select class="select select-sm w-24" bind:value={newUser.role}>
              <option value="user">user</option>
              <option value="admin">admin</option>
            </select>
          </fieldset>
          <button class="btn btn-sm">Add</button>
        </form>
      </section>
    {/if}
  {/if}
</div>

{#if confirming}
  <div class="modal modal-open" role="dialog">
    <div class="modal-box">
      <h3 class="text-lg font-medium">Delete {confirming.name}?</h3>
      <p class="py-3 text-sm opacity-70">
        Their conversations, sessions and API keys go with the account. Anything they
        trained or downloaded stays: those belong to the machine.
      </p>
      <div class="modal-action">
        <button class="btn btn-sm" onclick={() => (confirming = null)}>Cancel</button>
        <button
          class="btn btn-sm btn-error"
          onclick={async () => {
            const u = confirming;
            confirming = null;
            await act("Could not delete", () => api(`/api/users/${u.id}`, { method: "DELETE" }));
          }}
        >
          Delete
        </button>
      </div>
    </div>
    <button class="modal-backdrop" aria-label="Cancel" onclick={() => (confirming = null)}></button>
  </div>
{/if}
