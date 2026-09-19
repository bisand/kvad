<script>
  // The whole of the app before anybody is signed in: either a sign-in form,
  // or — on a server that has no accounts yet — the form that makes the first
  // one against the token printed in the terminal.
  import { auth } from "../lib/auth.svelte.js";

  let name = $state("");
  let password = $state("");
  let token = $state("");
  let busy = $state(false);
  let problem = $state(null);

  const setting_up = $derived(auth.needsSetup);

  async function submit(event) {
    event.preventDefault();
    busy = true;
    problem = null;
    try {
      if (setting_up) await auth.setUp(token.trim(), name.trim(), password);
      else await auth.signIn(name.trim(), password);
    } catch (e) {
      problem = e.message;
      password = "";
    } finally {
      busy = false;
    }
  }
</script>

<div class="grid min-h-screen place-items-center p-4">
  <div class="w-full max-w-sm">
    <div class="mb-6 text-center">
      <p class="text-2xl font-semibold tracking-tight">kvad</p>
      <p class="mt-1 text-sm opacity-60">transformers from scratch</p>
    </div>

    <form class="card bg-base-100 border-base-300 border" onsubmit={submit}>
      <div class="card-body gap-4">
        {#if setting_up}
          <div>
            <h1 class="text-lg font-medium">Set this server up</h1>
            <p class="mt-1 text-sm opacity-70">
              It has no accounts yet. The one-time token is in the terminal where you
              started it.
            </p>
          </div>
          <fieldset class="fieldset">
            <legend class="fieldset-legend">Setup token</legend>
            <input
              class="input w-full font-mono text-xs"
              bind:value={token}
              autocomplete="off"
              spellcheck="false"
              required
            />
          </fieldset>
        {:else}
          <h1 class="text-lg font-medium">Sign in</h1>
        {/if}

        <fieldset class="fieldset">
          <legend class="fieldset-legend">Name</legend>
          <!-- svelte-ignore a11y_autofocus -->
          <input
            class="input w-full"
            bind:value={name}
            autocomplete="username"
            autofocus={!setting_up}
            required
          />
        </fieldset>

        <fieldset class="fieldset">
          <legend class="fieldset-legend">Password</legend>
          <input
            class="input w-full"
            type="password"
            bind:value={password}
            autocomplete={setting_up ? "new-password" : "current-password"}
            required
          />
          {#if setting_up}
            <p class="mt-1 text-xs opacity-60">At least 8 characters.</p>
          {/if}
        </fieldset>

        {#if problem}
          <div role="alert" class="alert alert-error text-sm">{problem}</div>
        {/if}

        <button class="btn btn-primary" disabled={busy}>
          {#if busy}<span class="loading loading-spinner loading-sm"></span>{/if}
          {setting_up ? "Create the first account" : "Sign in"}
        </button>
      </div>
    </form>

    {#if auth.mode === "basic"}
      <p class="mt-4 text-center text-xs opacity-60">
        This server is in <code>basic</code> mode. Scripts can send the same name and
        password as an HTTP Basic header.
      </p>
    {/if}
  </div>
</div>
