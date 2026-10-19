<script>
  // The whole of the app before anybody is signed in: either a sign-in form,
  // or — on a server that has no accounts yet — the form that makes the first
  // one against the token printed in the terminal.
  //
  // Password managers decide what a form is from its fields' id, name and
  // autocomplete, and offer to save a password only for a form they took to
  // be a sign-up or a sign-in. So the token says it is a one-time code and is
  // not the username, the username and password are called that, and setup
  // asks for the password twice, as a sign-up form does. A first account
  // made with a generated password nobody saved has no way back in but
  // `kvad-serve password` on the machine.
  import { auth } from "../lib/auth.svelte.js";
  import Secret from "../lib/components/Secret.svelte";

  let name = $state("");
  let password = $state("");
  let again = $state("");
  let shown = $state(false);
  let token = $state("");
  let busy = $state(false);
  // Either something this page just tried, or what the provider sent us back
  // with — the OIDC callback has nowhere to put an error but the URL.
  let problem = $state(new URLSearchParams(location.search).get("signin_error"));

  const setting_up = $derived(auth.needsSetup);
  const through_provider = $derived(auth.mode === "oidc");

  const differ = $derived(setting_up && again !== "" && again !== password);

  async function submit(event) {
    event.preventDefault();
    if (setting_up && password !== again) {
      problem = "The two passwords are not the same.";
      return;
    }
    busy = true;
    problem = null;
    try {
      if (setting_up) await auth.setUp(token.trim(), name.trim(), password);
      else await auth.signIn(name.trim(), password);
    } catch (e) {
      problem = e.message;
      password = "";
      again = "";
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

    {#if through_provider}
      <div class="card bg-base-100 border-base-300 border">
        <div class="card-body gap-4">
          <h1 class="text-lg font-medium">Sign in</h1>
          <p class="text-sm opacity-70">
            This server hands identity to a provider. You will come back here once it
            knows who you are.
          </p>
          {#if problem}
            <div role="alert" class="alert alert-error text-sm">{problem}</div>
          {/if}
          <!-- An ordinary link, not a fetch: the provider answers with a
               redirect to itself, which the browser has to follow. -->
          <a class="btn btn-primary" href="/api/auth/oidc/start">Continue to the provider</a>
        </div>
      </div>
    {:else}
    <!-- `post`, so that a submit that ever got past the handler would not
         put the password in a URL. -->
    <form class="card bg-base-100 border-base-300 border" method="post" onsubmit={submit}>
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
            <label class="fieldset-legend" for="setup-token">Setup token</label>
            <input
              id="setup-token"
              name="setup-token"
              class="input w-full font-mono text-xs"
              bind:value={token}
              autocomplete="one-time-code"
              autocapitalize="off"
              spellcheck="false"
              required
            />
          </fieldset>
        {:else}
          <h1 class="text-lg font-medium">Sign in</h1>
        {/if}

        <fieldset class="fieldset">
          <label class="fieldset-legend" for="username">Name</label>
          <!-- svelte-ignore a11y_autofocus -->
          <input
            id="username"
            name="username"
            class="input w-full"
            bind:value={name}
            autocomplete="username"
            autocapitalize="off"
            spellcheck="false"
            autofocus={!setting_up}
            required
          />
        </fieldset>

        <fieldset class="fieldset">
          <label class="fieldset-legend" for="password">Password</label>
          <Secret
            id="password"
            name="password"
            bind:value={password}
            bind:shown
            autocomplete={setting_up ? "new-password" : "current-password"}
            minlength={setting_up ? 8 : undefined}
            required
          />
          {#if setting_up}
            <p class="mt-1 text-xs opacity-60">
              At least 8 characters. Let your password manager save it: there is no
              reset by email.
            </p>
          {/if}
        </fieldset>

        {#if setting_up}
          <fieldset class="fieldset">
            <label class="fieldset-legend" for="password-again">The same again</label>
            <Secret
              id="password-again"
              name="password-again"
              bind:value={again}
              bind:shown
              autocomplete="new-password"
              aria-invalid={differ}
              required
            />
            {#if differ}
              <p class="text-error mt-1 text-xs">Not the same as the one above.</p>
            {/if}
          </fieldset>
        {/if}

        {#if problem}
          <div role="alert" class="alert alert-error text-sm">{problem}</div>
        {/if}

        <button class="btn btn-primary" disabled={busy}>
          {#if busy}<span class="loading loading-spinner loading-sm"></span>{/if}
          {setting_up ? "Create the first account" : "Sign in"}
        </button>
      </div>
    </form>
    {/if}

    {#if auth.mode === "basic"}
      <p class="mt-4 text-center text-xs opacity-60">
        This server is in <code>basic</code> mode. Scripts can send the same name and
        password as an HTTP Basic header.
      </p>
    {/if}
  </div>
</div>
