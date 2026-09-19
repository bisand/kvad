<script>
  import { router } from "./lib/router.svelte.js";
  import { pageFor } from "./lib/pages.js";
  import { health as fetchHealth } from "./lib/api.js";
  import { auth } from "./lib/auth.svelte.js";
  import { toasts } from "./lib/toasts.svelte.js";
  import Navbar from "./lib/components/Navbar.svelte";
  import Sidebar from "./lib/components/Sidebar.svelte";
  import Toasts from "./lib/components/Toasts.svelte";
  import Unbuilt from "./lib/components/Unbuilt.svelte";
  import Dashboard from "./routes/Dashboard.svelte";
  import Models from "./routes/Models.svelte";
  import Chat from "./routes/Chat.svelte";
  import Training from "./routes/Training.svelte";
  import Datasets from "./routes/Datasets.svelte";
  import Monitoring from "./routes/Monitoring.svelte";
  import Settings from "./routes/Settings.svelte";
  import SignIn from "./routes/SignIn.svelte";
  import NotFound from "./routes/NotFound.svelte";

  const DRAWER = "kvad-drawer";

  let health = $state(null);
  let healthError = $state(null);

  const page = $derived(pageFor(router.path));

  // Who this browser is, before anything else is drawn. Rendering the app and
  // then discovering every call is a 401 is a worse first impression than a
  // sign-in form.
  $effect(() => {
    auth.refresh();
  });

  // The server is polled rather than asked once, so that a restart while the
  // tab is open is noticed. Thirty seconds: often enough to spot a restart,
  // rare enough to be invisible in a request log.
  $effect(() => {
    if (!auth.signedIn) return;
    let alive = true;
    async function poll(first) {
      try {
        const next = await fetchHealth();
        if (!alive) return;
        if (healthError) toasts.success("Server is back.");
        health = next;
        healthError = null;
      } catch (e) {
        if (!alive) return;
        // Only the first failure is worth a toast; after that the navbar's
        // indicator says it, and a toast every thirty seconds would be noise.
        if (!healthError) toasts.error(e.message);
        healthError = e.message;
        if (first) health = null;
      }
    }
    poll(true);
    const timer = setInterval(() => poll(false), 30_000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  });
</script>

{#if !auth.ready}
  <!-- One frame, usually. Better than a flash of the app followed by a flash
       of the sign-in form. -->
  <div class="grid min-h-screen place-items-center">
    <span class="loading loading-spinner"></span>
  </div>
{:else if !auth.signedIn}
  <SignIn />
{:else}
<div class="drawer lg:drawer-open">
  <input id={DRAWER} type="checkbox" class="drawer-toggle" />

  <div class="drawer-content flex min-h-screen flex-col">
    <Navbar drawerId={DRAWER} title={page?.label ?? "Not found"} {health} error={healthError} />
    <main class="grow p-4 sm:p-6">
      {#if !page}
        <NotFound />
      {:else if page.path === "/"}
        <Dashboard {health} error={healthError} />
      {:else if page.path === "/models"}
        <Models />
      {:else if page.path === "/chat"}
        <Chat />
      {:else if page.path === "/training"}
        <Training />
      {:else if page.path === "/datasets"}
        <Datasets />
      {:else if page.path === "/monitoring"}
        <Monitoring />
      {:else if page.path === "/settings"}
        <Settings />
      {:else}
        <Unbuilt {page} />
      {/if}
    </main>
  </div>

  <Sidebar drawerId={DRAWER} />
</div>
{/if}

<Toasts />
