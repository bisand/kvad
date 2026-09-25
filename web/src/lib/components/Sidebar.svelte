<script>
  import { PAGES } from "../pages.js";
  import { router, navigate } from "../router.svelte.js";
  import Icon from "./Icon.svelte";

  // The id of the drawer's checkbox, so picking a page can close the drawer
  // on a narrow screen. On a wide one the checkbox is not what is holding the
  // sidebar open (`lg:drawer-open` is), so clearing it there does nothing.
  let { drawerId } = $props();

  function pick(event, path) {
    navigate(event, path);
    const toggle = document.getElementById(drawerId);
    if (toggle) toggle.checked = false;
  }
</script>

<div class="drawer-side z-20">
  <label for={drawerId} aria-label="close sidebar" class="drawer-overlay"></label>
  <!-- The border matters in the light theme, where base-200 and base-100 are
       close enough that the sidebar would otherwise have no edge. -->
  <nav class="bg-base-200 border-base-300 flex min-h-full w-64 flex-col border-r">
    <a
      href="/"
      onclick={(e) => pick(e, "/")}
      class="flex items-baseline gap-2 px-4 h-16 shrink-0"
    >
      <!-- The mark is docs/brand/kvad-mark.svg, inline so it takes the text
           colour and follows the theme, and sized in em so it sits on the
           word like a letter. At 20px it leaves the tagline room for one line. -->
      <span class="text-xl font-semibold tracking-tight"
        ><svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="5 7 54 48"
          class="mr-1 inline h-[0.9em] w-auto align-[-0.1em]"
          fill="none"
          stroke="currentColor"
          stroke-linecap="round"
          aria-hidden="true"
        >
          <path d="M32 28 C30 21.5 26 13.5 20 9.5 M32 28 C34 21.5 38 13.5 44 9.5" stroke-width="2.6" opacity=".45" />
          <path d="M32 39.5 C29 29.9 23.1 18.1 14.2 12.1 M32 39.5 C35 29.9 40.9 18.1 49.8 12.1" stroke-width="2.6" opacity=".7" />
          <path d="M32 52 C28 39 20 23 8 15 M32 52 C36 39 44 23 56 15" stroke-width="5" />
        </svg>kvad</span
      >
      <span class="text-xs opacity-60">transformers from scratch</span>
    </a>

    <ul class="menu w-full grow gap-0.5 px-2">
      {#each PAGES as page (page.path)}
        <li>
          <a
            href={page.path}
            onclick={(e) => pick(e, page.path)}
            class={router.path === page.path ? "menu-active" : ""}
            aria-current={router.path === page.path ? "page" : undefined}
          >
            <Icon path={page.icon} />
            <span class="grow">{page.label}</span>
          </a>
        </li>
      {/each}
    </ul>

    <p class="px-4 py-3 text-xs opacity-50">
      Several models in memory, one generation at a time. Continuous batching is not here yet.
    </p>
  </nav>
</div>
