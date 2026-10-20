<script>
  // A password field, with a button that shows what is in it: a password a
  // manager generated is otherwise a row of dots nobody can check or copy.
  // `shown` is bindable so that a password and its confirmation are shown
  // and hidden together. Everything else — id, name, autocomplete — goes to
  // the <input>, where password managers look for it.
  import Icon from "./Icon.svelte";

  let { value = $bindable(""), shown = $bindable(false), small = false, ...rest } = $props();

  const EYE = "M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z";
  const EYE_OFF =
    "M3 3l18 18M10.6 10.6a2 2 0 0 0 2.8 2.8M9.9 5.1A9.8 9.8 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-2.2 3.2M6.6 6.6A16 16 0 0 0 2 12s3.5 7 10 7a9.6 9.6 0 0 0 5.4-1.6";
</script>

<label class={small ? "input input-sm" : "input w-full"}>
  <input type={shown ? "text" : "password"} bind:value spellcheck="false" {...rest} />
  <button
    type="button"
    class="btn btn-ghost btn-circle btn-xs -me-1"
    aria-label={shown ? "Hide the password" : "Show the password"}
    aria-pressed={shown}
    onclick={() => (shown = !shown)}
  >
    <Icon path={shown ? EYE_OFF : EYE} size={small ? 14 : 16} />
  </button>
</label>
