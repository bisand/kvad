// Light, dim, or whatever the operating system says.
//
// Three states rather than two, because "dark mode" is a question the OS has
// usually already answered and overriding that by default is rude. Nothing is
// written to `data-theme` while the choice is `system`: daisyUI's
// `--prefersdark` handles it in CSS, which is also why the inline script in
// `index.html` can be three lines.

const KEY = "kvad.theme";

/** The choices, in the order the menu shows them. */
export const THEMES = [
  { id: "system", label: "System" },
  { id: "light", label: "Light" },
  { id: "dim", label: "Dim" },
];

function stored() {
  try {
    const value = localStorage.getItem(KEY);
    return value === "light" || value === "dim" ? value : "system";
  } catch {
    // Private browsing, or storage turned off. Following the OS is the right
    // answer when we cannot remember being told otherwise.
    return "system";
  }
}

class Theme {
  current = $state(stored());

  set(id) {
    this.current = id;
    try {
      if (id === "system") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, id);
    } catch {}
    // Mirrors the inline script in index.html, which runs before this module
    // is even fetched.
    if (id === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = id;
  }
}

export const theme = new Theme();
