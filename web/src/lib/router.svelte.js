// A router, in about forty lines.
//
// The pages are a fixed list known at build time and there is no nesting, so
// the whole job is: keep the current path in state, intercept clicks on
// internal links, and answer the back button. A routing library would be more
// code than the thing it routes.
//
// Reloading `/models` works because the server hands back `index.html` for
// any path that does not look like a file; see `assets.rs`.

class Router {
  path = $state(window.location.pathname);

  constructor() {
    window.addEventListener("popstate", () => {
      this.path = window.location.pathname;
    });
  }

  go(path) {
    if (path === this.path) return;
    window.history.pushState({}, "", path);
    this.path = path;
    window.scrollTo(0, 0);
  }
}

export const router = new Router();

/**
 * A click on an internal link, handled here instead of by the browser.
 *
 * Modified clicks are left alone on purpose: ctrl-click, middle-click and
 * shift-click mean "somewhere else", and a router that swallowed them would
 * break the one habit every browser user has.
 */
export function navigate(event, path) {
  if (event.defaultPrevented) return;
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  if (event.button !== undefined && event.button !== 0) return;
  event.preventDefault();
  router.go(path);
}
