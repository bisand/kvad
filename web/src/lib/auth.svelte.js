// Who is signed in, and how to stop being.
//
// `/api/auth` is the one route that answers a request with no credential: it
// says which mode the server is in, whether it is waiting to be set up, and
// who — if anyone — this browser already is. Everything else in the UI waits
// for that answer, because rendering the app and then discovering it is all
// 401s is a worse first impression than a login form.

import { api } from "./api.js";
import { toasts } from "./toasts.svelte.js";

class Auth {
  /** null until the first answer arrives. */
  mode = $state(null);
  needsSetup = $state(false);
  who = $state(null);
  error = $state(null);

  get ready() {
    return this.mode !== null;
  }

  get signedIn() {
    return this.who !== null;
  }

  /** True when the server has accounts, so there is a sign-out to offer. */
  get hasAccounts() {
    return this.mode !== null && this.mode !== "none";
  }

  get isAdmin() {
    return this.who?.role === "admin";
  }

  async refresh() {
    try {
      const state = await api("/api/auth");
      this.mode = state.mode;
      this.needsSetup = state.needs_setup;
      this.who = state.signed_in;
      this.error = null;
    } catch (e) {
      this.error = e.message;
    }
  }

  async signIn(name, password) {
    this.who = await api("/api/auth/login", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, password }),
    });
    await this.refresh();
  }

  async setUp(token, name, password) {
    this.who = await api("/api/auth/setup", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ token, name, password }),
    });
    await this.refresh();
  }

  async signOut() {
    try {
      await api("/api/auth/logout", { method: "POST" });
    } catch (e) {
      toasts.error(e.message);
    }
    this.who = null;
    await this.refresh();
    // Everything cached belongs to whoever just left.
    window.location.href = "/";
  }
}

export const auth = new Auth();
