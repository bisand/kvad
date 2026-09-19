// Somewhere for the whole app to say something short.
//
// A list rather than a single message, because two things can go wrong at
// once and the second must not erase the first. Errors stay until dismissed —
// an error that vanishes before it is read may as well not have happened —
// and everything else clears itself.

let next = 0;

class Toasts {
  items = $state([]);

  /**
   * @param {string} message
   * @param {"info"|"success"|"warning"|"error"} kind
   */
  push(message, kind = "info") {
    const id = next++;
    this.items.push({ id, message, kind });
    if (kind !== "error") setTimeout(() => this.dismiss(id), 4000);
    return id;
  }

  info(message) {
    return this.push(message, "info");
  }
  success(message) {
    return this.push(message, "success");
  }
  warning(message) {
    return this.push(message, "warning");
  }
  error(message) {
    return this.push(message, "error");
  }

  dismiss(id) {
    this.items = this.items.filter((t) => t.id !== id);
  }
}

export const toasts = new Toasts();
