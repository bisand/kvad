// Text to video: the request being made, and the videos already kept.
//
// The page talks to `/v1/videos` like any other client. That endpoint is a
// job, not a stream: asking for a video answers at once, and the video is
// the server's to make from then on. So the page asks, then watches the
// gallery, which is where a video being made lives as well as a finished
// one. Closing the tab stops the watching and nothing else; the video is
// there when the page is opened again.

import { api } from "./api.js";
import { toasts } from "./toasts.svelte.js";
import { models } from "./models.svelte.js";

/** Sizes worth offering. LTX-2.5 wants both sides a multiple of 64, and was
 *  measured here at 768×512 and 1536×1024; anything else can be typed. */
export const SIZES = [
  { label: "Landscape", width: 768, height: 512 },
  { label: "Portrait", width: 512, height: 768 },
  { label: "Wide", width: 1024, height: 576 },
  { label: "Square", width: 640, height: 640 },
  { label: "Large landscape", width: 1536, height: 1024 },
  { label: "Large portrait", width: 1024, height: 1536 },
];

/** How often the gallery is asked again while something in it is live. */
const POLL_MS = 2000;

/** A video that is still to be made or being made. */
export const live = (v) => v.status === "queued" || v.status === "in_progress";

class Videos {
  /** Which video model to ask: a repo id, or a resident's `repo@backend`. */
  model = $state(null);
  prompt = $state("a lighthouse on a cliff at dusk, waves breaking on the rocks below, gulls calling");
  /** Blank means the model's own default. */
  width = $state(null);
  height = $state(null);
  seconds = $state(null);
  fps = $state(null);
  seed = $state(null);
  fixSeed = $state(false);
  sound = $state(true);

  /** Loading the model, or sending the request. */
  starting = $state(false);
  gallery = $state([]);
  #timer = null;
  #watchers = 0;

  /** Video models on this machine that could be asked, in memory first. */
  get choices() {
    const here = models.videoResidents.map((r) => ({ id: r.id, label: `${r.repo} · ${r.backend}`, resident: r }));
    const onDisk = models.all
      .filter((m) => m.kind === "video" && m.runnable && !models.resident(m.id))
      .map((m) => ({ id: m.id, label: `${m.id} (not loaded)`, resident: null }));
    return [...here, ...onDisk];
  }

  get chosen() {
    return this.choices.find((c) => c.id === this.model) ?? this.choices[0] ?? null;
  }

  /** What the chosen model does with a knob left blank, and its limits,
   *  once it is loaded. */
  get defaults() {
    return this.chosen?.resident?.video ?? null;
  }

  /** The frames a length in seconds makes: the nearest the model can, which
   *  for LTX is 8k + 1. */
  frames(d = this.defaults) {
    if (!d || !this.seconds) return null;
    const fps = Number(this.fps || d.fps);
    return Math.max(1, Math.round((Number(this.seconds) * fps) / d.frame_step)) * d.frame_step + 1;
  }

  /** Whether what is in the form is more than the model makes here. */
  get tooBig() {
    const d = this.defaults;
    if (!d) return false;
    const w = Number(this.width || d.width);
    const h = Number(this.height || d.height);
    const f = this.frames() ?? d.frames;
    return w * h * f > d.max_volume || f > d.max_frames;
  }

  /** The videos being made. */
  get running() {
    return this.gallery.filter(live);
  }

  async refresh() {
    try {
      this.gallery = (await api("/v1/videos?limit=100")).data;
    } catch (e) {
      toasts.error(e.message);
    }
    this.#schedule();
  }

  /** Ask again while anything is live and anybody is watching. */
  #schedule() {
    clearTimeout(this.#timer);
    this.#timer = null;
    if (this.#watchers > 0 && this.running.length > 0) {
      this.#timer = setTimeout(() => this.refresh(), POLL_MS);
    }
  }

  /** The page is open: keep the gallery current. Returns the undo. */
  watch() {
    this.#watchers += 1;
    this.refresh();
    return () => {
      this.#watchers -= 1;
      this.#schedule();
    };
  }

  async make() {
    const chosen = this.chosen;
    if (this.starting || !chosen || !this.prompt.trim()) return;
    this.starting = true;
    try {
      // A model that is not in memory is loaded first, as the Images page
      // does, so that a load shows as a load.
      let target = chosen.id;
      if (!chosen.resident) {
        if (!(await models.load(chosen.id))) return;
        target = models.videoResidents.find((r) => r.repo === chosen.id)?.id ?? chosen.id;
        this.model = target;
      }
      const body = { model: target, prompt: this.prompt, audio: this.sound };
      if (this.width && this.height) body.size = `${this.width}x${this.height}`;
      const frames = this.frames();
      if (frames) body.frames = frames;
      if (this.fps) body.fps = Number(this.fps);
      if (this.fixSeed && this.seed != null && this.seed !== "") body.seed = Number(this.seed);
      const video = await api("/v1/videos", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      // A seed nobody fixed is still worth keeping: it is how this video is
      // made again.
      this.seed = video.kvad.seed;
      this.gallery = [video, ...this.gallery];
      this.#schedule();
    } catch (e) {
      toasts.error(e.message);
    } finally {
      this.starting = false;
    }
  }

  /** Put an old video's settings back in the form. */
  reuse(v) {
    const k = v.kvad;
    this.prompt = v.prompt;
    this.width = k.width;
    this.height = k.height;
    this.seconds = Number(v.seconds);
    this.fps = k.fps;
    this.seed = k.seed;
    this.fixSeed = true;
    this.sound = k.audio;
    const resident = models.videoResidents.find((r) => r.repo === v.model);
    this.model = resident?.id ?? v.model;
  }

  /** Delete a video; one still being made stops at its next step. */
  async remove(v) {
    try {
      await api(`/v1/videos/${v.id}`, { method: "DELETE" });
    } catch (e) {
      toasts.error(e.message);
    }
    await this.refresh();
  }
}

export const videos = new Videos();
