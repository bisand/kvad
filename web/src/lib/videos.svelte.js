// Text to video: the request being made, and the videos already kept.
//
// The page talks to `/v1/videos` like any other client. That endpoint is a
// job, not a stream: asking for a video answers at once, and the video is
// the server's to make from then on. So the page asks, then follows each
// video still being made on its own stream of events (kvad's
// `/v1/videos/{id}/events`), which sends the video again at every step and
// when it ends. The gallery is where a video being made lives as well as a
// finished one. Closing the tab stops the following and nothing else; the
// video is there when the page is opened again.
//
// A video can start from a picture. The request is then a form with the
// picture as its `input_reference` file, as OpenAI's SDK sends one; without
// one it is JSON.

import { api, sse } from "./api.js";
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

/** A video that is still to be made or being made. */
export const live = (v) => v.status === "queued" || v.status === "in_progress";

/** `4.04 s`, or, before the model has chosen a length, what it is waiting
 *  on. */
export const lengthOf = (v) => (v.seconds ? `${v.seconds} s` : "length from the prompt");

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
  /** Guidance: any of these runs the guided (dev) pipeline. Blank is the
   *  pipeline's own default, and all blank is the fast unguided one. */
  steps = $state(null);
  guidance = $state(null);
  negative = $state("");
  /** The picture to start from, a `File`, and a link to show it by. */
  picture = $state(null);
  pictureUrl = $state(null);
  /** Its size, once the page has shown it. */
  pictureSize = $state(null);

  /** Loading the model, or sending the request. */
  starting = $state(false);
  gallery = $state([]);
  /** The videos being followed, by id, and how to stop following each. */
  #following = new Map();
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

  /** The longest clip the model makes here at the size in the form, in
   *  frames: what it may choose up to, when it chooses. */
  get longest() {
    const d = this.defaults;
    if (!d) return null;
    const w = Number(this.width || d.width);
    const h = Number(this.height || d.height);
    const most = Math.min(d.max_frames, Math.floor(d.max_volume / (w * h)));
    return Math.max(1, Math.floor((most - 1) / d.frame_step) * d.frame_step + 1);
  }

  /** Whether what is in the form is more than the model makes here. A
   *  length the model chooses is at least a second. */
  get tooBig() {
    const d = this.defaults;
    if (!d) return false;
    const w = Number(this.width || d.width);
    const h = Number(this.height || d.height);
    const fps = Number(this.fps || d.fps);
    const f = this.frames() ?? (d.duration ? fps : d.frames);
    return w * h * f > d.max_volume || f > d.max_frames;
  }

  /** Start from `file`, or from nothing. */
  choosePicture(file) {
    if (this.pictureUrl) URL.revokeObjectURL(this.pictureUrl);
    this.picture = file ?? null;
    this.pictureUrl = file ? URL.createObjectURL(file) : null;
    this.pictureSize = null;
  }

  /** What scaling the picture to cover the video cuts off: `null` when the
   *  shapes match, else which sides and how much of the picture goes. */
  get cut() {
    const d = this.defaults;
    const p = this.pictureSize;
    if (!p) return null;
    const w = Number(this.width || d?.width || 0);
    const h = Number(this.height || d?.height || 0);
    if (!w || !h) return null;
    const [pa, va] = [p.width / p.height, w / h];
    if (Math.abs(pa / va - 1) < 0.01) return null;
    return pa > va
      ? { sides: "left and right", share: 1 - va / pa }
      : { sides: "top and bottom", share: 1 - pa / va };
  }

  /** Whether the form asks for guidance, and so the dev pipeline. */
  get guided() {
    return Boolean(this.steps || this.guidance || this.negative.trim());
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
    this.#follow();
  }

  /** Follow every live video not already followed, while the page is open. */
  #follow() {
    if (this.#watchers === 0) return;
    for (const v of this.running) {
      if (this.#following.has(v.id)) continue;
      const stop = new AbortController();
      this.#following.set(v.id, stop);
      const put = (data) => {
        const video = JSON.parse(data);
        this.gallery = this.gallery.map((g) => (g.id === video.id ? video : g));
      };
      sse(
        `/v1/videos/${v.id}/events`,
        undefined,
        {
          "video.updated": put,
          "video.completed": put,
          "video.failed": put,
          "video.deleted": () => (this.gallery = this.gallery.filter((g) => g.id !== v.id)),
        },
        stop.signal,
        "GET",
      )
        .catch((e) => {
          if (e.name !== "AbortError") toasts.error(e.message);
        })
        .finally(() => this.#following.delete(v.id));
    }
  }

  /** The page is open: keep the gallery current. Returns the undo. */
  watch() {
    this.#watchers += 1;
    this.refresh();
    return () => {
      this.#watchers -= 1;
      if (this.#watchers === 0) {
        for (const stop of this.#following.values()) stop.abort();
        this.#following.clear();
      }
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
      if (this.steps) body.steps = Number(this.steps);
      if (this.guidance) body.guidance_scale = Number(this.guidance);
      if (this.negative.trim()) body.negative_prompt = this.negative.trim();
      let request;
      if (this.picture) {
        // A form, whose content type the browser writes with its boundary.
        const form = new FormData();
        for (const [k, v] of Object.entries(body)) form.append(k, String(v));
        form.append("input_reference", this.picture, this.picture.name || "picture");
        request = { method: "POST", body: form };
      } else {
        request = { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) };
      }
      const video = await api("/v1/videos", request);
      // A seed nobody fixed is still worth keeping: it is how this video is
      // made again.
      this.seed = video.kvad.seed;
      this.gallery = [video, ...this.gallery];
      this.#follow();
    } catch (e) {
      toasts.error(e.message);
    } finally {
      this.starting = false;
    }
  }

  /** Put an old video's settings back in the form, and its picture. */
  async reuse(v) {
    const k = v.kvad;
    this.prompt = v.prompt;
    this.width = k.width;
    this.height = k.height;
    // A length the model chose is left to it again: the same prompt makes
    // the same choice.
    this.seconds = k.length_chosen ? null : Number(v.seconds);
    this.fps = k.fps;
    this.seed = k.seed;
    this.fixSeed = true;
    this.sound = k.audio;
    this.steps = k.guided?.steps ?? null;
    this.guidance = k.guided?.guidance ?? null;
    this.negative = k.guided?.negative_prompt ?? "";
    const resident = models.videoResidents.find((r) => r.repo === v.model);
    this.model = resident?.id ?? v.model;
    if (!k.picture_url) return this.choosePicture(null);
    try {
      const blob = await (await fetch(k.picture_url)).blob();
      this.choosePicture(new File([blob], `video-${k.id}-picture`, { type: blob.type }));
    } catch (e) {
      toasts.error(`Could not fetch the picture ${v.id} started from (${e.message}).`);
    }
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
