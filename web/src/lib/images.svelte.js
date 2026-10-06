// Text to image: the request being made, and the pictures already kept.
//
// The page talks to `/v1/images/generations` like any other client, with
// `stream: true` so each denoising step arrives as it happens and a rough
// preview with it. The finished picture comes back as a link to the server's
// copy rather than as three megabytes of base64: the server keeps every image
// anyway, and the gallery shows the same file.

import { api, sse } from "./api.js";
import { toasts } from "./toasts.svelte.js";
import { models } from "./models.svelte.js";

/** Sizes worth offering, as SDXL was trained on them: about a megapixel, in
 *  the aspect ratios of its training buckets. Anything else can be typed. */
export const SIZES = [
  { label: "Square", width: 1024, height: 1024 },
  { label: "Landscape", width: 1216, height: 832 },
  { label: "Portrait", width: 832, height: 1216 },
  { label: "Wide", width: 1344, height: 768 },
  { label: "Small square", width: 768, height: 768 },
];

class Images {
  /** Which image model to ask: a repo id, or a resident's `repo@backend`. */
  model = $state(null);
  prompt = $state("a lighthouse on a cliff at dusk, oil painting");
  negative = $state("");
  /** Blank means the model's own default. */
  width = $state(null);
  height = $state(null);
  steps = $state(null);
  guidance = $state(null);
  seed = $state(null);
  fixSeed = $state(false);
  /** LoRAs to apply, `{ name, scale }`, each as it was pulled. */
  loras = $state([]);
  /** A picture to start from, `{ name, url }` with `url` a `data:` URL, or
   *  null to draw from the prompt alone. */
  source = $state(null);
  /** Where the picture may change: a `data:` URL of a mask, white where it
   *  is drawn anew, or null for all of it. */
  mask = $state(null);
  /** How far the picture is noised, 0 to 1; null for the default, which is
   *  0.75, or 1 where there is a mask. */
  strength = $state(null);

  running = $state(false);
  /** `{ step, total, started, preview }` while a picture is being made. */
  progress = $state(null);
  /** The one just made, as the gallery lists it. */
  last = $state(null);
  gallery = $state([]);
  #stop = null;

  /** Image models on this machine that could be asked, in memory first. */
  get choices() {
    const here = models.imageResidents.map((r) => ({ id: r.id, label: `${r.repo} · ${r.backend}`, resident: r }));
    const onDisk = models.all
      .filter((m) => m.kind === "image" && m.runnable && !models.resident(m.id))
      .map((m) => ({ id: m.id, label: `${m.id} (not loaded)`, resident: null }));
    return [...here, ...onDisk];
  }

  /** LoRAs on this machine for the chosen model: those whose layers' names
   *  say they were made for its pipeline, and those whose names say
   *  nothing. The file does not say which model it fits; one that does not
   *  fit fails the request, naming what it adapts that the model has not. */
  get loraChoices() {
    const repo = this.chosen?.resident?.repo ?? this.chosen?.id;
    const pipeline = models.all.find((m) => m.id === repo)?.pipeline ?? null;
    return models.all.filter((m) => m.lora && (!m.adapts || !pipeline || m.adapts === pipeline)).map((m) => m.id);
  }

  /** Add the first LoRA not already chosen, at full strength. */
  addLora() {
    const name = this.loraChoices.find((n) => !this.loras.some((l) => l.name === n));
    if (name) this.loras = [...this.loras, { name, scale: 1 }];
  }

  /** Change the `i`th LoRA: replaced, so that the page sees it. */
  setLora(i, change) {
    this.loras = this.loras.map((l, j) => (j === i ? { ...l, ...change } : l));
  }

  removeLora(i) {
    this.loras = this.loras.filter((_, j) => j !== i);
  }

  /** The chosen model, or the first there is. */
  get chosen() {
    return this.choices.find((c) => c.id === this.model) ?? this.choices[0] ?? null;
  }

  /** What the chosen model does with a knob left blank, once it is loaded. */
  get defaults() {
    return this.chosen?.resident?.image ?? null;
  }

  async refresh() {
    try {
      this.gallery = await api("/api/images");
    } catch (e) {
      toasts.error(e.message);
    }
  }

  async make() {
    const chosen = this.chosen;
    if (this.running || !chosen || !this.prompt.trim()) return;
    this.running = true;
    this.progress = { step: 0, total: this.steps ?? this.defaults?.steps ?? 0, started: Date.now(), preview: null };
    this.#stop = new AbortController();

    // A model that is not in memory is loaded first, on the Models page's
    // machinery, so that the minute a load takes shows as a load and not as
    // a generation that has not started.
    let target = chosen.id;
    if (!chosen.resident) {
      this.progress = { ...this.progress, loading: true };
      const ok = await models.load(chosen.id);
      if (!ok) {
        this.running = false;
        this.progress = null;
        return;
      }
      target = models.imageResidents.find((r) => r.repo === chosen.id)?.id ?? chosen.id;
      this.model = target;
    }

    const body = {
      model: target,
      prompt: this.prompt,
      stream: true,
      preview: true,
      response_format: "url",
    };
    // A model that takes no guidance refuses both, so neither is sent.
    const guided = this.defaults?.takes_guidance ?? true;
    if (guided && this.negative.trim()) body.negative_prompt = this.negative;
    if (this.width && this.height) body.size = `${this.width}x${this.height}`;
    if (this.steps) body.steps = this.steps;
    if (guided && this.guidance != null && this.guidance !== "") body.guidance_scale = Number(this.guidance);
    if (this.fixSeed && this.seed != null && this.seed !== "") body.seed = Number(this.seed);
    // A model that takes none refuses them, as it refuses guidance; one not
    // loaded yet says only when it is, so they are sent and it answers.
    if (this.loras.length && this.defaults?.takes_loras !== false) {
      body.loras = this.loras.map((l) => ({ name: l.name, scale: Number(l.scale) }));
    }

    // From a picture, it is an edit: the same request and the picture,
    // each file a `data:` URL, to the route that reads one.
    if (this.source) {
      body.image = this.source.url;
      if (this.mask) body.mask = this.mask;
      if (this.strength != null && this.strength !== "") body.strength = Number(this.strength);
    }

    this.progress = { ...this.progress, loading: false, started: Date.now() };
    try {
      await sse(
        this.source ? "/v1/images/edits" : "/v1/images/generations",
        body,
        {
          "image_generation.step": (data) => {
            const s = JSON.parse(data);
            this.progress = { ...this.progress, step: s.step, total: s.total };
          },
          "image_generation.partial_image": (data) => {
            const p = JSON.parse(data);
            this.progress = { ...this.progress, preview: `data:image/png;base64,${p.b64_json}` };
          },
          "image_generation.completed": (data) => {
            const k = JSON.parse(data).kvad;
            this.last = k;
            // A seed nobody fixed is still worth keeping: it is how this
            // picture is made again.
            this.seed = k.seed;
          },
          error: (data) => {
            throw new Error(JSON.parse(data).error);
          },
        },
        this.#stop.signal,
      );
    } catch (e) {
      if (e.name !== "AbortError") toasts.error(e.message);
    } finally {
      this.running = false;
      this.progress = null;
      this.#stop = null;
      await Promise.all([this.refresh(), models.refresh()]);
    }
  }

  /** Stop at the next step. Closing the stream is what tells the server. */
  stop() {
    this.#stop?.abort();
  }

  /** Start from a picture: a `File`, or a `Blob` with a name. */
  async startFrom(file, name = file.name) {
    try {
      this.source = { name, url: await dataUrl(file) };
      this.mask = null;
    } catch (e) {
      toasts.error(`Could not read ${name}: ${e.message}.`);
    }
  }

  /** Start from a picture in the gallery: an edit of what was made. */
  async startFromKept(image) {
    try {
      await this.startFrom(await blob(image.url), `image ${image.id}`);
      this.prompt = image.prompt;
      const resident = models.imageResidents.find((r) => r.repo === image.model);
      this.model = resident?.id ?? image.model;
    } catch (e) {
      toasts.error(e.message);
    }
  }

  /** Draw from the prompt alone again. */
  clearSource() {
    this.source = null;
    this.mask = null;
    this.strength = null;
  }

  /** Put an old picture's settings back in the form: for an edit, the
   *  picture it was made from and its mask too. */
  async reuse(image) {
    this.prompt = image.prompt;
    this.negative = image.negative_prompt ?? "";
    this.width = image.width;
    this.height = image.height;
    this.steps = image.steps;
    this.guidance = image.guidance;
    this.seed = image.seed;
    this.fixSeed = true;
    this.loras = (image.loras ?? []).map((l) => ({ name: l.name, scale: l.scale }));
    const resident = models.imageResidents.find((r) => r.repo === image.model);
    this.model = resident?.id ?? image.model;
    if (!image.input_url) return this.clearSource();
    try {
      this.source = { name: `what image ${image.id} was made from`, url: await dataUrl(await blob(image.input_url)) };
      this.mask = image.mask_url ? await dataUrl(await blob(image.mask_url)) : null;
      this.strength = image.strength;
    } catch (e) {
      toasts.error(e.message);
    }
  }

  async remove(id) {
    try {
      await api(`/api/images/${id}`, { method: "DELETE" });
      if (this.last?.id === id) this.last = null;
    } catch (e) {
      toasts.error(e.message);
    }
    await this.refresh();
  }
}

export const images = new Images();

/** A file from this server, as a `Blob`. */
async function blob(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} failed: ${response.status} ${response.statusText}.`);
  return await response.blob();
}

/** A file's bytes as a `data:` URL. */
function dataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error ?? new Error("the file could not be read"));
    reader.readAsDataURL(file);
  });
}
