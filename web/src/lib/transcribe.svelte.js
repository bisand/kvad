// Speech to text: a recording, the Whisper to hear it, and what it heard.
//
// The page talks to `/v1/audio/transcriptions` (or `/translations`) as any
// OpenAI client does, with a form and `stream=true`. The server answers with
// a `transcript.text.delta` event for each 30-second window it has done,
// which carries that window's timed segments under `kvad`, and a
// `transcript.text.done` at the end. So the segments arrive as they are
// heard, and a long recording shows that it is moving.
//
// Nothing is kept on the server, so nothing is kept here either: the
// transcript lives in this store until the next one, which is long enough to
// read, copy and download it, and survives a visit to another page.
//
// The downloads are written here from the segments rather than asked for
// again, since asking again is listening again. `render` is
// `kvad::speech::Transcript::render` (crates/llm/src/speech/transcribe.rs)
// line for line, so a file saved here is the file `kvad transcribe` and the
// server write for the same transcript.

import { sse } from "./api.js";
import { models } from "./models.svelte.js";
import { toasts } from "./toasts.svelte.js";

/** The server's limit on a recording (`speech::MAX_FILE`). Checked here too,
 *  so 300 MB is not uploaded only to be refused. */
export const MAX_FILE = 200 * 1024 * 1024;

/** Every language a Whisper has a token for, in its own order. */
const CODES = (
  "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt " +
  "la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc " +
  "ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue"
).split(" ");

/** The same, named and in alphabetical order. `jw` is Whisper's own code
 *  for Javanese, which the browser does not know by it. */
export const LANGUAGES = (() => {
  let names;
  try {
    names = new Intl.DisplayNames(["en"], { type: "language" });
  } catch {}
  const name = (c) => (c === "jw" ? "Javanese" : (names?.of(c) ?? c));
  return CODES.map((code) => ({ code, name: name(code) })).sort((a, b) => a.name.localeCompare(b.name));
})();

// ---------------------------------------------------------------------------
// What a transcript is written out as: `Transcript::render`, in JavaScript
// ---------------------------------------------------------------------------

/** Rust's `str::trim`: Unicode's White_Space, which is not quite the set
 *  JavaScript's own `trim` takes off (it adds U+FEFF and lacks U+0085). */
const trim = (s) => s.replace(/^\p{White_Space}+|\p{White_Space}+$/gu, "");

/**
 * `hh:mm:ss,mmm` — with a full stop instead of the comma for VTT.
 *
 * The times are `f32` on the server and `(seconds as f64 * 1000.0).round()`
 * there; `Math.fround` gives back that `f32` from the JSON, so a time on a
 * half millisecond rounds the same way here.
 */
export function clock(seconds, comma) {
  const ms = Math.round(Math.max(Math.fround(seconds), 0) * 1000);
  const two = (n) => String(n).padStart(2, "0");
  return `${two(Math.floor(ms / 3_600_000))}:${two(Math.floor(ms / 60_000) % 60)}:${two(Math.floor(ms / 1000) % 60)}${comma}${String(ms % 1000).padStart(3, "0")}`;
}

/** Numbered (SRT) or bare (VTT) cues, one per segment. */
function cues(segments, comma, numbered) {
  let out = "";
  segments.forEach((s, i) => {
    if (numbered) out += `${i + 1}\n`;
    out += `${clock(s.start, comma)} --> ${clock(s.end, comma)}\n${trim(s.text)}\n\n`;
  });
  return out;
}

/**
 * The transcript in `format`: `text`, `srt` or `vtt`.
 *
 * @param {{ start: number, end: number, text: string }[]} segments
 * @param {"text"|"srt"|"vtt"} format
 */
export function render(segments, format) {
  switch (format) {
    case "text":
      return `${trim(segments.map((s) => s.text).join(""))}\n`;
    case "srt":
      return cues(segments, ",", true);
    case "vtt":
      return `WEBVTT\n\n${cues(segments, ".", false)}`;
  }
  throw new Error(`no such format: ${format}`);
}

const TYPES = { text: "text/plain", srt: "application/x-subrip", vtt: "text/vtt" };
const EXTENSIONS = { text: "txt", srt: "srt", vtt: "vtt" };

class Transcribe {
  /** Which Whisper: a repo id, or a resident's `repo@backend`. */
  model = $state(null);
  /** The recording, a `File`. */
  file = $state(null);
  /** "" for the checkpoint's own, "auto" to have it guessed, or a code. */
  language = $state("");
  task = $state("transcribe");
  timestamps = $state(true);
  prompt = $state("");

  /** Between the button and the last event. */
  running = $state(false);
  /** Where a running one is: "loading", "sending", "listening". */
  phase = $state(null);
  /** The run being shown: what was asked, and what has come back. */
  segments = $state([]);
  heard = $state(0);
  total = $state(0);
  /** The `transcript.text.done` event's, once it has come. */
  done = $state(null);
  error = $state(null);
  stopped = $state(false);
  /** How the run was asked for, which decides what it can be saved as. */
  asked = $state(null);
  startedAt = $state(null);
  finishedAt = $state(null);

  #controller = null;

  /** Whispers on this machine that could be asked, in memory first. */
  get choices() {
    const here = models.residents
      .filter((r) => r.kind === "speech")
      .map((r) => ({ id: r.id, label: `${r.repo} · ${r.backend}`, resident: r }));
    const onDisk = models.all
      .filter((m) => m.kind === "speech" && m.runnable && !models.resident(m.id))
      .map((m) => ({ id: m.id, label: `${m.id} (not loaded)`, resident: null }));
    return [...here, ...onDisk];
  }

  get chosen() {
    return this.choices.find((c) => c.id === this.model) ?? this.choices[0] ?? null;
  }

  /** The whole text so far: the server's at the end, the segments' before. */
  get text() {
    return this.done ? this.done.text : trim(this.segments.map((s) => s.text).join(""));
  }

  choose(file) {
    if (!file) return;
    if (file.size > MAX_FILE) {
      toasts.error(`${file.name} is ${(file.size / 2 ** 20).toFixed(0)} MiB; the server takes a recording of at most 200 MiB.`);
      return;
    }
    this.file = file;
  }

  async start() {
    const chosen = this.chosen;
    if (this.running || !chosen || !this.file) return;
    const controller = new AbortController();
    this.#controller = controller;
    this.running = true;
    this.segments = [];
    this.heard = 0;
    this.total = 0;
    this.done = null;
    this.error = null;
    this.stopped = false;
    this.asked = { task: this.task, timestamps: this.timestamps, name: this.file.name };
    this.startedAt = Date.now();
    this.finishedAt = null;
    try {
      // A Whisper that is not in memory is loaded first, as the Images and
      // Videos pages do, so that a load shows as a load.
      let target = chosen.id;
      if (!chosen.resident) {
        this.phase = "loading";
        if (!(await models.load(chosen.id))) {
          this.error = `${chosen.id} did not load.`;
          return;
        }
        target = models.residents.find((r) => r.kind === "speech" && r.repo === chosen.id)?.id ?? chosen.id;
        this.model = target;
        if (controller.signal.aborted) throw new DOMException("stopped", "AbortError");
      }
      this.phase = "sending";
      const form = new FormData();
      form.append("model", target);
      form.append("stream", "true");
      form.append("timestamps", String(this.timestamps));
      if (this.language) form.append("language", this.language);
      if (this.prompt.trim()) form.append("prompt", this.prompt.trim());
      form.append("file", this.file, this.file.name || "recording");
      const path = this.task === "translate" ? "/v1/audio/translations" : "/v1/audio/transcriptions";
      await sse(
        path,
        form,
        {
          "transcript.text.delta": (data) => {
            const d = JSON.parse(data);
            this.segments = [...this.segments, ...d.kvad.segments];
            this.heard = d.kvad.done;
            this.total = d.kvad.total;
          },
          "transcript.text.done": (data) => {
            const d = JSON.parse(data);
            this.done = { text: d.text, language: d.kvad.language, duration: d.kvad.duration };
            this.heard = this.total = d.kvad.duration;
          },
          error: (data) => {
            throw new Error(JSON.parse(data).error);
          },
        },
        controller.signal,
        "POST",
        () => (this.phase = "listening"),
      );
      if (!this.done) this.error = "The server stopped before it had heard the whole recording.";
    } catch (e) {
      if (e.name === "AbortError") this.stopped = true;
      else this.error = e.message;
    } finally {
      this.running = false;
      this.phase = null;
      this.finishedAt = Date.now();
      this.#controller = null;
    }
  }

  /** Hang up. The server finishes the window it is on and stops there. */
  cancel() {
    this.#controller?.abort();
  }

  async copy() {
    try {
      await navigator.clipboard.writeText(this.text);
      toasts.success("Copied the text.");
    } catch (e) {
      toasts.error(`The browser would not copy it (${e.message}).`);
    }
  }

  /** @param {"text"|"srt"|"vtt"} format */
  download(format) {
    const blob = new Blob([render(this.segments, format)], { type: `${TYPES[format]};charset=utf-8` });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${(this.asked?.name ?? "transcript").replace(/\.[^.]*$/, "") || "transcript"}.${EXTENSIONS[format]}`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}

export const transcribe = new Transcribe();
