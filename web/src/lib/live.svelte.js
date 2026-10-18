// Live transcription from the browser's microphone: `/v1/realtime`.
//
// The microphone goes through an AudioWorklet that resamples it to 16 kHz
// mono and sends it, as 16-bit PCM, to the server in binary frames of a
// tenth of a second. The server answers with OpenAI's Realtime events for a
// transcription session: a `delta` for text that has settled, which never
// changes after, and kvad's own `kvad.transcription.partial`, the guess at
// the words after it, each replacing the last. So the page shows settled
// text as it is, and the partial greyed after it.
//
// Stopping is a commit: what is still a partial is settled, the server sends
// `completed`, and then the socket is closed. Nothing is kept on the server;
// the text lives here until the next session.

import { models } from "./models.svelte.js";
import { toasts } from "./toasts.svelte.js";

/** What the worklet does, as a string: it is loaded from a Blob, so the
 *  build needs no second entry point for it. */
const WORKLET = `
// The microphone at the context's rate, as 16 kHz 16-bit PCM in pieces of
// 100 ms: each output sample a windowed sinc over the input around it, cut
// off below 8 kHz, so that what is above it does not fold down.
class Pcm16k extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / 16000;
    this.cutoff = Math.min(1, 16000 / sampleRate) * 0.9;
    this.half = Math.ceil(8 * Math.max(1, this.step));
    this.history = new Float32Array(0);
    this.at = this.half; // where the next output falls, in history samples
    this.out = new Int16Array(1600);
    this.n = 0;
  }
  kernel(x) {
    if (x === 0) return this.cutoff;
    const w = 0.5 + 0.5 * Math.cos((Math.PI * x) / (this.half + 1));
    return (Math.sin(Math.PI * this.cutoff * x) / (Math.PI * x)) * w;
  }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    const h = new Float32Array(this.history.length + input.length);
    h.set(this.history);
    h.set(input, this.history.length);
    while (this.at + this.half < h.length) {
      const c = Math.floor(this.at);
      let sum = 0;
      for (let i = c - this.half + 1; i <= c + this.half; i++) sum += h[i] * this.kernel(this.at - i);
      const v = Math.max(-1, Math.min(1, sum));
      this.out[this.n++] = v < 0 ? v * 32768 : v * 32767;
      if (this.n === this.out.length) {
        this.port.postMessage(this.out.buffer, [this.out.buffer]);
        this.out = new Int16Array(1600);
        this.n = 0;
      }
      this.at += this.step;
    }
    // Keep what the next outputs still reach back to.
    const keep = Math.max(0, Math.floor(this.at) - this.half);
    this.history = h.slice(keep);
    this.at -= keep;
    return true;
  }
}
registerProcessor("pcm-16k", Pcm16k);
`;

class Live {
  /** Which Whisper: a repo id, or a resident's `repo@backend`. */
  model = $state(null);
  language = $state("");
  prompt = $state("");

  /** "loading", "opening", "listening", "stopping", or null. */
  phase = $state(null);
  /** Text settled, and the guess after it. */
  settled = $state("");
  partial = $state("");
  /** How far behind the voice the text is, in seconds, by the last event. */
  lag = $state(0);
  /** Seconds of audio sent. */
  sent = $state(0);
  language_heard = $state(null);
  error = $state(null);
  startedAt = $state(null);

  #socket = null;
  #audio = null;
  #stream = null;

  get running() {
    return this.phase !== null;
  }

  /** Whispers on this machine, in memory first; small is the one for
   *  live, which `live_speed` measured. */
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
    const c = this.choices;
    return c.find((x) => x.id === this.model) ?? c.find((x) => /small/.test(x.id)) ?? c[0] ?? null;
  }

  /** The text, settled and not, as it reads now. */
  get text() {
    return `${this.settled}${this.partial}`.trim();
  }

  async start() {
    const chosen = this.chosen;
    if (this.running || !chosen) return;
    this.settled = "";
    this.partial = "";
    this.lag = 0;
    this.sent = 0;
    this.error = null;
    this.language_heard = null;
    this.startedAt = null;
    try {
      let target = chosen.id;
      if (!chosen.resident) {
        this.phase = "loading";
        if (!(await models.load(chosen.id))) throw new Error(`${chosen.id} did not load.`);
        target = models.residents.find((r) => r.kind === "speech" && r.repo === chosen.id)?.id ?? chosen.id;
        this.model = target;
        if (this.phase !== "loading") return;
      }
      this.phase = "opening";
      // Asked first: a refused microphone is an answer before any socket.
      this.#stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      if (this.phase !== "opening") return this.#release();
      await this.#open(target);
      this.#audio = new AudioContext();
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }));
      await this.#audio.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      const source = this.#audio.createMediaStreamSource(this.#stream);
      const node = new AudioWorkletNode(this.#audio, "pcm-16k");
      node.port.onmessage = (e) => {
        if (this.#socket?.readyState === WebSocket.OPEN && this.phase === "listening") {
          this.#socket.send(e.data);
          this.sent += e.data.byteLength / 2 / 16000;
        }
      };
      source.connect(node);
      this.phase = "listening";
      this.startedAt = Date.now();
    } catch (e) {
      this.error = e.name === "NotAllowedError" ? "The browser was not allowed to use the microphone." : e.message;
      this.#release();
      this.phase = null;
    }
  }

  /** Open the socket, and resolve once the session is set up. */
  #open(target) {
    return new Promise((resolve, reject) => {
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(`${scheme}://${location.host}/v1/realtime?intent=transcription`);
      this.#socket = socket;
      socket.onopen = () => {
        const transcription = { model: target };
        if (this.language) transcription.language = this.language;
        if (this.prompt.trim()) transcription.prompt = this.prompt.trim();
        socket.send(
          JSON.stringify({
            type: "session.update",
            session: { type: "transcription", audio: { input: { format: { type: "audio/pcm", rate: 16000 }, transcription } } },
          }),
        );
      };
      socket.onmessage = (m) => {
        const e = JSON.parse(m.data);
        switch (e.type) {
          case "session.updated":
            resolve();
            break;
          case "conversation.item.input_audio_transcription.delta":
            this.settled += e.delta;
            this.lag = e.kvad?.lag ?? this.lag;
            break;
          case "kvad.transcription.partial":
            this.partial = e.text;
            this.lag = e.kvad?.lag ?? this.lag;
            break;
          case "conversation.item.input_audio_transcription.completed":
            this.partial = "";
            this.language_heard = e.kvad?.language ?? null;
            if (this.phase === "stopping") this.#finish();
            break;
          case "error":
            this.error = e.error?.message ?? "The server failed.";
            reject(new Error(this.error));
            this.#finish();
            break;
        }
      };
      socket.onerror = () => reject(new Error("The server would not open a live session."));
      socket.onclose = () => {
        if (this.phase && this.phase !== "stopping" && !this.error) this.error = "The server closed the session.";
        reject(new Error(this.error ?? "closed"));
        this.#finish();
      };
    });
  }

  /** Stop listening: what is still a guess is settled, then the socket
   *  closes. */
  stop() {
    if (this.phase === "listening" && this.#socket?.readyState === WebSocket.OPEN) {
      this.phase = "stopping";
      this.#stopAudio();
      this.#socket.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
    } else {
      // Loading or opening: nothing has been said yet.
      this.phase = null;
      this.#finish();
    }
  }

  async copy() {
    try {
      await navigator.clipboard.writeText(this.text);
      toasts.success("Copied the text.");
    } catch (e) {
      toasts.error(`The browser would not copy it (${e.message}).`);
    }
  }

  #stopAudio() {
    this.#stream?.getTracks().forEach((t) => t.stop());
    this.#stream = null;
    this.#audio?.close();
    this.#audio = null;
  }

  #release() {
    this.#stopAudio();
    const s = this.#socket;
    this.#socket = null;
    if (s && s.readyState <= WebSocket.OPEN) s.close();
  }

  #finish() {
    this.#release();
    this.phase = null;
  }
}

export const live = new Live();
