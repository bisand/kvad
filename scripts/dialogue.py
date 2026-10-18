#!/usr/bin/env python3
"""A conversation whose right answer is known to the sample: voices from
macOS's `say`, taking turns, with chosen gaps and overlaps.

Not a benchmark. A fixture for the diarization steps of
`docs/diarization-plan.md`: who spoke when is written out exactly, because
the script put every clip where it is, so a test that must not depend on a
dataset still has a reference to score against.

Every speaker is one `say` voice (Nora by default, which reads Norwegian),
pitched apart: the first as it is, the others resampled lower or higher and
retimed back to the first's tempo, so the words keep their length. Each clip
has its leading and trailing silence trimmed, so a turn's edges are its
speech's.

    python3 scripts/dialogue.py --out /tmp/dialogue
    python3 scripts/dialogue.py --script meeting.txt --out /tmp/meeting

With no `--script`, the diarization probe's dialogue: eight turns between two
voices, 27 s, 0.2 to 0.35 s between turns. A script is a turn a line, its
speaker, a colon and its text; a number in brackets before the speaker is the
pause before the turn, in seconds, and a negative one starts it that long
before the last turn ends, which is an overlap:

    A: Hei, og velkommen til møtet.
    (0.3) B: Takk.
    (-0.6) A: Ja, nettopp.

It writes, into `--out`:

- `dialogue.wav`: 16 kHz mono 16-bit, with half a second of silence at each
  end;
- `dialogue.rttm`: the true turns, as `score::rttm` reads them;
- `dialogue.json`: the same with each turn's text;
- `dialogue.txt`: `A: …`, a turn a line, the transcript with speakers.

Needs macOS (`say`), `ffmpeg` and numpy; no torch.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
import wave

import numpy as np

SR = 16000

PROBE = """\
A: Hei, og velkommen til møtet. I dag skal vi snakke om budsjettet for neste år.
(0.25) B: Takk. Jeg har sett på tallene, og de ser bedre ut enn i fjor.
(0.3) A: Det er godt å høre.
(0.2) B: Men vi må fortsatt kutte litt i reisekostnadene.
(0.35) A: Ja.
(0.25) B: Kan vi ta det opp med styret på torsdag?
(0.3) A: Det tror jeg er lurt. Jeg sender en innkalling i ettermiddag, så får alle lest saken før møtet.
(0.2) B: Flott, da gjør vi det slik.
"""

# How far each speaker's voice is moved, in order of first appearance: the
# probe's second voice was 0.78.
PITCHES = [1.0, 0.78, 1.2, 0.88, 1.1, 0.7]


def parse(script, gap):
    turns = []
    for n, line in enumerate(script.splitlines()):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(r"\s*(?:\((-?[0-9.]+)\)\s*)?([^:\s]+)\s*:\s*(.+)", line)
        if not m:
            raise SystemExit(f"script line {n + 1}: want `SPEAKER: text`, or `(0.3) SPEAKER: text`: {line}")
        pause = float(m.group(1)) if m.group(1) is not None else gap
        turns.append((pause, m.group(2), m.group(3).strip()))
    return turns


def clip(text, voice, pitch, tmp, i):
    """One turn, spoken, at 16 kHz, its silence trimmed."""
    aiff, wav = os.path.join(tmp, f"t{i}.aiff"), os.path.join(tmp, f"t{i}.wav")
    subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
    # `say` writes 22.05 kHz: claiming another rate moves the pitch and the
    # tempo together, and `atempo` puts the tempo back.
    shift = "" if pitch == 1.0 else f"asetrate=22050*{pitch},atempo={1 / pitch:.3f},"
    trim = "silenceremove=start_periods=1:start_threshold=-50dB"
    flt = f"{shift}aresample={SR},{trim},areverse,{trim},areverse"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", aiff, "-af", flt, "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", wav], check=True)
    with wave.open(wav) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--script", help="a turn a line; the probe's dialogue if left out")
    ap.add_argument("--voice", default="Nora")
    ap.add_argument("--gap", type=float, default=0.25, help="the pause before a turn whose line gives none")
    ap.add_argument("--pitch", help="each speaker's, in order of first appearance, comma-separated")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    turns = parse(open(args.script).read() if args.script else PROBE, args.gap)
    pitches = [float(p) for p in args.pitch.split(",")] if args.pitch else PITCHES
    speakers = list(dict.fromkeys(who for _, who, _ in turns))
    if len(speakers) > len(pitches):
        raise SystemExit(f"{len(speakers)} speakers and {len(pitches)} pitches: give --pitch")

    # Where each clip goes, in samples, and then the clips laid down and
    # summed, so that overlapping turns are both heard.
    placed, at, end = [], int(0.5 * SR), None
    with tempfile.TemporaryDirectory() as tmp:
        for i, (pause, who, text) in enumerate(turns):
            x = clip(text, args.voice, pitches[speakers.index(who)], tmp, i)
            if end is not None:
                at = end + int(pause * SR)
                if at < 0 or (placed and at < placed[-1][1]):
                    raise SystemExit(f"turn {i + 1} would start before the turn before it does")
            placed.append((who, at, x, text))
            end = at + len(x) if end is None else max(end, at + len(x))
    mix = np.zeros(end + int(0.5 * SR), dtype=np.float32)
    for _, at, x, _ in placed:
        mix[at : at + len(x)] += x
    pcm = np.clip(np.round(mix), -32768, 32767).astype(np.int16)

    truth = [{"speaker": who, "start": round(at / SR, 3), "end": round((at + len(x)) / SR, 3), "text": text} for who, at, x, text in placed]
    with wave.open(os.path.join(args.out, "dialogue.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    with open(os.path.join(args.out, "dialogue.rttm"), "w") as f:
        for t in truth:
            f.write(f"SPEAKER dialogue 1 {t['start']:.3f} {t['end'] - t['start']:.3f} <NA> <NA> {t['speaker']} <NA> <NA>\n")
    with open(os.path.join(args.out, "dialogue.json"), "w") as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "dialogue.txt"), "w") as f:
        f.writelines(f"{t['speaker']}: {t['text']}\n" for t in truth)
    print(f"{len(pcm) / SR:.2f} s, {len(truth)} turns, {len(speakers)} speakers, to {args.out}")


if __name__ == "__main__":
    main()
