#!/usr/bin/env python3
"""The diarization evaluation sets, laid out as kvad's scorer reads them.

`docs/diarization-plan.md` measures on three sets, none of them committed.
This turns each, fetched by hand into `--data` as below, into a directory of
whole recordings with their reference beside them:

    <data>/<set>/<name>.wav     16 kHz mono 16-bit
    <data>/<set>/<name>.rttm    who spoke when
    <data>/<set>/<name>.json    the reference's words, a turn an entry

- `ami`: the 16 test meetings' headset mix, with `pyannote/AMI-diarization-setup`'s
  `only_words` references, the ones pyannote's and NVIDIA's cards score on.
- `npsc`: the Hub copy's four test days of Stortinget (`NbAiLab/NPSC`), one
  MP3 a sentence, each laid back at its time in the sitting.
- `samtale`: NB Samtale, one WAV a segment, laid back at its time in its
  conversation.

Laying segments back leaves silence wherever no segment is, which makes false
alarms easier to avoid than in the original recording. Two segments that
overlap must carry the same audio where they do (correlated above 0.9, as NB
Samtale's differ by a few per cent in level); a conversation where they do
not is refused, and said so.

    D=~/data/diar
    git clone --depth 1 https://github.com/pyannote/AMI-diarization-setup $D/ami/setup
    for m in $(cat $D/ami/setup/lists/test.meetings.txt); do
      curl -fLo $D/ami/$m.Mix-Headset.wav \\
        https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$m/audio/$m.Mix-Headset.wav
    done
    for d in 20170207 20171122 20171219 20180530; do for f in $d.json ${d}_16K_mp3.tar.gz; do
      curl -fLo $D/npsc/$f https://huggingface.co/datasets/NbAiLab/NPSC/resolve/main/data/test/$f
    done; done
    curl -fLo $D/samtale/nb_samtale.zip https://www.nb.no/sbfil/taledata/nb_samtale.zip
    python3 scripts/diar-sets.py --data $D --out $D/sets

Needs numpy and `ffmpeg`.
"""
import argparse
import glob
import json
import os
import subprocess
import tarfile
import wave
import zipfile

import numpy as np

RATE = 16000


def decode(path):
    """Any audio file as 16 kHz mono int16 samples."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ar", str(RATE), "-ac", "1", "-f", "s16le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype=np.int16)


def write(out, name, audio, turns):
    """`turns` are (speaker, start, end, text), in seconds."""
    with wave.open(os.path.join(out, name + ".wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(audio.astype(np.int16).tobytes())
    turns = sorted(turns, key=lambda t: t[1])
    with open(os.path.join(out, name + ".rttm"), "w") as f:
        for s, a, b, _ in turns:
            f.write(f"SPEAKER {name} 1 {a:.3f} {b - a:.3f} <NA> <NA> {s} <NA> <NA>\n")
    with open(os.path.join(out, name + ".json"), "w") as f:
        json.dump([{"speaker": s, "start": a, "end": b, "text": t} for s, a, b, t in turns], f, ensure_ascii=False, indent=0)
    speakers = len({t[0] for t in turns})
    print(f"{name}: {len(audio) / RATE / 3600:.2f} h, {len(turns)} turns, {speakers} speakers")


def lay(clips, length=None):
    """Clips (start in samples, samples) laid at their times; None if two
    overlapping clips disagree where they overlap."""
    end = max(s + len(c) for s, c in clips)
    audio = np.zeros(max(end, length or 0), dtype=np.int16)
    set_ = np.zeros(len(audio), dtype=bool)
    for s, c in clips:
        old, seen = audio[s : s + len(c)], set_[s : s + len(c)]
        # NB Samtale's segments differ by a few per cent in level where they
        # overlap, so agreeing is correlating, not matching sample for sample.
        if seen.sum() > RATE // 10 and np.corrcoef(old[seen], c[seen])[0, 1] < 0.9:
            return None
        audio[s : s + len(c)] = c
        set_[s : s + len(c)] = True
    return audio


def ami(data, out):
    setup = os.path.join(data, "ami", "setup")
    for m in open(os.path.join(setup, "lists", "test.meetings.txt")).read().split():
        audio = decode(os.path.join(data, "ami", f"{m}.Mix-Headset.wav"))
        turns = []
        for line in open(os.path.join(setup, "only_words", "rttms", "test", f"{m}.rttm")):
            f = line.split()
            turns.append((f[7], float(f[3]), float(f[3]) + float(f[4]), ""))
        write(out, m, audio, turns)


def npsc(data, out):
    root = os.path.join(data, "npsc")
    for day in sorted(glob.glob(os.path.join(root, "????????.json"))):
        d = os.path.basename(day)[:8]
        rows = [json.loads(l) for l in open(day)]
        with tarfile.open(os.path.join(root, f"{d}_16K_mp3.tar.gz")) as tar:
            tmp = os.path.join(root, d)
            tar.extractall(tmp, filter="data")
        clips, turns = [], []
        for r in rows:
            a, b = r["start_time"] / 1000, r["end_time"] / 1000
            # An MP3 runs up to 2 ms past its sentence, into the next one's 10 ms gap.
            clip = decode(os.path.join(tmp, r["path"] + ".mp3"))[: round((b - a) * RATE)]
            clips.append((round(a * RATE), clip))
            turns.append((f"npsc_{r['speaker_id']}", a, b, r["text"]))
        audio = lay(clips, round(rows[0]["total_duration"] / 1000 * RATE))
        if audio is None:
            print(f"{d}: refused, overlapping sentences disagree")
            continue
        write(out, f"npsc_{d}", audio, turns)


def samtale(data, out):
    # Every split's segments: no system here was trained on any of them, and
    # a conversation is only whole with all of them.
    root = os.path.join(data, "samtale", "nb_samtale")
    by = {}
    for line in open(os.path.join(root, "metadata.jsonl")):
        r = json.loads(line)
        by.setdefault(r["source_file_id"], []).append(r)
    for name, rows in sorted(by.items()):
        clips, turns = [], []
        for r in rows:
            a, b = (int(t) / 1000 for t in r["segment_id"].split("-"))
            clip = decode(os.path.join(root, r["file_name"]))[: round((b - a) * RATE)]
            clips.append((round(a * RATE), clip))
            # P0 is music and noise: audio, but nobody speaking.
            if r["speaker_id"] != "P0":
                turns.append((r["speaker_id"], a, b, r["transcription"]))
        audio = lay(clips)
        if audio is None:
            print(f"{name}: refused, overlapping segments disagree")
            continue
        write(out, f"samtale_{name}", audio, turns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--set", action="append", choices=["ami", "npsc", "samtale"])
    a = p.parse_args()
    for s in a.set or ["ami", "npsc", "samtale"]:
        out = os.path.join(a.out, s)
        os.makedirs(out, exist_ok=True)
        globals()[s](a.data, out)


if __name__ == "__main__":
    main()
