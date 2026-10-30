#!/usr/bin/env python3
"""pyannote's own pipeline on the diarization sets, for kvad's table to have
the field's reference beside it.

For every `<set>/<name>.wav` that `scripts/diar-sets.py` laid out, runs
`pyannote/speaker-diarization-3.1` (or `--model`) and writes
`<set>/hyp/<model>-<device>/<name>.rttm`,
and the seconds it took to `seconds.txt` beside them, as kvad's runs do. The
pipeline is gated on the Hub: accept its terms, and segmentation-3.0's, with
the account whose token is in `~/.cache/huggingface/token`.

    /opt/homebrew/bin/python3.14 -m venv /tmp/pyannote-venv
    /tmp/pyannote-venv/bin/pip install pyannote.audio
    /tmp/pyannote-venv/bin/python scripts/pyannote-diarize.py --sets ~/data/diar/sets --device mps

pyannote.audio 4 runs 3.1 with a file of `speaker-diarization-community-1`,
so that one's terms are needed too, and `--model` runs it as well (3.x, which
3.1 was released with, needs a torchaudio older than Python 3.14's). The
audio is passed in memory, as pyannote's card suggests for speed; a recording already
done is skipped, so a run can be stopped and resumed.
"""
import argparse
import glob
import os
import time
import wave

import numpy as np
import torch
from pyannote.audio import Pipeline

MODEL = "pyannote/speaker-diarization-3.1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", required=True)
    p.add_argument("--set", action="append")
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", default=MODEL)
    a = p.parse_args()
    pipeline = Pipeline.from_pretrained(a.model).to(torch.device(a.device))
    for set_ in sorted(glob.glob(os.path.join(a.sets, "*"))):
        if a.set and os.path.basename(set_) not in a.set:
            continue
        out = os.path.join(set_, "hyp", f"{a.model.split('/')[-1]}-{a.device}")
        os.makedirs(out, exist_ok=True)
        for wav in sorted(glob.glob(os.path.join(set_, "*.wav"))):
            name = os.path.basename(wav)[:-4]
            if os.path.exists(os.path.join(out, name + ".rttm")):
                continue
            with wave.open(wav) as w:
                audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            t = time.time()
            result = pipeline({"waveform": torch.from_numpy(audio)[None], "sample_rate": 16000})
            took = time.time() - t
            # pyannote.audio 4 wraps the annotation; 3.x returns it.
            result = getattr(result, "speaker_diarization", result)
            with open(os.path.join(out, name + ".rttm"), "w") as f:
                result.write_rttm(f)
            with open(os.path.join(out, "seconds.txt"), "a") as f:
                f.write(f"{name} {took:.2f}\n")
            print(f"{os.path.basename(set_)}/{name}: {len(audio) / 16000 / 3600:.2f} h in {took:.0f} s", flush=True)


if __name__ == "__main__":
    main()
