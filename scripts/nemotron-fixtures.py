#!/usr/bin/env python3
"""Reference outputs of transformers' Nemotron-3-Diarization, for kvad's to be
compared with.

`crates/llm/src/speech/nemotron.rs` is the model written out by hand: the
spectrogram, the stacked frames, the encoder, the head, and the speaker cache
that carries who is who from one 27-second chunk to the next. Each can be a
little wrong and still label speakers plausibly, so this runs transformers on
the same 16 kHz WAV, offline, in f32 on the CPU, and writes each stage:

- `features`: the log-mel spectrogram, [frames, 128], one row per 10 ms
- `embeds`: the stacked and projected frames the encoder reads, [frames / 8, 512]
- `encoded`: the encoder's output for the first chunk alone, [chunk + look-ahead, 512]
- `probs`: each speaker's probability in every 10 ms frame of the whole
  recording, cache and all, [frames, 8]

and in `meta.json`, the model, the number of valid frames, and the segments
transformers' `extract_speaker_dict` makes of `probs`.

    /opt/homebrew/bin/python3.14 -m venv /tmp/nemotron-venv
    /tmp/nemotron-venv/bin/pip install torch transformers safetensors numpy librosa
    /tmp/nemotron-venv/bin/python scripts/nemotron-fixtures.py --wav meeting.wav --out /tmp/nemotron-fx
    KVAD_NEMOTRON_FIXTURES=/tmp/nemotron-fx cargo test --release -p kvad speech::nemotron -- --ignored --nocapture

The WAV must already be 16 kHz mono 16-bit (`ffmpeg -i IN -ar 16000 -ac 1
-c:a pcm_s16le OUT.wav`); it is copied into the fixture directory as
`audio.wav`. A recording longer than one chunk (27.2 s) is what exercises the
speaker cache: `scripts/dialogue.py --script` makes one.
"""
import argparse
import json
import os
import shutil
import wave

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

MODEL = "nvidia/Nemotron-3-Diarization"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=MODEL)
    a = p.parse_args()

    with wave.open(a.wav) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), "16 kHz mono 16-bit only"
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForAudioFrameClassification.from_pretrained(a.model, dtype=torch.float32).eval()
    inputs = processor(audio, sampling_rate=16000)

    with torch.inference_mode():
        features = inputs.input_features
        embeds = model.model.audio_tower.embedder(features)
        first = embeds[:, : model.config.chunk_length + model.config.chunk_right_context]
        encoded = model.model.audio_tower(inputs_embeds=first).last_hidden_state
        logits = model(**inputs).logits

    valid = int(inputs.attention_mask.sum())
    segments = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
    os.makedirs(a.out, exist_ok=True)
    save_file(
        {
            "features": features[0].contiguous(),
            "embeds": embeds[0].contiguous(),
            "encoded": encoded[0].contiguous(),
            "probs": logits[0].sigmoid().contiguous(),
        },
        os.path.join(a.out, "nemotron.safetensors"),
    )
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"model": a.model, "frames": valid, "segments": segments}, f, indent=1)
    shutil.copy(a.wav, os.path.join(a.out, "audio.wav"))
    speakers = sorted({s["Speaker"] for s in segments})
    print(f"{len(audio) / 16000:.1f} s, {valid} frames, {len(segments)} segments, speakers {speakers}")


if __name__ == "__main__":
    main()
