#!/usr/bin/env python3
"""Reference outputs of transformers' Whisper, for kvad's to be compared with.

`crates/llm/src/speech/` is a Whisper written out by hand: a log-mel
spectrogram, an encoder, a decoder with cross-attention and the decoding
loop. Each can be a little wrong and still produce a plausible transcript, so
this runs transformers on the same 16 kHz WAV, in f32 on the CPU, and writes
each stage for the test to compare against:

- `mel`: the log-mel spectrogram of the first 30 seconds, [n_mels, 3000]
- `encoded`: the encoder's output for it, [1500, d]
- `logits`: the decoder's scores after the prompt, [vocab]
- `tokens.json`: greedy tokens with and without timestamps, for one window,
  and the prompt they followed

    python3 -m venv /tmp/whisper-venv
    /tmp/whisper-venv/bin/pip install torch transformers safetensors numpy
    /tmp/whisper-venv/bin/python scripts/whisper-fixtures.py --model NbAiLab/nb-whisper-tiny \\
        --wav speech.wav --language no --out /tmp/whisper-fx
    KVAD_WHISPER_FIXTURES=/tmp/whisper-fx cargo test --release -p kvad speech::tests -- --ignored --nocapture

The WAV must already be 16 kHz mono 16-bit, so that both sides start from
the same samples (`ffmpeg -i IN -ar 16000 -ac 1 -c:a pcm_s16le OUT.wav`); it
is copied into the fixture directory as `audio.wav`. The venv wants a Python
that torch has wheels for, which a just-released one may not be.
"""
import argparse
import json
import os
import shutil
import wave

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--language", default="no")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with wave.open(args.wav) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, "16 kHz mono 16-bit, please"
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    audio = audio[: 30 * 16000]

    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    mel = proc(audio, sampling_rate=16000, return_tensors="pt").input_features

    tok = proc.tokenizer
    ids = lambda *names: [tok.convert_tokens_to_ids(n) for n in names]
    prompt = ids("<|startoftranscript|>", f"<|{args.language}|>", "<|transcribe|>", "<|notimestamps|>")
    with torch.no_grad():
        encoded = model.model.encoder(mel).last_hidden_state
        logits = model(encoder_outputs=(encoded,), decoder_input_ids=torch.tensor([prompt])).logits[0, -1]
        plain = model.generate(mel, language=args.language, task="transcribe", num_beams=1, do_sample=False)
        timed = model.generate(mel, language=args.language, task="transcribe", num_beams=1, do_sample=False, return_timestamps=True)

    def strip(seq):
        # Newer transformers return only what was generated, older ones the
        # prompt too: keep what follows it, without the end of text.
        seq = seq[0].tolist()
        while seq and seq[0] in prompt:
            seq = seq[1:]
        return [t for t in seq if t != tok.eos_token_id]

    save_file(
        {"mel": mel[0].contiguous(), "encoded": encoded[0].contiguous(), "logits": logits.contiguous()},
        os.path.join(args.out, "whisper.safetensors"),
    )
    with open(os.path.join(args.out, "tokens.json"), "w") as f:
        json.dump(
            {
                "model": args.model,
                "language": args.language,
                "prompt": prompt,
                "plain": strip(plain),
                "timed": strip(timed),
                "text": tok.decode(strip(plain), skip_special_tokens=True),
            },
            f,
            indent=1,
        )
    shutil.copy(args.wav, os.path.join(args.out, "audio.wav"))
    print(json.load(open(os.path.join(args.out, "tokens.json")))["text"])


if __name__ == "__main__":
    main()
