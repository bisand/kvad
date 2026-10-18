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
- `attention`: the alignment heads' cross-attention over the plain tokens,
  decoded again as `crates/llm/src/speech/align.rs` decodes them for word
  times, [heads, tokens, 1500]; and in `tokens.json`, the frame each token
  starts at by openai-whisper's `timing.py` arithmetic over those weights
  (`token_starts`), and transformers' own token-level timestamps, which are
  when each token ends (`token_ends`)

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


def timing(weights, columns, sot):
    """openai-whisper's `timing.py`, steps 2 to 6, over `weights`
    [heads, rows, frames]: the frame each text token starts at, and one more
    for end-of-text. Its own dtw and median filter, written out as there."""
    w = weights[:, :, : columns // 2]
    std, mean = torch.std_mean(w, dim=-2, keepdim=True, unbiased=False)
    w = (w - mean) / std
    w = median_filter(w, 7)
    matrix = w.mean(axis=0)[sot:-1]
    x = -matrix.double().numpy()
    n, m = x.shape
    cost = np.ones((n + 1, m + 1), dtype=np.float32) * np.inf
    trace = -np.ones((n + 1, m + 1), dtype=np.float32)
    cost[0, 0] = 0
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            c0, c1, c2 = cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]
            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2
            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t
    i, j = n, m
    trace[0, :] = 2
    trace[:, 0] = 1
    path = []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if trace[i, j] == 0:
            i, j = i - 1, j - 1
        elif trace[i, j] == 1:
            i -= 1
        else:
            j -= 1
    path = np.array(path[::-1]).T
    jumps = np.pad(np.diff(path[0]), (1, 0), constant_values=1).astype(bool)
    return [int(f) for f in path[1][jumps]]


def median_filter(x, width):
    """`timing.py`'s: along the last axis, the edges reflected."""
    pad = width // 2
    if x.shape[-1] <= pad:
        return x
    x = torch.nn.functional.pad(x[None], (pad, pad, 0, 0), mode="reflect")[0]
    return x.unfold(-1, width, 1).sort()[0][..., pad]


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
    # Eager attention, so that the weights can be returned at all.
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="eager").eval()
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

    # Words with times. The alignment pass: the prompt to the task,
    # <|notimestamps|>, the plain text and end-of-text, from an empty cache.
    heads = [tuple(h) for h in model.generation_config.alignment_heads]
    sot = prompt[:-1]
    text = strip(plain)
    rows = sot + [ids("<|notimestamps|>")[0]] + text + [tok.eos_token_id]
    with torch.no_grad():
        out = model(encoder_outputs=(encoded,), decoder_input_ids=torch.tensor([rows]), output_attentions=True)
    attention = torch.stack([out.cross_attentions[l][0, h] for l, h in heads])
    token_starts = timing(attention, len(audio) // 160, len(sot))
    # transformers' own, from the same greedy decode: it is told how much of
    # the window is recording by the attention mask.
    feats = proc(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    with torch.no_grad():
        timed_tokens = model.generate(
            feats.input_features, attention_mask=feats.attention_mask, language=args.language, task="transcribe",
            num_beams=1, do_sample=False, return_token_timestamps=True,
        )
    token_ends = [round(float(t) * 50) for t in timed_tokens["token_timestamps"][0][-len(text):]]

    save_file(
        {"mel": mel[0].contiguous(), "encoded": encoded[0].contiguous(), "logits": logits.contiguous(), "attention": attention.contiguous()},
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
                "heads": heads,
                "align_rows": rows,
                "token_starts": token_starts,
                "token_ends": token_ends,
            },
            f,
            indent=1,
        )
    shutil.copy(args.wav, os.path.join(args.out, "audio.wav"))
    print(json.load(open(os.path.join(args.out, "tokens.json")))["text"])


if __name__ == "__main__":
    main()
