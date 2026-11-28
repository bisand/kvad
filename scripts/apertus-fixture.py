#!/usr/bin/env python3
"""A tiny random Apertus and transformers' logits for it, for kvad's to be
compared with.

Apertus is Llama with three changes, each of which can be a little wrong and
still produce text:

- xIELU in an MLP with no gate, its two slopes learned per layer;
- Qwen3's per-head RMSNorm on the queries and keys, before the rotation;
- Llama 3.1's stretched rotation (`rope_type: llama3`), by wavelength.

So the model here is shaped to reach all three: the slopes are redrawn per
layer, and with `original_max_position_embeddings` 32 and a head of 16, one
frequency pair keeps its speed, two are blended and the rest are slowed.

    /opt/homebrew/bin/python3.14 -m venv /tmp/m3-venv
    /tmp/m3-venv/bin/pip install torch transformers safetensors
    /tmp/m3-venv/bin/python scripts/apertus-fixture.py --out crates/llm/tests/apertus

Writes `config.json`, `model.safetensors` and `logits.json` (`tokens`, and
`logits` [T][vocab] in f32, on the CPU). Small enough to commit, so `cargo
test` checks against the reference with no Python. Nothing is downloaded.
"""
import argparse
import json
import os

import torch
from transformers import ApertusConfig, AutoModelForCausalLM

VOCAB = 96
TOKENS = 24


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(0)

    config = ApertusConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        hidden_act="xielu",
        dtype=torch.float32,
        rope_parameters={
            "rope_type": "llama3",
            "rope_theta": 1000.0,
            "factor": 4.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 32,
        },
    )
    model = AutoModelForCausalLM.from_config(config).float().eval()
    # Every parameter redrawn, norms and the activation's slopes included: a
    # norm of ones checks nothing, and two layers with one slope check half.
    with torch.no_grad():
        for _, w in model.named_parameters():
            w.copy_(torch.randn_like(w) * (0.3 if w.dim() <= 1 else 0.08) + (1.0 if w.dim() <= 1 else 0.0))

    tokens = torch.randint(0, VOCAB, (1, TOKENS))
    with torch.inference_mode():
        logits = model(input_ids=tokens).logits[0].float()

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    for extra in ["generation_config.json"]:
        path = os.path.join(a.out, extra)
        if os.path.exists(path):
            os.remove(path)
    with open(os.path.join(a.out, "logits.json"), "w") as f:
        json.dump({"tokens": tokens[0].tolist(), "logits": [[round(v, 6) for v in row] for row in logits.tolist()]}, f)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
