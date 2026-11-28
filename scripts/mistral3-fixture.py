#!/usr/bin/env python3
"""A tiny random Mistral 3 and transformers' logits for it, for kvad's to be
compared with.

Ministral 3's language model is Mistral with two changes, each of which can be
a little wrong and still produce text:

- YaRN: the rotation's slow frequencies divided by `factor`, blended into the
  fast ones over a ramp, and the cos/sin scaled by an attention factor;
- Llama 4's query temperature: each query times `1 + beta * ln(1 + floor(pos /
  original_max_position_embeddings))`, after the rotation.

So the model here is shaped to reach both: `original_max_position_embeddings`
is 8 under a 24-token prompt, so the temperature steps at positions 8 and 16,
and YaRN's factor is 4 with `mscale` and `mscale_all_dim` unequal, so the
attention factor is not 1. It is the whole `Mistral3ForConditionalGeneration`,
a vision tower included, because that is what the real checkpoints are and
the loader has to find the language model inside it and skip the rest.

    /opt/homebrew/bin/python3.14 -m venv /tmp/m3-venv
    /tmp/m3-venv/bin/pip install torch transformers safetensors
    /tmp/m3-venv/bin/python scripts/mistral3-fixture.py --out crates/llm/tests/mistral3

Writes `config.json`, `model.safetensors` and `logits.json` (`tokens`, and
`logits` [T][vocab] in f32, on the CPU). Small enough to commit, so `cargo
test` checks against the reference with no Python. Nothing is downloaded.
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, Mistral3Config

VOCAB = 96
TOKENS = 24


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(0)

    text = {
        "model_type": "ministral3",
        "vocab_size": VOCAB,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "sliding_window": None,
        "rope_parameters": {
            "rope_type": "yarn",
            "type": "yarn",
            "rope_theta": 1000.0,
            "factor": 4.0,
            "original_max_position_embeddings": 8,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.5,
            "llama_4_scaling_beta": 0.3,
        },
    }
    vision = {
        "model_type": "pixtral",
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "head_dim": 16,
        "image_size": 28,
        "patch_size": 14,
    }
    config = Mistral3Config(text_config=text, vision_config=vision, image_token_index=VOCAB - 1, spatial_merge_size=2, tie_word_embeddings=False)
    model = AutoModelForImageTextToText.from_config(config).float().eval()
    # Every weight redrawn, norms included: a norm of ones checks nothing.
    with torch.no_grad():
        for _, w in model.named_parameters():
            w.copy_(torch.randn_like(w) * (0.3 if w.dim() == 1 else 0.08) + (1.0 if w.dim() == 1 else 0.0))

    tokens = torch.randint(0, VOCAB - 1, (1, TOKENS))
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
