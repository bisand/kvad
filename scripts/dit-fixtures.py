#!/usr/bin/env python3
"""Reference outputs of diffusers' DiT for a model trained by nervus.

`nervus/src/dit.rs` saves what `train_digits` trains as a diffusers pipeline
directory, whose transformer is a `DiTTransformer2DModel`. A layout that
round-trips through nervus's own save and load proves only that the two
agree with each other. This runs diffusers itself on the saved model, in f32
on the CPU, at a few noise levels and labels, and writes its outputs.
`dit::tests::agrees_with_diffusers` compares nervus's against them.

    python3 -m venv /tmp/dit-venv
    /tmp/dit-venv/bin/pip install torch diffusers safetensors
    /tmp/dit-venv/bin/python scripts/dit-fixtures.py --model out/digits/model --out /tmp/dit-fx.safetensors
    KVAD_DIT_MODEL=out/digits/model KVAD_DIT_FIXTURES=/tmp/dit-fx.safetensors \
        cargo test --release -p nervus dit::tests::agrees_with_diffusers -- --ignored --nocapture
"""
import argparse
import os

import torch
from diffusers import DiTTransformer2DModel
from safetensors.torch import save_file

# (t, label): near noise, the middle, near the image, and "no label", which
# is the row one past the last class.
CASES = [(0.95, 3), (0.5, 7), (0.05, 1), (0.5, None)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the pipeline directory nervus saved")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dit = DiTTransformer2DModel.from_pretrained(os.path.join(args.model, "transformer"), torch_dtype=torch.float32)
    dit.eval()
    c = dit.config
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, c.in_channels, c.sample_size, c.sample_size, generator=g)

    tensors = {"x": x[0].contiguous()}
    with torch.no_grad():
        for i, (t, label) in enumerate(CASES):
            label = c.num_embeds_ada_norm if label is None else label
            # The scheduler's timestep is σ·1000, and σ is nervus's t.
            out = dit(x, timestep=torch.tensor([t * 1000.0]), class_labels=torch.tensor([label])).sample
            tensors[f"case.{i}.out"] = out[0].contiguous()
            tensors[f"case.{i}.t"] = torch.tensor([t], dtype=torch.float32)
            tensors[f"case.{i}.label"] = torch.tensor([float(label)], dtype=torch.float32)
    save_file(tensors, args.out)
    print(f"wrote {len(CASES)} cases to {args.out}")


if __name__ == "__main__":
    main()
