#!/usr/bin/env python3
"""Check this engine's Qwen3.5 against the implementation it was read from.

`tests/qwen3_5.rs` compares the engine's gated delta net with a second one
written out longhand in Rust, and that catches algebra mistakes. It cannot
catch a *misreading*: a wrong understanding of the reference would sit in both
versions equally, and they would agree with each other all the way to the
wrong answer.

This closes that gap by running the real thing — `transformers`' own
`Qwen3_5ForCausalLM` — on the same weights and the same tokens.

    python3 -m venv /tmp/q35-venv
    /tmp/q35-venv/bin/pip install torch transformers
    KVAD_QWEN35_FIXTURE=/tmp/q35 cargo test -p kvad --test qwen3_5 -- --ignored
    /tmp/q35-venv/bin/python scripts/check-qwen3-5.py /tmp/q35

It is not part of `cargo test` because it needs Python and PyTorch, which is
a reasonable thing to install deliberately and a poor thing to require on
every build.

The fixture is written with this engine's tensor names, which carry the
multimodal checkpoint's `language_model.` level. `Qwen3_5ForCausalLM` is the
text model alone and does not, so the state dict is renamed on the way in —
the one difference between what the two implementations are handed.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Qwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


def main(fixture: Path) -> int:
    config = json.loads((fixture / "config.json").read_text())
    expected = json.loads((fixture / "logits.json").read_text())

    text = dict(config["text_config"])
    text.pop("model_type", None)
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**text))
    model.eval()

    raw = load_file(fixture / "model.safetensors")
    state = {}
    for name, tensor in raw.items():
        # `model.language_model.x` is the wrapper's spelling; the text model
        # alone calls it `model.x`.
        state[name.replace("model.language_model.", "model.")] = tensor.float()
    # The depthwise convolution is stored [channels, kernel] here and wants
    # [channels, 1, kernel].
    for name in list(state):
        if name.endswith("conv1d.weight") and state[name].dim() == 2:
            state[name] = state[name].unsqueeze(1)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"missing from the fixture: {sorted(missing)}", file=sys.stderr)
        return 2
    if unexpected:
        print(f"the fixture has tensors the model does not want: {sorted(unexpected)}",
              file=sys.stderr)
        return 2

    tokens = torch.tensor([expected["tokens"]], dtype=torch.long)
    with torch.no_grad():
        theirs = model(tokens).logits[0].float()
    ours = torch.tensor(expected["logits"], dtype=torch.float32)

    if theirs.shape != ours.shape:
        print(f"shape mismatch: theirs {tuple(theirs.shape)}, ours {tuple(ours.shape)}",
              file=sys.stderr)
        return 1

    worst = (theirs - ours).abs().max().item()
    scale = max(ours.abs().max().item(), 1.0)
    print(f"positions {tuple(ours.shape)}  worst absolute difference {worst:.3e}"
          f"  (scale {scale:.3f})")
    for i in range(ours.shape[0]):
        d = (theirs[i] - ours[i]).abs().max().item()
        agree = "ok" if d < 2e-3 * scale else "DISAGREE"
        print(f"  position {i}: {d:.3e}  {agree}")
    if worst >= 2e-3 * scale:
        print("the two implementations do not agree", file=sys.stderr)
        return 1
    print("they agree")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
