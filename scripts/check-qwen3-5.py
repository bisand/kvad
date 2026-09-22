#!/usr/bin/env python3
"""Check this engine's Qwen3.5 and Qwen3-Next against what they were read from.

`tests/qwen3_5.rs` compares the engine's gated delta net with a second one
written out longhand in Rust, and that catches algebra mistakes. It cannot
catch a *misreading*: a wrong understanding of the reference would sit in both
versions equally, and they would agree with each other all the way to the
wrong answer.

This closes that gap by running the real thing — `transformers`' own
`Qwen3_5ForCausalLM` or `Qwen3NextForCausalLM`, whichever the fixture says —
on the same weights and the same tokens.

    python3 -m venv /tmp/q35-venv
    /tmp/q35-venv/bin/pip install torch transformers
    KVAD_QWEN35_FIXTURE=/tmp/q35 cargo test -p kvad --test qwen3_5 -- --ignored
    /tmp/q35-venv/bin/python scripts/check-qwen3-5.py /tmp/q35

    KVAD_QWEN35_FAMILY=next KVAD_QWEN35_FIXTURE=/tmp/next \
        cargo test -p kvad --test qwen3_5 -- --ignored
    /tmp/q35-venv/bin/python scripts/check-qwen3-5.py /tmp/next

It is not part of `cargo test` because it needs Python and PyTorch, which is
a reasonable thing to install deliberately and a poor thing to require on
every build.

The fixture is written with this engine's tensor names, and two things about
them differ from what the Python class wants. Qwen3.5's carry the multimodal
checkpoint's `language_model.` level, which `Qwen3_5ForCausalLM` — the text
model alone — does not. And Qwen3-Next's experts are one matrix each, the way
the published checkpoints store them, where this version of `transformers`
keeps them stacked into one tensor per layer. Both are undone here rather than
in the fixture, so that what the engine reads stays what the Hub actually
ships.
"""

import json
import sys
from pathlib import Path

import re

import torch
from safetensors.torch import load_file


def build(config: dict):
    """The Python model this fixture is a fixture for, and its own config."""
    if config.get("model_type") == "qwen3_next":
        from transformers import Qwen3NextForCausalLM
        from transformers.models.qwen3_next.configuration_qwen3_next import (
            Qwen3NextConfig,
        )

        flat = {k: v for k, v in config.items() if k != "model_type"}
        return Qwen3NextForCausalLM(Qwen3NextConfig(**flat))

    from transformers import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    text = {k: v for k, v in config["text_config"].items() if k != "model_type"}
    return Qwen3_5ForCausalLM(Qwen3_5TextConfig(**text))


def stack_experts(state: dict) -> dict:
    """One matrix per expert, as the Hub ships them, into one tensor per layer.

    `Qwen3NextExperts` holds `gate_up_proj` as [experts, 2 * width, hidden] —
    gate above up, which is the order `chunk(2)` reads them back in — and
    `down_proj` as [experts, hidden, width].
    """
    layers: dict[str, int] = {}
    for name in state:
        m = re.match(r"(.*\.mlp)\.experts\.(\d+)\.", name)
        if m:
            layers[m.group(1)] = max(layers.get(m.group(1), 0), int(m.group(2)) + 1)
    for mlp, count in layers.items():
        names = [f"{mlp}.experts.{i}" for i in range(count)]
        state[f"{mlp}.experts.gate_up_proj"] = torch.stack(
            [torch.cat([state[f"{n}.gate_proj.weight"], state[f"{n}.up_proj.weight"]]) for n in names]
        )
        state[f"{mlp}.experts.down_proj"] = torch.stack(
            [state[f"{n}.down_proj.weight"] for n in names]
        )
        for n in names:
            for part in ("gate_proj", "up_proj", "down_proj"):
                del state[f"{n}.{part}.weight"]
    return state


def main(fixture: Path) -> int:
    config = json.loads((fixture / "config.json").read_text())
    expected = json.loads((fixture / "logits.json").read_text())

    model = build(config)
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
    state = stack_experts(state)

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
