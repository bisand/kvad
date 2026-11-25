#!/usr/bin/env python3
"""A tiny random Nemotron-H, and what `transformers` says it computes.

`crates/llm/src/model/nemotron_h.rs` is the architecture written out by hand:
Mamba-2 layers, attention without positions, ReLU-squared MLPs, and a
mixture whose experts run in a smaller latent space. Each of those can be a
little wrong and still produce numbers, so this builds the reference
`NemotronHForCausalLM` with one layer of every kind, fills it with random
weights, runs it in f32 on the CPU, and writes

- `config.json`, spelled as the Hub spells it (`hybrid_override_pattern`)
- `model.safetensors`, under the Hub's tensor names: `backbone.*`, and one
  matrix per expert where this version of `transformers` stacks them
- `logits.json`: the tokens, and the logits at every position

into `crates/llm/tests/nemotron_h/`, where `tests/nemotron_h.rs` reads them.
The fixture is a few hundred kilobytes and committed, so `cargo test` checks
the engine against the reference with no Python. Run this again only to
change the fixture:

    /opt/homebrew/bin/python3.14 -m venv /tmp/nh-venv
    /tmp/nh-venv/bin/pip install torch transformers safetensors
    /tmp/nh-venv/bin/python scripts/nemotron-h-fixture.py crates/llm/tests/nemotron_h

Written against transformers 5.24.1 and torch 2.15.0.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import NemotronHConfig, NemotronHForCausalLM

# Mamba, attention, the latent mixture, a dense MLP, Mamba again: every kind of
# layer, and two recurrent ones so a state is carried past an attention layer.
PATTERN = "M*E-M"

CONFIG = {
    "architectures": ["NemotronHForCausalLM"],
    "model_type": "nemotron_h",
    "vocab_size": 64,
    "hidden_size": 32,
    "hybrid_override_pattern": PATTERN,
    "num_hidden_layers": len(PATTERN),
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "attention_bias": False,
    "intermediate_size": 48,
    "mlp_hidden_act": "relu2",
    "mlp_bias": False,
    "mamba_num_heads": 4,
    "mamba_head_dim": 8,
    "n_groups": 2,
    "ssm_state_size": 8,
    "conv_kernel": 4,
    "use_conv_bias": True,
    "mamba_hidden_act": "silu",
    "mamba_proj_bias": False,
    # Small, so the reference's chunked scan crosses chunk boundaries within
    # seven tokens: the engine's one-token recurrence has to agree with both.
    "chunk_size": 3,
    "n_routed_experts": 8,
    "num_experts_per_tok": 3,
    "moe_intermediate_size": 16,
    "moe_latent_size": 16,
    "moe_shared_expert_intermediate_size": 24,
    "n_shared_experts": 1,
    "routed_scaling_factor": 2.5,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "layer_norm_epsilon": 1e-5,
    "tie_word_embeddings": False,
    "max_position_embeddings": 64,
    "use_cache": True,
    "num_nextn_predict_layers": 0,
}

TOKENS = [3, 17, 8, 0, 29, 11, 5]


def randomise(model: torch.nn.Module, gen: torch.Generator) -> None:
    """Weights with no structure a wrong reading could hide behind."""
    for name, p in model.named_parameters():
        with torch.no_grad():
            if name.endswith("A_log"):
                # A = -exp(A_log), so these are decays between e^-dt and e^-4dt.
                p.copy_(torch.log(1 + 3 * torch.rand(p.shape, generator=gen)))
            elif name.endswith(".D"):
                p.copy_(0.5 + torch.rand(p.shape, generator=gen))
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5 - 1.0)
            elif "norm" in name:
                p.copy_(1 + 0.2 * torch.randn(p.shape, generator=gen))
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.3)
    for name, b in model.named_buffers():
        if name.endswith("e_score_correction_bias"):
            b.copy_(torch.randn(b.shape, generator=gen) * 0.1)


def hub_names(state: dict) -> dict:
    """`transformers`' names, back into the ones the published files use.

    The decoder is `backbone` on the Hub, and each expert is its own
    `experts.{e}.up_proj.weight` and `down_proj.weight`, where this version
    keeps one `[experts, out, in]` tensor of each per layer.
    """
    out = {}
    for name, t in state.items():
        name = name.replace("model.", "backbone.", 1) if name.startswith("model.") else name
        if name.endswith("experts.up_proj") or name.endswith("experts.down_proj"):
            stem, proj = name.rsplit(".", 1)
            for e in range(t.shape[0]):
                out[f"{stem}.{e}.{proj}.weight"] = t[e].contiguous()
            continue
        if name.endswith("conv1d.weight"):
            # Hub checkpoints store the depthwise filters as [channels, 1, kernel].
            assert t.dim() == 3
        out[name] = t.contiguous()
    return out


def main(out: Path) -> int:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1234)
    config = NemotronHConfig(**CONFIG)
    config._attn_implementation = "eager"
    model = NemotronHForCausalLM(config).float().eval()
    randomise(model, gen)

    state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    state = hub_names(state)

    with torch.no_grad():
        logits = model(torch.tensor([TOKENS])).logits[0].float()
        # The same tokens one at a time through the cache: the recurrence the
        # engine runs, rather than the chunked scan, has to give the same.
        step = []
        cache = None
        for i, t in enumerate(TOKENS):
            r = model(torch.tensor([[t]]), past_key_values=cache, use_cache=True)
            cache = r.past_key_values
            step.append(r.logits[0, -1].float())
        step = torch.stack(step)
    gap = (logits - step).abs().max().item()
    print(f"reference, whole prompt against one token at a time: {gap:.2e}")
    if gap > 1e-4:
        print("the reference disagrees with itself; not writing a fixture", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    save_file(state, out / "model.safetensors")
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    (out / "logits.json").write_text(
        json.dumps({"tokens": TOKENS, "logits": [[round(v, 6) for v in row] for row in logits.tolist()]})
        + "\n"
    )
    print(f"wrote {len(state)} tensors to {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
