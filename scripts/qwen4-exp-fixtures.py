#!/usr/bin/env python3
"""A tiny random Qwen4-Exp and transformers' logits for it, for kvad's to be
compared with.

`qwen4_exp` (Qwen3.8-Flash-Next) is Qwen3.5-MoE's hybrid with three things
added, each of which can be a little wrong and still produce text:

- hyper-connections: four residual streams, mixed down to one before each
  mixer and MLP and written back to all four after it;
- QSA: a top-k block selection in front of every full-attention layer, which
  only does anything once a query sees more than `indexer_budget` tokens;
- PLE: on the layers in `ple_layer_ids`, hashed bigram and trigram
  embeddings gated into the streams through a dilated convolution.

So the model here is small enough to write in a second and shaped to reach all
three: a budget of 8 tokens in blocks of 2, a 24-token prompt, an EOS inside
it (the n-gram hash does not look across one), and the n-gram table in 12
shards (the loader concatenates them in numeric order, not string order).
Every weight is drawn again after construction, because the reference zeroes
the PLE convolution and every norm at init and a zero weight checks nothing.

    /opt/homebrew/bin/python3.14 -m venv /tmp/qwen4-venv
    /tmp/qwen4-venv/bin/pip install torch transformers safetensors
    /tmp/qwen4-venv/bin/python scripts/qwen4-exp-fixtures.py --out /tmp/qwen4-fx
    KVAD_QWEN4_FIXTURES=/tmp/qwen4-fx cargo test --release -p kvad --test qwen4_exp -- --ignored

Writes the model (`config.json`, `model.safetensors`) and `reference.safetensors`
holding `tokens` [T] and `logits` [T, vocab], f32, on the CPU. Nothing is
downloaded.
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

VOCAB, EOS = 96, 95


def config():
    text = dict(
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=4,
        # The published config says `full_attention`; the reference renames it.
        layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rope_parameters=dict(
            rope_type="default", rope_theta=10000.0, partial_rotary_factor=0.25,
            mrope_section=[2, 1, 1], mrope_interleaved=True,
        ),
        partial_rotary_factor=0.25,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        norm_topk_prob=True,
        hc_count=4,
        hc_lowrank=32,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=50,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=12,
        seed=1234,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
        output_gate_type="sigmoid",
        hidden_act="silu",
        rms_norm_eps=1e-6,
        eos_token_id=EOS,
        bos_token_id=EOS,
        tie_word_embeddings=False,
        max_position_embeddings=256,
    )
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                  patch_size=4, num_position_embeddings=16)
    return Qwen4ExpConfig(text_config=text, vision_config=vision, tie_word_embeddings=False)


def pack_experts(path):
    """Store the experts as the published checkpoint does.

    `save_pretrained` writes one `experts.N.{gate,up,down}_proj` per expert,
    and Qwen3.8-Flash-Next ships them packed: `experts.gate_up_proj` as
    `[experts, 2 * width, hidden]`, gate rows first, and `experts.down_proj`
    as `[experts, hidden, width]`. A fixture in the other spelling would test a
    loader against a file nobody publishes.
    """
    from safetensors.torch import load_file
    t = load_file(path)
    out, layers = {}, {}
    for k, v in t.items():
        parts = k.split(".")
        if ".mlp.experts." in k:
            i = parts.index("experts")
            layers.setdefault(".".join(parts[:i + 1]), {})[(int(parts[i + 1]), parts[i + 2])] = v
        else:
            out[k] = v
    for prefix, e in layers.items():
        n = 1 + max(x for x, _ in e)
        out[f"{prefix}.gate_up_proj"] = torch.stack(
            [torch.cat([e[(x, "gate_proj")], e[(x, "up_proj")]]) for x in range(n)]).contiguous()
        out[f"{prefix}.down_proj"] = torch.stack([e[(x, "down_proj")] for x in range(n)]).contiguous()
    save_file(out, path, metadata={"format": "pt"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()

    torch.manual_seed(0)
    model = Qwen4ExpForConditionalGeneration(config()).eval().float()
    with torch.no_grad():
        for name, t in model.named_parameters():
            if t.dim() == 1:
                # Norms are offsets from one here (or scales near one, for
                # the gated norm); the delta net's A_log and dt_bias too.
                t.copy_(torch.randn_like(t) * 0.2 + (1.0 if name.endswith("linear_attn.norm.weight") else 0.0))
            else:
                t.copy_(torch.randn_like(t) * 0.15)

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    pack_experts(os.path.join(a.out, "model.safetensors"))

    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, VOCAB - 1, (24,), generator=g)
    tokens[9] = EOS
    with torch.inference_mode():
        logits = model(input_ids=tokens[None], use_cache=False).logits[0]
        # The same model through its own cache, a token at a time: the two
        # must agree before either is worth comparing kvad against.
        out = model(input_ids=tokens[None, :13], use_cache=True)
        stepped = [out.logits[0]]
        cache = out.past_key_values
        for i in range(13, 24):
            out = model(input_ids=tokens[None, i:i + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            stepped.append(out.logits[0])
        stepped = torch.cat(stepped)
    gap = (stepped - logits).abs().max().item()
    print(f"whole vs cached, max |diff|: {gap:.2e}")
    assert gap < 1e-4, gap

    # The packed file is what transformers reads back: the experts' layout
    # is the published one and not a guess.
    again = Qwen4ExpForConditionalGeneration.from_pretrained(a.out, dtype=torch.float32).eval()
    with torch.inference_mode():
        gap = (again(input_ids=tokens[None], use_cache=False).logits[0] - logits).abs().max().item()
    print(f"reloaded from the packed file, max |diff|: {gap:.2e}")
    assert gap < 1e-5, gap

    # And the selection is reached: with a budget nothing exceeds, the
    # last positions' logits move. A fixture QSA does not change checks
    # nothing about QSA.
    again.model.language_model.layers[2].self_attn.indexer.token_budget = 1 << 20
    again.model.language_model.layers[2].self_attn.indexer.block_topk = 1 << 20
    with torch.inference_mode():
        full = again(input_ids=tokens[None], use_cache=False).logits[0]
    moved = (full - logits).abs().max(dim=-1).values
    print("positions QSA changes:", [i for i, m in enumerate(moved.tolist()) if m > 1e-5])
    assert moved[-1] > 1e-3

    save_file({"tokens": tokens.to(torch.int64), "logits": logits.float().contiguous()},
              os.path.join(a.out, "reference.safetensors"))
    names = sorted(json.load(open(os.path.join(a.out, "model.safetensors.index.json")))["weight_map"]) \
        if os.path.exists(os.path.join(a.out, "model.safetensors.index.json")) else None
    print("wrote", a.out, "" if names is None else f"({len(names)} tensors)")


if __name__ == "__main__":
    main()
