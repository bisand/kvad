#!/usr/bin/env python3
"""Reference outputs of a model with a LoRA, for kvad's to be compared with.

kvad applies a LoRA at run time, as a side path beside each layer it adapts
(`crates/gpu/src/image/lora.rs`). This runs diffusers with PEFT, which does
the same, on the same weights and inputs, and writes what the model makes
without the LoRA, with it, and with it at half strength, in f32 on the CPU.

`--pipeline qwen`: Qwen-Image's transformer is 82 GB in f32, so this builds
its first `--blocks` blocks only, from the cached bf16 shards, and the
LoRA's pairs for those blocks; `qwen::tests::a_lora_agrees_with_peft` builds
the same.

    HF_HUB_OFFLINE=1 python3 scripts/lora-fixtures.py --out /tmp/lora-fx \\
        --lora ~/.cache/huggingface/hub/models--lightx2v--Qwen-Image-Lightning/snapshots/*/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors
    KVAD_LORA_FIXTURES=/tmp/lora-fx cargo test --release -p kvad-gpu a_lora_agrees -- --ignored --nocapture

`--pipeline sdxl` or `sd15`: the base's text encoders and UNet, whole, with
a LoRA as diffusers' `load_lora_weights` reads it, kohya's names and all.
What each text encoder makes of the prompt, and one UNet call at 64 × 64
latents on the reference's own hidden states, so that each part's LoRA is
checked on its own. Written as `{pipeline}_{name}.safetensors`, the name
kvad knows the LoRA by with `/` as `--` and `:` as `@@`, for
`lora::tests::sd_loras_agree_with_peft` to find it by:

    HF_HUB_OFFLINE=1 python3 scripts/lora-fixtures.py --out /tmp/lora-fx --pipeline sdxl \\
        --name nerijs/pixel-art-xl --lora ~/.cache/huggingface/hub/models--nerijs--pixel-art-xl/snapshots/*/pixel-art-xl.safetensors
    KVAD_LORA_FIXTURES=/tmp/lora-fx cargo test --release -p kvad-gpu sd_loras_agree -- --ignored --nocapture
"""
import argparse
import glob
import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = "Qwen/Qwen-Image"
PROMPT = "a red fox sitting in fresh snow, photograph"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--pipeline", default="qwen", choices=["qwen", "flux", "sdxl", "sd15"])
    ap.add_argument("--name", help="the name kvad knows the LoRA by (sdxl, sd15)")
    ap.add_argument("--blocks", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.pipeline == "qwen":
        qwen(a)
    elif a.pipeline == "flux":
        flux(a)
    else:
        sd(a)


def sd(a):
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    xl = a.pipeline == "sdxl"
    repo = "stabilityai/stable-diffusion-xl-base-1.0" if xl else "stable-diffusion-v1-5/stable-diffusion-v1-5"
    cls = StableDiffusionXLPipeline if xl else StableDiffusionPipeline
    extra = {} if xl else dict(safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe = cls.from_pretrained(repo, variant="fp16", torch_dtype=torch.float32, vae=None, **extra)

    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 4, 64, 64, generator=g)
    t = 481
    time_ids = torch.tensor([[512.0, 512.0, 0.0, 0.0, 512.0, 512.0]])

    def run():
        with torch.no_grad():
            if xl:
                hidden, _, pooled, _ = pipe.encode_prompt(PROMPT, device="cpu", num_images_per_prompt=1, do_classifier_free_guidance=False)
                eps = pipe.unet(x, t, encoder_hidden_states=hidden, added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
                return {"hidden": hidden, "pooled": pooled, "eps": eps}
            hidden, _ = pipe.encode_prompt(PROMPT, "cpu", 1, False)
            return {"hidden": hidden, "eps": pipe.unet(x, t, encoder_hidden_states=hidden).sample}

    # diffusers 0.40 cannot load kohya's text-encoder pairs: `get_peft_kwargs`
    # fails on an empty rank list. So it loads the UNet's, and the text
    # encoders' are merged into their weights here, `W + (alpha/r)·s·B·A`,
    # in f32, where that is the same arithmetic as a side path.
    from safetensors.torch import load_file

    lora = load_file(a.lora)
    te = {"lora_te_": pipe.text_encoder, "lora_te1_": pipe.text_encoder, "lora_te2_": getattr(pipe, "text_encoder_2", None)}
    te_keys = [k for k in lora if k.startswith(tuple(te))]
    originals = {}

    def merge(scale):
        for m, w in originals.items():
            m.weight.data.copy_(w)
        for k in te_keys:
            if not k.endswith(".lora_down.weight"):
                continue
            base = k[: -len(".lora_down.weight")]
            prefix = next(p for p in te if base.startswith(p))
            # Newer transformers have no `text_model.` level; kohya's names do.
            modules = {k.replace(".", "_"): m for n, m in te[prefix].named_modules() for k in (n, f"text_model.{n}")}
            module = modules[base[len(prefix):]]
            down, up = lora[k].float(), lora[f"{base}.lora_up.weight"].float()
            alpha = lora[f"{base}.alpha"].item() if f"{base}.alpha" in lora else down.shape[0]
            originals.setdefault(module, module.weight.data.clone())
            module.weight.data += (up @ down) * (alpha / down.shape[0]) * scale

    out = {"x": x, "t": torch.tensor([float(t)])}
    out.update({f"plain_{k}": v for k, v in run().items()})
    pipe.load_lora_weights({k: v for k, v in lora.items() if k not in te_keys}, adapter_name="l")
    merge(1.0)
    out.update({f"adapted_{k}": v for k, v in run().items()})
    pipe.set_adapters(["l"], adapter_weights=[0.5])
    merge(0.5)
    out.update({f"half_{k}": v for k, v in run().items()})
    name = a.name.replace("/", "--").replace(":", "@@")
    save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(a.out, f"{a.pipeline}_{name}.safetensors"))
    d = lambda u, v: 10 * torch.log10((u**2).mean() / ((u - v) ** 2).mean()).item()
    print(f"{a.name}: the LoRA moves the hidden states to {d(out['plain_hidden'], out['adapted_hidden']):.1f} dB of them, "
          f"and the UNet's answer to {d(out['plain_eps'], out['adapted_eps']):.1f} dB of it")


def flux(a):
    """FLUX.1-schnell's first double and single blocks, `--blocks` of each,
    from the cached bf16 shards, with a LoRA as diffusers' FLUX loader reads
    it: its own conversion of kohya's names in Black Forest Labs' layout,
    fused `qkv` and `linear1` split. Written as `flux_{name}.safetensors`."""
    from diffusers import FluxTransformer2DModel
    from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin
    from huggingface_hub import snapshot_download

    root = snapshot_download("black-forest-labs/FLUX.1-schnell", allow_patterns=["transformer/*"])
    cfg = json.load(open(os.path.join(root, "transformer/config.json")))
    cfg["num_layers"] = cfg["num_single_layers"] = a.blocks
    model = FluxTransformer2DModel.from_config(cfg).to(torch.float32).eval()
    block = lambda k: re.match(r"(single_)?transformer_blocks\.(\d+)\.", k)
    keep = lambda k: not (m := block(k)) or int(m.group(2)) < a.blocks
    sd = {}
    for shard in sorted(glob.glob(os.path.join(root, "transformer/*.safetensors"))):
        with safe_open(shard, "pt") as f:
            for k in f.keys():
                if keep(k):
                    sd[k] = f.get_tensor(k).to(torch.float32)
    model.load_state_dict(sd, strict=True)

    # An 8 × 8 grid of packed latents, 12 tokens of T5, CLIP's pooled
    # vector, and the positions the pipeline gives them: (0, row, column)
    # for the picture, zeros for the text.
    g = torch.Generator().manual_seed(0)
    rows = cols = 8
    x = torch.randn(1, rows * cols, cfg["in_channels"], generator=g)
    txt = torch.randn(1, 12, cfg["joint_attention_dim"], generator=g)
    pooled = torch.randn(1, cfg["pooled_projection_dim"], generator=g)
    img_ids = torch.stack(torch.meshgrid(torch.zeros(1), torch.arange(rows).float(), torch.arange(cols).float(), indexing="ij"), -1).reshape(-1, 3)
    txt_ids = torch.zeros(12, 3)
    sigma = 0.6

    def run():
        with torch.no_grad():
            return model(hidden_states=x, encoder_hidden_states=txt, pooled_projections=pooled, timestep=torch.tensor([sigma]),
                         img_ids=img_ids, txt_ids=txt_ids, return_dict=False)[0]

    plain = run()
    lora = FluxLoraLoaderMixin.lora_state_dict(a.lora)
    lora = {k: v.to(torch.float32) for k, v in lora.items() if k.startswith("transformer.") and keep(k.removeprefix("transformer."))}
    model.load_lora_adapter(lora, adapter_name="l", prefix="transformer")
    adapted = run()
    model.set_adapters(["l"], weights=[0.5])
    half = run()
    name = a.name.replace("/", "--").replace(":", "@@")
    save_file(
        {"x": x, "txt": txt, "pooled": pooled, "sigma": torch.tensor([sigma]), "plain": plain.contiguous(), "adapted": adapted.contiguous(), "half": half.contiguous()},
        os.path.join(a.out, f"flux_{name}.safetensors"),
    )
    d = lambda u, v: 10 * torch.log10((u**2).mean() / ((u - v) ** 2).mean()).item()
    print(f"{a.name}: the LoRA moves the output to {d(plain, adapted):.1f} dB of it")


def qwen(a):
    from diffusers import QwenImageTransformer2DModel
    from diffusers.loaders.lora_pipeline import QwenImageLoraLoaderMixin
    from huggingface_hub import snapshot_download

    root = snapshot_download(REPO, allow_patterns=["transformer/*"])
    cfg = json.load(open(os.path.join(root, "transformer/config.json")))
    cfg["num_layers"] = a.blocks
    model = QwenImageTransformer2DModel.from_config(cfg).to(torch.float32).eval()

    # The weights of the blocks kept, and everything outside the blocks.
    keep = lambda k: not (m := re.match(r"transformer_blocks\.(\d+)\.", k)) or int(m.group(1)) < a.blocks
    sd = {}
    for shard in sorted(glob.glob(os.path.join(root, "transformer/*.safetensors"))):
        with safe_open(shard, "pt") as f:
            for k in f.keys():
                if keep(k):
                    sd[k] = f.get_tensor(k).to(torch.float32)
    model.load_state_dict(sd, strict=True)

    # Inputs: an 8 × 8 grid of packed latents, 12 tokens of text, a noise
    # level in the middle.
    g = torch.Generator().manual_seed(0)
    rows = cols = 8
    x = torch.randn(1, rows * cols, cfg["in_channels"], generator=g)
    txt = torch.randn(1, 12, cfg["joint_attention_dim"], generator=g)
    sigma = 0.6

    def run():
        with torch.no_grad():
            return model(
                hidden_states=x,
                encoder_hidden_states=txt,
                encoder_hidden_states_mask=torch.ones(1, txt.shape[1]),
                timestep=torch.tensor([sigma]),
                img_shapes=[[(1, rows, cols)]],
                return_dict=False,
            )[0]

    plain = run()
    lora = QwenImageLoraLoaderMixin.lora_state_dict(a.lora)
    # Its names come back under `transformer.`, diffusers' pipeline's name
    # for the model.
    lora = {k: v.to(torch.float32) for k, v in lora.items() if keep(k.removeprefix("transformer."))}
    model.load_lora_adapter(lora, adapter_name="l", prefix="transformer")
    adapted = run()
    model.set_adapters(["l"], weights=[0.5])
    half = run()
    save_file(
        {"x": x, "txt": txt, "sigma": torch.tensor([sigma]), "plain": plain.contiguous(), "adapted": adapted.contiguous(), "half": half.contiguous()},
        os.path.join(a.out, "qwen_lora.safetensors"),
    )
    d = lambda u, v: 10 * torch.log10((u**2).mean() / ((u - v) ** 2).mean()).item()
    print(f"wrote {a.out}/qwen_lora.safetensors: the LoRA moves the output to {d(plain, adapted):.1f} dB of it")


if __name__ == "__main__":
    main()
