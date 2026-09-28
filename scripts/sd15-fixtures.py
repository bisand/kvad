#!/usr/bin/env python3
"""Reference outputs of Stable Diffusion 1.5's parts, for kvad's to be
compared with.

`crates/gpu/src/image/sd15.rs` runs SD 1.5 on the pieces SDXL's pipeline
already has. A picture that looks right cannot say whether a part is a little
wrong, so this runs diffusers itself on the same weights and inputs, in f32 on
the CPU, and writes what each part makes: the text encoder's hidden states,
one UNet call, and the VAE's decode. `sd15::tests::agrees_with_diffusers`
compares kvad's against them.

    python3 -m venv /tmp/sd-venv
    /tmp/sd-venv/bin/pip install torch diffusers transformers safetensors
    HF_HUB_OFFLINE=1 /tmp/sd-venv/bin/python scripts/sd15-fixtures.py --out /tmp/sd15-fx
    KVAD_SD15_FIXTURES=/tmp/sd15-fx cargo test --release -p kvad-gpu sd15::tests -- --ignored --nocapture

`--preview` also runs the whole pipeline once, in f16 on MPS, and fits the
latent-to-colour matrix `sd15.rs`'s previews use to its final latent and the
image decoded from it, as `sdxl.rs`'s was fitted.
"""
import argparse
import os

import torch
from safetensors.torch import save_file

REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROMPT = "a red fox sitting in fresh snow, photograph"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    kw = dict(variant="fp16", torch_dtype=torch.float32)
    tok = CLIPTokenizer.from_pretrained(a.repo, subfolder="tokenizer")
    text = CLIPTextModel.from_pretrained(a.repo, subfolder="text_encoder", **kw).eval()
    unet = UNet2DConditionModel.from_pretrained(a.repo, subfolder="unet", **kw).eval()
    vae = AutoencoderKL.from_pretrained(a.repo, subfolder="vae", **kw).eval()

    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        # The prompt as the pipeline encodes it: padded to 77 with the end
        # marker, the last hidden state.
        ids = tok(PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        hidden = text(ids)[0]
        # One UNet call, at 64 × 64 latents and a timestep in the middle.
        x = torch.randn(1, 4, 64, 64, generator=g)
        t = 481
        eps = unet(x, t, encoder_hidden_states=hidden).sample
        # The VAE's decode of a latent the size of a 512² image, as the
        # pipeline scales it.
        z = torch.randn(1, 4, 64, 64, generator=g)
        pixels = vae.decode(z / vae.config.scaling_factor).sample
    save_file(
        {
            "ids": ids.to(torch.float32).contiguous(),
            "hidden": hidden.contiguous(),
            "x": x,
            "t": torch.tensor([float(t)]),
            "eps": eps.contiguous(),
            "z": z,
            "pixels": pixels.contiguous(),
        },
        os.path.join(a.out, "sd15.safetensors"),
    )
    print(f"wrote {a.out}/sd15.safetensors")

    if a.preview:
        preview(a)


def preview(a):
    """Fit rows of RGB to SD 1.5's latent channels, as sdxl.rs's PREVIEW was:
    each final latent pixel against the mean of the 8 × 8 pixels it decodes
    to, by least squares, with a bias."""
    from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        a.repo, variant="fp16", torch_dtype=torch.float16, safety_checker=None, feature_extractor=None, requires_safety_checker=False
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("mps")
    prompt = "a busy market street in marrakech, vivid colours, photograph"
    g = torch.Generator("mps").manual_seed(7)
    lat = pipe(prompt, num_inference_steps=25, generator=g, output_type="latent").images
    with torch.no_grad():
        img = pipe.vae.decode(lat / pipe.vae.config.scaling_factor).sample.float().cpu()
    lat = (lat.float().cpu() / pipe.vae.config.scaling_factor)[0]  # [4, 64, 64]
    img = img[0].clamp(-1, 1)  # [3, 512, 512], in [-1, 1]
    blocks = img.reshape(3, 64, 8, 64, 8).mean(dim=(2, 4))  # [3, 64, 64]
    X = torch.cat([lat.reshape(4, -1).T, torch.ones(64 * 64, 1)], 1)  # [n, 5]
    Y = blocks.reshape(3, -1).T  # [n, 3]
    sol = torch.linalg.lstsq(X, Y).solution  # [5, 3]
    resid = ((X @ sol - Y) ** 2).sum(0)
    total = ((Y - Y.mean(0)) ** 2).sum(0)
    explained = 1 - resid / total
    rows = ", ".join("[" + ", ".join(f"{v:.4f}" for v in sol[i].tolist()) + "]" for i in range(4))
    print(f"PREVIEW = [{rows}]")
    print("PREVIEW_BIAS = [" + ", ".join(f"{v:.4f}" for v in sol[4].tolist()) + "]")
    print("explained: " + ", ".join(f"{v * 100:.0f}%" for v in explained.tolist()) + " of R, G, B")


if __name__ == "__main__":
    main()
