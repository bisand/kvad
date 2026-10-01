#!/usr/bin/env python3
"""Reference outputs of diffusers' VAE encoders, for kvad's to be compared with.

`crates/gpu/src/image/vae.rs` has an encoder for the VAEs of SD 1.5, SDXL and
FLUX, which are all diffusers' `AutoencoderKL`, and `qwen.rs` one for
Qwen-Image's, a video VAE run on one frame (`AutoencoderKLQwenImage`). A round trip through kvad's
own encoder and decoder can look right and still be a little wrong in both
halves, so this runs diffusers itself on the same picture, in f32 on the CPU,
and writes, for each VAE: the picture as both sides see it, the mean and
log-variance the encoder makes of it, and what the decoder makes of the mean.
`vae::tests::the_encoder_agrees_with_diffusers` compares kvad's against them,
and `qwen::tests::the_encoder_agrees_with_diffusers` Qwen-Image's.

    python3 -m venv /tmp/vae-venv
    /tmp/vae-venv/bin/pip install torch diffusers safetensors pillow
    HF_HUB_OFFLINE=1 /tmp/vae-venv/bin/python scripts/vae-fixtures.py --image some.png --out /tmp/vae-fx
    KVAD_VAE_FIXTURES=/tmp/vae-fx cargo test --release -p kvad-gpu vae::tests -- --ignored --nocapture

The picture is cut to a square from its middle and scaled to 512×512 here,
so that kvad, which reads no PNG, starts from exactly the same pixels.
"""
import argparse
import os

import torch
from diffusers import AutoencoderKL, AutoencoderKLQwenImage
from PIL import Image
from safetensors.torch import save_file

# The VAEs kvad's pipelines load, by the name the test knows them by; the
# variant of the weights file where the repo only has that one; the
# precision the pipeline runs the VAE in; and its class.
VAES = {
    "sdxl": ("madebyollin/sdxl-vae-fp16-fix", None, None, torch.float16, AutoencoderKL),
    "flux": ("black-forest-labs/FLUX.1-schnell", "vae", None, torch.bfloat16, AutoencoderKL),
    "sd15": ("stable-diffusion-v1-5/stable-diffusion-v1-5", "vae", "fp16", torch.float16, AutoencoderKL),
    "qwen": ("Qwen/Qwen-Image", "vae", None, torch.bfloat16, AutoencoderKLQwenImage),
}
SIDE = 512


def picture(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((SIDE, SIDE), Image.LANCZOS)
    x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).reshape(SIDE, SIDE, 3).permute(2, 0, 1)
    return (x.float() / 127.5 - 1.0).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="one of " + ", ".join(VAES))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    x = picture(args.image)

    for name, (repo, sub, variant, half, cls) in VAES.items():
        if args.only and name != args.only:
            continue
        vae = cls.from_pretrained(repo, subfolder=sub, variant=variant, torch_dtype=torch.float32)
        vae.eval()
        # A video VAE takes `[B, C, T, H, W]`: the picture is a clip of one
        # frame, and its frame axis is dropped again from what comes out.
        video = cls is AutoencoderKLQwenImage
        frame = (lambda t: t.unsqueeze(2)) if video else (lambda t: t)
        still = (lambda t: t.squeeze(2)) if video else (lambda t: t)
        with torch.no_grad():
            posterior = vae.encode(frame(x)).latent_dist
            decoded = still(vae.decode(posterior.mean).sample)
            posterior.mean, posterior.logvar = still(posterior.mean), still(posterior.logvar)
        tensors = {
            "image": x[0].contiguous(),
            "mean": posterior.mean[0].contiguous(),
            "logvar": posterior.logvar[0].contiguous(),
            "decoded": decoded[0].contiguous(),
        }
        # diffusers' own mean in the pipeline's precision, on the GPU: how far
        # half precision moves the reference, to judge kvad's by.
        if torch.backends.mps.is_available():
            with torch.no_grad():
                h = still(vae.to("mps", half).encode(frame(x).to("mps", half)).latent_dist.mean)
            tensors["mean_half"] = h[0].float().cpu().contiguous()
        save_file(tensors, os.path.join(args.out, f"{name}.safetensors"))
        err = (decoded - x).pow(2).mean().item()
        psnr = 10 * torch.log10(torch.tensor(4.0 / err)).item()  # the range is 2 wide
        print(f"{name}: latent {tuple(posterior.mean.shape[1:])}, diffusers' own round trip {psnr:.2f} dB")


if __name__ == "__main__":
    main()
