#!/usr/bin/env python3
"""Reference outputs of LTX-2.5's decoders and text path, for kvad's to be
compared with.

`crates/gpu/src/video` reimplements LTX-2.5's convolutional video decoder, its
audio decoder, vocoder and bandwidth extension, and its text path: Gemma 4,
the aggregate projections and the two connectors. Unit tests there check the
pieces against their definitions, and that catches algebra mistakes. It
cannot catch a *misreading* of the reference. This runs the reference itself,
Lightricks' own `ltx-core`, on the same weights and the same latents, and
writes what it makes for the examples to compare against.

    python3 -m venv /tmp/ltx-venv
    /tmp/ltx-venv/bin/pip install torch torchaudio einops safetensors av colour-science
    git clone --depth 1 https://github.com/Lightricks/LTX-2 /tmp/LTX-2

    VIDEO=$(cargo run -q --release -p kvad-gpu --example ltx_decode -- --where)
    AUDIO=$(cargo run -q --release -p kvad-gpu --example ltx_audio -- --where)
    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --video "$VIDEO" --audio "$AUDIO" --out /tmp/ltx-fx

    F=/tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_decode -- --cpu \\
        --latent $F/random_latent.safetensors --expect $F/random_frames.safetensors
    cargo run --release -p kvad-gpu --example ltx_decode -- \\
        --latent $F/pattern_latent.safetensors --expect $F/pattern_frames.safetensors --out pattern.mp4
    cargo run --release -p kvad-gpu --example ltx_audio -- \\
        --latent $F/audio_sound_latent.safetensors --expect $F/audio_sound_out.safetensors \\
        --expect-bwe $F/audio_random_bwe.safetensors --out sound.wav

    TEXT=$(cargo run -q --release -p kvad-gpu --example ltx_text -- --where | head -1)
    DIT=$(cargo run -q --release -p kvad-gpu --example ltx_text -- --where | tail -1)
    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --text "$TEXT" --dit "$DIT" --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_text -- --fixtures /tmp/ltx-fx [--quant q8]

    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --dit "$DIT" --contexts /tmp/ltx-fx/text_contexts_f32.safetensors --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_dit -- --fixtures /tmp/ltx-fx [--quant q8] [--f32]

    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src:/tmp/LTX-2/packages/ltx-pipelines/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --video "$VIDEO" --picture synthetic --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_encode -- --fixtures /tmp/ltx-fx --picture /tmp/ltx-fx/picture.png

    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --video "$VIDEO" --clip 384x256x25 --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_encode -- --fixtures /tmp/ltx-fx --clip 384x256x25

    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --dit "$DIT" --contexts random --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_dit -- --fixtures /tmp/ltx-fx --held
    cargo run --release -p kvad-gpu --example ltx_dit -- --fixtures /tmp/ltx-fx --blind
    cargo run --release -p kvad-gpu --example ltx_dit -- --fixtures /tmp/ltx-fx --deaf

    LORA=$(cargo run -q --release -p kvad-gpu --example ltx_fetch -- loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors | cut -d' ' -f1)
    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --dit "$DIT" --contexts random --lora "$LORA" --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_dit -- --fixtures /tmp/ltx-fx --lora "$LORA"
    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src:/tmp/LTX-2/packages/ltx-pipelines/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --dit "$DIT" --contexts random --guided --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_dit -- --fixtures /tmp/ltx-fx --guided
    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --dit "$DIT" --contexts random --conditioned --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_cond -- --fixtures /tmp/ltx-fx

    UP=$(cargo run -q --release -p kvad-gpu --example ltx_upsample -- --where | head -1)
    DETAILING=$(cargo run -q --release -p kvad-gpu --example ltx_fetch -- --repo Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler \
        ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors | cut -d' ' -f1)
    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src:/tmp/LTX-2/packages/ltx-pipelines/src /tmp/ltx-venv/bin/python \
        scripts/ltx-fixtures.py --dit "$DIT" --contexts random --dfr --detailing "$DETAILING" \
        --upsampler "$UP" --temporal "$(cargo run -q --release -p kvad-gpu --example ltx_upsample -- --temporal --where | head -1)" \
        --vae "$VIDEO" --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_dfr -- --fixtures /tmp/ltx-fx --detailing "$DETAILING"

    cargo run --release -p kvad-gpu --example ltx_duration -- --contexts /tmp/ltx-fx "a door slams shut" "…"
    HEAD=$(cargo run -q --release -p kvad-gpu --example ltx_duration -- --where)
    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --duration "$HEAD" --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_duration -- --fixtures /tmp/ltx-fx

    DIFFVAE=$(cargo run -q --release -p kvad-gpu --example ltx_diffvae -- --where)
    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --diffvae "$DIFFVAE" --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_diffvae -- --fixtures /tmp/ltx-fx [--f32 | --cpu]

    UP=$(cargo run -q --release -p kvad-gpu --example ltx_upsample -- --where | head -1)
    cargo run --release -p kvad-gpu --example ltx -- --prompt "…" --stages 1 \\
        --width 512 --height 320 --frames 25 --latents /tmp/stage1.safetensors
    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --upsampler "$UP" --vae "$VIDEO" --latent /tmp/stage1.safetensors --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_upsample -- --fixtures /tmp/ltx-fx [--cpu | --f32]

    TUP=$(cargo run -q --release -p kvad-gpu --example ltx_upsample -- --where --temporal | head -1)
    cargo run --release -p kvad-gpu --example ltx -- --prompt "…" \\
        --width 768 --height 512 --frames 49 --latents /tmp/stage2.safetensors
    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --upsampler "$TUP" --vae "$VIDEO" --latent /tmp/stage2.safetensors --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_upsample -- --fixtures /tmp/ltx-fx --temporal [--cpu | --f32]

(the text fixtures also want `transformers` 5.8 to 5.14 in the venv, the
picture `pillow`, and `--guided` `torchaudio` and `scipy`, for the pipelines
package it imports the guided denoiser from). The
`--where` lines download each file first. The repo is gated: accept its
licence on the Hub and have a token saved.)

What to expect, as measured on 2026-09-24 on an M5 Pro:

- The video decoder in f32, on the CPU or on Metal, agrees with the
  reference to about 122 dB PSNR: f32 rounding, 0.00 of an 8-bit level.
- In bf16 on Metal it is 59 dB from the reference's f32. That is no worse
  than the reference's own bf16, which this script measures on MPS: 58 dB.
- The audio path, which kvad runs in f32 throughout, agrees to 124 dB at the
  spectrogram and 91-97 dB at 16 and 48 kHz.
- Tokens are identical, including a prompt cut at 1024 tokens.
- Gemma's first six layers in f32 agree to 108-115 dB at every hidden state.
  In bf16 on Metal they are 42-55 dB from the reference's f32, the same as
  the reference's own bf16 on MPS, to within half a dB at every state.
- All 48 layers in bf16 on Metal are 36-56 dB from the reference's bf16 on
  MPS (two bf16 computations drifting apart); at q8, 27-46 dB.
- The DiT's first two blocks and its output heads, at 512x320x25 and
  sigma 0.9875: identical positions and tokens; 115-120 dB in f32 on the CPU
  and 118-123 on Metal. In bf16 on Metal, video 46.2/45.0 dB after the
  blocks and 42.0 at the velocity, audio 44.6-45.1, where the reference's
  own bf16 on MPS is 46.8/46.6, 43.6 and 45.5-46.1. At q8, much the same.
- A picture for image-to-video (1000x700, drawn here): scaled and cut to
  384x256 and 768x512 exactly as the reference does (150 dB); the encoder
  105.9-108.0 dB in f32 on the CPU and 37.4-41.0 in bf16 on Metal, where the
  reference's own bf16 on MPS is 37.0-40.1. kvad's H.264 round trip, by
  ffmpeg, is 48.2 dB from the reference's, by PyAV.
- A clip through the encoder (`--clip 384x256x25`, drawn here): 108.4 dB in
  f32 on the CPU, 107.9 on Metal, and 36.9 in bf16 on Metal, where the
  reference's own bf16 on MPS is 35.9.
- The DiT with its first latent frame held at sigma 0 (`--held`), on seeded
  contexts: 107-118 dB in f32 on the CPU; in bf16 on Metal video 47.1/46.0
  and velocity 42.9 dB, where the reference's own bf16 is 46.9/46.3 and 43.7.
- The duration head, on eight prompts' contexts from kvad's text path: within
  5e-7 of the reference's seconds in f32 on the CPU and on Metal, and the same
  frames for each; the reference's own bf16 is within 1% and the same frames.
- Guidance's perturbations, on the DiT's first two blocks: the last block's
  self-attention skipped (`--blind`) 112-120 dB in f32, and every audio-video
  attention skipped (`--deaf`) 115-120 dB; in bf16 on Metal each where the
  reference's own bf16 is. The distilled LoRA fused (`--lora`): 115-121 dB,
  its fused weights rounded to bf16 as the reference rounds them. One guided
  prediction, four passes combined (`--guided`): video 102.6 and audio
  99.2 dB in f32; 35.6 and 27.4 in bf16, where the reference's own is 35.7
  and 27.1. Any DiT of this architecture serves for these: `ltx_dit --path`
  loads the dev model's.
- The diffusion decoder (DiffVAE), each stage from the reference's input to
  it, on a seeded 3x8x8 latent: 120.1, 125.0 and 123.8 dB in f32 on Metal
  (stages 1-3, 4 and 5); in bf16 43.4, 47.9 and 47.5, where the reference's
  own bf16 on MPS is 43.6, 48.8 and 50.8.
- The latent upsampler, on a 512x320x25 latent from a generation: 98 dB in
  f32 on the CPU and on Metal. In bf16 it is 27 dB from exact, and so is
  the reference's own bf16 on MPS; kvad runs it in f32.

Every reference model here runs in f32 on the CPU, so the files are what the
architecture computes, not what one GPU's rounding makes of it. It is not part
of `cargo test` because it needs Python, PyTorch and a clone of `LTX-2`.
"""

import argparse
import dataclasses
import json
import math
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def read(path):
    with safe_open(path, framework="pt") as f:
        meta = {k: json.loads(v) if v.strip().startswith("{") else v for k, v in f.metadata().items()}
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    return meta, tensors


def load(model, tensors, rename):
    """Load `tensors` into `model` under the names `rename` gives, and refuse
    to go on if the model wanted a weight that was not there."""
    sd = {}
    for k, v in tensors.items():
        n = rename(k)
        if n is not None:
            sd[n] = v
    missing, _ = model.load_state_dict(sd, strict=False)
    assert not missing, missing
    return model.float().eval()


def psnr(a, b):
    return 10 * math.log10(1 / max((a - b).pow(2).mean().item(), 1e-30))


def video(path, out):
    from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator, VideoEncoderConfigurator

    meta, tensors = read(path)
    stats = lambda k: k if k.startswith("per_channel_statistics.") else None
    dec = load(VideoDecoderConfigurator.from_metadata(meta), tensors, lambda k: k[len("decoder."):] if k.startswith("decoder.") else stats(k))
    enc = load(VideoEncoderConfigurator.from_metadata(meta), tensors, lambda k: k[len("encoder."):] if k.startswith("encoder.") else stats(k))

    def decode(z):
        with torch.no_grad():
            return dec(z).float().add(1).mul(0.5).clamp(0, 1)[0]  # [3, T, H, W], as kvad's --expect wants

    # 1. A seeded random latent: pure arithmetic, nothing to look at.
    z = torch.randn(1, 128, 3, 4, 6, generator=torch.Generator().manual_seed(0))
    save_file({"latent": z[0].contiguous()}, f"{out}/random_latent.safetensors")
    save_file({"frames": decode(z).contiguous()}, f"{out}/random_frames.safetensors")

    # 2. A clip to look at: a disc moving over gradients, above a row of bars,
    # encoded by the reference's own encoder and decoded again.
    T, H, W = 17, 256, 384
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    clip = torch.zeros(3, T, H, W)
    for f in range(T):
        cx, cy = 60 + 14 * f, 128 + 50 * math.sin(f / 3)
        disc = (((xx - cx) ** 2 + (yy - cy) ** 2) < 40**2).float()
        bars = ((xx // 24).long() % 2).float() * (yy > 200).float()
        clip[0, f] = 0.2 + 0.6 * xx / W
        clip[1, f] = 0.3 + 0.5 * yy / H
        clip[2, f] = 0.5 + 0.4 * torch.sin((xx + 8 * f) / 20)
        for c, v in enumerate((1.0, 0.85, 0.1)):
            clip[c, f] = clip[c, f] * (1 - disc) + v * disc
        clip[:, f] = clip[:, f] * (1 - bars) + bars
    with torch.no_grad():
        z = enc(clip.mul(2).sub(1)[None])
    frames = decode(z)
    print(f"video: the reference's own round trip is {psnr(frames, clip):.1f} dB from the source clip")
    save_file({"latent": z[0].contiguous()}, f"{out}/pattern_latent.safetensors")
    save_file({"frames": frames.contiguous()}, f"{out}/pattern_frames.safetensors")
    save_file({"frames": clip.contiguous()}, f"{out}/pattern_source.safetensors")

    # 3. How far bf16 alone moves the reference, on MPS if there is one (on
    # the CPU, bf16 3D convolutions take the better part of an hour).
    if torch.backends.mps.is_available():
        d16 = dec.to("mps", torch.bfloat16)
        with torch.no_grad():
            x = d16(z.to("mps", torch.bfloat16)).float().add(1).mul(0.5).clamp(0, 1)[0].cpu()
        print(f"video: the reference in bf16 on MPS is {psnr(x, frames):.1f} dB from its f32")
        dec.float().cpu()


def clip(path, out, frames, height, width):
    """A clip through the reference's video encoder, which is causal: a disc
    moving over gradients and changing colour, above bars that scroll, so
    that every latent frame differs from the one before. f32 on the CPU, and
    bf16 on MPS for the drift bf16 costs the reference."""
    from ltx_core.model.video_vae.model_configurator import VideoEncoderConfigurator

    meta, tensors = read(path)
    stats = lambda k: k if k.startswith("per_channel_statistics.") else None
    enc = load(VideoEncoderConfigurator.from_metadata(meta), tensors, lambda k: k[len("encoder."):] if k.startswith("encoder.") else stats(k))
    T, H, W = frames, height, width
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    rgb = torch.zeros(3, T, H, W)
    for f in range(T):
        cx, cy = W * (0.15 + 0.7 * f / max(T - 1, 1)), H * (0.45 + 0.2 * math.sin(f / 3))
        disc = (((xx - cx) ** 2 + (yy - cy) ** 2) < (H / 6) ** 2).float()
        bars = (((xx + 6 * f) // (W / 16)).long() % 2).float() * (yy > 0.8 * H).float()
        rgb[0, f] = 0.2 + 0.6 * xx / W
        rgb[1, f] = 0.3 + 0.5 * yy / H
        rgb[2, f] = 0.5 + 0.4 * torch.sin((xx + 8 * f) / 20)
        for c, v in enumerate((1.0, 0.85 - 0.6 * f / T, 0.1 + 0.8 * f / T)):
            rgb[c, f] = rgb[c, f] * (1 - disc) + v * disc
        rgb[:, f] = rgb[:, f] * (1 - bars) + bars
    pixels = rgb.mul(2).sub(1)
    with torch.no_grad():
        z = enc(pixels[None])
    # Frames first, as kvad holds a clip: [T, 3, H, W].
    fx = {"frames": pixels.permute(1, 0, 2, 3).contiguous(), "latent": z[0].contiguous()}
    if torch.backends.mps.is_available():
        e16 = enc.to("mps", torch.bfloat16)
        with torch.no_grad():
            fx["latent_bf16"] = e16(pixels[None].to("mps", torch.bfloat16)).float()[0].cpu().contiguous()
        enc.float().cpu()
    save_file(fx, f"{out}/clip_{W}x{H}x{T}.safetensors")
    print(f"clip: {W}×{H}×{T} to latent {tuple(z.shape[1:])}")


def media_io():
    """The reference's own picture handling, `ltx_pipelines.utils.media_io`'s
    `decode` and `resize`, without the package `__init__`s above them, which
    import every pipeline and model. OpenImageIO is only for EXR stills, and
    stands in as an empty module."""
    import importlib
    import os
    import sys
    import types

    import ltx_pipelines

    root = os.path.dirname(ltx_pipelines.__file__)
    for name in ("ltx_pipelines.utils", "ltx_pipelines.utils.media_io"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(root, *name.split(".")[1:])]
            sys.modules[name] = m
    sys.modules.setdefault("OpenImageIO", types.ModuleType("OpenImageIO"))
    return importlib.import_module("ltx_pipelines.utils.media_io.decode"), importlib.import_module("ltx_pipelines.utils.media_io.resize")


def picture(path, image, out):
    """A picture through the reference's image-to-video preparation and its
    video encoder: decoded, re-compressed as one H.264 frame at CRF 18 (the
    value `detect_params` gives a 2.5 checkpoint), then, at stage 1's size
    and stage 2's, scaled to cover, cut from the middle, and encoded."""
    from ltx_core.model.video_vae.model_configurator import VideoEncoderConfigurator

    decode, resize = media_io()
    if image is None:
        # Something to look at, deliberately not the shape of either target,
        # so that both the scaling and the cut are exercised.
        from PIL import Image

        H, W = 700, 1000
        yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
        rgb = torch.stack([0.2 + 0.6 * xx / W, 0.3 + 0.5 * yy / H, 0.5 + 0.4 * torch.sin(xx / 37 + yy / 53)])
        disc = (((xx - 420) ** 2 + (yy - 330) ** 2) < 150**2).float()
        bars = ((xx // 40).long() % 2).float() * (yy > 560).float()
        for c, v in enumerate((1.0, 0.85, 0.1)):
            rgb[c] = rgb[c] * (1 - disc) + v * disc
        rgb = rgb * (1 - bars) + bars
        image = f"{out}/picture.png"
        Image.fromarray((rgb.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()).save(image)
    rgb = decode.preprocess(decode.decode_image(image), crf=18)

    meta, tensors = read(path)
    stats = lambda k: k if k.startswith("per_channel_statistics.") else None
    enc = load(VideoEncoderConfigurator.from_metadata(meta), tensors, lambda k: k[len("encoder."):] if k.startswith("encoder.") else stats(k))
    fx = {"rgb": torch.from_numpy(rgb.copy()).contiguous()}
    fx16 = {}
    for w, h in ((384, 256), (768, 512)):
        pixels = resize.resize_and_center_crop(torch.tensor(rgb, dtype=torch.float32), h, w) / 127.5 - 1.0
        with torch.no_grad():
            z = enc(pixels)
        fx[f"pixels_{w}x{h}"] = pixels[0, :, 0].contiguous()
        fx[f"latent_{w}x{h}"] = z[0].contiguous()
        if torch.backends.mps.is_available():
            e16 = enc.to("mps", torch.bfloat16)
            with torch.no_grad():
                fx16[f"latent_{w}x{h}"] = e16(pixels.to("mps", torch.bfloat16)).float()[0].cpu().contiguous()
            enc.float().cpu()
        print(f"picture: {rgb.shape[1]}×{rgb.shape[0]} to {w}×{h}, latent {tuple(z.shape[1:])}")
    save_file(fx, f"{out}/picture_f32.safetensors")
    if fx16:
        save_file(fx16, f"{out}/picture_bf16.safetensors")


def duration(path, out):
    """The duration head, on contexts kvad's text path made
    (`examples/ltx_duration.rs --contexts`): the head reads nothing else, so
    the same contexts give the reference's prediction for the same prompts."""
    from ltx_core.duration_head.model_configurator import DURATION_HEAD_KEY_OPS, DurationHeadConfigurator

    meta, tensors = read(path)
    head = load(DurationHeadConfigurator.from_metadata(meta), tensors, lambda k: k[len("duration_head."):] if k.startswith("duration_head.") else None)
    ctx = load_file(f"{out}/duration_contexts.safetensors")
    n = len([k for k in ctx if k.startswith("video_")])

    def run(device, dtype):
        h = head.to(device, dtype)
        with torch.no_grad():
            return {
                f"seconds_{i}": h(ctx[f"video_{i}"][None].to(device, dtype), ctx[f"audio_{i}"][None].to(device, dtype)).float().cpu()
                for i in range(n)
            }

    f32 = run("cpu", torch.float32)
    save_file(f32, f"{out}/duration_f32.safetensors")
    if torch.backends.mps.is_available():
        save_file(run("mps", torch.bfloat16), f"{out}/duration_bf16.safetensors")
    print("duration: " + ", ".join(f"{f32[f'seconds_{i}'].item():.3f} s" for i in range(n)))


def diffvae_model(path):
    """The diffusion decoder (DiffVAE) as the reference builds it, in f32, on
    its eager neighbourhood-attention path: NATTEN's semantics written out
    in torch, and for keyframes its joint attention, `joint_eager`."""
    from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator
    from ltx_core.model.video_vae import diffusion_tiling
    from ltx_core.model.video_vae.transformer.apply import apply_diffvae_config
    from ltx_core.model.video_vae.transformer.config import DiffVAEBlockKind, DiffVAEConfig, NAttentionKind

    meta, tensors = read(path)
    # The loader's renames: `decoder.` off, the fused qkv into its three, the
    # timestep MLP under its module's names.
    sd = {}
    for k, v in tensors.items():
        if k.startswith("per_channel_statistics."):
            sd[k] = v
        if not k.startswith("decoder."):
            continue
        k = k[len("decoder."):].replace("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1.").replace("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2.")
        if ".qkv." in k:
            d = v.shape[0] // 3
            leaf = k.rsplit(".", 1)[1]
            for i, name in enumerate(("to_q", "to_k", "to_v")):
                sd[k.replace(f"qkv.{leaf}", f"qkv.{name}.{leaf}")] = v[i * d : (i + 1) * d]
        else:
            sd[k] = v
    dec = VideoDecoderConfigurator.from_metadata(meta)
    missing, unexpected = dec.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    dec = apply_diffvae_config(
        dec.float().eval(),
        DiffVAEConfig(block=DiffVAEBlockKind.COMBINED, w_chunks=1, natten_backend=None, attention=NAttentionKind.EAGER_SDPA, compile_blocks=False, compile_det_stages=False),
    )
    return dec


def diffvae(path, out):
    """The diffusion decoder (DiffVAE), stage by stage, by the reference's own
    modules. A seeded latent of 3 x 8 x 8 (17 frames of 256 x 256), the
    smallest the stages' windows allow in space; the stage-5 noise is drawn
    here and saved, so that kvad's decoder starts from the same."""
    from ltx_core.model.video_vae import diffusion_tiling

    dec = diffvae_model(path)
    g = torch.Generator().manual_seed(5)
    z = torch.randn(1, 128, 3, 8, 8, generator=g)
    with torch.no_grad():
        t = time.time()
        padded = diffusion_tiling.pad_trailing_latent_for_natten_border(z, dec._natten_trailing_pad_latent_frames)
        feat = dec.forward_stages_1_to_3(padded, drop_leading_frame=True)
        print(f"diffvae: stages 1-3 to {tuple(feat.shape)} in {time.time() - t:.1f} s")
        t = time.time()
        context = dec.forward_stage_4(feat, drop_leading_frame=True, pad_trailing=True)
        print(f"diffvae: stage 4 to {tuple(context.shape)} in {time.time() - t:.1f} s")
        f5, h5, w5 = context.shape[1], context.shape[2] * dec.patch_size, context.shape[3] * dec.patch_size
        noise = torch.randn(1, 3, f5, h5, w5, generator=g)
        t = time.time()
        pixels = dec.forward_diff_step(dec._context_and_x_for_diff_step(context, noise), torch.ones(1))
        print(f"diffvae: stage 5 to {tuple(pixels.shape)} in {time.time() - t:.1f} s")
    save_file(
        {
            "latent": z[0].contiguous(),
            # Channels-last, as the reference keeps them: [T, H, W, C].
            "stage4_input": feat[0].contiguous(),
            "context": context[0].contiguous(),
            # Frames first, as kvad keeps them: [T, 3, H, W].
            "noise": noise[0].permute(1, 0, 2, 3).contiguous(),
            "pixels": pixels[0].permute(1, 0, 2, 3).contiguous(),
        },
        f"{out}/diffvae_f32.safetensors",
    )

    # The reference's own bf16 on MPS, each stage from its f32 input, as
    # kvad's check runs it: the drift bf16 costs the reference itself.
    if torch.backends.mps.is_available():
        m = dec.to("mps", torch.bfloat16)
        on = lambda x: x.to("mps", torch.bfloat16)
        with torch.no_grad():
            t = time.time()
            f16 = m.forward_stages_1_to_3(on(padded), drop_leading_frame=True).float().cpu()
            c16 = m.forward_stage_4(on(feat), drop_leading_frame=True, pad_trailing=True).float().cpu()
            p16 = m.forward_diff_step(m._context_and_x_for_diff_step(on(context), on(noise)), torch.ones(1, device="mps")).float().cpu()
            print(f"diffvae: the reference in bf16 on MPS, {time.time() - t:.1f} s")
        save_file(
            {"stage4_input": f16[0].contiguous(), "context": c16[0].contiguous(), "pixels": p16[0].permute(1, 0, 2, 3).contiguous()},
            f"{out}/diffvae_bf16.safetensors",
        )


def diffvae_keyframes(path, out):
    """The DiffVAE's keyframe-aware decode, stage by stage, as DFR runs it:
    the same seeded latent as `diffvae`, and two keyframe planes, seeded
    latents at pixel frames 8 and 16. Both streams through stages 1-3, stage
    4 and the one step of stage 5, with the stage-5 noise of both drawn here
    and saved. The joint attention is the reference's `joint_eager`: each
    video query's window, cut at the edges, and the same rows and columns on
    its two nearest planes, in one softmax; each plane's own window and the
    same on its two nearest frames."""
    import sys
    import types

    from ltx_core.model.video_vae import diffusion_tiling
    from ltx_core.model.video_vae.keyframes import DecodeKeyframes

    sys.modules.setdefault("OpenImageIO", types.ModuleType("OpenImageIO"))
    dec = diffvae_model(path)
    g = torch.Generator().manual_seed(5)
    z = torch.randn(1, 128, 3, 8, 8, generator=g)
    gk = torch.Generator().manual_seed(6)
    planes = torch.randn(1, 128, 2, 8, 8, generator=gk)
    indices = torch.tensor([8, 16])
    keys = DecodeKeyframes(latents=planes, pixel_frame_indices=indices)

    def run(m, on, z, feat=None, stream=None, context=None, kcontext=None, noise=None, knoise=None):
        """Each stage from the f32 run's input to it, when given."""
        out = {}
        with torch.no_grad():
            padded = diffusion_tiling.pad_trailing_latent_for_natten_border(on(z), m._natten_trailing_pad_latent_frames)
            f, st = m.forward_stages_1_to_3_with_keyframes(padded, dataclasses.replace(keys, latents=on(keys.latents)), drop_leading_frame=True)
            out["stage4_input"], out["kf_stage4_input"] = f[0].float().cpu(), st.x[0].float().cpu()
            if feat is not None:
                f, st = on(feat)[None], dataclasses.replace(st, x=on(stream)[None])
            c, kst = m.forward_stage_4_with_keyframes(f, st, indices, drop_leading_frame=True, pad_trailing=True)
            out["context"], out["kf_context"], out["kf_times"] = c[0].float().cpu(), kst.x[0].float().cpu(), kst.times.float().cpu()
            if context is not None:
                c, kst = on(context)[None], dataclasses.replace(kst, x=on(kcontext)[None])
            f5, h5, w5 = c.shape[1], c.shape[2] * m.patch_size, c.shape[3] * m.patch_size
            if noise is None:
                noise = torch.randn(1, 3, f5, h5, w5, generator=g)
                knoise = torch.randn(1, 3, kst.x.shape[1], h5, w5, generator=gk)
            out["noise"], out["kf_noise"] = noise, knoise
            px, kpx = m.forward_diff_step_with_keyframes(
                m._context_and_x_for_diff_step(c, on(noise)), m._keyframe_context_and_x_for_diff_step(kst.x, on(knoise), kst.valid),
                torch.ones(1, device=c.device, dtype=c.dtype), kst.times, kst.valid,
            )
            out["pixels"], out["kf_pixels"] = px[0].float().cpu(), kpx[0].float().cpu()
        return out

    t = time.time()
    r = run(dec, lambda x: x, z)
    print(f"diffvae: keyframe decode, 2 planes, f32 on the CPU, {time.time() - t:.1f} s")
    frames = lambda x: x.permute(1, 0, 2, 3).contiguous()
    save_file(
        {
            "latent": z[0].contiguous(),
            "planes": planes[0].contiguous(),
            "indices": indices.float(),
            # Channels-last, as the reference keeps them: [T, H, W, C].
            "stage4_input": r["stage4_input"].contiguous(),
            "kf_stage4_input": r["kf_stage4_input"].contiguous(),
            "context": r["context"].contiguous(),
            "kf_context": r["kf_context"].contiguous(),
            "kf_times": r["kf_times"].contiguous(),
            # Frames first, as kvad keeps them: [T, 3, H, W].
            "noise": frames(r["noise"][0]),
            "kf_noise": frames(r["kf_noise"][0]),
            "pixels": frames(r["pixels"]),
            "kf_pixels": frames(r["kf_pixels"]),
        },
        f"{out}/diffvae_kf_f32.safetensors",
    )
    # The reference's own bf16 on MPS, each stage from its f32 input.
    if torch.backends.mps.is_available():
        m = dec.to("mps", torch.bfloat16)
        on = lambda x: x.to("mps", torch.bfloat16)
        t = time.time()
        b = run(m, on, z, r["stage4_input"], r["kf_stage4_input"], r["context"], r["kf_context"], r["noise"], r["kf_noise"])
        print(f"diffvae: keyframe decode, the reference in bf16 on MPS, {time.time() - t:.1f} s")
        save_file(
            {"stage4_input": b["stage4_input"].contiguous(), "kf_stage4_input": b["kf_stage4_input"].contiguous(),
             "context": b["context"].contiguous(), "kf_context": b["kf_context"].contiguous(), "pixels": frames(b["pixels"])},
            f"{out}/diffvae_kf_bf16.safetensors",
        )


def audio(path, out):
    from ltx_core.model.audio_vae.audio_vae import encode_audio
    from ltx_core.model.audio_vae.model_configurator import (
        AudioDecoderConfigurator,
        AudioEncoderConfigurator,
        VocoderConfigurator,
    )
    from ltx_core.types import Audio

    meta, tensors = read(path)
    prefix = "audio_vae.per_channel_statistics."
    stats = lambda k: "per_channel_statistics." + k[len(prefix):] if k.startswith(prefix) else None
    under = lambda p: lambda k: k[len(p):] if k.startswith(p) else stats(k)
    dec = load(AudioDecoderConfigurator.from_metadata(meta), tensors, under("audio_vae.decoder."))
    enc = load(AudioEncoderConfigurator.from_metadata(meta), tensors, under("audio_vae.encoder."))
    voc = load(VocoderConfigurator.from_metadata(meta), tensors, lambda k: k.removeprefix("vocoder.") if k.startswith("vocoder.") else None)

    def run(z):
        with torch.no_grad():
            mel = dec(z)
            return {"mel": mel[0], "low": voc.vocoder(mel)[0], "high": voc(mel)[0]}

    # 1. A seeded random latent, and the bandwidth extension's insides for it.
    z = torch.randn(1, 8, 26, 16, generator=torch.Generator().manual_seed(1))
    save_file({"latent": z[0].contiguous()}, f"{out}/audio_random_latent.safetensors")
    save_file({k: v.contiguous() for k, v in run(z).items()}, f"{out}/audio_random_out.safetensors")
    with torch.no_grad():
        low = voc.vocoder(dec(z))
        mel = voc._compute_mel(low)
        insides = {"bwe_mel": mel[0], "residual": voc.bwe_generator(mel.transpose(2, 3))[0], "skip": voc.resampler(low)[0]}
    save_file({k: v.contiguous() for k, v in insides.items()}, f"{out}/audio_random_bwe.safetensors")

    # 2. Sound: a chord on the left, a rising chirp on the right, clicks on
    # both, encoded by the reference's own encoder.
    sr = 48000
    t = torch.arange(2 * sr) / sr
    left = 0.2 * sum(torch.sin(2 * math.pi * f * t) for f in (220, 277.2, 329.6))
    right = 0.4 * torch.sin(2 * math.pi * (200 * t + 900 * t * t))
    clicks = ((torch.arange(2 * sr) % (sr // 4)) < 24).float() * 0.6
    wave = torch.stack([left + clicks, right + clicks]).clamp(-1, 1)[None]
    with torch.no_grad():
        z = encode_audio(Audio(waveform=wave, sampling_rate=sr), enc)
    save_file({"latent": z[0].contiguous()}, f"{out}/audio_sound_latent.safetensors")
    save_file({k: v.contiguous() for k, v in run(z).items()}, f"{out}/audio_sound_out.safetensors")
    save_file({"wave": wave[0].contiguous()}, f"{out}/audio_sound_source.safetensors")


PROMPTS = [
    "A golden retriever running through a sunny meadow, cinematic lighting",
    "  Ein Fuchs springt über den Zaun \u2014 狐が柵を飛び越える.\n",
    " ".join(["a slow dolly shot across a rain-soaked neon street at night"] * 120),
]


def gemma_text_model(assets, device, dtype, layers=None):
    """transformers' own Gemma 4 text tower, with the LTX file's weights,
    built on `device` one tensor at a time (the whole model never sits in
    memory twice, which in bf16 would be 48 GB)."""
    from transformers.models.gemma4_unified import modeling_gemma4_unified as m

    from ltx_core.text_encoders.gemma.gemma_assets import build_gemma_hf_config

    cfg = build_gemma_hf_config(assets).text_config
    if layers is not None:
        cfg.num_hidden_layers = layers
        cfg.layer_types = cfg.layer_types[:layers]
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = m.Gemma4UnifiedTextModel(cfg)
    model = model.to_empty(device=device).to(dtype)
    with safe_open(assets.weight_paths[0], framework="pt") as f:
        for name, t in list(model.named_parameters()) + list(model.named_buffers()):
            key = "model." + name
            if key in f.keys():
                t.data.copy_(f.get_tensor(key).to(device=device, dtype=t.dtype))
    # Buffers that are computed rather than stored.
    fresh = m.Gemma4UnifiedTextRotaryEmbedding(cfg)
    for name, b in fresh.named_buffers():
        getattr(model.rotary_emb, name).data.copy_(b.to(device))
    model.embed_tokens.embed_scale = torch.tensor(model.embed_tokens.scalar_embed_scale, device=device)
    return model.eval()


def text(path, dit, out):
    from ltx_core.text_encoders.gemma.embeddings_connector import (
        AudioEmbeddings1DConnectorConfigurator,
        Embeddings1DConnectorConfigurator,
    )
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
    from ltx_core.text_encoders.gemma.encoders.base_encoder import build_gemma_tokenizer
    from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2
    from ltx_core.text_encoders.gemma.gemma_assets import GemmaAssets

    assets = GemmaAssets.load(path)
    tok = build_gemma_tokenizer(assets)

    # 1. Tokens: the ids and mask LTX feeds Gemma, for prompts that trim,
    # mix scripts and overrun 1024 tokens.
    tokens = []
    for p in PROMPTS:
        pairs = tok.tokenize_with_weights(p)["gemma"]
        ids = [int(t) for t, m in pairs if int(m) == 1]
        tokens.append({"prompt": p, "ids": ids})
    with open(f"{out}/text_tokens.json", "w") as f:
        json.dump(tokens, f, ensure_ascii=False)
    print("text: token counts", [len(t["ids"]) for t in tokens])

    def run(model, ids, device, dtype):
        n = len(ids)
        full = torch.tensor([[0] * (1024 - n) + ids], device=device)
        mask = torch.tensor([[0] * (1024 - n) + [1] * n], device=device)
        with torch.no_grad():
            hs = model(input_ids=full, attention_mask=mask, output_hidden_states=True).hidden_states
        return torch.stack([h[0, 1024 - n :].float().cpu() for h in hs])  # [states, n, width]

    ids = tokens[0]["ids"]
    # 2. The first six layers (one of them global) in f32 on the CPU: exact.
    six = run(gemma_text_model(assets, "cpu", torch.float32, layers=6), ids, "cpu", torch.float32)
    save_file({"states": six.contiguous()}, f"{out}/text_gemma6_f32.safetensors")
    print("text: six-layer states", tuple(six.shape))

    # 3. All 48 layers in bf16 on MPS: what LTX itself runs.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = gemma_text_model(assets, device, torch.bfloat16)
    t = time.time()
    full = run(model, ids, device, torch.bfloat16)
    print(f"text: 49 states {tuple(full.shape)} in bf16 on {device}, {time.time() - t:.1f} s")
    save_file({"states": full.contiguous()}, f"{out}/text_gemma_bf16.safetensors")
    del model
    if device == "mps":
        torch.mps.empty_cache()

    # 4. From those states, the projections and both connectors in f32: the
    # rest of the text path, exactly. They need the DiT's file.
    if dit is None:
        print("text: no --dit, so no contexts")
        return
    # Only the connectors: the file is 42 GB and the rest is the DiT's.
    with safe_open(dit, framework="pt") as f:
        meta = {k: json.loads(v) if v.strip().startswith("{") else v for k, v in f.metadata().items()}
        tensors = {k: f.get_tensor(k) for k in f.keys() if "_embeddings_connector." in k}
    vconn = load(Embeddings1DConnectorConfigurator.from_metadata(meta), tensors, lambda k: k[len("model.diffusion_model.video_embeddings_connector."):] if k.startswith("model.diffusion_model.video_embeddings_connector.") else None)
    aconn = load(AudioEmbeddings1DConnectorConfigurator.from_metadata(meta), tensors, lambda k: k[len("model.diffusion_model.audio_embeddings_connector."):] if k.startswith("model.diffusion_model.audio_embeddings_connector.") else None)
    del tensors
    with safe_open(path, framework="pt") as f:
        def linear(name):
            w = f.get_tensor(f"text_embedding_projection.{name}.weight").float()
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data, lin.bias.data = w, f.get_tensor(f"text_embedding_projection.{name}.bias").float()
            return lin
        fx = FeatureExtractorV2(video_aggregate_embed=linear("video_aggregate_embed"), embedding_dim=full.shape[2], audio_aggregate_embed=linear("audio_aggregate_embed"))
    proc = EmbeddingsProcessor(feature_extractor=fx, video_connector=vconn, audio_connector=aconn).eval()
    n = full.shape[1]
    padded = tuple(torch.cat([torch.zeros(1024 - n, full.shape[2]), s])[None] for s in full)
    mask = torch.tensor([[0] * (1024 - n) + [1] * n])
    with torch.no_grad():
        o = proc.process_hidden_states(padded, mask)
    save_file({"video": o.video_encoding[0].contiguous(), "audio": o.audio_encoding[0].contiguous()}, f"{out}/text_contexts_f32.safetensors")
    print("text: contexts", tuple(o.video_encoding.shape), tuple(o.audio_encoding.shape))


def transformer(path, out, contexts, blocks, lora=None, guided=False, conditioned=False, dfr=None):
    """The DiT's first `blocks` blocks, with everything around them: the
    patchify projections, the eight adaLN modules, the keyframe embedding,
    the RoPE tables built from the reference's own positions, and the output
    heads. The whole DiT is 22B parameters, 88 GB in f32; two blocks of it
    and the rest are 1.2B. What is left out is only more of the same block."""
    from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
    from ltx_core.tools import AudioLatentTools, VideoLatentTools
    from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape

    prefix = "model.diffusion_model."
    with safe_open(path, framework="pt") as f:
        meta = {k: json.loads(v) if v.strip().startswith("{") else v for k, v in f.metadata().items()}
        keep = lambda k: k.startswith(prefix) and "_embeddings_connector." not in k and not (
            k.startswith(prefix + "transformer_blocks.") and int(k.split(".")[3]) >= blocks
        )
        tensors = {k: f.get_tensor(k) for k in f.keys() if keep(k)}
    meta["config"]["transformer"]["num_layers"] = blocks

    # 512×320, 25 frames at 24 fps: 4 latent frames of 10×16, 640 video
    # tokens, and 26 audio latents. Small enough for f32 on the CPU.
    width, height, frames, fps = 512, 320, 25, 24.0
    pixels = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)
    vtools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), VideoLatentShape.from_pixel_shape(pixels), fps)
    atools = AudioLatentTools(AudioPatchifier(patch_size=1), AudioLatentShape.from_video_pixel_shape(pixels))
    vstate = vtools.create_initial_state("cpu", torch.float32)
    astate = atools.create_initial_state("cpu", torch.float32)
    g = torch.Generator().manual_seed(7)
    vlat = torch.randn(vstate.latent.shape, generator=g)
    alat = torch.randn(astate.latent.shape, generator=g)
    sigma = 0.9875
    if contexts == "random":
        # Any contexts check the DiT's arithmetic, and seeded ones need no
        # 12B Gemma in f32 to make.
        gc = torch.Generator().manual_seed(11)
        ctx = {"video": torch.randn(64, 4096, generator=gc), "audio": torch.randn(64, 2048, generator=gc)}
    else:
        ctx = load_file(contexts)
    # A negative prompt's contexts, for guidance: seeded too.
    gn = torch.Generator().manual_seed(12)
    neg = {"video": torch.randn(64, 4096, generator=gn), "audio": torch.randn(64, 2048, generator=gn)}
    # Image-to-video: the first latent frame is the picture, held at σ = 0
    # while the rest is denoised. Its tokens are the first h·w.
    frame = vstate.latent.shape[1] // VideoLatentShape.from_pixel_shape(pixels).frames

    def perturbed(kind, device, dtype):
        """Guidance's perturbations: `blind`, spatio-temporal guidance on the
        last block loaded (LTX-2.5's is block 28, past what two blocks reach),
        for video and audio; `deaf`, both audio-video attentions skipped in
        every block."""
        from ltx_core.guidance.perturbations import (
            BatchedPerturbationConfig,
            Perturbation,
            PerturbationConfig,
            PerturbationType,
        )

        last = [blocks - 1]
        ptb = {
            "blind": [Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=last), Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=last)],
            "deaf": [Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None), Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None)],
        }[kind]
        return BatchedPerturbationConfig([PerturbationConfig(ptb)], num_blocks=blocks, device=device, dtype=dtype)

    def run(device, dtype, held=False, kind=None):
        m = LTXModelConfigurator.from_metadata(meta)
        m = load(m, tensors, lambda k: k[len(prefix):])
        m = m.to(device=device, dtype=dtype)
        caught = {}
        for i, b in enumerate(m.transformer_blocks):
            b.register_forward_hook(lambda _m, _i, o, i=i: caught.update({f"video_{i}": o[0].x[0].float().cpu(), f"audio_{i}": o[1].x[0].float().cpu()}))

        def modality(state, lat, context):
            s = torch.tensor([sigma], device=device)
            mask = None if state.keyframes_mask is None else state.keyframes_mask.to(device)
            denoise = state.denoise_mask.clone()
            if held and state is vstate:
                denoise[:, :frame] = 0
            return Modality(
                latent=lat.to(device=device, dtype=dtype),
                sigma=s,
                timesteps=(denoise.to(device) * s).float(),
                positions=state.positions.to(device),
                context=context.to(device=device, dtype=dtype)[None],
                keyframes_mask=mask,
            )

        with torch.no_grad():
            t = time.time()
            ptb = perturbed(kind, device, dtype) if kind else None
            v, a = m(modality(vstate, vlat, ctx["video"]), modality(astate, alat, ctx["audio"]), ptb)
            print(f"dit: {blocks} blocks in {dtype} on {device}{', the first frame held' if held else ''}{', ' + kind if kind else ''}, {time.time() - t:.1f} s")
        caught.update({"video_out": v[0].float().cpu(), "audio_out": a[0].float().cpu()})
        return {k: v.contiguous() for k, v in caught.items()}

    save_file(
        {
            "video": vlat[0].contiguous(),
            "audio": alat[0].contiguous(),
            # The same, unpatchified by the reference: [128, F, h, w], [8, T, 16].
            "video_latent": vtools.patchifier.unpatchify(vlat, vtools.target_shape)[0].contiguous(),
            "audio_latent": atools.patchifier.unpatchify(alat, atools.target_shape)[0].contiguous(),
            "video_positions": vstate.positions[0].contiguous(),
            "audio_positions": astate.positions[0].contiguous(),
            "sigma": torch.tensor([sigma]),
            "shape": torch.tensor([width, height, frames, fps]),
            "video_context": ctx["video"].contiguous(),
            "audio_context": ctx["audio"].contiguous(),
            "video_negative": neg["video"].contiguous(),
            "audio_negative": neg["audio"].contiguous(),
        },
        f"{out}/dit_inputs.safetensors",
    )
    def run_guided(device, dtype):
        """One guided prediction by the reference's own `_guided_denoise`:
        four passes (the prompt, the negative prompt, STG on the last block
        loaded, the streams without each other) combined by LTX-2.5's
        guidance, CFG 3 and 7, STG 1, modality 3, rescale 0.7."""
        import dataclasses
        import sys
        import types

        sys.modules.setdefault("OpenImageIO", types.ModuleType("OpenImageIO"))
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        from ltx_core.model.transformer import X0Model
        from ltx_pipelines.utils.denoisers import _guided_denoise

        m = load(LTXModelConfigurator.from_metadata(meta), tensors, lambda k: k[len(prefix):]).to(device=device, dtype=dtype)
        def on(st, lat):
            moved = {f.name: getattr(st, f.name).to(device) for f in dataclasses.fields(st) if isinstance(getattr(st, f.name), torch.Tensor)}
            moved["latent"] = lat.to(device, dtype)
            return dataclasses.replace(st, **moved)

        guide = lambda cfg, n: MultiModalGuider(
            params=MultiModalGuiderParams(cfg_scale=cfg, stg_scale=1.0, stg_blocks=[blocks - 1], rescale_scale=0.7, modality_scale=3.0),
            negative_context=n.to(device, dtype)[None],
        )
        with torch.no_grad():
            t = time.time()
            v, a = _guided_denoise(
                X0Model(m), on(vstate, vlat), on(astate, alat), torch.tensor(sigma, device=device), guide(3.0, neg["video"]), guide(7.0, neg["audio"]),
                ctx["video"].to(device, dtype)[None], ctx["audio"].to(device, dtype)[None], last_denoised_video=None, last_denoised_audio=None, step_index=0,
            )
            print(f"dit: one guided prediction, {blocks} blocks in {dtype} on {device}, {time.time() - t:.1f} s")
        return {"video": v.denoised[0].float().cpu().contiguous(), "audio": a.denoised[0].float().cpu().contiguous()}

    def run_conditioned(device, dtype):
        """DFR's conditioning tokens on one state, built by the reference's
        own items at the 60 fps DFR conditions at: the target video; two
        anchor keyframes (`VideoConditionByKeyframeIndex`, strength 0.95, in
        bf16 as DFR's carried keyframes are) at pixel frames 0 and 16; two
        generated keyframe slots at 8 and 24, seeded with initial latents;
        and a half-size reference latent (`VideoConditionByReferenceLatent`,
        downscale 2, strength 1). Noised as `GaussianNoiser` noises, at 0.975,
        from noise drawn here and saved. The sound frozen, as DFR's temporal
        tiles freeze it: σ 0 everywhere."""
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
        from ltx_core.conditioning.types.keyframe_slots import VideoGeneratedKeyframeSlots
        from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent

        ctools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), VideoLatentShape.from_pixel_shape(pixels), 60.0)
        target = vtools.patchifier.unpatchify(vlat, vtools.target_shape)
        st = ctools.create_initial_state("cpu", torch.float32, initial_latent=target)
        gk = torch.Generator().manual_seed(21)
        lat = target.shape
        anchors = torch.randn(1, 128, 2, lat[3], lat[4], generator=gk).to(torch.bfloat16)
        initials = torch.randn(1, 128, 2, lat[3], lat[4], generator=gk)
        reference = torch.randn(1, 128, lat[2], lat[3] // 2, lat[4] // 2, generator=gk)
        for i, f in enumerate((0, 16)):
            st = VideoConditionByKeyframeIndex(anchors[:, :, i : i + 1], frame_idx=f, strength=0.95).apply_to(st, ctools)
        st = VideoGeneratedKeyframeSlots((8, 24), initial_keyframes=initials).apply_to(st, ctools)
        st = VideoConditionByReferenceLatent(reference, downscale_factor=2, strength=1.0).apply_to(st, ctools)
        noise = torch.randn(st.latent.shape, generator=gk)
        noiser = GaussianNoiser(torch.Generator())
        noiser._sample_noise = lambda _st: noise
        st = noiser(st, noise_scale=0.975)

        m = load(LTXModelConfigurator.from_metadata(meta), tensors, lambda k: k[len(prefix):]).to(device=device, dtype=dtype)
        s = torch.tensor([sigma], device=device)
        zero = torch.tensor([0.0], device=device)
        video = Modality(
            latent=st.latent.to(device=device, dtype=dtype),
            sigma=s,
            timesteps=(st.denoise_mask.to(device) * s).float(),
            positions=st.positions.to(device),
            context=ctx["video"].to(device=device, dtype=dtype)[None],
            keyframes_mask=st.keyframes_mask.to(device),
        )
        audio = Modality(
            latent=alat.to(device=device, dtype=dtype),
            sigma=zero,
            timesteps=torch.zeros_like(astate.denoise_mask).to(device),
            positions=astate.positions.to(device),
            context=ctx["audio"].to(device=device, dtype=dtype)[None],
        )
        with torch.no_grad():
            t = time.time()
            v, a = m(video, audio, None)
            print(f"dit: {blocks} blocks, conditioned, {st.latent.shape[1]} video tokens, in {dtype} on {device}, {time.time() - t:.1f} s")
        state = {
            "anchors": anchors.float()[0].contiguous(),
            "initials": initials[0].contiguous(),
            "reference": reference[0].contiguous(),
            "noise": noise[0].contiguous(),
            "noised": st.latent[0].contiguous(),
            "clean": st.clean_latent[0].contiguous(),
            "mask": st.denoise_mask[0, :, 0].contiguous(),
            "positions": st.positions[0].contiguous(),
            "marks": st.keyframes_mask[0, :, 0].contiguous(),
        }
        return state, {"video_out": v[0].float().cpu().contiguous(), "audio_out": a[0].float().cpu().contiguous()}

    def run_dfr(device, dtype, replay=None):
        """DFR's stages 1 and 2 as `DFRPipeline.__call__` runs them, through
        the reference's own `DiffusionStage`, with the DiT cut to its first
        blocks: 512×320 × 49 frames at 48 fps, so the canvas has keyframes
        at 24 and 48 and the DiT is told 60 fps; stage 1 at 256×160, eight
        plain Euler steps with the keyframe slots; the video and the slots
        upsampled, apart, by the spatial upsampler; stage 2 at 512×320,
        three steps, the slots seeded with the upsampled ones and stage 1's
        video as the reference latent. With the temporal upsampler, then
        two of DFR's temporal rounds on stage 2's answer, as its `__call__`
        runs them: 97 frames at 96 fps in two tiles, then 193 at 192 in
        four, each tile four ancestral steps at η 0.5 with its seams held and
        the sound frozen. The noise every stage and step draws is saved, and
        replayed in `replay`'s place when given.

        The one departure: the ancestral loop keeps its latents in the
        run's dtype, where DFR leaves them in its default bf16, so that the
        f32 run is f32 throughout. The anchors are rounded to bf16 all the
        same, as DFR's are when it carries them."""
        import contextlib
        import sys
        import types
        from types import SimpleNamespace

        # The pipelines' media reader, which nothing here reads with.
        sys.modules.setdefault("OpenImageIO", types.ModuleType("OpenImageIO"))
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.conditioning import VideoConditionByReferenceLatent, VideoGeneratedKeyframeSlots
        from ltx_core.model.transformer import X0Model
        from ltx_core.model.upsampler.model import upsample_video
        from ltx_core.model.upsampler.model_configurator import LatentUpsamplerConfigurator
        from ltx_core.model.video_vae.ops import PerChannelStatistics
        from ltx_pipelines.dfr_layout import resolve_canvas
        from ltx_pipelines.dfr_pipeline import _conditioning_fps
        from ltx_pipelines.utils.blocks import DiffusionStage
        from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.types import ModalitySpec

        from dataclasses import replace
        from functools import partial

        from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep
        from ltx_pipelines.dfr_layout import TemporalTilePlan
        from ltx_pipelines.dfr_pipeline import (
            _ANCHOR_KEYFRAME_STRENGTH,
            _TEMPORAL_ANCESTRAL_ETA,
            _audio_latent_for_tile,
            _keyframe_conditionings_from_latents,
            _merge_carry_forward_keyframes,
            _slot_initials_from_video,
        )
        from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop

        up_path, vae, detailing, downscale, temporal = dfr
        width, height, frames, fps = 512, 320, 49, 48.0
        frames, _, positions = resolve_canvas(frames)

        def stage(sd):
            m = X0Model(load(LTXModelConfigurator.from_metadata(meta), sd, lambda k: k[len(prefix):]).to(device=device, dtype=dtype))
            st = DiffusionStage(SimpleNamespace(checkpoint=f"the first {blocks} blocks"), dtype, torch.device(device))
            st._transformer_ctx = lambda **_kw: contextlib.nullcontext(m)
            st._assert_supports_conditionings = lambda _v: None
            return st

        umeta, utensors = read(up_path)
        up = load(LatentUpsamplerConfigurator.from_metadata(umeta), utensors, lambda k: k).to(device, dtype)
        stats = PerChannelStatistics()
        with safe_open(vae, framework="pt") as f:
            for n in ("std-of-means", "mean-of-means"):
                stats.get_buffer(n).copy_(f.get_tensor(f"per_channel_statistics.{n}").float())
        enc = SimpleNamespace(per_channel_statistics=stats.to(device))
        upsample = lambda z: upsample_video(z, enc, up)
        if temporal:
            tmeta, ttensors = read(temporal)
            tup = load(LatentUpsamplerConfigurator.from_metadata(tmeta), ttensors, lambda k: k).to(device, dtype)

        drawn = []
        noiser = GaussianNoiser(torch.Generator(device).manual_seed(5))
        own = noiser._sample_noise
        def sample(state):
            n = own(state) if replay is None else replay[len(drawn)].to(state.latent.device, state.latent.dtype)[None]
            drawn.append(n[0].float().cpu().contiguous())
            return n
        noiser._sample_noise = sample
        def draw(x, generator):
            """The ancestral loop's noise, drawn or replayed, and saved."""
            n = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device) if replay is None else replay[len(drawn)].to(x.device, x.dtype).reshape(x.shape)
            drawn.append(n[0].float().cpu().contiguous())
            return n
        vc, ac = ctx["video"].to(device, dtype)[None], ctx["audio"].to(device, dtype)[None]
        cfps = _conditioning_fps(fps)

        with torch.no_grad():
            t = time.time()
            s1 = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
            v1, a1 = stage(tensors)(
                denoiser=SimpleDenoiser(vc, ac), sigmas=s1, noiser=noiser, width=width // 2, height=height // 2, frames=frames, fps=cfps, audio_fps=fps,
                video=ModalitySpec(context=vc, conditionings=[VideoGeneratedKeyframeSlots(pixel_frame_indices=positions)]),
                audio=ModalitySpec(context=ac),
            )
            half = v1.latent[:1].detach().clone()
            keys = upsample(v1.generated_keyframes)
            upv = upsample(half)
            s2 = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
            sd2 = tensors if detailing is None else fused(detailing, 0.5)
            v2, a2 = stage(sd2)(
                denoiser=SimpleDenoiser(vc, ac), sigmas=s2, noiser=noiser, width=width, height=height, frames=frames, fps=cfps, audio_fps=fps,
                video=ModalitySpec(
                    context=vc, noise_scale=s2[0].item(), initial_latent=upv,
                    conditionings=[
                        VideoGeneratedKeyframeSlots(pixel_frame_indices=positions, initial_keyframes=keys),
                        VideoConditionByReferenceLatent(latent=half, downscale_factor=downscale, strength=1.0),
                    ],
                ),
                audio=ModalitySpec(context=ac, noise_scale=s2[0].item(), initial_latent=a1.latent),
            )
            print(f"dit: DFR stages 1 and 2, {blocks} blocks{', detailing' if detailing else ''}, in {dtype} on {device}, {time.time() - t:.1f} s")
        c = lambda x: x[0].float().cpu().contiguous()
        rounds = {}
        if temporal:
            # `DFRPipeline.__call__`'s temporal rounds, as it has them, less
            # the pictures a request may give.
            video_state, num_frames, current_fps = v2, frames, fps
            carry_positions, carry_keyframes = list(positions), v2.generated_keyframes
            temporal_sigmas = DISTILLED_SIGMAS[4:].to(dtype=torch.float32, device=device)
            stage_1_audio_latent, stage_1_duration = a1.latent, frames / fps
            seed = 10
            # DFR's `self.stage`: the distilled DiT, without the detailing LoRA.
            base = stage(tensors)
            with torch.no_grad():
                for round_idx in (1, 2):
                    t = time.time()
                    video_latent = upsample_video(video_state.latent[:1], enc, tup)
                    rounds[f"round_{round_idx}_upsampled"] = c(video_latent)
                    num_frames = 2 * (num_frames - 1) + 1
                    current_fps = 2 * current_fps
                    seam_positions = [2 * position for position in carry_positions]
                    anchor_keyframes = carry_keyframes
                    seam_to_index = {seam: index for index, seam in enumerate(seam_positions)}
                    cond_fps = _conditioning_fps(current_fps)
                    windows = TemporalTilePlan(seam_positions, num_frames, 2**round_idx, 8)
                    tile_latents, slot_positions, slot_latent_slices = [], [], []
                    for tile_index, (interval, pixel_start, pixel_end, anchor_global, slot_global) in enumerate(windows):
                        local_frames = (interval.end - interval.start - 1) * 8 + 1
                        tile_video = video_latent[:, :, interval.start : interval.end]
                        round_conditionings = []
                        if anchor_global:
                            anchor_latents = torch.cat([anchor_keyframes[:, :, seam_to_index[p] : seam_to_index[p] + 1] for p in anchor_global], dim=2)
                            round_conditionings.extend(
                                _keyframe_conditionings_from_latents(
                                    anchor_latents.to(torch.bfloat16),
                                    [int(position) - pixel_start for position in anchor_global],
                                    strength=_ANCHOR_KEYFRAME_STRENGTH,
                                )
                            )
                        if slot_global:
                            slot_local = [int(position) - pixel_start for position in slot_global]
                            round_conditionings.append(
                                VideoGeneratedKeyframeSlots(pixel_frame_indices=slot_local, initial_keyframes=_slot_initials_from_video(tile_video, slot_local, 8))
                            )
                        tile_state, _ = base(
                            denoiser=SimpleDenoiser(vc, ac), sigmas=temporal_sigmas, noiser=noiser, width=width, height=height, frames=local_frames, fps=cond_fps,
                            video=ModalitySpec(context=vc, conditionings=round_conditionings, noise_scale=temporal_sigmas[0].item(), initial_latent=tile_video),
                            audio=ModalitySpec(
                                context=ac, frozen=True, noise_scale=0.0,
                                initial_latent=_audio_latent_for_tile(
                                    stage_1_audio_latent, pixel_start=pixel_start, local_frames=local_frames, playback_fps=current_fps,
                                    source_duration=stage_1_duration, cond_fps=cond_fps,
                                ),
                            ),
                            stepper=EulerAncestralDiffusionStep(eta=_TEMPORAL_ANCESTRAL_ETA),
                            loop=partial(euler_ancestral_denoising_loop, noise_seed=seed + 1000 * round_idx + tile_index, new_noise_fn=draw, model_dtype=dtype),
                        )
                        tile_latents.append(tile_state.latent[:1, :, interval.left_ramp :])
                        if slot_global:
                            slot_positions.extend(slot_global)
                            slot_latent_slices.append(tile_state.generated_keyframes)
                    stitched = torch.cat(tile_latents, dim=2)
                    video_state = replace(video_state, latent=stitched, generated_keyframes=None)
                    slot_latents = torch.cat(slot_latent_slices, dim=2) if slot_latent_slices else None
                    if slot_positions and slot_latents is not None:
                        first_index = {}
                        for index, position in enumerate(slot_positions):
                            first_index.setdefault(position, index)
                        slot_positions = sorted(first_index)
                        slot_latents = torch.cat([slot_latents[:, :, first_index[p] : first_index[p] + 1] for p in slot_positions], dim=2)
                    carry_positions, carry_keyframes = _merge_carry_forward_keyframes(seam_positions, anchor_keyframes, slot_positions, slot_latents)
                    rounds[f"round_{round_idx}_video"] = c(stitched)
                    rounds[f"round_{round_idx}_keyframes"] = c(carry_keyframes)
                    rounds[f"round_{round_idx}_positions"] = torch.tensor(carry_positions, dtype=torch.float32)
                    print(f"dit: DFR round {round_idx}, {len(windows)} tiles, {num_frames} frames at {current_fps} fps, in {dtype} on {device}, {time.time() - t:.1f} s")
        return {
            "shape": torch.tensor([width, height, frames, fps]),
            "positions": torch.tensor(positions, dtype=torch.float32),
            "downscale": torch.tensor([float(downscale)]),
            **{f"noise_{i}": n for i, n in enumerate(drawn)},
            "stage_1_video": c(v1.latent), "stage_1_keyframes": c(v1.generated_keyframes), "stage_1_audio": c(a1.latent),
            "upsampled_video": c(upv), "upsampled_keyframes": c(keys),
            "stage_2_video": c(v2.latent), "stage_2_keyframes": c(v2.generated_keyframes), "stage_2_audio": c(a2.latent),
            **rounds,
        }

    def fused(lora, strength):
        """The DiT's kept tensors with `lora` fused in at `strength`, as the
        reference fuses it by its own `apply_loras`: `(B · strength) @ A`,
        plus the weight, rounded to the weight's bf16. On MPS, as it does
        here, where it aggregates in f32; on the CPU it aggregates in bf16,
        on one core, for most of an hour. The LoRA's names are the DiT's
        without `model.`."""
        from ltx_core.loader.fuse_loras import apply_loras
        from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict

        with safe_open(lora, framework="pt") as f:
            lsd = {"model." + k: f.get_tensor(k) for k in f.keys() if keep("model." + k.split(".lora_")[0] + ".weight")}
        n = sum(1 for k in lsd if k.endswith(".lora_A.weight"))
        fuse = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        on = lambda d: {k: v.to(fuse) for k, v in d.items()}
        sd = apply_loras(StateDict(on(tensors), fuse, 0, set()), [LoraStateDictWithStrength(StateDict(on(lsd), fuse, 0, set()), strength)])
        print(f"dit: fused {n} LoRA pairs at {strength} into the first {blocks} blocks and the rest")
        return {k: v.cpu() for k, v in sd.sd.items()}

    runs = ((False, None, "dit"), (True, None, "dit_held"), (False, "blind", "dit_blind"), (False, "deaf", "dit_deaf"))
    if dfr:
        runs = ()
        f32 = run_dfr("cpu", torch.float32)
        save_file(f32, f"{out}/dfr_f32.safetensors")
        if torch.backends.mps.is_available():
            replay = [f32[f"noise_{i}"] for i in range(sum(1 for k in f32 if k.startswith("noise_")))]
            save_file(run_dfr("mps", torch.bfloat16, replay), f"{out}/dfr_bf16.safetensors")
    if conditioned:
        runs = ()
        state, f32 = run_conditioned("cpu", torch.float32)
        save_file(state, f"{out}/dit_cond_state.safetensors")
        save_file(f32, f"{out}/dit_cond_f32.safetensors")
        if torch.backends.mps.is_available():
            save_file(run_conditioned("mps", torch.bfloat16)[1], f"{out}/dit_cond_bf16.safetensors")
    if lora:
        # Every run below with the LoRA fused in at 1.
        tensors = fused(lora, 1.0)
        runs = ((False, None, "dit_lora"),)
    if guided:
        runs = ()
        save_file(run_guided("cpu", torch.float32), f"{out}/dit_guided_f32.safetensors")
        if torch.backends.mps.is_available():
            save_file(run_guided("mps", torch.bfloat16), f"{out}/dit_guided_bf16.safetensors")
    for held, kind, name in runs:
        save_file(run("cpu", torch.float32, held, kind), f"{out}/{name}_f32.safetensors")
        if torch.backends.mps.is_available():
            save_file(run("mps", torch.bfloat16, held, kind), f"{out}/{name}_bf16.safetensors")
    print(f"dit: video {tuple(vlat.shape)}, audio {tuple(alat.shape)}, σ {sigma}")


def upsampler(path, vae, latent, out):
    """A latent upsampler on a real latent, through the reference's own
    `upsample_video`: un-normalise with the VAE's statistics, upsample,
    normalise again. The spatial one (on a stage-1 latent) writes
    `upsample_*`; the temporal one, which DFR's temporal rounds run on stage
    2's latent, writes `temporal_*`."""
    from types import SimpleNamespace

    from ltx_core.model.upsampler.model import upsample_video
    from ltx_core.model.upsampler.model_configurator import LatentUpsamplerConfigurator
    from ltx_core.model.video_vae.ops import PerChannelStatistics

    meta, tensors = read(path)
    up = load(LatentUpsamplerConfigurator.from_metadata(meta), tensors, lambda k: k)
    name = "temporal" if up.temporal_upsample else "upsample"
    stats = PerChannelStatistics()
    with safe_open(vae, framework="pt") as f:
        for n in ("std-of-means", "mean-of-means"):
            stats.get_buffer(n).copy_(f.get_tensor(f"per_channel_statistics.{n}").float())
    z = load_file(latent)
    z = (z["video"] if "video" in z else z["latent"])[None].float()

    def run(device, dtype):
        enc = SimpleNamespace(per_channel_statistics=stats.to(device))
        with torch.no_grad():
            t = time.time()
            y = upsample_video(z.to(device, dtype), enc, up.to(device, dtype))[0].float().cpu()
        print(f"upsampler: {tuple(z.shape[1:])} to {tuple(y.shape)} in {dtype} on {device}, {time.time() - t:.1f} s")
        return y

    save_file({"latent": z[0].contiguous(), "upsampled": run("cpu", torch.float32).contiguous()}, f"{out}/{name}_f32.safetensors")
    if torch.backends.mps.is_available():
        save_file({"upsampled": run("mps", torch.bfloat16).contiguous()}, f"{out}/{name}_bf16.safetensors")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", help="vae/ltx-2.5-video-vae-conv-bf16.safetensors")
    p.add_argument("--audio", help="vae/ltx-2.5-audio-vae-bf16.safetensors")
    p.add_argument("--text", help="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    p.add_argument("--dit", help="diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors (for the connectors)")
    p.add_argument("--blocks", type=int, default=2, help="how many DiT blocks to run with --dit --contexts")
    p.add_argument("--lora", help="with --dit --contexts: a LoRA fused into the DiT, writing dit_lora_*")
    p.add_argument("--guided", action="store_true", help="with --dit --contexts: one guided prediction, writing dit_guided_*")
    p.add_argument("--conditioned", action="store_true", help="with --dit --contexts: DFR's conditioning tokens (anchors, slots, a reference latent) and frozen sound, writing dit_cond_*")
    p.add_argument("--dfr", action="store_true", help="with --dit --contexts --upsampler (the spatial one) --vae: DFR's stages 1 and 2, writing dfr_*")
    p.add_argument("--temporal", help="with --dfr: the temporal upsampler, for two temporal rounds after stage 2")
    p.add_argument("--detailing", help="with --dfr: the detailing IC-LoRA, fused in at 0.5 for stage 2")
    p.add_argument("--contexts", help="text_contexts_f32.safetensors from --text, or `random`: the DiT's first blocks against them")
    p.add_argument("--duration", help="model_patches/ltx-2.5-duration-head-bf16.safetensors, on OUT/duration_contexts.safetensors")
    p.add_argument("--diffvae", help="vae/ltx-2.5-video-vae-bf16.safetensors: the diffusion decoder, stage by stage")
    p.add_argument("--keyframes", action="store_true", help="with --diffvae: the keyframe-aware decode, writing diffvae_kf_*")
    p.add_argument("--picture", help="with --video: a picture for image-to-video, `synthetic` for one drawn here")
    p.add_argument("--clip", help="with --video: a clip drawn here, WxHxT (T = 8k + 1), through the encoder, writing clip_WxHxT")
    p.add_argument("--upsampler", help="latent_upscale_models/ltx-2.5-latent-{spatial,temporal}-upscaler-x2-bf16-1.0.safetensors, with --vae and --latent")
    p.add_argument("--vae", help="vae/ltx-2.5-video-vae-conv-bf16.safetensors, for the upsampler's statistics")
    p.add_argument("--latent", help="a stage-1 latent to upsample: a safetensors file with `video`, [128, F, h, w], as examples/ltx.rs --latents writes")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import os

    os.makedirs(a.out, exist_ok=True)
    started = time.time()
    if a.video and a.clip:
        w, h, t = (int(v) for v in a.clip.split("x"))
        clip(a.video, a.out, t, h, w)
    elif a.video and a.picture:
        picture(a.video, None if a.picture == "synthetic" else a.picture, a.out)
    elif a.video:
        video(a.video, a.out)
    if a.audio:
        audio(a.audio, a.out)
    if a.text:
        text(a.text, a.dit, a.out)
    if a.dit and a.contexts:
        dfr = None
        if a.dfr:
            # The reference latent's downscale: the detailing LoRA's metadata
            # says it, as DFR reads it; 2 without one, as its name says.
            downscale = 2
            if a.detailing:
                with safe_open(a.detailing, framework="pt") as f:
                    downscale = int((f.metadata() or {}).get("reference_downscale_factor", 1))
            dfr = (a.upsampler, a.vae, a.detailing, downscale, a.temporal)
        transformer(a.dit, a.out, a.contexts, a.blocks, a.lora, a.guided, a.conditioned, dfr)
    if a.duration:
        duration(a.duration, a.out)
    if a.diffvae and a.keyframes:
        diffvae_keyframes(a.diffvae, a.out)
    elif a.diffvae:
        diffvae(a.diffvae, a.out)
    if a.upsampler and a.latent:
        upsampler(a.upsampler, a.vae, a.latent, a.out)
    print(f"wrote {a.out} in {time.time() - started:.0f} s")
