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
    /tmp/ltx-venv/bin/pip install torch torchaudio einops safetensors av
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

    cargo run --release -p kvad-gpu --example ltx_duration -- --contexts /tmp/ltx-fx "a door slams shut" "…"
    HEAD=$(cargo run -q --release -p kvad-gpu --example ltx_duration -- --where)
    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --duration "$HEAD" --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_duration -- --fixtures /tmp/ltx-fx

    UP=$(cargo run -q --release -p kvad-gpu --example ltx_upsample -- --where | head -1)
    cargo run --release -p kvad-gpu --example ltx -- --prompt "…" --stages 1 \\
        --width 512 --height 320 --frames 25 --latents /tmp/stage1.safetensors
    PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src /tmp/ltx-venv/bin/python \\
        scripts/ltx-fixtures.py --upsampler "$UP" --vae "$VIDEO" --latent /tmp/stage1.safetensors --out /tmp/ltx-fx
    cargo run --release -p kvad-gpu --example ltx_upsample -- --fixtures /tmp/ltx-fx [--cpu | --f32]

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
- The latent upsampler, on a 512x320x25 latent from a generation: 98 dB in
  f32 on the CPU and on Metal. In bf16 it is 27 dB from exact, and so is
  the reference's own bf16 on MPS; kvad runs it in f32.

Every reference model here runs in f32 on the CPU, so the files are what the
architecture computes, not what one GPU's rounding makes of it. It is not part
of `cargo test` because it needs Python, PyTorch and a clone of `LTX-2`.
"""

import argparse
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


def transformer(path, out, contexts, blocks, lora=None, guided=False):
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
    if lora:
        # The LoRA fused as the reference fuses it, by its own `apply_loras`:
        # `(B · strength) @ A`, plus the weight, rounded to the weight's
        # bf16. On MPS, as it does here, where it aggregates in f32; on the
        # CPU it aggregates in bf16, on one core, for most of an hour. The
        # LoRA's names are the DiT's without `model.`.
        from ltx_core.loader.fuse_loras import apply_loras
        from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict

        with safe_open(lora, framework="pt") as f:
            lsd = {"model." + k: f.get_tensor(k) for k in f.keys() if keep("model." + k.split(".lora_")[0] + ".weight")}
        n = sum(1 for k in lsd if k.endswith(".lora_A.weight"))
        fuse = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        on = lambda d: {k: v.to(fuse) for k, v in d.items()}
        sd = apply_loras(StateDict(on(tensors), fuse, 0, set()), [LoraStateDictWithStrength(StateDict(on(lsd), fuse, 0, set()), 1.0)])
        tensors = {k: v.cpu() for k, v in sd.sd.items()}
        print(f"dit: fused {n} LoRA pairs into the first {blocks} blocks and the rest")

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

    runs = ((False, None, "dit"), (True, None, "dit_held"), (False, "blind", "dit_blind"), (False, "deaf", "dit_deaf"))
    if lora:
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
    """The spatial latent upsampler on a real stage-1 latent, through the
    reference's own `upsample_video`: un-normalise with the VAE's statistics,
    upsample, normalise again."""
    from types import SimpleNamespace

    from ltx_core.model.upsampler.model import upsample_video
    from ltx_core.model.upsampler.model_configurator import LatentUpsamplerConfigurator
    from ltx_core.model.video_vae.ops import PerChannelStatistics

    meta, tensors = read(path)
    up = load(LatentUpsamplerConfigurator.from_metadata(meta), tensors, lambda k: k)
    stats = PerChannelStatistics()
    with safe_open(vae, framework="pt") as f:
        for n in ("std-of-means", "mean-of-means"):
            stats.get_buffer(n).copy_(f.get_tensor(f"per_channel_statistics.{n}").float())
    z = load_file(latent)["video"][None].float()

    def run(device, dtype):
        enc = SimpleNamespace(per_channel_statistics=stats.to(device))
        with torch.no_grad():
            t = time.time()
            y = upsample_video(z.to(device, dtype), enc, up.to(device, dtype))[0].float().cpu()
        print(f"upsampler: {tuple(z.shape[1:])} to {tuple(y.shape)} in {dtype} on {device}, {time.time() - t:.1f} s")
        return y

    save_file({"latent": z[0].contiguous(), "upsampled": run("cpu", torch.float32).contiguous()}, f"{out}/upsample_f32.safetensors")
    if torch.backends.mps.is_available():
        save_file({"upsampled": run("mps", torch.bfloat16).contiguous()}, f"{out}/upsample_bf16.safetensors")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", help="vae/ltx-2.5-video-vae-conv-bf16.safetensors")
    p.add_argument("--audio", help="vae/ltx-2.5-audio-vae-bf16.safetensors")
    p.add_argument("--text", help="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    p.add_argument("--dit", help="diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors (for the connectors)")
    p.add_argument("--blocks", type=int, default=2, help="how many DiT blocks to run with --dit --contexts")
    p.add_argument("--lora", help="with --dit --contexts: a LoRA fused into the DiT, writing dit_lora_*")
    p.add_argument("--guided", action="store_true", help="with --dit --contexts: one guided prediction, writing dit_guided_*")
    p.add_argument("--contexts", help="text_contexts_f32.safetensors from --text, or `random`: the DiT's first blocks against them")
    p.add_argument("--duration", help="model_patches/ltx-2.5-duration-head-bf16.safetensors, on OUT/duration_contexts.safetensors")
    p.add_argument("--picture", help="with --video: a picture for image-to-video, `synthetic` for one drawn here")
    p.add_argument("--upsampler", help="latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors, with --vae and --latent")
    p.add_argument("--vae", help="vae/ltx-2.5-video-vae-conv-bf16.safetensors, for the upsampler's statistics")
    p.add_argument("--latent", help="a stage-1 latent to upsample: a safetensors file with `video`, [128, F, h, w], as examples/ltx.rs --latents writes")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import os

    os.makedirs(a.out, exist_ok=True)
    started = time.time()
    if a.video and a.picture:
        picture(a.video, None if a.picture == "synthetic" else a.picture, a.out)
    elif a.video:
        video(a.video, a.out)
    if a.audio:
        audio(a.audio, a.out)
    if a.text:
        text(a.text, a.dit, a.out)
    if a.dit and a.contexts:
        transformer(a.dit, a.out, a.contexts, a.blocks, a.lora, a.guided)
    if a.duration:
        duration(a.duration, a.out)
    if a.upsampler:
        upsampler(a.upsampler, a.vae, a.latent, a.out)
    print(f"wrote {a.out} in {time.time() - started:.0f} s")
