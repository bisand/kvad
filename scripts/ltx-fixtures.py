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

(the text fixtures also want `transformers` 5.8 to 5.14 in the venv). The
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", help="vae/ltx-2.5-video-vae-conv-bf16.safetensors")
    p.add_argument("--audio", help="vae/ltx-2.5-audio-vae-bf16.safetensors")
    p.add_argument("--text", help="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    p.add_argument("--dit", help="diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors (for the connectors)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import os

    os.makedirs(a.out, exist_ok=True)
    started = time.time()
    if a.video:
        video(a.video, a.out)
    if a.audio:
        audio(a.audio, a.out)
    if a.text:
        text(a.text, a.dit, a.out)
    print(f"wrote {a.out} in {time.time() - started:.0f} s")
