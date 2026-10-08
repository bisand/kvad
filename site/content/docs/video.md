---
title: Video
description: Clips with sound from LTX-2.5, from a prompt or from a picture.
---

Video runs on the GPU engine, so it needs a Mac, and LTX-2.5 is a large model:
the download is about 52 GB.

```bash
kvad pull Lightricks/LTX-2.5
kvad videos make "a lighthouse on a cliff at dusk, waves breaking below" --out lighthouse.mp4
```

A video takes minutes, and it is the server's job from the moment it is asked
for. `make` follows it and writes the file, and stopping the wait does not
stop the video.

```bash
kvad videos                # ls
kvad videos show 3         # how far along it is
kvad videos watch 3        # follow it again
kvad videos get 3 --out clip.mp4
kvad videos rm 3           # delete it, stopping it if it is still being made
```

## Length and size

```bash
kvad videos make "…" --size 768x512 --seconds 5
kvad videos make "…" --frames 121 --fps 24
```

With no `--seconds` or `--frames`, LTX-2.5 chooses the length from the prompt.
`--silent` leaves the sound out.

## From a picture

```bash
kvad videos make "the boat drifts slowly to the left" --image boat.jpg
```

The picture becomes the first frame, scaled to cover the video's size and cut
from the middle. This needs `ffmpeg` on the server.

## Quality and speed

| Option | What it does |
|---|---|
| *(nothing)* | LTX-2.5's distilled model in two stages. The fast path. |
| `--steps`, `--guidance`, `--negative` | Runs the dev model with guidance instead: 30 steps at guidance 3 unless given, about four times as long. Its files come from `kvad pull Lightricks/LTX-2.5 --dev`. |
| `--pipeline dfr` | The reference's production pipeline: generated keyframes, a detailing pass and a keyframe-aware decode. Slower, and finer. |
| `--fps` above 30 | Runs DFR, making the clip at half or a quarter of the rate and doubling it: 48, 50, 60, 96, 100 or 120 fps. |
| `--epilogue` | Runs DFR at half the width and height, then details the clip at the size asked for, in tiles. For large sizes: see below. |
| `--decoder conv` | The convolutional decoder instead of the diffusion one: lighter, and about twice as fast to decode. |
| `--lora NAME[:SCALE]` | Applies a LoRA to every DiT the pipeline runs. |

DFR's detailing LoRA is gated on Hugging Face, and the first DFR video fetches
it, so you need to have accepted its terms there.

## Large sizes

DFR's second stage holds the whole clip at once, which sets how large a clip
this machine can make. `--epilogue` gets past that: the clip is made at half
its width and height, then upsampled once more and detailed at the size asked
for, with each pass over it cut into overlapping tiles.

```bash
kvad videos make "…" --size 1536x1024 --seconds 3 --fps 48 --epilogue
```

Both sides must be multiples of 128, and it takes several times as long. On
an M5 Pro, that clip took 18 minutes to denoise and 2 to decode, and held
27 GB at its peak.

The clip on [the front page](/) is that one:

```bash
kvad videos make "a fox trots through fresh snow towards the camera, snow falling, photograph" --size 1536x1024 --seconds 3 --fps 48 --epilogue --seed 3
```

It is for sizes DFR cannot make without it. At 768×512 the clip is worse than
plain DFR's, because its first stage is then only 192×128.

## Compression

Kvad writes an MP4 itself that compresses nothing, about 14 MB a second at
768×512. With `ffmpeg` available the server compresses each video once it is
made, to H.264 and AAC, at about a thirtieth of the size.

```toml
[videos]
ffmpeg = "auto"     # or "off", or a path
```

`auto` looks on `PATH` and then in `/opt/homebrew/bin`, `/usr/local/bin` and
`/usr/bin`, because a background service's `PATH` has only the system
directories.

## From the API

`POST /v1/videos` is OpenAI's video endpoint: a job that answers at once and
is watched until it is done.

```bash
curl http://127.0.0.1:5823/v1/videos \
  -H 'content-type: application/json' \
  -d '{"model": "Lightricks/LTX-2.5", "prompt": "waves breaking below a lighthouse", "size": "768x512", "seconds": 5}'

curl http://127.0.0.1:5823/v1/videos/3            # status and progress
curl http://127.0.0.1:5823/v1/videos/3/content -o clip.mp4
```
