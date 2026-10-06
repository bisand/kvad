---
title: OpenAI-compatible API
description: Chat completions, images and videos in OpenAI's shapes, so existing clients and SDKs work unchanged.
---

The base URL is the server's address with `/v1`:

```text
http://127.0.0.1:5823/v1
```

With the default configuration there is no authentication on loopback, and
clients that insist on an API key can be given any string. With an
[authentication](/docs/authentication/) mode configured, send an API key as a
bearer token.

## With an SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5823/v1", api_key="unused")

reply = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Say hello in Norwegian."}],
)
print(reply.choices[0].message.content)
```

```javascript
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://127.0.0.1:5823/v1", apiKey: "unused" });

const stream = await client.chat.completions.create({
  model: "Qwen/Qwen2.5-1.5B-Instruct",
  messages: [{ role: "user", content: "Say hello in Norwegian." }],
  stream: true,
});
for await (const chunk of stream) process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
```

## Chat completions

`POST /v1/chat/completions`

```bash
curl http://127.0.0.1:5823/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "system", "content": "You are terse."},
      {"role": "user", "content": "Why is the sky blue?"}
    ],
    "stream": true
  }'
```

The request takes `messages`, `stream`, `temperature`, `top_p`, `max_tokens`
(or `max_completion_tokens`), `seed`, `tools` and `tool_choice`.

`stream: true` returns `text/event-stream` in OpenAI's chunk format, ending
with `[DONE]`. Otherwise the answer is one JSON object. The numbers for each
generation, prefill and decode rates and cached tokens, are in an extension
field.

### Naming a model

`model` is a repository, or `repo@backend` for a particular backend:

```json
{ "model": "Qwen/Qwen3-14B@gpu-q4" }
```

| The model is | What happens |
|---|---|
| in memory | It answers. |
| on disk, and fits beside what is loaded | It is loaded, then answers. |
| on disk, and does not fit | `409`, naming what holds the memory. Nothing is unloaded. |
| not on disk | `404`. Pull it first. |
| left out | The one model in memory. Refused when there are several. |

### Reasoning

A reasoning model's working arrives as `reasoning_content`, the field
DeepSeek's API established, and `content` is the answer alone. Both stream.

### Tools

`tools` are passed to the model's own chat template and its calls come back as
`tool_calls`. See [Tool calls and agents](/docs/agents/).

## Models

`GET /v1/models` lists every model on the machine in OpenAI's shape, with a
`kvad` object on each:

```json
{
  "id": "Qwen/Qwen2.5-1.5B-Instruct",
  "object": "model",
  "owned_by": "huggingface",
  "kvad": { "tools": true, "resident": false, "kind": "chat" }
}
```

`kind` is `chat`, `image` or `video`. `tools` says whether the model's
template has a place for tool definitions. `resident` says whether it is in
memory now.

## Images

`POST /v1/images/generations`

OpenAI's fields are `prompt`, `model`, `n` (at most 4), `size` as
`WIDTHxHEIGHT`, and `response_format` of `b64_json` (the default) or `url`.
Beside them are the settings OpenAI has no field for:

| Field | |
|---|---|
| `negative_prompt` | |
| `steps` | or `num_inference_steps` |
| `guidance_scale` | |
| `seed` | |
| `loras` | `[{ "name": "…", "scale": 1 }]`, at most 4 |
| `stream`, `partial_images` | An event per denoising step, with a preview. |

Each image in the answer carries a `kvad` object with its id, its seed, its
settings and the time each stage took.

When streaming, the events are `image_generation.step`,
`image_generation.partial_image` and `image_generation.completed`.

`POST /v1/images/edits`

OpenAI's edits endpoint, as a multipart form or as JSON with each file a
`data:` URL. It takes everything above, and:

| Field | |
|---|---|
| `image` | The picture to start from. One a request. |
| `mask` | Optional. Where the picture may change: transparent there, or white on black. The picture's size. |
| `strength` | Above 0, at most 1. 0.75 without a mask, 1 with one. |

It runs image-to-image on SDXL and SD 1.5: the prompt describes the result,
not the change. The `kvad` object adds `strength`, `input_url` and `mask_url`.
See [Images](/docs/images/#from-a-picture).

## Videos

`POST /v1/videos` starts a job and answers at once.

| Route | |
|---|---|
| `POST /v1/videos` | Start one. JSON, or `multipart/form-data` as OpenAI's SDKs send it. |
| `GET /v1/videos` | The list, newest first. |
| `GET /v1/videos/{id}` | One video, and how far along it is. |
| `GET /v1/videos/{id}/events` | Follow it until it ends. |
| `GET /v1/videos/{id}/content` | The file. |
| `DELETE /v1/videos/{id}` | Delete it, stopping it if it is still being made. |

The request takes `prompt`, `model`, `size`, and `seconds` or `frames`, plus
`fps`, `seed`, `audio`, `input_reference`, `steps`, `guidance_scale`,
`negative_prompt`, `decoder`, `pipeline` and `loras`. [Video](/docs/video/)
explains what each one does.

## Everything else

The server's own API is under `/api`: models, conversations, jobs, datasets,
evals, benchmarks, metrics, accounts. It is documented by the server itself.

- `http://127.0.0.1:5823/api` is the reference, as a page.
- `/api/openapi.json` is the same thing as OpenAPI. It is generated from a
  table that a test compares against the router, so a route that exists is a
  route that is documented.
- `kvad api` lists every route, and `kvad api GET /api/health` calls one.

## What is not there

- `tool_choice` accepts `auto` and `none`. `required` and a named function are
  refused, because forcing a call would need a constrained sampler.
- There is no embeddings endpoint and no legacy `/v1/completions`. Raw
  completion without a chat template is `POST /api/playground/complete`.
- Requests are answered one at a time.
