---
title: Tool calls and agents
description: Function calling through the model's own chat template, and how to drive a coding agent such as OpenCode with a local model.
---

An agent is not asking for prose. It sends a list of functions it is willing
to run and expects the model to answer with a call, runs it, and sends the
result back. Kvad supports that loop through `/v1/chat/completions`, in
OpenAI's shape.

## How it works

The tools go into the model's **own chat template**, which writes them into
the prompt in the wording that model was fine-tuned on. The call comes back as
text, and the server parses it out of the reply as it streams and sends it as
`tool_calls`, with a `finish_reason` of `tool_calls`.

```bash
curl http://127.0.0.1:5823/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "What is the weather in Oslo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "The current weather in a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

Send the result back as a `tool` message carrying the same id, and the model
answers from it.

## Which models can

A model whose template never mentions tools would render the same prompt
whether or not any were offered, and answer in prose. Kvad refuses that
request with a `400` naming the model, rather than letting a client mistake it
for a model that considered the tools and declined.

`GET /v1/models` reports `kvad.tools` for each model, and `kvad ls` marks them:

```text
Qwen/Qwen2.5-1.5B-Instruct                     llama          2.9 GB *  tools
Qwen/Qwen3-14B                                 llama         27.5 GB  tools
```

The Qwen instruct models can. SmolLM2, the DeepSeeks and GPT-2 cannot.

The parser reads the Hermes-style `<tool_call>` format that Qwen and most open
models adopted. Llama 3.1's bare object and DeepSeek's own markers are not
read yet.

## Choosing a model for an agent

Tool calling works on a model as small as Qwen2.5-0.5B, which is surprising
and not useful. The protocol does not care how big the model is; the work does.
What trying it with real agents turned up:

- **Avoid the Coder tunings.** Qwen2.5-Coder-7B ignores its own template's
  tool format and invents a different wrapper each run. The plain
  `Qwen/Qwen2.5-7B-Instruct` emits proper calls. Coder models are tuned for
  completion, and the symptom is a model that prints the code instead of
  writing the file.
- **Trim the tools.** A local model treats a large set of tools as somewhere
  to wander. A 14B model with OpenCode's `task` and `skill` tools denied wrote
  a real source file and reported accurately that it was incomplete. With them
  allowed it looped until OpenCode stopped it.
- **Check the files, not the summary.** A 7B model will report that it wrote
  and verified a server over an untouched template. A 14B model is much more
  honest about what it did.

Expect single-file work to succeed and open-ended multi-file work to fail. A
local model is a good agent for small, well-bounded jobs, and this page will
not tell you otherwise.

## OpenCode

[OpenCode](https://opencode.ai) talks to Kvad as a custom OpenAI-compatible
provider. In `~/.config/opencode/opencode.jsonc`:

```json
{
  "provider": {
    "kvad": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Kvad",
      "options": {
        "baseURL": "http://127.0.0.1:5823/v1",
        "apiKey": "unused"
      },
      "models": {
        "Qwen/Qwen3-14B": { "name": "Qwen3 14B" }
      }
    }
  },
  "agent": {
    "local": {
      "model": "kvad/Qwen/Qwen3-14B",
      "permission": { "task": "deny", "skill": "deny" }
    }
  }
}
```

Three things to know:

- OpenCode does not read `/v1/models`. It shows only the models written under
  `models`, so list the ones you have pulled, and only ones with `tools`.
- A model that is not on disk is a `404`. Pull it before you select it.
- If `opencode run` goes quiet, it is usually waiting on stdin for a
  permission prompt, and the server has seen no request. `kvad metrics` shows
  whether the request count is moving.

## Limits

`tool_choice` is `auto` or `none`. Forcing a call, with `required` or a named
function, would need the sampler constrained to the tokens that open a call,
and that is not written.
