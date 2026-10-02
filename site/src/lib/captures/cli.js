// Real output, copied from a terminal on an M5 Pro on 2026-10-02.

export const RUN = `$ kvad search qwen2.5
MODEL                             DOWNLOADS      SIZE  ARCH   STATUS
Qwen/Qwen2.5-7B-Instruct            8891777   14.2 GB  llama  chat · fits at f32
Qwen/Qwen2.5-0.5B-Instruct          8661907  942.3 MB  llama  chat · fits at f32
Qwen/Qwen2.5-1.5B-Instruct          7509634    2.9 GB  llama  downloaded
Qwen/Qwen2.5-32B-Instruct           1920796   61.0 GB  llama  chat · fits at q8

$ kvad load Qwen/Qwen3-14B
loaded Qwen/Qwen3-14B on metal q8
  llama · 40 layers · 40 heads (8 KV heads, 5x grouped) · 5120 embd · 40960 ctx
  14768.3M parameters · weights 14.6 GB · instruction-tuned

$ kvad ps
MODEL                  CHARGED  CONTEXT  CACHED
Qwen/Qwen3-14B@gpu-q8  19.6 GB  40960    0       tools

memory: 16.4 GB of 36.0 GB left · each charged for 32768 tokens of KV cache
`;

export const BENCH = `$ kvad bench run Qwen/Qwen2.5-1.5B-Instruct@cpu-q8 \\
      Qwen/Qwen2.5-1.5B-Instruct@gpu-q8 Qwen/Qwen2.5-1.5B-Instruct@gpu-q4
job 2 — 3 variants, 5 rounds (bench)

MODEL                                RUNS  DECODE TOK/S, MEDIAN (RANGE)  FIRST TOKEN MS
Qwen/Qwen2.5-1.5B-Instruct · cpu-q8  5     48.5 (47.2–49.3)              72.3
Qwen/Qwen2.5-1.5B-Instruct · gpu-q8  5     132.5 (131.1–133.9)           34.8
Qwen/Qwen2.5-1.5B-Instruct · gpu-q4  5     215.4 (213.2–218.3)           34.1
`;

export const IMAGE = `$ kvad images make "a red fox in fresh snow at the edge of a birch forest, \\
      early morning light, wildlife photograph" \\
      --model stabilityai/stable-diffusion-xl-base-1.0 --seed 5 --out fox.png
fox.png: 1024×1024, 30 steps, guidance 5.0, seed 5 — image 2 on the server
  denoise 82.5 s, decode 9.2 s
`;

export const API = `$ curl http://127.0.0.1:5823/v1/chat/completions \\
    -H 'content-type: application/json' \\
    -d '{
      "model": "Qwen/Qwen3-14B",
      "messages": [{"role": "user", "content": "What is a KV cache?"}],
      "stream": true
    }'
`;
