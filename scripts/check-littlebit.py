#!/usr/bin/env python3
"""Check this engine's LittleBit kernels against the reference's arithmetic.

`tests/littlebit.rs` checks the reader against a packing written out a second
time in Rust, which catches slips and cannot catch a misunderstanding of the
format shared by both. This reads the same checkpoint again with a
transcription of the reference — SamsungLabs/LittleBit's `binary_unpacker`
and `LittleBitLinear.forward` — in numpy and float64, straight from the
safetensors bytes, and compares.

    KVAD_LITTLEBIT_CHECKPOINT=/path/to/model KVAD_LITTLEBIT_FIXTURE=/tmp/lb \\
        cargo test --release -p kvad --test littlebit -- --ignored writes
    python3 scripts/check-littlebit.py /path/to/model /tmp/lb

numpy only: no PyTorch, and no copy of the reference's code. What it
transcribes is the arithmetic, which is short enough to read against the
original line by line.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np


def read_header(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    return f, 8 + n, json.loads(f.read(n))


def tensor(f, base, header, name):
    meta = header[name]
    a, b = meta["data_offsets"]
    f.seek(base + a)
    raw = f.read(b - a)
    kind = {"BF16": None, "F16": np.float16, "F32": np.float32, "I32": np.int32, "I64": np.int64}[meta["dtype"]]
    if kind is None:  # bf16 is the top half of an f32
        return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])
    return np.frombuffer(raw, kind).reshape(meta["shape"])


def unpack(packed, shape):
    """`binary_unpacker`: LSB first, bit 1 is -1, trimmed to the stored width."""
    rows, cols = (int(v) for v in shape)
    bits = (packed.view(np.uint32)[:, :, None] >> np.arange(32, dtype=np.uint32)) & 1
    return 1.0 - 2.0 * bits.reshape(rows, -1)[:, :cols].astype(np.float64)


def forward(get, stem, x):
    """`LittleBitLinear.forward`, both paths when there are two."""
    y = 0.0
    for s in ["", "_R"]:
        try:
            U = unpack(get(f"{stem}.U{s}_packed"), get(f"{stem}.U{s}_shape"))
        except KeyError:
            break
        V = unpack(get(f"{stem}.V{s}_packed"), get(f"{stem}.V{s}_shape"))
        u1, u2, v1, v2 = (get(f"{stem}.{v}{s}").astype(np.float64) for v in ["u1", "u2", "v1", "v2"])
        y = y + ((((x * v2) @ V.T) * (v1 * u2)) @ U.T) * u1
    return y


def activation(n, seed):
    """The activation `tests/littlebit.rs` builds, to the bit."""
    i = np.arange(n, dtype=np.uint64)
    return (((i * 2654435761 + seed * 40503) % 1000).astype(np.float32) / np.float32(500) - np.float32(1)).astype(np.float64)


def main():
    model, fixture = Path(sys.argv[1]), Path(sys.argv[2])
    f, base, header = read_header(model / "model.safetensors")
    get = lambda name: tensor(f, base, header, name if name in header else f"model.{name}")
    worst = 0.0
    for case in json.loads((fixture / "littlebit.json").read_text()):
        stem, rows, cols, m = case["stem"], case["rows"], case["cols"], case["m"]
        x = activation(cols, 0)
        xs = np.stack([activation(cols, i + 1) for i in range(m)])
        want, wants = forward(get, stem, x[None, :])[0], forward(get, stem, xs)
        got = np.array(case["matvec"])
        gots = np.array(case["matmul"]).reshape(m, rows)
        rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
        errs = rel(got, want), rel(gots, wants)
        worst = max(worst, *errs)
        print(f"{stem}: matvec {errs[0]:.1e}, matmul {errs[1]:.1e}, {case['bits_per_weight']:.4f} bits a weight")
    # f32 against float64, over sums of thousands of terms.
    if worst > 1e-5:
        sys.exit(f"worst relative difference {worst:.1e}: the engine does not compute what the reference does")
    print(f"agree: worst relative difference {worst:.1e}")


if __name__ == "__main__":
    main()
