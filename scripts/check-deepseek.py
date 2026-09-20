#!/usr/bin/env python3
"""Check this engine's DeepSeek against DeepSeek's own implementation.

`tests/deepseek.rs` already compares the engine's absorbed attention with a
second implementation written out longhand in Rust, and that catches algebra
mistakes. It cannot catch a *misreading*: a wrong understanding of the
reference would sit in both versions equally, and they would agree with each
other all the way to the wrong answer.

This closes that gap by running the actual thing — DeepSeek's own
`modeling_deepseek.py`, under PyTorch, on the same weights.

    python3 -m venv /tmp/ds-venv
    /tmp/ds-venv/bin/pip install torch transformers
    KVAD_DEEPSEEK_FIXTURE=/tmp/ds cargo test -p kvad --test deepseek -- --ignored
    /tmp/ds-venv/bin/python scripts/check-deepseek.py /tmp/ds

It is not part of `cargo test` because it needs Python, PyTorch, and
DeepSeek's published modelling code, which it downloads and executes. That is
a reasonable thing to do deliberately and a poor thing to do on every build —
which is also why it imports the files directly instead of going through
`trust_remote_code=True`, so that what is being run is visible on disk.
"""

import importlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

import torch

# Where each flavour's reference implementation comes from. Both repos call
# their file `modeling_deepseek.py`, which is why they are imported by path
# under separate module names rather than by `import`.
SOURCES = {
    "v2": ("deepseek-ai/DeepSeek-V2-Lite", "DeepseekV2Config", "DeepseekV2ForCausalLM"),
    "v3": ("deepseek-ai/DeepSeek-V3", "DeepseekV3Config", "DeepseekV3ForCausalLM"),
}


def fetch_reference(flavour: str, cache: Path) -> tuple[type, type]:
    """Download and import one flavour's published modelling code.

    As a package named after the flavour, because the file says
    `from .configuration_deepseek import ...` and because both repos call
    theirs `modeling_deepseek.py` — importing them flat would have the second
    one shadow the first.
    """
    repo, config_name, model_name = SOURCES[flavour]
    into = cache / flavour
    into.mkdir(parents=True, exist_ok=True)
    (into / "__init__.py").touch()
    for name in ("configuration_deepseek.py", "modeling_deepseek.py"):
        path = into / name
        if not path.exists():
            url = f"https://huggingface.co/{repo}/resolve/main/{name}"
            print(f"  fetching {url}")
            urllib.request.urlretrieve(url, path)

    if str(cache) not in sys.path:
        sys.path.insert(0, str(cache))
    config = importlib.import_module(f"{flavour}.configuration_deepseek")
    modelling = importlib.import_module(f"{flavour}.modeling_deepseek")
    return getattr(config, config_name), getattr(modelling, model_name)


def read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """The four lines the format actually is, so this script needs one fewer
    dependency than the thing it is checking."""
    blob = path.read_bytes()
    n = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8 : 8 + n])
    body = blob[8 + n :]
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        assert meta["dtype"] == "F32", meta["dtype"]
        start, end = meta["data_offsets"]
        values = torch.frombuffer(bytearray(body[start:end]), dtype=torch.float32)
        out[name] = values.reshape(meta["shape"])
    return out


def check(model_dir: Path, cache: Path) -> bool:
    flavour = model_dir.name
    saved = json.loads((model_dir / "logits.json").read_text())
    ours = torch.tensor(saved["logits"], dtype=torch.float32)

    print(f"{flavour}:")
    config_class, model_class = fetch_reference(flavour, cache)
    config = config_class(**json.loads((model_dir / "config.json").read_text()))
    model = model_class(config).to(torch.float32)
    missing, unexpected = model.load_state_dict(
        read_safetensors(model_dir / "model.safetensors"), strict=False
    )
    # A silently unloaded weight would make this whole exercise meaningless.
    assert not missing, f"the reference wanted weights the fixture has not: {missing}"
    assert not unexpected, f"the fixture has weights the reference does not: {unexpected}"
    model.eval()

    with torch.no_grad():
        theirs = model(torch.tensor([saved["tokens"]])).logits[0].float()

    gap = (ours - theirs).abs()
    # Not bit-identical: the two sum in different orders. A wrong rotary
    # convention or a missing normalisation moves logits by whole units.
    same = bool((ours.argmax(-1) == theirs.argmax(-1)).all())
    ok = gap.max().item() < 2e-3 and same
    print(
        f"  worst {gap.max().item():.2e}  mean {gap.mean().item():.2e}  "
        f"same argmax everywhere: {same}  -> {'ok' if ok else 'MISMATCH'}"
    )
    return ok


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ds")
    cache = root / "reference"
    dirs = sorted(d for d in root.iterdir() if (d / "logits.json").exists())
    if not dirs:
        print(f"no fixtures under {root}; run the ignored test first")
        return 2
    return 0 if all([check(d, cache) for d in dirs]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
