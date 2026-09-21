# nervus

Neural network training written from scratch, in Rust, with no dependencies at
all. It builds and trains GPT-style transformers on the CPU, and what it writes
is a safetensors checkpoint beside a `tokenizer.json` — the GPT-2 layout, which
a real runtime can load.

`[dependencies]` in this crate is empty, and stays empty. No BLAS, no CUDA
toolchain, no build script: if a machine has a Rust compiler, it can train here.

## Train a GPT on a text file

```rust
use nervus::model::{Gpt, GptConfig};
use nervus::rng::Rng;
use nervus::text::{self, CharTokenizer, Corpus, Report, Training};

fn main() -> std::io::Result<()> {
    let text = std::fs::read_to_string("corpus.txt")?;
    let tok = CharTokenizer::from_text(&text);
    let corpus = Corpus::new(tok.encode(&text).unwrap(), 0.1);

    let mut rng = Rng::new(42);
    let config = GptConfig {
        vocab: tok.vocab(),
        context: 128,
        d_model: 192,
        n_heads: 6,
        n_layers: 6,
    };
    let mut model = Gpt::new(config, &mut rng);

    // Writes the model whenever validation loss improves, so a run that is
    // interrupted still leaves the best one behind.
    let cfg = Training {
        steps: 2000,
        save: Some("out/model".into()),
        ..Training::default()
    };
    text::train(&mut model, &tok, &corpus, &cfg, &mut rng, &mut |report| {
        if let Report::Step { step, val_loss, .. } = report {
            println!("step {step}: validation loss {val_loss:.3}");
        }
    })?;

    let prompt = tok.encode("The ").unwrap();
    let written = text::generate(&mut model, &prompt, 200, 0.8, &mut rng);
    println!("{}", tok.decode(&written));
    Ok(())
}
```

## What is in it

| Module | What it holds |
|---|---|
| `matrix` | the three matrix products everything else is built from |
| `nn` | `Linear`, `Relu`, `Gelu`, `Mlp`, softmax cross-entropy, and the `Layer` trait |
| `attention` | causal self-attention, forward and backward |
| `norm` | `LayerNorm` and `RmsNorm` |
| `embedding`, `block` | token and position embeddings; the residual connection and the block |
| `model` | `Gpt` — the pieces above, stacked |
| `optim` | `AdamW`, warm-up and cosine decay, gradient clipping |
| `text` | char tokeniser, batching, the training loop, sampling, data-parallel replicas |
| `checkpoint` | safetensors, read and written in GPT-2 layout |
| `mnist` | the dataset loader, for training an MLP instead |

Two binaries come with it: `train_mnist` and `train_text`.

## Every derivative by hand

There is no autograd engine here. Each layer's `backward` was derived by hand
and is checked against a numerical estimate of the same gradient:

```
cargo test -p nervus
```

That check is the crate's own proof, and it runs in under a second.

## Where it sits

`nervus` is part of [kvad](https://github.com/bisand/kvad), which is the engine
that runs what this trains: the repository has a test that trains a model here,
loads it with the inference code written for OpenAI's own GPT-2 weights, and
requires the same logits and the same characters out of both.

MIT licensed.
