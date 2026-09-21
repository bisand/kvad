//! Saving a trained model, in a file that something else can open.
//!
//! # The one idea in this file
//!
//! A trained model is a list of named arrays of floats. That is all of it:
//! the architecture is a handful of integers, and everything the model learned
//! is in the arrays. So saving one is writing down names, shapes and floats,
//! and the only real decision is *which* names and shapes — because those are
//! a contract with whoever reads the file.
//!
//! This crate could invent its own. It uses GPT-2's instead, so that what
//! comes out of `train_text` is a GPT-2 checkpoint: the inference engine in
//! this repository (`kvad`) loads it with the code it uses for the real one,
//! and its tests check that the two crates then compute the same logits.
//!
//! # safetensors
//!
//! ```text
//! [8 bytes: N, the header's length, little-endian u64]
//! [N bytes: JSON, tensor name -> {"dtype", "shape", "data_offsets": [from, to]}]
//! [the floats, back to back, little-endian]
//! ```
//!
//! There is no code in the file and nothing to execute on load, which is the
//! format's reason for existing; the pickled checkpoints it replaced were
//! programs.
//!
//! # What has to move
//!
//! Less than you would fear. `Linear` here stores its weight `[in, out]` and
//! computes `x @ W`, which is exactly the convention GPT-2's checkpoint uses,
//! so most tensors are written as they are. Two things differ, and `place`
//! is the whole list:
//!
//! * **Query, key and value are one tensor.** GPT-2 computes all three with a
//!   single `[d, 3d]` matrix and slices the result. Ours are three `[d, d]`
//!   matrices, which go side by side into its columns.
//! * **The output head is transposed.** It is the one weight GPT-2 stores
//!   `[out, in]`, because there it is not a separate tensor at all: it is the
//!   token embedding table, `[vocab, d]`, used a second time.
//!
//! That second point is also where this file is *not* GPT-2. Our head is its
//! own tensor and has a bias, so the checkpoint carries `lm_head.weight` and
//! `lm_head.bias`, and `config.json` says `tie_word_embeddings: false`. A
//! reader that insists on tying would run a different model.
//!
//! # What is not saved
//!
//! The optimiser. AdamW keeps two numbers per weight, and a run resumed
//! without them starts its averages again from nothing. That sounds worse
//! than it measures: stopping `train_text` at step 1000 and resuming, against
//! the same run left alone, the training loss over the next 50 steps was
//! higher by 0.06 and 0.04 on two seeds and equal on a third, and by step 100
//! there was nothing to see. A long run with a learning-rate schedule would
//! have more to lose.

use crate::json::{object, Json};
use crate::model::{Gpt, GptConfig};
use crate::rng::Rng;
use std::io::{self, Write};
use std::path::Path;

pub const WEIGHTS_FILE: &str = "model.safetensors";
pub const CONFIG_FILE: &str = "config.json";

pub(crate) fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// An error from the file system does not say which file. This one does.
fn named(path: &Path, e: io::Error) -> io::Error {
    io::Error::new(e.kind(), format!("{}: {e}", path.display()))
}

pub(crate) fn read_text(path: &Path) -> io::Result<String> {
    std::fs::read_to_string(path).map_err(|e| named(path, e))
}

/// Write `bytes` so that `path` either keeps its old contents or gets all of
/// the new ones. Saving over the only copy of a model is exactly when a
/// half-written file would hurt.
pub(crate) fn write_whole(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let partial = path.with_extension("partial");
    let mut file = std::fs::File::create(&partial)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    std::fs::rename(&partial, path)
}

// ---------------------------------------------------------------------------
// The file format
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn write_safetensors(path: &Path, tensors: &[Tensor]) -> io::Result<()> {
    let mut entries = Vec::new();
    let mut offset = 0;
    for t in tensors {
        assert_eq!(t.data.len(), t.shape.iter().product::<usize>(), "{}: shape and data disagree", t.name);
        let end = offset + 4 * t.data.len();
        let entry = object([
            ("dtype", "F32".into()),
            ("shape", Json::Array(t.shape.iter().map(|&n| n.into()).collect())),
            ("data_offsets", Json::Array(vec![offset.into(), end.into()])),
        ]);
        entries.push((t.name.clone(), entry));
        offset = end;
    }

    // JSON ignores trailing spaces, so they are free padding: enough of them
    // and the floats start on an 8-byte boundary, where a reader that maps
    // the file can use them in place.
    let mut header = Json::Object(entries).to_string().into_bytes();
    header.resize(header.len().next_multiple_of(8), b' ');

    let mut bytes = Vec::with_capacity(8 + header.len() + offset);
    bytes.extend((header.len() as u64).to_le_bytes());
    bytes.extend(header);
    for t in tensors {
        for v in &t.data {
            bytes.extend(v.to_le_bytes());
        }
    }
    write_whole(path, &bytes)
}

pub fn read_safetensors(path: &Path) -> io::Result<Vec<Tensor>> {
    let bytes = std::fs::read(path).map_err(|e| named(path, e))?;
    let bad = |what: &str| invalid(format!("{}: {what}", path.display()));

    let length = bytes.get(..8).ok_or_else(|| bad("too short to be a safetensors file"))?;
    // Checked, because in a damaged file these eight bytes can say anything.
    let floats_at = usize::try_from(u64::from_le_bytes(length.try_into().unwrap()))
        .ok()
        .and_then(|n| n.checked_add(8))
        .filter(|&at| at <= bytes.len())
        .ok_or_else(|| bad("the header runs past the end of the file"))?;
    let header = std::str::from_utf8(&bytes[8..floats_at]).map_err(|_| bad("the header is not text"))?;
    let header = Json::parse(header).map_err(|e| bad(&e))?;
    let floats = &bytes[floats_at..];

    let mut tensors = Vec::new();
    for (name, entry) in header.as_object().ok_or_else(|| bad("the header is not an object"))? {
        if name == "__metadata__" {
            continue;
        }
        let bad = |what: &str| bad(&format!("tensor `{name}` {what}"));
        let dtype = entry.get("dtype").and_then(Json::as_str).ok_or_else(|| bad("has no dtype"))?;
        if dtype != "F32" {
            return Err(bad(&format!("is {dtype}; only F32 is read here")));
        }
        let sizes = |key: &str| -> Option<Vec<usize>> {
            entry.get(key)?.as_array()?.iter().map(Json::as_usize).collect()
        };
        let shape = sizes("shape").ok_or_else(|| bad("has no shape"))?;
        let (from, to) = match sizes("data_offsets").as_deref() {
            Some(&[from, to]) if from <= to && to <= floats.len() => (from, to),
            _ => return Err(bad("points outside the file")),
        };
        if to - from != 4 * shape.iter().product::<usize>() {
            return Err(bad("has a shape that does not match its size"));
        }
        let data = floats[from..to].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        tensors.push(Tensor { name: name.clone(), shape, data });
    }
    Ok(tensors)
}

// ---------------------------------------------------------------------------
// The contract: where each of our tensors lives in a GPT-2 checkpoint
// ---------------------------------------------------------------------------

/// Where one of this crate's tensors goes inside a GPT-2 checkpoint.
struct Place {
    /// The tensor in the file, and its shape there.
    tensor: String,
    shape: Vec<usize>,
    /// The shape of ours. A vector counts as one row.
    rows: usize,
    cols: usize,
    how: How,
}

enum How {
    /// The same numbers in the same order, starting at this column of the
    /// file's tensor. For everything except query, key and value that is
    /// column 0 of a tensor exactly as wide as ours.
    FromColumn(usize),
    /// Rows and columns swapped.
    Transposed,
}

impl Place {
    /// Where element `[r, j]` of our tensor sits in the file's flat data.
    /// Saving and loading both go through here, so they cannot disagree.
    fn index(&self, r: usize, j: usize) -> usize {
        match self.how {
            How::FromColumn(first) => r * self.shape[self.shape.len() - 1] + first + j,
            How::Transposed => j * self.rows + r,
        }
    }
}

/// The whole mapping, from the names `Gpt::params` reports to GPT-2's.
fn place(name: &str, config: &GptConfig) -> Option<Place> {
    let GptConfig { vocab, context, d_model: d, .. } = *config;
    let whole = |tensor: String, rows: usize, cols: usize| {
        let shape = if rows == 1 { vec![cols] } else { vec![rows, cols] };
        Some(Place { tensor, shape, rows, cols, how: How::FromColumn(0) })
    };

    if let Some(rest) = name.strip_prefix("blocks.") {
        let (layer, rest) = rest.split_once('.')?;
        let h = |s: &str| format!("h.{layer}.{s}");

        // Query, key and value, side by side in that order.
        for (third, projection) in ["wq", "wk", "wv"].iter().enumerate() {
            let how = How::FromColumn(third * d);
            if rest == format!("attn.1.{projection}.weight") {
                return Some(Place { tensor: h("attn.c_attn.weight"), shape: vec![d, 3 * d], rows: d, cols: d, how });
            }
            if rest == format!("attn.1.{projection}.bias") {
                return Some(Place { tensor: h("attn.c_attn.bias"), shape: vec![3 * d], rows: 1, cols: d, how });
            }
        }
        return match rest {
            "attn.0.gamma" => whole(h("ln_1.weight"), 1, d),
            "attn.0.beta" => whole(h("ln_1.bias"), 1, d),
            "attn.1.wo.weight" => whole(h("attn.c_proj.weight"), d, d),
            "attn.1.wo.bias" => whole(h("attn.c_proj.bias"), 1, d),
            "mlp.0.gamma" => whole(h("ln_2.weight"), 1, d),
            "mlp.0.beta" => whole(h("ln_2.bias"), 1, d),
            "mlp.1.weight" => whole(h("mlp.c_fc.weight"), d, 4 * d),
            "mlp.1.bias" => whole(h("mlp.c_fc.bias"), 1, 4 * d),
            "mlp.3.weight" => whole(h("mlp.c_proj.weight"), 4 * d, d),
            "mlp.3.bias" => whole(h("mlp.c_proj.bias"), 1, d),
            _ => None,
        };
    }
    match name {
        "tokens.table" => whole("wte.weight".into(), vocab, d),
        "positions.table" => whole("wpe.weight".into(), context, d),
        "norm.gamma" => whole("ln_f.weight".into(), 1, d),
        "norm.beta" => whole("ln_f.bias".into(), 1, d),
        "head.weight" => Some(Place {
            tensor: "lm_head.weight".into(),
            shape: vec![vocab, d],
            rows: d,
            cols: vocab,
            how: How::Transposed,
        }),
        "head.bias" => whole("lm_head.bias".into(), 1, vocab),
        _ => None,
    }
}

/// The model's weights, named and laid out as GPT-2's.
pub fn to_tensors(model: &mut Gpt) -> Vec<Tensor> {
    let config = model.config();
    let mut tensors: Vec<Tensor> = Vec::new();
    for p in model.params() {
        // A parameter added to the model and not to `place` would otherwise
        // be left out of every checkpoint without a word.
        let place = place(&p.name, &config).unwrap_or_else(|| panic!("no place in a checkpoint for `{}`", p.name));
        assert_eq!(p.value.len(), place.rows * place.cols, "`{}` is not the size its place expects", p.name);

        let at = tensors.iter().position(|t| t.name == place.tensor).unwrap_or_else(|| {
            let data = vec![0.0; place.shape.iter().product()];
            tensors.push(Tensor { name: place.tensor.clone(), shape: place.shape.clone(), data });
            tensors.len() - 1
        });
        for r in 0..place.rows {
            for j in 0..place.cols {
                tensors[at].data[place.index(r, j)] = p.value[r * place.cols + j];
            }
        }
    }
    tensors
}

/// Build a model of this shape and fill it from `tensors`.
pub fn from_tensors(config: GptConfig, tensors: &[Tensor]) -> Result<Gpt, String> {
    // The values it is built with are all overwritten, so the seed is moot.
    let mut model = Gpt::new(config, &mut Rng::new(0));
    for p in model.params() {
        let place = place(&p.name, &config).ok_or_else(|| format!("no place in a checkpoint for `{}`", p.name))?;
        let tensor = tensors
            .iter()
            .find(|t| t.name == place.tensor)
            .ok_or_else(|| format!("the checkpoint has no tensor `{}`", place.tensor))?;
        if tensor.shape != place.shape {
            return Err(format!(
                "tensor `{}` has shape {:?}, and this model needs {:?}",
                place.tensor, tensor.shape, place.shape
            ));
        }
        for r in 0..place.rows {
            for j in 0..place.cols {
                p.value[r * place.cols + j] = tensor.data[place.index(r, j)];
            }
        }
    }
    Ok(model)
}

// ---------------------------------------------------------------------------
// A directory holding a model
// ---------------------------------------------------------------------------

/// The architecture, in the words GPT-2's `config.json` uses for it.
fn config_json(config: &GptConfig) -> Json {
    // Through text, so that the f32 `1e-5` is written as 0.00001 and not as
    // the f64 nearest to it, 0.000009999999747.
    let eps: f64 = crate::norm::EPS.to_string().parse().unwrap();
    object([
        ("model_type", "gpt2".into()),
        ("architectures", Json::Array(vec!["GPT2LMHeadModel".into()])),
        ("vocab_size", config.vocab.into()),
        ("n_positions", config.context.into()),
        ("n_embd", config.d_model.into()),
        ("n_head", config.n_heads.into()),
        ("n_layer", config.n_layers.into()),
        ("layer_norm_epsilon", Json::Number(eps)),
        // The tanh approximation in `nn::Gelu`, by the name GPT-2 gives it.
        ("activation_function", "gelu_new".into()),
        ("tie_word_embeddings", Json::Bool(false)),
    ])
}

fn config_from_json(json: &Json) -> Result<GptConfig, String> {
    let size = |key: &str| {
        json.get(key).and_then(Json::as_usize).filter(|&n| n > 0).ok_or_else(|| format!("config: no usable `{key}`"))
    };
    let config = GptConfig {
        vocab: size("vocab_size")?,
        context: size("n_positions")?,
        d_model: size("n_embd")?,
        n_heads: size("n_head")?,
        n_layers: size("n_layer")?,
    };
    if config.d_model % config.n_heads != 0 {
        return Err(format!("config: {} heads do not divide a width of {}", config.n_heads, config.d_model));
    }
    Ok(config)
}

/// Write `model.safetensors` and `config.json` into `dir`, creating it.
pub fn save(dir: &Path, model: &mut Gpt) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    write_safetensors(&dir.join(WEIGHTS_FILE), &to_tensors(model))?;
    write_whole(&dir.join(CONFIG_FILE), config_json(&model.config()).to_string().as_bytes())
}

pub fn load(dir: &Path) -> io::Result<Gpt> {
    let config = read_text(&dir.join(CONFIG_FILE))?;
    let config = Json::parse(&config).and_then(|json| config_from_json(&json)).map_err(invalid)?;
    let tensors = read_safetensors(&dir.join(WEIGHTS_FILE))?;
    from_tensors(config, &tensors).map_err(invalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::scramble;

    /// No two sizes alike, so that none can stand in for another unnoticed.
    const CONFIG: GptConfig = GptConfig { vocab: 11, context: 6, d_model: 8, n_heads: 2, n_layers: 3 };
    const IDS: [usize; 6] = [3, 10, 0, 3, 7, 1];

    /// A model in which no two numbers are alike. Fresh from `Gpt::new` every
    /// bias is 0 and every gamma is 1, and a checkpoint that lost or swapped
    /// them would load back looking perfect.
    fn model(seed: u64) -> Gpt {
        let mut rng = Rng::new(seed);
        let mut model = Gpt::new(CONFIG, &mut rng);
        scramble(model.params(), &mut rng);
        model
    }

    fn snapshot(model: &mut Gpt) -> Vec<(String, Vec<f32>)> {
        model.params().into_iter().map(|p| (p.name, p.value.to_vec())).collect()
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("nervus-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// The format, checked against its description rather than against our
    /// own reader: two functions that share a misunderstanding agree.
    #[test]
    fn a_file_is_a_length_a_header_and_the_floats() {
        let dir = scratch("format");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("two.safetensors");
        let tensors = [
            Tensor { name: "a".into(), shape: vec![2, 3], data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0] },
            Tensor { name: "b \"quoted\"".into(), shape: vec![2], data: vec![-0.0, f32::MIN_POSITIVE] },
        ];
        write_safetensors(&path, &tensors).unwrap();
        let bytes = std::fs::read(&path).unwrap();

        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(n % 8, 0, "the floats should start on an 8-byte boundary");
        assert_eq!(bytes.len(), 8 + n + 4 * 8, "nothing but the length, the header and eight floats");

        let header = Json::parse(std::str::from_utf8(&bytes[8..8 + n]).unwrap()).unwrap();
        let b = header.get("b \"quoted\"").unwrap();
        assert_eq!(b.get("dtype").and_then(Json::as_str), Some("F32"));
        assert_eq!(b.get("shape"), Some(&Json::Array(vec![2.into()])));
        assert_eq!(b.get("data_offsets"), Some(&Json::Array(vec![24.into(), 32.into()])));
        // The fifth float of `a`, read by hand from where the header says.
        let at = 8 + n + 4 * 4;
        assert_eq!(bytes[at..at + 4], 5.0f32.to_le_bytes());

        // Bit for bit, which `==` would not check: -0.0 == 0.0.
        let back = read_safetensors(&path).unwrap();
        assert_eq!(back, tensors);
        assert_eq!(back[1].data[0].to_bits(), (-0.0f32).to_bits());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_saved_model_comes_back_bit_for_bit() {
        let dir = scratch("roundtrip");
        let mut original = model(1);
        save(&dir, &mut original).unwrap();
        // Saved twice: the second write has to replace the first.
        save(&dir, &mut original).unwrap();
        let mut loaded = load(&dir).unwrap();

        assert_eq!(snapshot(&mut loaded), snapshot(&mut original));
        assert_eq!(loaded.forward(&IDS), original.forward(&IDS));
        assert_eq!(loaded.config().n_heads, CONFIG.n_heads);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Every float in the file is written by exactly one parameter. Two
    /// parameters sharing a place, or a place nobody fills, are both caught
    /// here before they become a model that is subtly not the one you trained.
    #[test]
    fn every_float_in_the_file_belongs_to_one_parameter() {
        let mut model = model(2);
        let mut written: Vec<Tensor> = to_tensors(&mut model);
        written.iter_mut().for_each(|t| t.data.fill(0.0));
        for p in model.params() {
            let place = place(&p.name, &CONFIG).unwrap();
            let tensor = written.iter_mut().find(|t| t.name == place.tensor).unwrap();
            for r in 0..place.rows {
                for j in 0..place.cols {
                    tensor.data[place.index(r, j)] += 1.0;
                }
            }
        }
        for t in &written {
            assert!(t.data.iter().all(|&n| n == 1.0), "`{}` has floats written {:?} times", t.name, t.data);
        }
        let floats: usize = written.iter().map(|t| t.data.len()).sum();
        assert_eq!(floats, model.param_count());
    }

    /// The two rearrangements, read off by hand. The round trip cannot see a
    /// mistake made the same way in both directions; this can, and the test
    /// in the `kvad` crate that runs the file through another implementation
    /// is the one that settles it.
    #[test]
    fn the_layout_is_gpt2s() {
        let mut model = model(3);
        let tensors = to_tensors(&mut model);
        let params = snapshot(&mut model);
        let ours = |name: &str| &params.iter().find(|(n, _)| n == name).unwrap().1;
        let theirs = |name: &str| tensors.iter().find(|t| t.name == name).unwrap();
        let (d, vocab) = (CONFIG.d_model, CONFIG.vocab);

        let fused = theirs("h.1.attn.c_attn.weight");
        assert_eq!(fused.shape, [d, 3 * d]);
        for (third, name) in ["wq", "wk", "wv"].iter().enumerate() {
            let w = ours(&format!("blocks.1.attn.1.{name}.weight"));
            let b = ours(&format!("blocks.1.attn.1.{name}.bias"));
            // Row 5, column 2 of each projection.
            assert_eq!(fused.data[5 * 3 * d + third * d + 2], w[5 * d + 2]);
            assert_eq!(theirs("h.1.attn.c_attn.bias").data[third * d + 2], b[2]);
        }

        let head = theirs("lm_head.weight");
        assert_eq!(head.shape, [vocab, d]);
        // Ours is [d, vocab]: the weight from feature 5 to token 9.
        assert_eq!(head.data[9 * d + 5], ours("head.weight")[5 * vocab + 9]);

        assert_eq!(theirs("wpe.weight").shape, [CONFIG.context, d]);
        assert_eq!(theirs("h.0.mlp.c_fc.weight").shape, [d, 4 * d]);
        assert_eq!(theirs("h.0.mlp.c_proj.weight").shape, [4 * d, d]);
        assert_eq!(theirs("h.0.ln_2.bias").data, *ours("blocks.0.mlp.0.beta"));
        assert_eq!(tensors.len(), 2 + 12 * CONFIG.n_layers + 2 + 2);
    }

    #[test]
    fn the_config_says_what_the_model_is() {
        let json = Json::parse(&config_json(&CONFIG).to_string()).unwrap();
        let back = config_from_json(&json).unwrap();
        assert_eq!(format!("{back:?}"), format!("{CONFIG:?}"));
        assert_eq!(json.get("layer_norm_epsilon"), Some(&Json::Number(1e-5)));
        assert_eq!(json.get("tie_word_embeddings"), Some(&Json::Bool(false)));
    }

    #[test]
    fn a_checkpoint_for_another_model_is_refused_with_a_reason() {
        let tensors = to_tensors(&mut model(4));

        let wider = GptConfig { d_model: 16, ..CONFIG };
        let error = from_tensors(wider, &tensors).err().unwrap();
        assert!(error.contains("wte.weight") && error.contains("[11, 16]"), "{error}");

        let without: Vec<Tensor> = tensors.iter().filter(|t| t.name != "h.2.ln_2.bias").cloned().collect();
        let error = from_tensors(CONFIG, &without).err().unwrap();
        assert!(error.contains("no tensor `h.2.ln_2.bias`"), "{error}");

        let bad = Json::parse(r#"{"vocab_size":11,"n_positions":6,"n_embd":8,"n_head":3,"n_layer":3}"#).unwrap();
        assert!(config_from_json(&bad).unwrap_err().contains("3 heads"));
    }

    #[test]
    fn a_damaged_file_is_an_error_not_a_panic() {
        let dir = scratch("damaged");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.safetensors");
        write_safetensors(&path, &to_tensors(&mut model(5))).unwrap();
        let whole = std::fs::read(&path).unwrap();

        for keep in [0, 7, 8, 200, whole.len() - 1] {
            std::fs::write(&path, &whole[..keep]).unwrap();
            assert!(read_safetensors(&path).is_err(), "a file cut to {keep} bytes was accepted");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
