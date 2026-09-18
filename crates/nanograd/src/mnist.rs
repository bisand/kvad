//! Loader for the MNIST handwritten-digit dataset in its original IDX format.
//!
//! IDX is about as simple as a binary format gets: a big-endian magic number
//! that encodes the element type and the number of dimensions, then one
//! big-endian `u32` per dimension, then the raw bytes. No compression, no
//! metadata, no headers per record.

use crate::matrix::Matrix;
use std::fs;
use std::path::Path;

pub struct Dataset {
    /// [n_examples, 784], normalised.
    pub images: Matrix,
    /// Digit 0-9 per example.
    pub labels: Vec<usize>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Gather the given example indices into one [batch, 784] matrix.
    pub fn batch(&self, indices: &[usize]) -> (Matrix, Vec<usize>) {
        let cols = self.images.cols;
        let mut x = Matrix::zeros(indices.len(), cols);
        let mut y = Vec::with_capacity(indices.len());
        for (r, &i) in indices.iter().enumerate() {
            x.row_mut(r).copy_from_slice(self.images.row(i));
            y.push(self.labels[i]);
        }
        (x, y)
    }

    /// Render one example as ASCII art, so you can see what the network sees.
    pub fn render(&self, i: usize) -> String {
        const RAMP: &[u8] = b" .:-=+*#%@";
        let mut s = String::new();
        for r in 0..28 {
            for c in 0..28 {
                // Undo the normalisation to get back to roughly [0, 1].
                let v = self.images.get(i, r * 28 + c) * 0.3081 + 0.1307;
                let idx = ((v.clamp(0.0, 1.0) * (RAMP.len() - 1) as f32) as usize).min(RAMP.len() - 1);
                s.push(RAMP[idx] as char);
            }
            s.push('\n');
        }
        s
    }
}

fn read_be_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn read_idx(path: &Path) -> std::io::Result<(Vec<usize>, Vec<u8>)> {
    let bytes = fs::read(path)?;
    if bytes.len() < 4 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "file too short"));
    }
    // Magic: two zero bytes, then the element-type code, then the rank.
    let magic = read_be_u32(&bytes, 0);
    if magic >> 16 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: not an IDX file (magic 0x{magic:08x})", path.display()),
        ));
    }
    let elem_type = (magic >> 8) & 0xff;
    if elem_type != 0x08 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: expected u8 elements, got type 0x{elem_type:02x}", path.display()),
        ));
    }
    let rank = (magic & 0xff) as usize;
    let mut dims = Vec::with_capacity(rank);
    for d in 0..rank {
        dims.push(read_be_u32(&bytes, 4 + d * 4) as usize);
    }
    let payload_start = 4 + rank * 4;
    let expected: usize = dims.iter().product();
    let payload = bytes[payload_start..].to_vec();
    if payload.len() != expected {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{}: expected {expected} bytes of payload, found {}",
                path.display(),
                payload.len()
            ),
        ));
    }
    Ok((dims, payload))
}

/// Load one split. `split` is "train" or "t10k".
pub fn load(dir: &Path, split: &str) -> std::io::Result<Dataset> {
    let img_path = dir.join(format!("{split}-images-idx3-ubyte"));
    let lbl_path = dir.join(format!("{split}-labels-idx1-ubyte"));

    if !img_path.exists() || !lbl_path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "MNIST files not found in {}.\nRun:  ./scripts/get-mnist.sh",
                dir.display()
            ),
        ));
    }

    let (img_dims, img_bytes) = read_idx(&img_path)?;
    let (_, labels) = read_idx(&lbl_path)?;
    let n = img_dims[0];
    let pixels = img_dims[1] * img_dims[2];

    // Scale to [0,1], then standardise using MNIST's global mean and standard
    // deviation. Centred, unit-variance inputs keep the first layer's
    // activations in a sane range, which is the same reason the weights are
    // initialised the way they are.
    let data = img_bytes
        .iter()
        .map(|&b| (b as f32 / 255.0 - 0.1307) / 0.3081)
        .collect::<Vec<f32>>();

    Ok(Dataset {
        images: Matrix::from_vec(n, pixels, data),
        labels: labels.iter().map(|&b| b as usize).collect(),
    })
}
