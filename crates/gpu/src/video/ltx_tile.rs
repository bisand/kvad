//! A DiT call over a large frame, made as several over overlapping tiles of
//! it and blended: the reference's `TiledDiffusionModel` with a
//! `TileCountConfig`, as DFR's spatial epilogue uses it.
//!
//! The sampler keeps one latent for the whole frame and steps it once. Only
//! the model's prediction is made in pieces. The rows and the columns are
//! each cut into `count` spans that overlap their neighbours by `overlap`
//! latent cells ([`split`], the reference's `split_by_count`); a tile is a
//! span of rows by a span of columns, over every frame. Each tile's tokens
//! are the video's own inside it and every appended token whose place
//! overlaps it ([`Tiling::keeps`]); their places are moved so that the tile
//! begins at row and column 0, and the DiT runs on them as on a clip of the
//! tile's size, which is the point: a size it was trained at.
//!
//! The predictions of the clean latent are then summed, each of the video's
//! own tokens weighted by its tile's *ramp* on rows times that on columns
//! ([`ramp`], the reference's trapezoidal mask: 1 inside, falling linearly
//! over the overlap), and each appended token by one over the tiles that
//! kept it.
//!
//! **The ramps do not always sum to 1.** Two neighbours' ramps do. But a
//! span shorter than twice the overlap is overlapped from both sides at
//! once, its two ramps multiply in the middle, and a third tile reaches the
//! same cells: with the epilogue's four tiles and overlap of 10 that is any
//! side under 50 latent cells, 1600 pixels, where the reference's weights
//! come to as much as 1.074 (32 cells) and it does not divide by them. That
//! is a prediction 7% too large in bands. [`Tiling::normalised`] divides by
//! the sum, which changes nothing where it is 1; without it the sums are the
//! reference's, for comparing against it.

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A span of latent cells `start..end` along one side, and how many cells
/// at each end it shares with a neighbour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub left: usize,
    pub right: usize,
}

impl Span {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }
}

/// `size` cells in `count` spans overlapping by `overlap`: the reference's
/// `split_by_count`, after its `clamp_dim_tiling`. The spans are
/// `(size + overlap·(count − 1)) / count` long, the first of them a cell
/// longer where that leaves a remainder. An overlap the side cannot hold is
/// cut to `size − count`, and a side of fewer cells than tiles is one span.
pub fn split(size: usize, count: usize, overlap: usize) -> Vec<Span> {
    let whole = vec![Span { start: 0, end: size, left: 0, right: 0 }];
    if count <= 1 || size < count {
        return whole;
    }
    let overlap = overlap.min(size - count);
    let total = size + overlap * (count - 1);
    let (len, extra) = (total / count, total % count);
    if size - extra <= len {
        return whole;
    }
    let step = len - overlap;
    (0..count)
        .map(|i| {
            // The first `extra` spans are a cell longer, and the rest moved.
            let (shift, grow) = (i.min(extra), (i < extra) as usize);
            let end = if i == count - 1 { size - extra } else { i * step + len };
            Span { start: i * step + shift, end: end + shift + grow, left: if i == 0 { 0 } else { overlap }, right: if i == count - 1 { 0 } else { overlap } }
        })
        .collect()
}

/// A span's blend weights: the reference's `compute_trapezoidal_mask_1d`. 1
/// inside, and over `left` cells at the start `k / (left + 1)` for `k` from
/// 1, over `right` at the end the same falling. Where the two meet they
/// multiply.
pub fn ramp(span: &Span) -> Vec<f32> {
    let n = span.len();
    let mut w = vec![1f32; n];
    // `torch.linspace(0, 1, steps)`, in f32 as torch fills it: from the
    // start in the first half and back from the end in the second.
    let linspace = |from: f32, to: f32, steps: usize, i: usize| -> f32 {
        let by = (to - from) / (steps - 1) as f32;
        if i < steps / 2 { from + by * i as f32 } else { to - by * (steps - i - 1) as f32 }
    };
    let (left, right) = (span.left.min(n), span.right.min(n));
    for (k, x) in w.iter_mut().take(left).enumerate() {
        *x *= linspace(0.0, 1.0, left + 2, k + 1);
    }
    for (k, x) in w.iter_mut().skip(n - right).enumerate() {
        *x *= linspace(1.0, 0.0, right + 2, k + 1);
    }
    w.iter().map(|x| x.clamp(0.0, 1.0)).collect()
}

/// How a call is tiled: `count` tiles each way, overlapping by `overlap`
/// cells, and whether the weights are divided by their sum (see the module
/// notes) or left as the reference leaves them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tiling {
    pub count: usize,
    pub overlap: usize,
    pub normalised: bool,
}

/// One tile: its rows and columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tile {
    pub rows: Span,
    pub cols: Span,
}

/// What a tile's call is made of: which of the state's tokens, in order,
/// the video's own first; where they are, moved to the tile's corner; and
/// each one's weight in the sum.
pub struct Piece {
    pub tokens: Vec<u32>,
    pub positions: [Vec<f32>; 3],
    pub weights: Vec<f32>,
}

impl Tiling {
    /// The tiles of a frame of `rows × cols` cells, rows first, as the
    /// reference's `create_tiles` orders them.
    pub fn tiles(&self, rows: usize, cols: usize) -> Vec<Tile> {
        let (r, c) = (split(rows, self.count, self.overlap), split(cols, self.count, self.overlap));
        r.iter().flat_map(|&rows| c.iter().map(move |&cols| Tile { rows, cols })).collect()
    }

    /// Each tile's [`Piece`] of a state of `frames × rows × cols` video
    /// tokens followed by appended ones, whose places are `positions`
    /// (time, rows, columns: the middle of each token's span, in seconds
    /// and pixels) and whose spans on rows and columns reach `reach` pixels
    /// either side of it.
    ///
    /// An appended token goes to every tile its span overlaps on rows and
    /// on columns, as the reference's `_all_tiles_cond_keep` has it; every
    /// tile spans the whole clip in time, so time keeps none out.
    pub fn pieces(&self, frames: usize, rows: usize, cols: usize, positions: &[Vec<f32>; 3], reach: &[f32]) -> Res<Vec<Piece>> {
        let own = frames * rows * cols;
        let n = positions[0].len();
        if n < own || reach.len() != n || positions.iter().any(|p| p.len() != n) {
            return Err(format!("{n} places and {} reaches for {frames}×{rows}×{cols} video tokens", reach.len()).into());
        }
        let tiles = self.tiles(rows, cols);
        // The sum of the weights at each cell, to divide by.
        let mut sum = vec![0f32; rows * cols];
        for t in &tiles {
            let (wr, wc) = (ramp(&t.rows), ramp(&t.cols));
            for (i, r) in (t.rows.start..t.rows.end).enumerate() {
                for (j, c) in (t.cols.start..t.cols.end).enumerate() {
                    sum[r * cols + c] += wr[i] * wc[j];
                }
            }
        }
        // A tile's bounds in pixels, the spans of its own tokens.
        let bounds = |s: &Span| ((32 * s.start) as f32, (32 * s.end) as f32);
        let keeps = |t: &Tile, k: usize| {
            let inside = |axis: usize, s: &Span| {
                let (lo, hi) = bounds(s);
                positions[axis][k] - reach[k] < hi && positions[axis][k] + reach[k] > lo
            };
            inside(1, &t.rows) && inside(2, &t.cols)
        };
        let kept: Vec<f32> = (own..n).map(|k| tiles.iter().filter(|t| keeps(t, k)).count() as f32).collect();
        let mut out = Vec::with_capacity(tiles.len());
        for t in &tiles {
            let (wr, wc) = (ramp(&t.rows), ramp(&t.cols));
            let (mut tokens, mut weights) = (Vec::new(), Vec::new());
            for f in 0..frames {
                for (i, r) in (t.rows.start..t.rows.end).enumerate() {
                    for (j, c) in (t.cols.start..t.cols.end).enumerate() {
                        tokens.push((f * rows * cols + r * cols + c) as u32);
                        let w = wr[i] * wc[j];
                        weights.push(if self.normalised { w / sum[r * cols + c] } else { w });
                    }
                }
            }
            for k in own..n {
                if keeps(t, k) {
                    tokens.push(k as u32);
                    weights.push(1.0 / kept[k - own]);
                }
            }
            // Moved so that the tile's own tokens begin at 0 on rows and
            // columns; they begin at 0 in time already.
            let (top, left) = (bounds(&t.rows).0, bounds(&t.cols).0);
            let at = |axis: usize, by: f32| tokens.iter().map(|&k| positions[axis][k as usize] - by).collect::<Vec<f32>>();
            out.push(Piece { positions: [at(0, 0.0), at(1, top), at(2, left)], tokens, weights });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference's `split_by_count(n, overlap)(size)`, asked of its
    /// 1.4.2: starts, ends and ramps.
    #[test]
    fn spans_are_the_reference_s() {
        let got = |size, count, overlap| split(size, count, overlap).iter().map(|s| (s.start, s.end, s.left, s.right)).collect::<Vec<_>>();
        assert_eq!(got(32, 4, 10), vec![(0, 16, 0, 10), (6, 22, 10, 10), (12, 27, 10, 10), (17, 32, 10, 0)]);
        assert_eq!(got(48, 4, 10), vec![(0, 20, 0, 10), (10, 30, 10, 10), (20, 39, 10, 10), (29, 48, 10, 0)]);
        assert_eq!(got(50, 4, 10), vec![(0, 20, 0, 10), (10, 30, 10, 10), (20, 40, 10, 10), (30, 50, 10, 0)]);
        assert_eq!(got(120, 4, 10), vec![(0, 38, 0, 10), (28, 66, 10, 10), (56, 93, 10, 10), (83, 120, 10, 0)]);
        assert_eq!(got(32, 2, 10), vec![(0, 21, 0, 10), (11, 32, 10, 0)]);
        assert_eq!(got(13, 2, 10), vec![(0, 12, 0, 10), (2, 13, 10, 0)]);
        assert_eq!(got(21, 4, 10), vec![(0, 13, 0, 10), (3, 16, 10, 10), (6, 19, 10, 10), (9, 21, 10, 0)]);
        // An overlap the side cannot hold, and a side of fewer cells than tiles.
        assert_eq!(got(10, 4, 10), vec![(0, 7, 0, 6), (1, 8, 6, 6), (2, 9, 6, 6), (3, 10, 6, 0)]);
        assert_eq!(got(3, 4, 10), vec![(0, 3, 0, 0)]);
        assert_eq!(got(20, 1, 10), vec![(0, 20, 0, 0)]);
    }

    #[test]
    fn two_neighbours_ramps_sum_to_one_and_three_tiles_do_not() {
        let sums = |size: usize, count: usize| {
            let mut acc = vec![0f32; size];
            for s in split(size, count, 10) {
                for (i, w) in ramp(&s).into_iter().enumerate() {
                    acc[s.start + i] += w;
                }
            }
            (acc.iter().cloned().fold(f32::MAX, f32::min), acc.iter().cloned().fold(0.0, f32::max))
        };
        // A falling ramp of 10: 10/11 down to 1/11.
        let r = ramp(&Span { start: 0, end: 21, left: 0, right: 10 });
        assert!((r[10] - 1.0).abs() < 1e-6 && (r[11] - 10.0 / 11.0).abs() < 1e-6 && (r[20] - 1.0 / 11.0).abs() < 1e-6);
        for (size, count, most) in [(32, 2, 1.0), (50, 4, 1.0), (120, 4, 1.0), (32, 4, 1.0744), (48, 4, 1.0083)] {
            let (lo, hi) = sums(size, count);
            assert!((lo - 1.0).abs() < 1e-4 && (hi - most).abs() < 1e-4, "{size} in {count}: {lo} to {hi}");
        }
    }

    /// A frame of 32 × 48 cells in sixteen tiles, with one appended plane
    /// and a half-size reference: every token is in some tile, normalised
    /// weights sum to 1 at each, and a tile's places begin at its corner.
    #[test]
    fn pieces_cover_the_state_and_their_weights_sum_to_one() {
        let (frames, rows, cols) = (2, 32, 48);
        let mut positions = [Vec::new(), Vec::new(), Vec::new()];
        let mut reach = Vec::new();
        let mut put = |t: f32, r: usize, c: usize, scale: usize| {
            positions[0].push(t);
            positions[1].push((32 * scale * r + 16 * scale) as f32);
            positions[2].push((32 * scale * c + 16 * scale) as f32);
            reach.push((16 * scale) as f32);
        };
        for f in 0..frames {
            (0..rows).for_each(|r| (0..cols).for_each(|c| put(f as f32, r, c, 1)));
        }
        (0..rows).for_each(|r| (0..cols).for_each(|c| put(0.5, r, c, 1)));
        (0..rows / 2).for_each(|r| (0..cols / 2).for_each(|c| put(0.0, r, c, 2)));
        let n = reach.len();
        for normalised in [true, false] {
            let pieces = Tiling { count: 4, overlap: 10, normalised }.pieces(frames, rows, cols, &positions, &reach).unwrap();
            assert_eq!(pieces.len(), 16);
            let mut sum = vec![0f32; n];
            for p in &pieces {
                p.tokens.iter().zip(&p.weights).for_each(|(&k, w)| sum[k as usize] += w);
                assert_eq!((p.positions[1][0], p.positions[2][0]), (16.0, 16.0));
            }
            let (lo, hi) = (sum.iter().cloned().fold(f32::MAX, f32::min), sum.iter().cloned().fold(0.0, f32::max));
            match normalised {
                true => assert!((lo - 1.0).abs() < 1e-5 && (hi - 1.0).abs() < 1e-5, "{lo} to {hi}"),
                // Rows of 32 reach 1.0744 and columns of 48 1.0083.
                false => assert!((lo - 1.0).abs() < 1e-5 && (hi - 1.0744 * 1.0083).abs() < 1e-3, "{lo} to {hi}"),
            }
        }
        // The first tile, rows 0..16 and columns 0..20: a reference token
        // spans two cells, so its row 8 and column 10 are just outside.
        let first = &Tiling { count: 4, overlap: 10, normalised: true }.pieces(frames, rows, cols, &positions, &reach).unwrap()[0];
        assert_eq!(first.tokens.len(), 2 * 16 * 20 + 16 * 20 + 8 * 10);
    }
}
