//! Moving MNIST: two digits bouncing around a box, made up on the spot.
//!
//! A video model needs videos, and the smallest honest one is this: take two
//! handwritten digits, drop them into a 64×64 box at random places, give each
//! a direction, and let them fly, bouncing off the walls, for a few frames.
//! Where they overlap, the brighter pixel wins. (Srivastava et al. made the
//! dataset this way in 2015, and it has been the video models' MNIST since.)
//!
//! Nothing is downloaded: the digits are the MNIST images already on disk,
//! and a seeded [`Rng`] decides the rest, so the same seed is the same clip.
//! It also comes with the ground truth a real video lacks. Every clip moves
//! smoothly, by construction, so a model whose clips do not is wrong in a way
//! that can be measured: see [`flicker`].

use crate::rng::Rng;

/// Pixels on a side of an MNIST digit.
pub const DIGIT: usize = 28;

/// Where the digits start, and which way they go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    /// Anywhere in the box, in any direction: Moving MNIST as published.
    Anywhere,
    /// Both in the middle, one leaving for the top-left corner and the other
    /// for the bottom-right: "a 3 and a 7 moving apart". Nothing about a clip
    /// is left to chance but the handwriting.
    Apart,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motion {
    pub start: Start,
    /// Pixels on a side of the box. At least a digit wide.
    pub side: usize,
    /// Pixels on a side of a digit: [`DIGIT`] as MNIST has them, or smaller
    /// after [`shrink`].
    pub digit: usize,
    pub frames: usize,
    /// Pixels a digit moves each frame.
    pub speed: f32,
}

impl Default for Motion {
    /// 64×64 and 3 pixels a frame, as the original dataset. 8 frames rather
    /// than its 20, so that a clip is 512 tokens at 8×8 patches.
    fn default() -> Self {
        Motion { start: Start::Anywhere, side: 64, digit: DIGIT, frames: 8, speed: 3.0 }
    }
}

impl Motion {
    /// The same clips at half the size: 14-pixel digits in a 32-pixel box,
    /// moving half as far. At 4×4 patches that is the 512 tokens the full
    /// size needs 8×8 patches for — and an 8×8 patch is 64 numbers for one
    /// token to carry, which a model 128 wide did not manage (see `train_video`).
    pub fn half() -> Self {
        Motion { start: Start::Anywhere, side: 32, digit: DIGIT / 2, frames: 8, speed: 1.5 }
    }

    /// Half size, and the two digits moving apart from the middle.
    ///
    /// Why this exists: with digits anywhere, a model told "a 3 and a 7" has
    /// learned little about *what* to draw before it has learned *where*, and
    /// the labels barely matter. Measured on one frame of the half-size clips
    /// after 12,000 steps: validation 0.1354 with the right labels and 0.1387
    /// with wrong ones, a gap of 0.0033, where the digit model's is 0.059.
    /// It drew strokes, mostly not the digits asked for. Take the where away
    /// and what is left is the digit model's problem, plus time.
    pub fn apart() -> Self {
        Motion { start: Start::Apart, ..Motion::half() }
    }
}

/// A `DIGIT × DIGIT` image at `1/factor` of its size, each pixel the mean of
/// the `factor × factor` it replaces.
pub fn shrink(image: &[f32], factor: usize) -> Vec<f32> {
    assert_eq!(DIGIT % factor, 0, "{DIGIT} pixels do not shrink by {factor}");
    let small = DIGIT / factor;
    let mut out = vec![0.0; small * small];
    for y in 0..DIGIT {
        for x in 0..DIGIT {
            out[(y / factor) * small + x / factor] += image[y * DIGIT + x] / (factor * factor) as f32;
        }
    }
    out
}

/// One clip: `frames` images of `side × side`, frame after frame, each pixel
/// from 0 (paper) to 1 (ink); and the two digits in it.
pub struct Clip {
    pub pixels: Vec<f32>,
    pub labels: [usize; 2],
}

/// Where a digit's top-left corner is in each frame, starting somewhere in
/// the box and moving `speed` a frame in a random direction, turning back
/// from each wall it would cross.
pub fn path(motion: &Motion, rng: &mut Rng) -> Vec<(usize, usize)> {
    let room = (motion.side - motion.digit) as f32;
    let (x, y) = (rng.uniform() * room, rng.uniform() * room);
    let angle = rng.uniform() * std::f32::consts::TAU;
    path_from(motion, (x, y), angle)
}

/// A digit's corners, starting at `(x, y)` and heading at `angle`.
fn path_from(motion: &Motion, (mut x, mut y): (f32, f32), angle: f32) -> Vec<(usize, usize)> {
    let room = (motion.side - motion.digit) as f32;
    let (mut dx, mut dy) = (motion.speed * angle.cos(), motion.speed * angle.sin());
    let mut out = Vec::with_capacity(motion.frames);
    for _ in 0..motion.frames {
        out.push((x.round() as usize, y.round() as usize));
        // A bounce is a reflection: whatever went past the wall comes back
        // by as much, and the velocity turns round.
        x += dx;
        y += dy;
        if x < 0.0 || x > room {
            dx = -dx;
            x = if x < 0.0 { -x } else { 2.0 * room - x };
        }
        if y < 0.0 || y > room {
            dy = -dy;
            y = if y < 0.0 { -y } else { 2.0 * room - y };
        }
    }
    out
}

/// A clip of two digits from `images`, each `digit × digit` from 0 to 1,
/// moving independently.
pub fn clip(motion: &Motion, images: &[Vec<f32>], labels: &[usize], rng: &mut Rng) -> Clip {
    let d = motion.digit;
    assert!(motion.side >= d, "a {}-pixel box cannot hold a {d}-pixel digit", motion.side);
    assert_eq!(images[0].len(), d * d, "the images are not {d}×{d}");
    let (a, b) = (rng.below(images.len()), rng.below(images.len()));
    let n = motion.side;
    let mut pixels = vec![0.0f32; motion.frames * n * n];
    let middle = ((n - d) / 2) as f32;
    for (i, which) in [a, b].into_iter().enumerate() {
        let path = match motion.start {
            Start::Anywhere => path(motion, rng),
            // Up and left for the first label, down and right for the second.
            Start::Apart => path_from(motion, (middle, middle), std::f32::consts::FRAC_PI_4 * if i == 0 { 5.0 } else { 1.0 }),
        };
        for (f, (x0, y0)) in path.into_iter().enumerate() {
            let frame = &mut pixels[f * n * n..(f + 1) * n * n];
            for y in 0..d {
                for x in 0..d {
                    let at = &mut frame[(y0 + y) * n + x0 + x];
                    *at = at.max(images[which][y * d + x]);
                }
            }
        }
    }
    Clip { pixels, labels: [labels[a], labels[b]] }
}

/// How much a clip changes from one frame to the next: the mean absolute
/// difference between neighbouring frames, over every pixel.
///
/// A model that draws each frame as a good picture, but not the *same*
/// picture a moment later, flickers: digits change shape, jump, or fade in
/// and out between frames. Each frame would pass any test of one frame. This
/// number would not, because real clips of digits moving three pixels a
/// frame change by a known, small amount, and a flickering one changes by a
/// lot more.
pub fn flicker(pixels: &[f32], frames: usize) -> f32 {
    // One frame has nothing to change from.
    if frames < 2 {
        return 0.0;
    }
    let size = pixels.len() / frames;
    let mut total = 0.0;
    for f in 1..frames {
        let (prev, next) = (&pixels[(f - 1) * size..f * size], &pixels[f * size..(f + 1) * size]);
        total += prev.iter().zip(next).map(|(a, b)| (a - b).abs()).sum::<f32>();
    }
    total / ((frames - 1) * size) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A digit that is a solid square, so that where it went is easy to see.
    fn squares() -> (Vec<Vec<f32>>, Vec<usize>) {
        (vec![vec![1.0; DIGIT * DIGIT], vec![0.5; DIGIT * DIGIT]], vec![3, 7])
    }

    #[test]
    fn a_digit_moves_at_its_speed_and_stays_in_the_box() {
        let motion = Motion { frames: 200, ..Motion::default() };
        let mut rng = Rng::new(1);
        for _ in 0..20 {
            let p = path(&motion, &mut rng);
            for (a, b) in p.iter().zip(&p[1..]) {
                assert!(b.0 <= motion.side - motion.digit && b.1 <= motion.side - motion.digit, "{b:?} left the box");
                // Rounding each position can add up to a pixel either way.
                let step = ((a.0 as f32 - b.0 as f32).powi(2) + (a.1 as f32 - b.1 as f32).powi(2)).sqrt();
                assert!(step <= motion.speed + 1.5, "moved {step} in one frame");
            }
        }
    }

    /// Over 200 frames every digit hits a wall, and it comes back.
    #[test]
    fn a_digit_bounces() {
        let motion = Motion { frames: 200, ..Motion::default() };
        let p = path(&motion, &mut Rng::new(2));
        let xs: Vec<usize> = p.iter().map(|&(x, _)| x).collect();
        assert!(xs.iter().any(|&x| x == 0 || x == motion.side - DIGIT) || p.iter().any(|&(_, y)| y == 0 || y == motion.side - DIGIT));
        let turns = xs.windows(3).filter(|w| (w[1] as i32 - w[0] as i32) * (w[2] as i32 - w[1] as i32) < 0).count();
        assert!(turns > 0, "never turned round");
    }

    #[test]
    fn a_seed_is_a_clip() {
        let (images, labels) = squares();
        let motion = Motion::default();
        let one = clip(&motion, &images, &labels, &mut Rng::new(3));
        let two = clip(&motion, &images, &labels, &mut Rng::new(3));
        assert_eq!(one.pixels, two.pixels);
        assert_eq!(one.labels, two.labels);
        assert_eq!(one.pixels.len(), 8 * 64 * 64);
    }

    /// Two digits in every frame, and where they overlap the brighter wins.
    #[test]
    fn every_frame_holds_both_digits() {
        let (images, labels) = squares();
        let motion = Motion::default();
        let mut rng = Rng::new(4);
        for _ in 0..10 {
            let c = clip(&motion, &images, &labels, &mut rng);
            let n = motion.side * motion.side;
            for f in 0..motion.frames {
                let frame = &c.pixels[f * n..(f + 1) * n];
                let ink: f32 = frame.iter().sum();
                // Two squares, which may be the same one twice and may
                // overlap: at least the dim one alone, at most the bright
                // one twice, side by side.
                let square = (DIGIT * DIGIT) as f32;
                assert!((0.5 * square - 1e-3..=2.0 * square + 1e-3).contains(&ink), "frame {f}: {ink}");
                assert!(frame.iter().all(|&v| v == 0.0 || v == 0.5 || v == 1.0));
            }
        }
    }

    #[test]
    fn a_shrunk_digit_keeps_its_ink() {
        let mut image = vec![0.0; DIGIT * DIGIT];
        image[0] = 1.0; // one pixel of the top-left 2×2
        image[DIGIT * DIGIT - 1] = 0.5;
        let small = shrink(&image, 2);
        assert_eq!(small.len(), 14 * 14);
        assert_eq!(small[0], 0.25);
        assert_eq!(small[14 * 14 - 1], 0.125);
        assert!((small.iter().sum::<f32>() * 4.0 - image.iter().sum::<f32>()).abs() < 1e-6);

        let (images, labels) = squares();
        let half: Vec<Vec<f32>> = images.iter().map(|i| shrink(i, 2)).collect();
        let c = clip(&Motion::half(), &half, &labels, &mut Rng::new(6));
        assert_eq!(c.pixels.len(), 8 * 32 * 32);
    }

    /// Moving apart: starting together in the middle, never touching a wall
    /// in 8 frames, and further apart each frame.
    #[test]
    fn apart_is_the_same_journey_every_time() {
        let motion = Motion::apart();
        let middle = (motion.side - motion.digit) / 2;
        let start = (middle as f32, middle as f32);
        let a = path_from(&motion, start, std::f32::consts::FRAC_PI_4 * 5.0);
        let b = path_from(&motion, start, std::f32::consts::FRAC_PI_4);
        assert_eq!((a[0], b[0]), ((middle, middle), (middle, middle)));
        for f in 1..motion.frames {
            assert!(a[f].0 < a[f - 1].0 && a[f].1 < a[f - 1].1, "frame {f}: {:?}", a[f]);
            assert!(b[f].0 > b[f - 1].0 && b[f].1 > b[f - 1].1, "frame {f}: {:?}", b[f]);
        }
        let last = motion.frames - 1;
        assert!(a[last].0 > 0 && b[last].0 < motion.side - motion.digit, "reached a wall");
    }

    /// With the path fixed, a clip depends on its seed only through which
    /// two digits it drew: two seeds that draw the same two are the same clip.
    #[test]
    fn apart_clips_differ_only_in_their_digits() {
        let motion = Motion::apart();
        // Four different digits, so a clip's labels say which were drawn.
        let images: Vec<Vec<f32>> = (0..4).map(|i| shrink(&vec![0.2 * (i + 1) as f32; DIGIT * DIGIT], 2)).collect();
        let labels = [0, 1, 2, 3];
        let clips: Vec<Clip> = (0..40).map(|seed| clip(&motion, &images, &labels, &mut Rng::new(seed))).collect();
        let mut compared = 0;
        for i in 0..clips.len() {
            for j in 0..i {
                if clips[i].labels == clips[j].labels {
                    assert_eq!(clips[i].pixels, clips[j].pixels, "seeds {i} and {j}");
                    compared += 1;
                }
            }
        }
        assert!(compared > 0, "no two seeds drew the same digits");
    }

    #[test]
    fn a_still_clip_does_not_flicker_and_a_shuffled_one_does() {
        let (images, labels) = squares();
        let still = clip(&Motion { speed: 0.0, ..Motion::default() }, &images, &labels, &mut Rng::new(5));
        assert_eq!(flicker(&still.pixels, 8), 0.0);
        let moving = clip(&Motion::default(), &images, &labels, &mut Rng::new(5));
        let n = 64 * 64;
        // The same frames in an order that jumps about.
        let order = [0, 5, 2, 7, 1, 6, 3, 4];
        let shuffled: Vec<f32> = order.iter().flat_map(|&f| moving.pixels[f * n..(f + 1) * n].to_vec()).collect();
        assert!(flicker(&shuffled, 8) > flicker(&moving.pixels, 8));
    }
}
