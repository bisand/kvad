//! Video: frames, sound, and the files they are saved as.
//!
//! Everything that *makes* a video lives in `kvad-gpu` (see
//! `docs/video-plan.md`). What is here is what the rest of the engine has to
//! name without depending on candle, starting with the part every later step
//! needs before it can be seen at all: a file a browser will play.
//!
//! # A video file, by hand
//!
//! [`Image::png`](crate::image::Image::png) writes a PNG that compresses
//! nothing, using deflate's *stored* blocks. This module does the same for
//! video: every codec and container below is the real thing, and nothing is
//! compressed.
//!
//! - **Video is H.264, every macroblock `I_PCM`.** H.264 has a macroblock
//!   type that carries its 16×16 luma and two 8×8 chroma samples as plain
//!   bytes, with no prediction, no transform and no entropy coding. A
//!   stream made only of those is legal Baseline-profile H.264, which every
//!   browser decodes. It is what stored blocks are to deflate. What it costs
//!   is size: 1.5 bytes a pixel, about 14 MB a second at 768×512 and 24 fps.
//! - **Sound is FLAC, every subframe `VERBATIM`.** FLAC likewise has a
//!   subframe type that stores the samples as they are. Browsers play FLAC
//!   inside MP4, where AAC or Opus would each be an encoder of their own.
//! - **The container is MP4** (ISO base media), written with its index
//!   (`moov`) in front of the data (`mdat`), so a browser can start playing
//!   before the whole file has arrived.
//!
//! [`Audio::wav`] writes the sound on its own as well, because a WAV plays
//! everywhere and is four header fields and the samples.
//!
//! The colours are BT.709, limited range, 4:2:0, and the stream says so. That
//! is what the LTX reference writes, and what a player assumes when a stream
//! says nothing; saying it anyway means no player has to guess.

/// Frames to be shown one after another, as 8-bit RGB.
#[derive(Debug, Clone, PartialEq)]
pub struct Video {
    pub width: usize,
    pub height: usize,
    /// Frames per second. Whole numbers only, which covers what the models
    /// here are trained at (24, 25, 48, 50).
    pub fps: u32,
    /// Every frame, one after another, each `width × height × 3` bytes.
    pub rgb: Vec<u8>,
}

/// Sound, as samples in [-1, 1], interleaved by channel.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    /// Samples per second, per channel.
    pub rate: u32,
    pub channels: usize,
    pub samples: Vec<f32>,
}

impl Video {
    pub fn frames(&self) -> usize {
        self.rgb.len() / (self.width * self.height * 3)
    }

    /// The video, and the sound if there is some, as an MP4 file.
    ///
    /// Width and height must be even, because 4:2:0 keeps one chroma sample
    /// per 2×2 pixels; they need not be multiples of 16, which the stream
    /// pads to and then crops off again.
    pub fn mp4(&self, audio: Option<&Audio>) -> Vec<u8> {
        assert!(self.width % 2 == 0 && self.height % 2 == 0, "4:2:0 needs an even width and height");
        assert!(self.width > 0 && self.height > 0 && self.fps > 0);
        assert_eq!(self.rgb.len() % (self.width * self.height * 3), 0, "rgb is not a whole number of frames");

        let h264 = H264::new(self.width, self.height, self.fps);
        let frame = self.width * self.height * 3;
        let video = Track {
            kind: Kind::Video { width: self.width, height: self.height, avcc: h264.avcc() },
            timescale: self.fps,
            samples: (0..self.frames())
                .map(|i| h264.frame(&self.rgb[i * frame..(i + 1) * frame], i))
                .collect(),
            durations: vec![1; self.frames()],
        };
        let mut tracks = vec![video];
        if let Some(a) = audio {
            // One FLAC frame per video frame, so the two interleave one for
            // one in the file and a player never waits on either.
            let block = (a.rate as usize / self.fps as usize).max(16);
            tracks.push(flac_track(a, block));
        }
        mp4(&tracks)
    }
}

impl Audio {
    /// The sound as a 16-bit PCM WAV file.
    pub fn wav(&self) -> Vec<u8> {
        let data = self.samples.len() * 2;
        let mut out = Vec::with_capacity(44 + data);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // integer PCM
        out.extend_from_slice(&(self.channels as u16).to_le_bytes());
        out.extend_from_slice(&self.rate.to_le_bytes());
        out.extend_from_slice(&(self.rate * self.channels as u32 * 2).to_le_bytes());
        out.extend_from_slice(&(self.channels as u16 * 2).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data as u32).to_le_bytes());
        for &s in &self.samples {
            out.extend_from_slice(&pcm16(s).to_le_bytes());
        }
        out
    }

    fn frames(&self) -> usize {
        self.samples.len() / self.channels
    }
}

/// A sample in [-1, 1] as a signed 16-bit integer.
fn pcm16(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

/// One frame of RGB as the three planes of BT.709 limited-range 4:2:0,
/// padded to `pw × ph` by repeating the last row and column.
///
/// Luma is `0.2126 R + 0.7152 G + 0.0722 B`, stored as `16 + 219·Y`; the two
/// colour differences are scaled into ±0.5 and stored as `128 + 224·C`. So
/// black is (16, 128, 128) and white (235, 128, 128), and no sample is ever 0
/// — which is as well, because a run of zero bytes is the one thing an H.264
/// payload must not contain unescaped.
///
/// Each chroma sample is the mean of the four pixels it covers, as the LTX
/// reference computes it.
fn yuv420(rgb: &[u8], w: usize, h: usize, pw: usize, ph: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut y, mut cb, mut cr) = (vec![0u8; pw * ph], vec![0f32; pw * ph / 4], vec![0f32; pw * ph / 4]);
    for py in 0..ph {
        for px in 0..pw {
            let i = (py.min(h - 1) * w + px.min(w - 1)) * 3;
            let [r, g, b] = [rgb[i], rgb[i + 1], rgb[i + 2]].map(|v| v as f32 / 255.0);
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            y[py * pw + px] = (16.0 + 219.0 * luma).round() as u8;
            let c = (py / 2) * (pw / 2) + px / 2;
            cb[c] += (b - luma) / 1.8556 / 4.0;
            cr[c] += (r - luma) / 1.5748 / 4.0;
        }
    }
    let chroma = |p: Vec<f32>| p.into_iter().map(|c| (128.0 + 224.0 * c).round() as u8).collect();
    (y, chroma(cb), chroma(cr))
}

// ---------------------------------------------------------------------------
// Bits
// ---------------------------------------------------------------------------

/// A bit writer, most significant bit first, as both H.264 and FLAC write.
struct Bits {
    out: Vec<u8>,
    /// Bits already used in the last byte of `out`, 0 when it is full.
    used: u8,
}

impl Bits {
    fn new() -> Self {
        Bits { out: Vec::new(), used: 0 }
    }

    fn bit(&mut self, b: bool) {
        if self.used == 0 {
            self.out.push(0);
        }
        if b {
            *self.out.last_mut().unwrap() |= 0x80 >> self.used;
        }
        self.used = (self.used + 1) % 8;
    }

    fn bits(&mut self, v: u64, n: u32) {
        for i in (0..n).rev() {
            self.bit(v >> i & 1 == 1);
        }
    }

    /// Exp-Golomb, H.264's `ue(v)`: `v + 1` in binary, preceded by one fewer
    /// zeros than it has digits. 0 is `1`, 1 is `010`, 2 is `011`, 3 is
    /// `00100`.
    fn ue(&mut self, v: u32) {
        let x = v as u64 + 1;
        let n = 64 - x.leading_zeros();
        self.bits(0, n - 1);
        self.bits(x, n);
    }

    /// Signed Exp-Golomb, `se(v)`: 1, −1, 2, −2 … become 1, 2, 3, 4 … for `ue`.
    fn se(&mut self, v: i32) {
        self.ue(if v > 0 { 2 * v as u32 - 1 } else { 2 * v.unsigned_abs() });
    }

    fn align(&mut self) {
        self.used = 0;
    }

    /// Whole bytes, which must start on a byte boundary.
    fn bytes(&mut self, b: &[u8]) {
        assert_eq!(self.used, 0, "bytes must start on a byte boundary");
        self.out.extend_from_slice(b);
    }

    /// H.264's `rbsp_trailing_bits`: a single 1, then zeros to the byte.
    fn trailing(mut self) -> Vec<u8> {
        self.bit(true);
        self.align();
        self.out
    }
}

// ---------------------------------------------------------------------------
// H.264
// ---------------------------------------------------------------------------

/// The parts of an H.264 stream that stay the same from frame to frame.
struct H264 {
    width: usize,
    height: usize,
    /// Width and height in 16×16 macroblocks, rounded up.
    mbw: usize,
    mbh: usize,
    fps: u32,
}

/// Levels, from H.264's Table A-1: the smallest that allows a stream's frame
/// size, macroblock rate and bit rate is the one it declares.
///
/// `(level_idc, max macroblocks a second, max macroblocks a frame, max
/// kbit/s)`.
const LEVELS: [(u8, u64, u64, u64); 19] = [
    (10, 1485, 99, 64),
    (11, 3000, 396, 192),
    (12, 6000, 396, 384),
    (13, 11880, 396, 768),
    (20, 11880, 396, 2000),
    (21, 19800, 792, 4000),
    (22, 20250, 1620, 4000),
    (30, 40500, 1620, 10000),
    (31, 108000, 3600, 14000),
    (32, 216000, 5120, 20000),
    (40, 245760, 8192, 20000),
    (41, 245760, 8192, 50000),
    (42, 522240, 8704, 50000),
    (50, 589824, 22080, 135000),
    (51, 983040, 36864, 240000),
    (52, 2073600, 36864, 240000),
    (60, 4177920, 139264, 240000),
    (61, 8355840, 139264, 480000),
    (62, 16711680, 139264, 800000),
];

impl H264 {
    fn new(width: usize, height: usize, fps: u32) -> Self {
        H264 { width, height, mbw: width.div_ceil(16), mbh: height.div_ceil(16), fps }
    }

    /// The level this stream needs.
    ///
    /// Every macroblock is 384 bytes, so the bit rate is known exactly and is
    /// high: 768×512 at 24 fps is 113 Mbit/s, which is level 5.0. Levels
    /// also cap compression ratio from below (`MinCR`), which an uncompressed
    /// stream cannot meet at any level; decoders do not check it.
    fn level(&self) -> u8 {
        let fs = (self.mbw * self.mbh) as u64;
        let mbps = fs * self.fps as u64;
        let kbps = (fs * 384 * 8 * self.fps as u64).div_ceil(1000);
        let side = self.mbw.max(self.mbh) as u64;
        LEVELS
            .iter()
            .find(|&&(_, max_mbps, max_fs, max_kbps)| {
                // A frame's width and height may each be at most √(8·MaxFS)
                // macroblocks, so a very long thin frame needs a higher level.
                mbps <= max_mbps && fs <= max_fs && kbps <= max_kbps && side * side <= 8 * max_fs
            })
            .map_or(62, |l| l.0)
    }

    /// The sequence parameter set: size, profile, level, colour, frame rate.
    fn sps(&self) -> Vec<u8> {
        let mut b = Bits::new();
        b.bits(66, 8); // profile_idc: Baseline
        // constraint_set0 and set1: also decodable as Constrained Baseline
        // and Main, which is what browsers ask for.
        b.bits(0b1100_0000, 8);
        b.bits(self.level() as u64, 8);
        b.ue(0); // seq_parameter_set_id
        b.ue(0); // log2_max_frame_num_minus4: frame_num is always 0 here
        // pic_order_cnt_type 2: display order is decoding order, so no
        // picture-order counts are sent at all.
        b.ue(2);
        b.ue(1); // max_num_ref_frames
        b.bit(false); // gaps_in_frame_num_value_allowed_flag
        b.ue(self.mbw as u32 - 1);
        b.ue(self.mbh as u32 - 1);
        b.bit(true); // frame_mbs_only_flag: frames, never fields
        b.bit(true); // direct_8x8_inference_flag
        // Cropping is in units of two pixels for 4:2:0.
        let (right, bottom) = ((self.mbw * 16 - self.width) / 2, (self.mbh * 16 - self.height) / 2);
        b.bit(right > 0 || bottom > 0);
        if right > 0 || bottom > 0 {
            b.ue(0);
            b.ue(right as u32);
            b.ue(0);
            b.ue(bottom as u32);
        }

        b.bit(true); // vui_parameters_present_flag
        b.bit(true); // aspect_ratio_info_present_flag
        b.bits(1, 8); // square pixels
        b.bit(false); // overscan_info_present_flag
        b.bit(true); // video_signal_type_present_flag
        b.bits(5, 3); // video_format: unspecified
        b.bit(false); // video_full_range_flag: limited range
        b.bit(true); // colour_description_present_flag
        b.bits(1, 8); // colour_primaries: BT.709
        b.bits(1, 8); // transfer_characteristics: BT.709
        b.bits(1, 8); // matrix_coefficients: BT.709
        b.bit(false); // chroma_loc_info_present_flag
        b.bit(true); // timing_info_present_flag
        // A tick is half a frame (a field) in H.264's timing, whether or not
        // the stream has fields.
        b.bits(1, 32); // num_units_in_tick
        b.bits(2 * self.fps as u64, 32); // time_scale
        b.bit(true); // fixed_frame_rate_flag
        b.bit(false); // nal_hrd_parameters_present_flag
        b.bit(false); // vcl_hrd_parameters_present_flag
        b.bit(false); // pic_struct_present_flag
        b.bit(true); // bitstream_restriction_flag
        b.bit(true); // motion_vectors_over_pic_boundaries_flag
        b.ue(0); // max_bytes_per_pic_denom: no limit
        b.ue(0); // max_bits_per_mb_denom: no limit
        b.ue(11); // log2_max_mv_length_horizontal
        b.ue(11); // log2_max_mv_length_vertical
        // No frame waits for a later one, so a decoder may show each as soon
        // as it has it.
        b.ue(0); // max_num_reorder_frames
        b.ue(1); // max_dec_frame_buffering
        nal(0x67, &b.trailing())
    }

    /// The picture parameter set: CAVLC, one slice group, deblocking
    /// switchable per slice.
    fn pps(&self) -> Vec<u8> {
        let mut b = Bits::new();
        b.ue(0); // pic_parameter_set_id
        b.ue(0); // seq_parameter_set_id
        b.bit(false); // entropy_coding_mode_flag: CAVLC
        b.bit(false); // bottom_field_pic_order_in_frame_present_flag
        b.ue(0); // num_slice_groups_minus1
        b.ue(0); // num_ref_idx_l0_default_active_minus1
        b.ue(0); // num_ref_idx_l1_default_active_minus1
        b.bit(false); // weighted_pred_flag
        b.bits(0, 2); // weighted_bipred_idc
        b.se(0); // pic_init_qp_minus26
        b.se(0); // pic_init_qs_minus26
        b.se(0); // chroma_qp_index_offset
        b.bit(true); // deblocking_filter_control_present_flag
        b.bit(false); // constrained_intra_pred_flag
        b.bit(false); // redundant_pic_cnt_present_flag
        nal(0x68, &b.trailing())
    }

    /// The `avcC` box body: the decoder configuration an MP4 carries once,
    /// in place of parameter sets in the stream.
    fn avcc(&self) -> Vec<u8> {
        let (sps, pps) = (self.sps(), self.pps());
        let mut out = vec![1, sps[1], sps[2], sps[3]];
        out.push(0xfc | 3); // NAL lengths are 4 bytes
        out.push(0xe0 | 1); // one SPS
        out.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        out.extend_from_slice(&sps);
        out.push(1); // one PPS
        out.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        out.extend_from_slice(&pps);
        out
    }

    /// One frame as one IDR slice, length-prefixed as MP4 stores it.
    ///
    /// Every frame is an IDR picture: decodable on its own, a place a player
    /// can seek to, and a reference for nothing after it.
    fn frame(&self, rgb: &[u8], index: usize) -> Vec<u8> {
        let (pw, ph) = (self.mbw * 16, self.mbh * 16);
        let (y, cb, cr) = yuv420(rgb, self.width, self.height, pw, ph);

        let mut b = Bits::new();
        b.ue(0); // first_mb_in_slice
        b.ue(7); // slice_type: I, and every slice in the picture is I
        b.ue(0); // pic_parameter_set_id
        b.bits(0, 4); // frame_num
        // Two IDR pictures in a row must have different ids, or a decoder may
        // take the second for more of the first.
        b.ue((index % 2) as u32); // idr_pic_id
        b.bit(false); // no_output_of_prior_pics_flag
        b.bit(false); // long_term_reference_flag
        b.se(0); // slice_qp_delta
        // PCM samples are not filtered anyway (their quantiser is 0, and the
        // filter's thresholds are 0 there); say so rather than rely on it.
        b.ue(1); // disable_deblocking_filter_idc

        for my in 0..self.mbh {
            for mx in 0..self.mbw {
                b.ue(25); // mb_type: I_PCM
                b.align(); // pcm_alignment_zero_bit
                for row in 0..16 {
                    let i = (my * 16 + row) * pw + mx * 16;
                    b.bytes(&y[i..i + 16]);
                }
                for plane in [&cb, &cr] {
                    for row in 0..8 {
                        let i = (my * 8 + row) * (pw / 2) + mx * 8;
                        b.bytes(&plane[i..i + 8]);
                    }
                }
            }
        }
        let slice = nal(0x65, &b.trailing());
        let mut out = (slice.len() as u32).to_be_bytes().to_vec();
        out.extend_from_slice(&slice);
        out
    }
}

/// A NAL unit: its header byte, then the payload with emulation prevention.
///
/// A decoder finds unit boundaries by the byte pattern `00 00 01`, so the
/// payload must never contain it. Wherever two zero bytes are followed by a
/// byte of 3 or less, a `03` is inserted between them, and the decoder takes
/// it out again. MP4 frames its units by length and does not need the
/// boundaries, but the escaping is part of the NAL syntax and is required
/// all the same.
fn nal(header: u8, rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 64 + 1);
    out.push(header);
    let mut zeros = 0;
    for &b in rbsp {
        if zeros == 2 && b <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    out
}

// ---------------------------------------------------------------------------
// FLAC
// ---------------------------------------------------------------------------

/// The sound as an MP4 track of FLAC frames, `block` samples each (the last
/// may be shorter), 16 bits a sample.
fn flac_track(a: &Audio, block: usize) -> Track {
    assert!((1..=8).contains(&a.channels), "FLAC holds 1 to 8 channels");
    assert!(block <= 65535);
    let n = a.frames();
    let mut samples = Vec::new();
    let mut durations = Vec::new();
    for (i, start) in (0..n).step_by(block).enumerate() {
        let len = block.min(n - start);
        samples.push(flac_frame(&a.samples[start * a.channels..(start + len) * a.channels], a.channels, i as u64, a.rate));
        durations.push(len as u32);
    }
    Track {
        kind: Kind::Audio { channels: a.channels, rate: a.rate, dfla: dfla(a, block) },
        timescale: a.rate,
        samples,
        durations,
    }
}

/// The `dfLa` box body: FLAC's `STREAMINFO` metadata block, which MP4 carries
/// where a FLAC file would have it after its `fLaC` marker.
fn dfla(a: &Audio, block: usize) -> Vec<u8> {
    let mut b = Bits::new();
    b.bits(0, 32); // version and flags
    b.bit(true); // last metadata block
    b.bits(0, 7); // type 0: STREAMINFO
    b.bits(34, 24); // its length
    b.bits(block as u64, 16); // minimum block size (the last block is exempt)
    b.bits(block as u64, 16); // maximum block size
    b.bits(0, 24); // minimum frame size: unknown
    b.bits(0, 24); // maximum frame size: unknown
    b.bits(a.rate as u64, 20);
    b.bits(a.channels as u64 - 1, 3);
    b.bits(15, 5); // bits per sample, minus one
    b.bits(a.frames() as u64, 36);
    b.bits(0, 64); // MD5 of the samples: zero, meaning not given
    b.bits(0, 64);
    b.out
}

/// One FLAC frame, every channel stored `VERBATIM`.
fn flac_frame(samples: &[f32], channels: usize, number: u64, rate: u32) -> Vec<u8> {
    let len = samples.len() / channels;
    let mut b = Bits::new();
    b.bits(0b11_1111_1111_1110, 14); // sync code
    b.bit(false); // reserved
    b.bit(false); // blocking strategy: fixed, so the frame number is counted
    b.bits(0b0111, 4); // block size: in 16 bits after the frame number
    // The rate if it has a code of its own, otherwise "as in STREAMINFO".
    let code = match rate {
        88200 => 1,
        176400 => 2,
        192000 => 3,
        8000 => 4,
        16000 => 5,
        22050 => 6,
        24000 => 7,
        32000 => 8,
        44100 => 9,
        48000 => 10,
        96000 => 11,
        _ => 0,
    };
    b.bits(code, 4);
    b.bits(channels as u64 - 1, 4); // each channel on its own
    b.bits(0b100, 3); // 16 bits a sample
    b.bit(false); // reserved
    b.bytes(&utf8_number(number));
    b.bits(len as u64 - 1, 16);
    let crc = crc8(&b.out);
    b.bits(crc as u64, 8);

    for c in 0..channels {
        b.bits(0b0000_0010, 8); // padding bit, type 000001 (VERBATIM), no wasted bits
        for i in 0..len {
            b.bits(pcm16(samples[i * channels + c]) as u16 as u64, 16);
        }
    }
    let crc = crc16(&b.out);
    b.bits(crc as u64, 16);
    b.out
}

/// A frame number the way FLAC writes it: in UTF-8's variable-length form,
/// extended to 36 bits.
fn utf8_number(n: u64) -> Vec<u8> {
    if n < 0x80 {
        return vec![n as u8];
    }
    // Continuation bytes carry six bits each; the lead byte carries what is
    // left after its run of ones and a zero.
    let extra = (1..=6).find(|&k| n < 1u64 << (6 * k + 6 - k)).unwrap();
    let mut out = vec![(0xff00u16 >> (extra + 1)) as u8 | (n >> (6 * extra)) as u8];
    for k in (0..extra).rev() {
        out.push(0x80 | (n >> (6 * k) & 0x3f) as u8);
    }
    out
}

/// FLAC's CRC-8 over a frame header: polynomial `x⁸ + x² + x + 1`, not
/// reflected, starting from 0.
fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            crc = (crc << 1) ^ (0x07 & (crc >> 7).wrapping_neg());
        }
    }
    crc
}

/// FLAC's CRC-16 over a whole frame: polynomial `0x8005`, not reflected,
/// starting from 0.
fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            crc = (crc << 1) ^ (0x8005 & (crc >> 15).wrapping_neg());
        }
    }
    crc
}

// ---------------------------------------------------------------------------
// MP4
// ---------------------------------------------------------------------------

struct Track {
    kind: Kind,
    /// Units a second that `durations` count in.
    timescale: u32,
    samples: Vec<Vec<u8>>,
    durations: Vec<u32>,
}

enum Kind {
    Video { width: usize, height: usize, avcc: Vec<u8> },
    Audio { channels: usize, rate: u32, dfla: Vec<u8> },
}

/// The file: `ftyp`, then `moov` (every track's index), then `mdat` (every
/// sample). Samples are interleaved one from each track in turn, each its
/// own chunk, so the file plays from the front as it arrives.
fn mp4(tracks: &[Track]) -> Vec<u8> {
    let mut order = Vec::new();
    for i in 0..tracks.iter().map(|t| t.samples.len()).max().unwrap_or(0) {
        for (t, track) in tracks.iter().enumerate() {
            if i < track.samples.len() {
                order.push((t, i));
            }
        }
    }
    let data: u64 = order.iter().map(|&(t, i)| tracks[t].samples[i].len() as u64).sum();

    let mut ftyp = Vec::new();
    boxed(&mut ftyp, b"ftyp", |b| {
        b.extend_from_slice(b"isom");
        b.extend_from_slice(&0x200u32.to_be_bytes());
        b.extend_from_slice(b"isomiso2avc1mp41");
    });
    // An mdat over 4 GiB needs the 64-bit size form, and its offsets the
    // 64-bit chunk table.
    let large = data + 16 > u32::MAX as u64;
    let head = if large { 16 } else { 8 };

    // The chunk offsets depend on the size of the moov that holds them, and
    // that size does not depend on their values: write it once to measure it
    // and once more to fill it in.
    let first = moov(tracks, &order, 0, large).len() as u64;
    let start = ftyp.len() as u64 + first + head;
    let moov = moov(tracks, &order, start, large);
    assert_eq!(moov.len() as u64, first);

    let mut out = Vec::with_capacity((start + data) as usize);
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&moov);
    if large {
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(b"mdat");
        out.extend_from_slice(&(data + 16).to_be_bytes());
    } else {
        out.extend_from_slice(&(data as u32 + 8).to_be_bytes());
        out.extend_from_slice(b"mdat");
    }
    for &(t, i) in &order {
        out.extend_from_slice(&tracks[t].samples[i]);
    }
    out
}

/// A box: 32-bit size, four-letter type, body.
fn boxed(out: &mut Vec<u8>, kind: &[u8; 4], body: impl FnOnce(&mut Vec<u8>)) {
    let start = out.len();
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(kind);
    body(out);
    let size = (out.len() - start) as u32;
    out[start..start + 4].copy_from_slice(&size.to_be_bytes());
}

/// A "full" box: a box whose body starts with a version byte and 24 bits of
/// flags.
fn full(out: &mut Vec<u8>, kind: &[u8; 4], version_flags: u32, body: impl FnOnce(&mut Vec<u8>)) {
    boxed(out, kind, |b| {
        b.extend_from_slice(&version_flags.to_be_bytes());
        body(b);
    });
}

fn u16s(b: &mut Vec<u8>, vs: &[u16]) {
    for v in vs {
        b.extend_from_slice(&v.to_be_bytes());
    }
}

fn u32s(b: &mut Vec<u8>, vs: &[u32]) {
    for v in vs {
        b.extend_from_slice(&v.to_be_bytes());
    }
}

/// The identity transform, in the 16.16 and 2.30 fixed point MP4 uses.
const MATRIX: [u32; 9] = [0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x4000_0000];

/// The movie's own clock, which `mvhd` and `tkhd` durations count in.
const MOVIE_TIMESCALE: u32 = 1000;

fn moov(tracks: &[Track], order: &[(usize, usize)], start: u64, large: bool) -> Vec<u8> {
    // Where each sample lands in the file, per track.
    let mut offsets = vec![Vec::new(); tracks.len()];
    let mut at = start;
    for &(t, i) in order {
        offsets[t].push(at);
        at += tracks[t].samples[i].len() as u64;
    }
    let seconds = |t: &Track| t.durations.iter().map(|&d| d as u64).sum::<u64>() as f64 / t.timescale as f64;
    let movie = |t: &Track| (seconds(t) * MOVIE_TIMESCALE as f64).round() as u32;

    let mut out = Vec::new();
    boxed(&mut out, b"moov", |b| {
        full(b, b"mvhd", 0, |b| {
            u32s(b, &[0, 0, MOVIE_TIMESCALE, tracks.iter().map(movie).max().unwrap_or(0)]);
            u32s(b, &[0x10000]); // rate 1.0
            u16s(b, &[0x100, 0]); // volume 1.0, reserved
            u32s(b, &[0, 0]);
            u32s(b, &MATRIX);
            u32s(b, &[0; 6]);
            u32s(b, &[tracks.len() as u32 + 1]); // next_track_ID
        });
        for (n, t) in tracks.iter().enumerate() {
            boxed(b, b"trak", |b| trak(b, t, n as u32 + 1, movie(t), &offsets[n], large));
        }
    });
    out
}

fn trak(b: &mut Vec<u8>, t: &Track, id: u32, movie_duration: u32, offsets: &[u64], large: bool) {
    let (w, h, volume) = match t.kind {
        Kind::Video { width, height, .. } => (width as u32, height as u32, 0),
        Kind::Audio { .. } => (0, 0, 0x100),
    };
    // Flags 3: the track is enabled and part of the presentation.
    full(b, b"tkhd", 3, |b| {
        u32s(b, &[0, 0, id, 0, movie_duration, 0, 0]);
        u16s(b, &[0, 0, volume, 0]); // layer, alternate group, volume, reserved
        u32s(b, &MATRIX);
        u32s(b, &[w << 16, h << 16]);
    });
    boxed(b, b"mdia", |b| {
        full(b, b"mdhd", 0, |b| {
            u32s(b, &[0, 0, t.timescale, t.durations.iter().sum()]);
            // Language "und", three 5-bit letters offset from 0x60.
            u16s(b, &[0x55c4, 0]);
        });
        let (handler, name): (&[u8; 4], &[u8]) = match t.kind {
            Kind::Video { .. } => (b"vide", b"VideoHandler\0"),
            Kind::Audio { .. } => (b"soun", b"SoundHandler\0"),
        };
        full(b, b"hdlr", 0, |b| {
            u32s(b, &[0]);
            b.extend_from_slice(handler);
            u32s(b, &[0, 0, 0]);
            b.extend_from_slice(name);
        });
        boxed(b, b"minf", |b| {
            match t.kind {
                Kind::Video { .. } => full(b, b"vmhd", 1, |b| u16s(b, &[0, 0, 0, 0])),
                Kind::Audio { .. } => full(b, b"smhd", 0, |b| u16s(b, &[0, 0])),
            }
            boxed(b, b"dinf", |b| {
                full(b, b"dref", 0, |b| {
                    u32s(b, &[1]);
                    // Flag 1: the data is in this file.
                    full(b, b"url ", 1, |_| {});
                });
            });
            boxed(b, b"stbl", |b| stbl(b, t, offsets, large));
        });
    });
}

/// The sample table: what the samples are, how long each lasts, how big each
/// is and where each is.
fn stbl(b: &mut Vec<u8>, t: &Track, offsets: &[u64], large: bool) {
    full(b, b"stsd", 0, |b| {
        u32s(b, &[1]);
        match &t.kind {
            Kind::Video { width, height, avcc } => boxed(b, b"avc1", |b| {
                u16s(b, &[0, 0, 0, 1]); // reserved, data_reference_index 1
                u16s(b, &[0, 0]);
                u32s(b, &[0, 0, 0]);
                u16s(b, &[*width as u16, *height as u16]);
                u32s(b, &[0x0048_0000, 0x0048_0000, 0]); // 72 dpi, reserved
                u16s(b, &[1]); // one frame per sample
                let mut name = [0u8; 32];
                name[0] = 4;
                name[1..5].copy_from_slice(b"kvad");
                b.extend_from_slice(&name);
                u16s(b, &[0x18, 0xffff]); // 24-bit colour, no colour table
                boxed(b, b"avcC", |b| b.extend_from_slice(avcc));
                // The colour space again, at the container level, for players
                // that read it there rather than from the stream.
                boxed(b, b"colr", |b| {
                    b.extend_from_slice(b"nclx");
                    u16s(b, &[1, 1, 1]);
                    b.push(0); // limited range
                });
                boxed(b, b"pasp", |b| u32s(b, &[1, 1]));
            }),
            Kind::Audio { channels, rate, dfla } => boxed(b, b"fLaC", |b| {
                u16s(b, &[0, 0, 0, 1]);
                u32s(b, &[0, 0]);
                u16s(b, &[*channels as u16, 16, 0, 0]);
                u32s(b, &[rate << 16]);
                boxed(b, b"dfLa", |b| b.extend_from_slice(dfla));
            }),
        }
    });

    // Durations, as runs of equal ones.
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &d in &t.durations {
        match runs.last_mut() {
            Some((count, delta)) if *delta == d => *count += 1,
            _ => runs.push((1, d)),
        }
    }
    full(b, b"stts", 0, |b| {
        u32s(b, &[runs.len() as u32]);
        for (count, delta) in runs {
            u32s(b, &[count, delta]);
        }
    });
    // No stss: its absence says every sample is a sync sample, which for
    // all-IDR video and for FLAC is true.
    full(b, b"stsc", 0, |b| u32s(b, &[1, 1, 1, 1])); // one sample per chunk
    full(b, b"stsz", 0, |b| {
        u32s(b, &[0, t.samples.len() as u32]);
        for s in &t.samples {
            u32s(b, &[s.len() as u32]);
        }
    });
    if large {
        full(b, b"co64", 0, |b| {
            u32s(b, &[offsets.len() as u32]);
            for &o in offsets {
                b.extend_from_slice(&o.to_be_bytes());
            }
        });
    } else {
        full(b, b"stco", 0, |b| {
            u32s(b, &[offsets.len() as u32]);
            for &o in offsets {
                u32s(b, &[o as u32]);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(f: impl FnOnce(&mut Bits)) -> String {
        let mut b = Bits::new();
        f(&mut b);
        let n = b.out.len() * 8 - if b.used == 0 { 0 } else { 8 - b.used as usize };
        b.out.iter().map(|x| format!("{x:08b}")).collect::<String>()[..n].to_string()
    }

    #[test]
    fn exp_golomb_codes_are_the_ones_the_standard_tabulates() {
        assert_eq!(bits(|b| b.ue(0)), "1");
        assert_eq!(bits(|b| b.ue(1)), "010");
        assert_eq!(bits(|b| b.ue(2)), "011");
        assert_eq!(bits(|b| b.ue(3)), "00100");
        assert_eq!(bits(|b| b.ue(25)), "000011010"); // I_PCM
        assert_eq!(bits(|b| b.se(0)), "1");
        assert_eq!(bits(|b| b.se(1)), "010");
        assert_eq!(bits(|b| b.se(-1)), "011");
        assert_eq!(bits(|b| b.se(-2)), "00101");
    }

    /// Undo [`nal`]'s escaping, as a decoder does.
    fn unescape(payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut zeros = 0;
        for &b in payload {
            if zeros == 2 && b == 3 {
                zeros = 0;
                continue;
            }
            out.push(b);
            zeros = if b == 0 { zeros + 1 } else { 0 };
        }
        out
    }

    #[test]
    fn emulation_prevention_escapes_every_start_code_and_undoes_cleanly() {
        assert_eq!(nal(0x65, &[0, 0, 1]), [0x65, 0, 0, 3, 1]);
        assert_eq!(nal(0x65, &[0, 0, 0, 0]), [0x65, 0, 0, 3, 0, 0]);
        assert_eq!(nal(0x65, &[0, 0, 4]), [0x65, 0, 0, 4]);
        // Bytes drawn mostly from 0..4, where escapes are densest.
        let raw: Vec<u8> = (0..10_000u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 29) as u8).collect();
        let escaped = nal(0x65, &raw);
        for w in escaped[1..].windows(3) {
            assert!(!(w[0] == 0 && w[1] == 0 && w[2] <= 2), "a start code survived: {w:?}");
        }
        assert_eq!(unescape(&escaped[1..]), raw);
    }

    #[test]
    fn colours_land_where_bt709_limited_range_puts_them() {
        let px = |rgb: [u8; 3]| {
            let (y, cb, cr) = yuv420(&rgb.repeat(4), 2, 2, 2, 2);
            (y[0], cb[0], cr[0])
        };
        assert_eq!(px([0, 0, 0]), (16, 128, 128));
        assert_eq!(px([255, 255, 255]), (235, 128, 128));
        // Pure red: Y = 16 + 219 × 0.2126, Cr at its maximum of 240.
        assert_eq!(px([255, 0, 0]), (63, 102, 240));
        assert_eq!(px([0, 0, 255]), (32, 240, 118));
    }

    #[test]
    fn the_level_is_the_smallest_that_fits_the_uncompressed_rate() {
        // 113 Mbit/s: over level 4.2's 50, under 5.0's 135.
        assert_eq!(H264::new(768, 512, 24).level(), 50);
        // 453 Mbit/s: over 5.2 and 6.0's 240, under 6.1's 480.
        assert_eq!(H264::new(1536, 1024, 24).level(), 61);
        assert_eq!(H264::new(512, 320, 24).level(), 41);
    }

    #[test]
    fn flac_checksums_and_frame_numbers_match_their_definitions() {
        // CRC-8/SMBUS and CRC-16/UMTS are these polynomials with these
        // settings, and these are their catalogued check values.
        assert_eq!(crc8(b"123456789"), 0xF4);
        assert_eq!(crc16(b"123456789"), 0xFEE8);
        assert_eq!(utf8_number(0), [0]);
        assert_eq!(utf8_number(0x7f), [0x7f]);
        // The same bytes UTF-8 uses for U+0080, U+07FF and U+0800.
        assert_eq!(utf8_number(0x80), [0xc2, 0x80]);
        assert_eq!(utf8_number(0x7ff), [0xdf, 0xbf]);
        assert_eq!(utf8_number(0x800), [0xe0, 0xa0, 0x80]);
        assert_eq!(utf8_number((1 << 36) - 1), [0xfe, 0xbf, 0xbf, 0xbf, 0xbf, 0xbf, 0xbf]);
    }

    #[test]
    fn a_flac_frame_carries_its_samples_verbatim_under_valid_checksums() {
        let samples = [0.0, 1.0, -1.0, 0.5, 2.0, -0.25];
        let f = flac_frame(&samples, 2, 300, 48000);
        assert_eq!(&f[..2], &[0xff, 0xf8]);
        assert_eq!(f[2], 0b0111_1010, "16-bit block size, 48 kHz");
        assert_eq!(f[3], 0b0001_1000, "two channels, 16 bits");
        assert_eq!(&f[4..6], &utf8_number(300)[..]);
        assert_eq!(u16::from_be_bytes([f[6], f[7]]), 2, "block size minus one");
        assert_eq!(crc8(&f[..8]), f[8]);
        let n = f.len();
        assert_eq!(crc16(&f[..n - 2]), u16::from_be_bytes([f[n - 2], f[n - 1]]));
        // Each channel: a VERBATIM subframe header, then its three samples.
        let read = |at: usize| i16::from_be_bytes([f[at], f[at + 1]]);
        assert_eq!(f[9], 0b10);
        assert_eq!([read(10), read(12), read(14)], [0, -32767, 32767]);
        assert_eq!(f[16], 0b10);
        assert_eq!([read(17), read(19), read(21)], [32767, 16384, -8192]);
    }

    #[test]
    fn a_wav_is_its_header_and_the_samples_as_16_bit() {
        let wav = Audio { rate: 48000, channels: 2, samples: vec![0.0, 1.0, -1.0, 0.5] }.wav();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()) as usize, wav.len() - 8);
        assert_eq!(&wav[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes(wav[28..32].try_into().unwrap()), 48000 * 4, "byte rate");
        let s: Vec<i16> = wav[44..].chunks(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(s, [0, 32767, -32767, 16384]);
    }

    /// The boxes directly inside `data`, as (type, body).
    fn boxes(data: &[u8]) -> Vec<(String, &[u8])> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < data.len() {
            let size = u32::from_be_bytes(data[i..i + 4].try_into().unwrap()) as usize;
            out.push((String::from_utf8_lossy(&data[i + 4..i + 8]).to_string(), &data[i + 8..i + size]));
            i += size;
        }
        out
    }

    fn find<'a>(data: &'a [u8], path: &[&str]) -> &'a [u8] {
        let (first, rest) = path.split_first().unwrap();
        let body = boxes(data).into_iter().find(|(k, _)| k == first).unwrap_or_else(|| panic!("no {first}")).1;
        if rest.is_empty() { body } else { find(body, rest) }
    }

    #[test]
    fn an_mp4_indexes_every_frame_where_it_is_and_the_pcm_inside_is_the_picture() {
        // 34×18 is two macroblocks by two, cropped: the padding and cropping
        // are exercised as well as the layout.
        let (w, h, n) = (34, 18, 3);
        let rgb: Vec<u8> = (0..w * h * 3 * n).map(|i| (i * 7 % 256) as u8).collect();
        let video = Video { width: w, height: h, fps: 24, rgb: rgb.clone() };
        let audio = Audio { rate: 48000, channels: 2, samples: vec![0.1; 2 * 6000] };
        let mp4 = video.mp4(Some(&audio));

        let top: Vec<String> = boxes(&mp4).into_iter().map(|b| b.0).collect();
        assert_eq!(top, ["ftyp", "moov", "mdat"], "the index comes before the data");

        let traks: Vec<&[u8]> = boxes(find(&mp4, &["moov"])).into_iter().filter(|b| b.0 == "trak").map(|b| b.1).collect();
        assert_eq!(traks.len(), 2);
        let stbl = find(traks[0], &["mdia", "minf", "stbl"]);
        let stsz = find(stbl, &["stsz"]);
        let stco = find(stbl, &["stco"]);
        assert_eq!(u32::from_be_bytes(stsz[8..12].try_into().unwrap()), n as u32);
        assert_eq!(u32::from_be_bytes(stco[4..8].try_into().unwrap()), n as u32);

        let (pw, ph) = (48, 32);
        for f in 0..n {
            let off = u32::from_be_bytes(stco[8 + 4 * f..12 + 4 * f].try_into().unwrap()) as usize;
            let size = u32::from_be_bytes(stsz[12 + 4 * f..16 + 4 * f].try_into().unwrap()) as usize;
            let len = u32::from_be_bytes(mp4[off..off + 4].try_into().unwrap()) as usize;
            assert_eq!(len + 4, size, "the sample is exactly one length-prefixed unit");
            assert_eq!(mp4[off + 4], 0x65, "an IDR slice");
            let rbsp = unescape(&mp4[off + 5..off + 4 + len]);

            // Read the slice back: its header, then six I_PCM macroblocks.
            let mut at = 0usize; // in bits
            let bit = |at: &mut usize| {
                let b = rbsp[*at / 8] >> (7 - *at % 8) & 1;
                *at += 1;
                b as u32
            };
            let ue = |at: &mut usize| {
                let mut zeros = 0;
                while bit(at) == 0 {
                    zeros += 1;
                }
                (0..zeros).fold(1u32, |v, _| v << 1 | bit(at)) - 1
            };
            assert_eq!([ue(&mut at), ue(&mut at), ue(&mut at)], [0, 7, 0]);
            at += 4; // frame_num
            assert_eq!(ue(&mut at), (f % 2) as u32, "idr_pic_id alternates");
            at += 2;
            assert_eq!([ue(&mut at), ue(&mut at)], [0, 1]); // qp delta 0 is se "1"; deblocking off

            let (y, cb, cr) = yuv420(&rgb[f * w * h * 3..(f + 1) * w * h * 3], w, h, pw, ph);
            for my in 0..ph / 16 {
                for mx in 0..pw / 16 {
                    assert_eq!(ue(&mut at), 25, "I_PCM");
                    at = at.div_ceil(8) * 8;
                    let mut take = |k: usize| {
                        let s = &rbsp[at / 8..at / 8 + k];
                        at += 8 * k;
                        s.to_vec()
                    };
                    for row in 0..16 {
                        let i = (my * 16 + row) * pw + mx * 16;
                        assert_eq!(take(16), &y[i..i + 16]);
                    }
                    for plane in [&cb, &cr] {
                        for row in 0..8 {
                            let i = (my * 8 + row) * (pw / 2) + mx * 8;
                            assert_eq!(take(8), &plane[i..i + 8]);
                        }
                    }
                }
            }
            assert_eq!(&rbsp[at / 8..], &[0x80], "then the trailing bits, and nothing else");
        }

        // The sound: 6000 samples in blocks of 2000, one per video frame.
        let stbl = find(traks[1], &["mdia", "minf", "stbl"]);
        let stts = find(stbl, &["stts"]);
        assert_eq!(&stts[4..16], &[0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0x07, 0xd0]);
    }
}
