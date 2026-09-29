//! A greyscale PNG writer, so that what a model draws can be looked at.
//!
//! Like [`json`](crate::json), this is here because it has to be and teaches
//! nothing about networks. It is short because it never compresses: a PNG's
//! pixels go through zlib, and zlib allows "stored" blocks, which are the
//! bytes as they are with a length in front. The file is bigger than it could
//! be, and every image viewer opens it.

use std::io;
use std::path::Path;

/// Write `pixels`, one byte each, `width` to a row, as an 8-bit grey PNG.
pub fn write_grey(path: &Path, width: usize, height: usize, pixels: &[u8]) -> io::Result<()> {
    assert_eq!(pixels.len(), width * height, "{} pixels for a {width}x{height} image", pixels.len());

    // Each row starts with its filter type; 0 is "none".
    let mut raw = Vec::with_capacity(height * (width + 1));
    for row in pixels.chunks(width) {
        raw.push(0);
        raw.extend_from_slice(row);
    }

    let mut zlib = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = raw.chunks(u16::MAX as usize).collect();
    for (i, block) in blocks.iter().enumerate() {
        zlib.push((i + 1 == blocks.len()) as u8); // last block?, and type 0: stored
        let len = block.len() as u16;
        zlib.extend(len.to_le_bytes());
        zlib.extend((!len).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    zlib.extend(adler32(&raw).to_be_bytes());

    let mut header = Vec::new();
    header.extend((width as u32).to_be_bytes());
    header.extend((height as u32).to_be_bytes());
    header.extend([8, 0, 0, 0, 0]); // 8 bits, greyscale, deflate, no filter set, no interlace

    let mut file = b"\x89PNG\r\n\x1a\n".to_vec();
    chunk(&mut file, b"IHDR", &header);
    chunk(&mut file, b"IDAT", &zlib);
    chunk(&mut file, b"IEND", &[]);
    std::fs::write(path, file)
}

fn chunk(file: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    file.extend((data.len() as u32).to_be_bytes());
    file.extend_from_slice(kind);
    file.extend_from_slice(data);
    file.extend(crc32(&[kind.as_slice(), data].concat()).to_be_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 == 1 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in bytes {
        a = (a + x as u32) % 65_521;
        b = (b + a) % 65_521;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The check values every implementation of these two is tested against.
    #[test]
    fn checksums_match_their_published_check_values() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn a_large_image_is_split_into_stored_blocks() {
        let path = std::env::temp_dir().join(format!("nervus-png-{}.png", std::process::id()));
        // 300 rows of 301 bytes is more than one 65535-byte block.
        write_grey(&path, 300, 300, &vec![128; 300 * 300]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&bytes[12..16], b"IHDR");
        // Signature, three chunks of 12 bytes' framing, the header, the zlib
        // header and checksum, two blocks' framing, and the pixels.
        assert_eq!(bytes.len(), 8 + 3 * 12 + 13 + 2 + 4 + 2 * 5 + 300 * 301);
    }
}
