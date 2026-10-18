// A QR code encoder, for the pairing link.
//
// The CLI draws the same link with the `qrcode` crate; the web UI has no such
// package, and the whole of what it needs — byte mode, one error-correction
// level, every version — is a page of arithmetic (ISO/IEC 18004). So it is
// written out here rather than added as a dependency.
//
// Level L, as the CLI uses: a pairing link is long (a node id, a knock, the
// relay, a token and a voucher), and the code is read off a screen a hand's
// width away, not off a scuffed label, so the smaller code is the better one.
//
// `encode(text)` gives a square of booleans, true for a dark module, with no
// quiet zone: the caller draws the four light modules of margin.

// Per version 1–40, at level L: error-correction codewords in each block, and
// how many blocks. Table 9 of the standard; the `qrcode` crate's agree.
const ECC_PER_BLOCK = [
  7, 10, 15, 20, 26, 18, 20, 24, 30, 18, 20, 24, 26, 30, 22, 24, 28, 30, 28, 28,
  28, 28, 30, 30, 26, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
];
const BLOCKS = [
  1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 7, 8,
  8, 9, 9, 10, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 24, 25,
];
// Level L's two bits in the format information. (M is 00, Q 11, H 10.)
const LEVEL_BITS = 0b01;

/** Every module a version has, less the fixed patterns: what data and error correction fill. */
function rawModules(version) {
  let n = (16 * version + 128) * version + 64;
  if (version >= 2) {
    const align = Math.floor(version / 7) + 2;
    n -= (25 * align - 10) * align - 55;
    if (version >= 7) n -= 36; // the two version-information blocks
  }
  return n;
}

const dataCodewords = (v) => Math.floor(rawModules(v) / 8) - ECC_PER_BLOCK[v - 1] * BLOCKS[v - 1];

/** Centres of the alignment patterns along one axis. */
function alignmentPositions(version) {
  if (version === 1) return [];
  const count = Math.floor(version / 7) + 2;
  const size = version * 4 + 17;
  // Evenly spaced back from the far edge, in an even step; version 32 is the
  // one the standard spaces by hand.
  const step = version === 32 ? 26 : Math.ceil((version * 4 + 4) / (count * 2 - 2)) * 2;
  const at = [6];
  for (let p = size - 7; at.length < count; p -= step) at.splice(1, 0, p);
  return at;
}

// Arithmetic in GF(256), modulo x⁸ + x⁴ + x³ + x² + 1 (0x11D).
function gfMul(x, y) {
  let z = 0;
  for (let i = 7; i >= 0; i--) {
    z = (z << 1) ^ ((z >>> 7) * 0x11d);
    z ^= ((y >>> i) & 1) * x;
  }
  return z;
}

/** The Reed–Solomon generator of a degree, (x − 2⁰)(x − 2¹)…, highest coefficient dropped. */
function generator(degree) {
  const g = new Array(degree).fill(0);
  g[degree - 1] = 1;
  let root = 1;
  for (let i = 0; i < degree; i++) {
    for (let j = 0; j < g.length; j++) {
      g[j] = gfMul(g[j], root);
      if (j + 1 < g.length) g[j] ^= g[j + 1];
    }
    root = gfMul(root, 0x02);
  }
  return g;
}

/** The error-correction codewords: the remainder of the data divided by the generator. */
function remainder(data, g) {
  const r = new Array(g.length).fill(0);
  for (const b of data) {
    const factor = b ^ r.shift();
    r.push(0);
    g.forEach((c, i) => (r[i] ^= gfMul(c, factor)));
  }
  return r;
}

/** The text as codewords: byte mode, its length, the bytes, then padding to fill the version. */
function codewords(bytes, version) {
  const bits = [];
  const put = (value, n) => {
    for (let i = n - 1; i >= 0; i--) bits.push((value >>> i) & 1);
  };
  put(0b0100, 4); // byte mode
  put(bytes.length, version <= 9 ? 8 : 16);
  bytes.forEach((b) => put(b, 8));
  const capacity = dataCodewords(version) * 8;
  put(0, Math.min(4, capacity - bits.length)); // the terminator, as much as fits
  put(0, (8 - (bits.length % 8)) % 8);
  const out = [];
  for (let i = 0; i < bits.length; i += 8) out.push(bits.slice(i, i + 8).reduce((a, b) => (a << 1) | b, 0));
  for (let pad = 0xec; out.length < capacity / 8; pad ^= 0xec ^ 0x11) out.push(pad);
  return out;
}

/** Splits the data into blocks, adds each one's error correction, and interleaves them. */
function interleave(data, version) {
  const blocks = BLOCKS[version - 1];
  const ecc = ECC_PER_BLOCK[version - 1];
  const raw = Math.floor(rawModules(version) / 8);
  // Some blocks are one codeword shorter than the rest, and come first.
  const short = blocks - (raw % blocks);
  const shortLength = Math.floor(raw / blocks);
  const g = generator(ecc);
  const all = [];
  for (let i = 0, k = 0; i < blocks; i++) {
    const d = data.slice(k, k + shortLength - ecc + (i < short ? 0 : 1));
    k += d.length;
    const e = remainder(d, g);
    if (i < short) d.push(null); // a gap, so the columns line up
    all.push(d.concat(e));
  }
  const out = [];
  for (let i = 0; i < all[0].length; i++) for (const b of all) if (b[i] !== null) out.push(b[i]);
  return out;
}

// The eight masks, by column x and row y.
const MASKS = [
  (x, y) => (x + y) % 2 === 0,
  (x, y) => y % 2 === 0,
  (x) => x % 3 === 0,
  (x, y) => (x + y) % 3 === 0,
  (x, y) => (Math.floor(x / 3) + Math.floor(y / 2)) % 2 === 0,
  (x, y) => ((x * y) % 2) + ((x * y) % 3) === 0,
  (x, y) => (((x * y) % 2) + ((x * y) % 3)) % 2 === 0,
  (x, y) => (((x + y) % 2) + ((x * y) % 3)) % 2 === 0,
];

/** The standard's penalty for a finished symbol: lower reads more reliably. */
function penalty(m) {
  const size = m.length;
  let score = 0;
  const lines = [];
  for (let i = 0; i < size; i++) {
    lines.push(m[i]);
    lines.push(m.map((row) => row[i]));
  }
  // Runs of five or more of one colour, in rows and columns.
  for (const line of lines) {
    let run = 1;
    for (let i = 1; i <= size; i++) {
      if (i < size && line[i] === line[i - 1]) run++;
      else {
        if (run >= 5) score += run - 2;
        run = 1;
      }
    }
  }
  // 2×2 squares of one colour.
  for (let y = 0; y + 1 < size; y++)
    for (let x = 0; x + 1 < size; x++) {
      const c = m[y][x];
      if (c === m[y][x + 1] && c === m[y + 1][x] && c === m[y + 1][x + 1]) score += 3;
    }
  // Anything that looks like a finder: dark-light-dark×3-light-dark beside four light.
  const finder = [1, 0, 1, 1, 1, 0, 1];
  for (const line of lines)
    for (let i = 0; i + 7 <= size; i++) {
      if (!finder.every((f, j) => +line[i + j] === f)) continue;
      const light = (from) => [0, 1, 2, 3].every((j) => !line[from + j]);
      if ((i >= 4 && light(i - 4)) || (i + 11 <= size && light(i + 7))) score += 40;
    }
  // Balance: ten for every five percent the dark modules are off half.
  const dark = m.flat().filter(Boolean).length;
  score += Math.floor(Math.abs(dark * 20 - size * size * 10) / (size * size)) * 10;
  return score;
}

/**
 * @param {string} text
 * @returns {boolean[][]} rows of modules, true for dark
 */
export function encode(text) {
  const bytes = [...new TextEncoder().encode(text)];
  let version = 1;
  while (4 + (version <= 9 ? 8 : 16) + bytes.length * 8 > dataCodewords(version) * 8) {
    if (++version > 40) throw new Error("This is too long for a QR code.");
  }
  const size = version * 4 + 17;
  const m = Array.from({ length: size }, () => new Array(size).fill(false));
  const fixed = Array.from({ length: size }, () => new Array(size).fill(false));
  const set = (x, y, dark) => {
    m[y][x] = dark;
    fixed[y][x] = true;
  };

  // Timing lines along row 6 and column 6.
  for (let i = 0; i < size; i++) {
    set(6, i, i % 2 === 0);
    set(i, 6, i % 2 === 0);
  }
  // Finders in three corners, each with its light separator.
  for (const [cx, cy] of [[3, 3], [size - 4, 3], [3, size - 4]])
    for (let dy = -4; dy <= 4; dy++)
      for (let dx = -4; dx <= 4; dx++) {
        const x = cx + dx, y = cy + dy;
        const d = Math.max(Math.abs(dx), Math.abs(dy));
        if (x >= 0 && x < size && y >= 0 && y < size) set(x, y, d !== 2 && d !== 4);
      }
  // Alignment patterns, except where a finder already is.
  const at = alignmentPositions(version);
  const last = at.length - 1;
  at.forEach((cx, i) =>
    at.forEach((cy, j) => {
      if ((i === 0 && j === 0) || (i === 0 && j === last) || (i === last && j === 0)) return;
      for (let dy = -2; dy <= 2; dy++)
        for (let dx = -2; dx <= 2; dx++) set(cx + dx, cy + dy, Math.max(Math.abs(dx), Math.abs(dy)) !== 1);
    }),
  );
  // The format information, fifteen bits drawn twice beside the finders.
  const drawFormat = (mask) => {
    const data = (LEVEL_BITS << 3) | mask;
    let r = data;
    for (let i = 0; i < 10; i++) r = (r << 1) ^ ((r >>> 9) * 0x537); // BCH(15, 5)
    const bits = ((data << 10) | r) ^ 0x5412;
    const bit = (i) => ((bits >>> i) & 1) === 1;
    for (let i = 0; i <= 5; i++) set(8, i, bit(i));
    set(8, 7, bit(6));
    set(8, 8, bit(7));
    set(7, 8, bit(8));
    for (let i = 9; i < 15; i++) set(14 - i, 8, bit(i));
    for (let i = 0; i < 8; i++) set(size - 1 - i, 8, bit(i));
    for (let i = 8; i < 15; i++) set(8, size - 15 + i, bit(i));
    set(8, size - 8, true); // the one module that is always dark
  };
  drawFormat(0); // reserves its modules; redrawn once the mask is chosen
  // The version, eighteen bits in two blocks, from version 7 on.
  if (version >= 7) {
    let r = version;
    for (let i = 0; i < 12; i++) r = (r << 1) ^ ((r >>> 11) * 0x1f25); // BCH(18, 6)
    const bits = (version << 12) | r;
    for (let i = 0; i < 18; i++) {
      const dark = ((bits >>> i) & 1) === 1;
      const a = size - 11 + (i % 3), b = Math.floor(i / 3);
      set(a, b, dark);
      set(b, a, dark);
    }
  }

  // The codewords, two columns at a time from the right, zigzagging up and
  // down and stepping over column 6, the vertical timing line.
  const all = interleave(codewords(bytes, version), version);
  let i = 0;
  for (let right = size - 1; right >= 1; right -= 2) {
    if (right === 6) right = 5;
    const upward = ((right + 1) & 2) === 0;
    for (let v = 0; v < size; v++)
      for (let j = 0; j < 2; j++) {
        const x = right - j;
        const y = upward ? size - 1 - v : v;
        if (fixed[y][x] || i >= all.length * 8) continue;
        m[y][x] = ((all[i >>> 3] >>> (7 - (i & 7))) & 1) === 1;
        i++;
      }
  }

  // Every mask, and keep the one the penalty likes best.
  const flip = (mask) => {
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) if (!fixed[y][x] && MASKS[mask](x, y)) m[y][x] = !m[y][x];
  };
  let best = 0, lowest = Infinity;
  for (let mask = 0; mask < 8; mask++) {
    flip(mask);
    drawFormat(mask);
    const p = penalty(m);
    if (p < lowest) [best, lowest] = [mask, p];
    flip(mask); // XOR again undoes it
  }
  flip(best);
  drawFormat(best);
  return m;
}
