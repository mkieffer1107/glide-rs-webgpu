//! BF16 projections packed as sixteen output rows by two input channels.

use half::bf16;

const ROWS: usize = 16;
const PAIR_BYTES: usize = ROWS * 4;

pub(crate) fn packed_size(rows: usize, cols: usize) -> Option<usize> {
    rows.div_ceil(ROWS)
        .checked_mul(cols.div_ceil(2))?
        .checked_mul(PAIR_BYTES)
}

#[inline]
pub(crate) fn round(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

pub(crate) fn silu(value: f32, lookup: &[f32; 65536]) -> f32 {
    lookup[(value.to_bits() >> 16) as usize]
}

pub(crate) fn activate_silu(row: &mut [f32], gate: Option<&[f32]>, lookup: &[f32; 65536]) {
    #[cfg(target_arch = "x86_64")]
    if row.len() % 16 == 0 && std::arch::is_x86_feature_detected!("avx512f") {
        // SAFETY: Every vector is a complete 16-element chunk. A supplied gate
        // is the same width as the row, and all lookup indices are BF16 bits.
        unsafe {
            if let Some(gate) = gate {
                assert_eq!(row.len(), gate.len());
                x86::activate::<true, false>(row, gate, lookup);
            } else {
                x86::activate::<false, true>(row, &[], lookup);
            }
        }
        return;
    }
    if let Some(gate) = gate {
        for (up, &g) in row.iter_mut().zip(gate) {
            *up = round(*up * round(silu(g, lookup)));
        }
    } else {
        for x in row {
            *x = round(silu(round(*x), lookup));
        }
    }
}

pub(crate) fn activate_gelu_gate(row: &mut [f32], gate: &[f32], lookup: &[f32; 65536]) {
    debug_assert_eq!(row.len(), gate.len());
    #[cfg(target_arch = "x86_64")]
    if row.len() % 16 == 0 && std::arch::is_x86_feature_detected!("avx512f") {
        // SAFETY: The rows have equal widths in complete 16-value chunks.
        unsafe { x86::activate::<true, true>(row, gate, lookup) };
        return;
    }
    for (x, &g) in row.iter_mut().zip(gate) {
        *x = round(lookup[(x.to_bits() >> 16) as usize] * g);
    }
}

#[inline]
pub(crate) fn weight(data: &[u8], cols: usize, row: usize, col: usize) -> f32 {
    let pair = row / ROWS * cols.div_ceil(2) + col / 2;
    let offset = pair * PAIR_BYTES + (row % ROWS * 2 + col % 2) * 2;
    bf16::from_le_bytes([data[offset], data[offset + 1]]).to_f32()
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{PAIR_BYTES, ROWS};
    use half::bf16;
    use rayon::prelude::*;
    use std::{arch::x86_64::*, mem::transmute};

    const TOKENS: usize = 16;
    const WIDE_TOKENS: usize = 6;
    const WIDE_ROWS: usize = 64;
    // oneDNN's BF16 projection sums 1024 input channels at a time,
    // then adds the FP32 partial sums before rounding the output.
    const PAIRS_PER_BLOCK: usize = 512;

    #[target_feature(enable = "avx512f")]
    unsafe fn bf16_bits(x: __m512) -> __m512i {
        let bits = _mm512_castps_si512(x);
        let high = _mm512_srli_epi32::<16>(bits);
        let lsb = _mm512_and_si512(high, _mm512_set1_epi32(1));
        let offset = _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7fff));
        let rounded = _mm512_srli_epi32::<16>(_mm512_add_epi32(bits, offset));
        let abs = _mm512_and_si512(bits, _mm512_set1_epi32(0x7fff_ffff));
        let nan = _mm512_cmpgt_epi32_mask(abs, _mm512_set1_epi32(0x7f80_0000));
        _mm512_mask_or_epi32(rounded, nan, high, _mm512_set1_epi32(0x40))
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn activate<const GATED: bool, const ACTIVATE_INPUT: bool>(
        row: &mut [f32],
        gate: &[f32],
        lookup: &[f32; 65536],
    ) {
        for (i, chunk) in row.as_chunks_mut::<16>().0.iter_mut().enumerate() {
            let input = _mm512_loadu_ps(chunk.as_ptr());
            let gate = if GATED {
                _mm512_loadu_ps(gate.as_ptr().add(i * 16))
            } else {
                input
            };
            let index = if GATED {
                let value = if ACTIVATE_INPUT { input } else { gate };
                _mm512_srli_epi32::<16>(_mm512_castps_si512(value))
            } else {
                bf16_bits(input)
            };
            let activated = _mm512_i32gather_ps::<4>(index, lookup.as_ptr());
            let output = if GATED {
                let rounded = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(bf16_bits(activated)));
                _mm512_mul_ps(if ACTIVATE_INPUT { gate } else { input }, rounded)
            } else {
                activated
            };
            let bits = _mm512_slli_epi32::<16>(bf16_bits(output));
            _mm512_storeu_ps(chunk.as_mut_ptr(), _mm512_castsi512_ps(bits));
        }
    }

    fn pack<const M: usize>(input: &[Vec<f32>], pairs: usize) -> Vec<Vec<[u32; M]>> {
        input
            .par_chunks(M)
            .map(|tokens| {
                let mut packed = vec![[0; M]; pairs];
                for (t, vals) in tokens.iter().enumerate() {
                    for (pair, values) in packed.iter_mut().zip(vals.chunks(2)) {
                        let low = bf16::from_f32(values[0]).to_bits();
                        let high = values.get(1).map_or(0, |&x| bf16::from_f32(x).to_bits());
                        pair[t] = u32::from(low) | (u32::from(high) << 16);
                    }
                }
                packed
            })
            .collect()
    }

    fn untile<const N: usize>(
        tiles: &[Vec<f32>],
        tokens: usize,
        weight_rows: usize,
    ) -> Vec<Vec<f32>> {
        (0..tokens)
            .into_par_iter()
            .map(|token| {
                let start = token * N;
                let mut output = Vec::with_capacity(weight_rows);
                for tile in tiles {
                    let count = N.min(weight_rows - output.len());
                    output.extend_from_slice(&tile[start..start + count]);
                }
                output
            })
            .collect()
    }

    pub(crate) fn matmul(
        weights: &[u8],
        weight_rows: usize,
        cols: usize,
        input: &[Vec<f32>],
    ) -> Option<Vec<Vec<f32>>> {
        if !is_x86_feature_detected!("avx512bf16") {
            return None;
        }
        let pairs = cols.div_ceil(2);
        if weight_rows.div_ceil(ROWS) % 4 == 0 && input.len() >= 128 {
            let packed_input = pack::<WIDE_TOKENS>(input, pairs);
            let tiles: Vec<_> = weights
                .par_chunks_exact(4 * pairs * PAIR_BYTES)
                .map(|tile| {
                    let mut out = vec![0.0; input.len() * WIDE_ROWS];
                    for (a, out) in packed_input
                        .iter()
                        .zip(out.chunks_mut(WIDE_TOKENS * WIDE_ROWS))
                    {
                        // SAFETY: Each pair has four 16-row vectors. Every output
                        // row slice has 64 entries, and AVX-512 BF16 is enabled.
                        unsafe { multiply_wide(tile, a, out) };
                    }
                    out
                })
                .collect();
            return Some(untile::<WIDE_ROWS>(&tiles, input.len(), weight_rows));
        }
        let packed_input = pack::<TOKENS>(input, pairs);
        let tiles: Vec<_> = weights
            .par_chunks_exact(pairs * PAIR_BYTES)
            .map(|tile| {
                let mut out = vec![0.0; input.len() * ROWS];
                for (a, out) in packed_input.iter().zip(out.chunks_mut(TOKENS * ROWS)) {
                    // SAFETY: the CPU supports AVX-512 BF16, and each weight tile
                    // holds one vector per packed input pair.
                    unsafe {
                        match out.len() / ROWS {
                            1 => multiply::<1>(tile, a, out),
                            2 => multiply::<2>(tile, a, out),
                            3..=4 => multiply::<4>(tile, a, out),
                            5..=8 => multiply::<8>(tile, a, out),
                            9..=12 => multiply::<12>(tile, a, out),
                            _ => multiply::<TOKENS>(tile, a, out),
                        }
                    }
                }
                out
            })
            .collect();
        Some(untile::<ROWS>(&tiles, input.len(), weight_rows))
    }

    #[target_feature(enable = "avx512bf16")]
    unsafe fn multiply<const M: usize>(weights: &[u8], a: &[[u32; TOKENS]], out: &mut [f32]) {
        for (block, activations) in a.chunks(PAIRS_PER_BLOCK).enumerate() {
            let first_pair = block * PAIRS_PER_BLOCK;
            let mut sums = [_mm512_setzero_ps(); M];
            for (i, packed) in activations.iter().enumerate() {
                let b =
                    _mm512_loadu_si512(weights.as_ptr().add((first_pair + i) * PAIR_BYTES).cast());
                let b = transmute::<__m512i, __m512bh>(b);
                for (sum, &pair) in sums.iter_mut().zip(packed) {
                    let x = transmute::<__m512i, __m512bh>(_mm512_set1_epi32(pair as i32));
                    *sum = _mm512_dpbf16_ps(*sum, b, x);
                }
            }
            let last = first_pair + activations.len() == a.len();
            for (row, sum) in sums.into_iter().take(out.len() / ROWS).enumerate() {
                store_sum(out.as_mut_ptr().add(row * ROWS), sum, last);
            }
        }
    }

    #[target_feature(enable = "avx512bf16")]
    unsafe fn multiply_wide(weights: &[u8], a: &[[u32; WIDE_TOKENS]], out: &mut [f32]) {
        let stride = a.len() * PAIR_BYTES;
        for (block, activations) in a.chunks(PAIRS_PER_BLOCK).enumerate() {
            let first_pair = block * PAIRS_PER_BLOCK;
            let mut sums = [[_mm512_setzero_ps(); 4]; WIDE_TOKENS];
            for (i, packed) in activations.iter().enumerate() {
                let ptr = weights.as_ptr().add((first_pair + i) * PAIR_BYTES);
                let w0 = transmute::<__m512i, __m512bh>(_mm512_loadu_si512(ptr.cast()));
                let w1 = transmute::<__m512i, __m512bh>(_mm512_loadu_si512(ptr.add(stride).cast()));
                let w2 =
                    transmute::<__m512i, __m512bh>(_mm512_loadu_si512(ptr.add(2 * stride).cast()));
                let w3 =
                    transmute::<__m512i, __m512bh>(_mm512_loadu_si512(ptr.add(3 * stride).cast()));
                for (s, &pair) in sums.iter_mut().zip(packed) {
                    let x = transmute::<__m512i, __m512bh>(_mm512_set1_epi32(pair as i32));
                    s[0] = _mm512_dpbf16_ps(s[0], w0, x);
                    s[1] = _mm512_dpbf16_ps(s[1], w1, x);
                    s[2] = _mm512_dpbf16_ps(s[2], w2, x);
                    s[3] = _mm512_dpbf16_ps(s[3], w3, x);
                }
            }
            let last = first_pair + activations.len() == a.len();
            for (t, row) in out.chunks_exact_mut(WIDE_ROWS).enumerate() {
                for (part, value) in sums[t].into_iter().enumerate() {
                    store_sum(row.as_mut_ptr().add(part * ROWS), value, last);
                }
            }
        }
    }

    #[target_feature(enable = "avx512bf16")]
    unsafe fn store_sum(out: *mut f32, sum: __m512, last: bool) {
        let sum = _mm512_add_ps(_mm512_loadu_ps(out), sum);
        if last {
            let bf = transmute::<__m256bh, __m256i>(_mm512_cvtneps_pbh(sum));
            let bits = _mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(bf));
            _mm512_storeu_ps(out, _mm512_castsi512_ps(bits));
        } else {
            _mm512_storeu_ps(out, sum);
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) use x86::matmul;
