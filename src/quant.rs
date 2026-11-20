//! Tiled Q4, Q6, and Q8 projections with a VNNI kernel and a scalar fallback.

use half::f16;

pub(crate) const BLOCK_COLS: usize = 32;
pub(crate) const TILE_ROWS: usize = 16;
const SCALE_BYTES: usize = TILE_ROWS * 2;
pub(crate) const Q4_CODEBOOK: [i8; 16] = [
    -127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127,
];

#[derive(Clone, Copy)]
pub(crate) enum TileQuant {
    Q4,
    Q4Codebook,
    Q6,
    Q8,
}

impl TileQuant {
    pub(crate) fn from_dtype(dtype: &str) -> Option<Self> {
        match dtype {
            "q4x16" => Some(Self::Q4),
            "q4nx16" => Some(Self::Q4Codebook),
            "q6x16" => Some(Self::Q6),
            "q8x16" => Some(Self::Q8),
            _ => None,
        }
    }

    pub(crate) fn block_bytes(self) -> usize {
        SCALE_BYTES
            + match self {
                Self::Q4 | Self::Q4Codebook => 4 * 64,
                Self::Q6 => 6 * 64,
                Self::Q8 => 8 * 64,
            }
    }
}

#[inline]
pub(crate) fn tiled_weight(block: &[u8], row: usize, col: usize, quant: TileQuant) -> f32 {
    let scale = f16::from_le_bytes([block[row * 2], block[row * 2 + 1]]).to_f32();
    let value = match quant {
        TileQuant::Q4 | TileQuant::Q4Codebook => {
            let packed = block[SCALE_BYTES + (col / 8) * 64 + row * 4 + col % 4];
            let nibble = if col % 8 < 4 {
                packed & 15
            } else {
                packed >> 4
            };
            match quant {
                TileQuant::Q4Codebook => i32::from(Q4_CODEBOOK[nibble as usize]),
                _ => i32::from(nibble) - 8,
            }
        }
        TileQuant::Q6 => {
            let part = SCALE_BYTES + col / 8 * 96;
            let half = col % 8 / 4;
            let low = (block[part + row * 4 + col % 4] >> (half * 4)) & 15;
            let high = (block[part + 64 + half * 16 + row] >> (2 * (col % 4))) & 3;
            i32::from(low | (high << 4)) - 32
        }
        TileQuant::Q8 => i32::from(block[SCALE_BYTES + (col / 4) * 64 + row * 4 + col % 4]) - 128,
    };
    scale * value as f32
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use rayon::prelude::*;
    use std::arch::x86_64::*;

    const TOKEN_TILE: usize = 12;

    struct Activations {
        values: [[i32; 8]; TOKEN_TILE],
        scale: [f32; TOKEN_TILE],
        sum: [i32; TOKEN_TILE],
    }

    impl Activations {
        #[target_feature(enable = "avx512f,avx512bw")]
        unsafe fn new(tokens: &[Vec<f32>], group: usize) -> Self {
            let mut block = Self {
                values: [[0; 8]; TOKEN_TILE],
                scale: [0.0; TOKEN_TILE],
                sum: [0; TOKEN_TILE],
            };
            let magnitude = _mm512_set1_epi32(0x7fff_ffff);
            for (t, token) in tokens.iter().enumerate() {
                let start = group * BLOCK_COLS;
                let values = &token[start..(start + BLOCK_COLS).min(token.len())];
                let mut tail = [0.0; BLOCK_COLS];
                let values = if values.len() == BLOCK_COLS {
                    values
                } else {
                    tail[..values.len()].copy_from_slice(values);
                    &tail
                };
                let low = _mm512_loadu_ps(values.as_ptr());
                let high = _mm512_loadu_ps(values.as_ptr().add(16));
                let abs_low =
                    _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(low), magnitude));
                let abs_high =
                    _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(high), magnitude));
                let max = _mm512_reduce_max_ps(_mm512_max_ps(abs_low, abs_high));
                if max == 0.0 {
                    continue;
                }
                let inverse = _mm512_set1_ps(127.0 / max);
                const ROUND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
                let q_low = _mm512_cvt_roundps_epi32::<ROUND>(_mm512_mul_ps(low, inverse));
                let q_high = _mm512_cvt_roundps_epi32::<ROUND>(_mm512_mul_ps(high, inverse));
                let packed = block.values[t].as_mut_ptr().cast::<__m128i>();
                _mm_storeu_si128(packed, _mm512_cvtepi32_epi8(q_low));
                _mm_storeu_si128(packed.add(1), _mm512_cvtepi32_epi8(q_high));
                block.scale[t] = max / 127.0;
                block.sum[t] = _mm512_reduce_add_epi32(_mm512_add_epi32(q_low, q_high));
            }
            block
        }
    }

    pub(crate) fn matmul(
        quant: TileQuant,
        weights: &[u8],
        rows: usize,
        cols: usize,
        input: &[Vec<f32>],
    ) -> Option<Vec<Vec<f32>>> {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vnni")
        {
            return None;
        }
        Some(match quant {
            TileQuant::Q4 => matmul_layout::<4, false>(weights, rows, cols, input),
            TileQuant::Q4Codebook => matmul_layout::<4, true>(weights, rows, cols, input),
            TileQuant::Q6 => matmul_layout::<6, false>(weights, rows, cols, input),
            TileQuant::Q8 => matmul_layout::<8, false>(weights, rows, cols, input),
        })
    }

    fn matmul_layout<const BITS: usize, const CODEBOOK: bool>(
        weights: &[u8],
        rows: usize,
        cols: usize,
        input: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let groups = cols.div_ceil(BLOCK_COLS);
        let activations: Vec<_> = input
            .par_chunks(TOKEN_TILE)
            .map(|tokens| {
                (0..groups)
                    // SAFETY: matmul checked the CPU features. Groups are
                    // wholly inside the input or padded on the final block.
                    .map(|g| unsafe { Activations::new(tokens, g) })
                    .collect::<Vec<_>>()
            })
            .collect();
        let tiles: Vec<_> = weights
            .par_chunks_exact(groups * (SCALE_BYTES + BITS * 64))
            .map(|weight_tile| {
                let mut result = vec![0.0; input.len() * TILE_ROWS];
                for (acts, output) in activations
                    .iter()
                    .zip(result.chunks_mut(TOKEN_TILE * TILE_ROWS))
                {
                    // SAFETY: the CPU features were checked above. The input contains a
                    // full 16-row tile per group, and output holds 16 rows per real token.
                    unsafe {
                        match output.len() / TILE_ROWS {
                            1 => multiply_tile::<BITS, 1, CODEBOOK>(weight_tile, acts, output),
                            2 => multiply_tile::<BITS, 2, CODEBOOK>(weight_tile, acts, output),
                            3..=4 => multiply_tile::<BITS, 4, CODEBOOK>(weight_tile, acts, output),
                            5..=8 => multiply_tile::<BITS, 8, CODEBOOK>(weight_tile, acts, output),
                            _ => multiply_tile::<BITS, TOKEN_TILE, CODEBOOK>(
                                weight_tile,
                                acts,
                                output,
                            ),
                        }
                    };
                }
                result
            })
            .collect();
        (0..input.len())
            .into_par_iter()
            .map(|token| {
                let start = token * TILE_ROWS;
                let mut output = Vec::with_capacity(rows);
                for tile in &tiles {
                    output.extend_from_slice(&tile[start..start + TILE_ROWS]);
                }
                output
            })
            .collect()
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn multiply_tile<const BITS: usize, const M: usize, const CODEBOOK: bool>(
        weights: &[u8],
        activations: &[Activations],
        out: &mut [f32],
    ) {
        let mut acc = [_mm512_setzero_ps(); M];
        let mask = _mm512_set1_epi32(0x0f0f0f0f);
        // The VNNI dot takes unsigned weights and signed activations.
        // Add 128 to every signed codebook value by flipping its sign bit.
        let codebook = _mm512_xor_si512(
            _mm512_broadcast_i32x4(_mm_loadu_si128(Q4_CODEBOOK.as_ptr().cast())),
            _mm512_set1_epi8(-128),
        );
        for (g, act) in activations.iter().enumerate() {
            let ptr = weights.as_ptr().add(g * (SCALE_BYTES + BITS * 64));
            let scales = _mm512_cvtph_ps(_mm256_loadu_si256(ptr.cast()));
            // Undo the bias on the unsigned weights once per group.
            let bias = if CODEBOOK { 128 } else { 1 << (BITS - 1) };
            let mut dot: [__m512i; M] =
                std::array::from_fn(|t| _mm512_set1_epi32(-bias * act.sum[t]));
            for part in 0..4 {
                let (first, second) = if BITS < 8 {
                    let start = ptr.add(SCALE_BYTES + part * (BITS * 16));
                    let packed = _mm512_loadu_si512(start.cast());
                    let low_first = _mm512_and_si512(packed, mask);
                    let low_second = _mm512_and_si512(_mm512_srli_epi32(packed, 4), mask);
                    if BITS == 6 {
                        (
                            _mm512_or_si512(low_first, q6_high_bits(start.add(64))),
                            _mm512_or_si512(low_second, q6_high_bits(start.add(80))),
                        )
                    } else if CODEBOOK {
                        (
                            _mm512_shuffle_epi8(codebook, low_first),
                            _mm512_shuffle_epi8(codebook, low_second),
                        )
                    } else {
                        (low_first, low_second)
                    }
                } else {
                    let start = ptr.add(SCALE_BYTES + part * 128);
                    (
                        _mm512_loadu_si512(start.cast()),
                        _mm512_loadu_si512(start.add(64).cast()),
                    )
                };
                for (t, sum) in dot.iter_mut().enumerate() {
                    *sum = _mm512_dpbusd_epi32(
                        *sum,
                        first,
                        _mm512_set1_epi32(act.values[t][2 * part]),
                    );
                    *sum = _mm512_dpbusd_epi32(
                        *sum,
                        second,
                        _mm512_set1_epi32(act.values[t][2 * part + 1]),
                    );
                }
            }
            for t in 0..M {
                let scale = _mm512_mul_ps(scales, _mm512_set1_ps(act.scale[t]));
                acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot[t]), scale, acc[t]);
            }
        }
        for (t, result) in acc.into_iter().take(out.len() / TILE_ROWS).enumerate() {
            _mm512_storeu_ps(out.as_mut_ptr().add(t * TILE_ROWS), result);
        }
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn q6_high_bits(bytes: *const u8) -> __m512i {
        let packed = _mm512_cvtepu8_epi32(_mm_loadu_si128(bytes.cast()));
        #[cfg(target_feature = "avx512vbmi")]
        {
            // Extract the four two-bit fields from each row's byte.
            let shifts = _mm512_set1_epi64(0x2624_2220_0604_0200);
            let pairs = _mm512_multishift_epi64_epi8(shifts, packed);
            _mm512_slli_epi32::<4>(_mm512_and_si512(pairs, _mm512_set1_epi32(0x0303_0303)))
        }
        #[cfg(not(target_feature = "avx512vbmi"))]
        {
            let a = _mm512_slli_epi32::<4>(_mm512_and_si512(packed, _mm512_set1_epi32(0x03)));
            let b = _mm512_slli_epi32::<10>(_mm512_and_si512(packed, _mm512_set1_epi32(0x0c)));
            let c = _mm512_slli_epi32::<16>(_mm512_and_si512(packed, _mm512_set1_epi32(0x30)));
            let d = _mm512_slli_epi32::<22>(_mm512_and_si512(packed, _mm512_set1_epi32(0xc0)));
            _mm512_or_si512(_mm512_or_si512(a, b), _mm512_or_si512(c, d))
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) use x86::matmul;
