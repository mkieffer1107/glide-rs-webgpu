//! Blocked noncausal attention for the native 256-wide GLiDE attention heads.

use crate::{bf16, model::sigmoid};
use rayon::prelude::*;
use std::{arch::x86_64::*, mem::transmute};

const HEAD: usize = 256;
const KEY_BLOCK: usize = 512;
const QUERIES: usize = 6;
const QUERY_BLOCK: usize = 8 * QUERIES;
const LANES: usize = 32;
const QUANTIZED_BF16_MIN_KEYS: usize = 16_384;

struct KeyBlock {
    count: usize,
    keys: Vec<[u32; LANES]>,
    values: Vec<[u32; LANES]>,
}

fn pack_pair(low: f32, high: f32) -> u32 {
    let low = half::bf16::from_f32(low).to_bits();
    let high = half::bf16::from_f32(high).to_bits();
    u32::from(low) | (u32::from(high) << 16)
}

impl KeyBlock {
    fn pack<const BF16: bool>(keys: &[Vec<f32>], values: &[Vec<f32>], group: usize) -> Self {
        let stride = if BF16 { 2 } else { 1 };
        let start = group * HEAD;
        let mut packed_keys = vec![[0; LANES]; keys.len().div_ceil(LANES) * HEAD / stride];
        for (tile, rows) in keys.chunks(LANES).enumerate() {
            let matrix = &mut packed_keys[tile * HEAD / stride..(tile + 1) * HEAD / stride];
            for (k, row) in rows.iter().enumerate() {
                for (d, channels) in row[start..start + HEAD].chunks_exact(stride).enumerate() {
                    matrix[d][k] = if BF16 {
                        pack_pair(channels[0], channels[1])
                    } else {
                        channels[0].to_bits()
                    };
                }
            }
        }

        let inner = values.len().div_ceil(stride);
        let mut packed_values = vec![[0; LANES]; inner * HEAD / LANES];
        for (i, rows) in values.chunks(stride).enumerate() {
            for tile in 0..HEAD / LANES {
                let column = start + tile * LANES;
                let packed = &mut packed_values[tile * inner + i];
                for (lane, word) in packed.iter_mut().enumerate() {
                    let c = column + lane;
                    *word = if BF16 {
                        pack_pair(rows[0][c], rows.get(1).map_or(0.0, |r| r[c]))
                    } else {
                        rows[0][c].to_bits()
                    };
                }
            }
        }
        Self {
            count: keys.len(),
            keys: packed_keys,
            values: packed_values,
        }
    }
}

pub(crate) fn attend(
    query: &[Vec<f32>],
    key: &[Vec<f32>],
    value: &[Vec<f32>],
    out: &mut [Vec<f32>],
    padding: usize,
    bf16: bool,
) -> bool {
    if key.len() < 256 || !is_x86_feature_detected!("avx512bf16") {
        return false;
    }
    if bf16 || key.len() >= QUANTIZED_BF16_MIN_KEYS {
        attend_prompt::<true>(query, key, value, out, padding, bf16);
    } else {
        attend_prompt::<false>(query, key, value, out, 0, false);
    }
    true
}

fn attend_prompt<const BF16: bool>(
    query: &[Vec<f32>],
    key: &[Vec<f32>],
    value: &[Vec<f32>],
    out: &mut [Vec<f32>],
    padding: usize,
    round_output: bool,
) {
    let groups = key[0].len() / HEAD;
    let heads = out[0].len() / HEAD;
    let first = KEY_BLOCK - padding % KEY_BLOCK;
    let starts: Vec<_> = std::iter::once(0)
        .chain((first..key.len()).step_by(KEY_BLOCK))
        .collect();
    let packed: Vec<Vec<_>> = (0..groups)
        .into_par_iter()
        .map(|group| {
            starts
                .par_iter()
                .enumerate()
                .map(|(i, &start)| {
                    let end = starts.get(i + 1).copied().unwrap_or(key.len());
                    KeyBlock::pack::<BF16>(&key[start..end], &value[start..end], group)
                })
                .collect()
        })
        .collect();
    let block_width = if key.len() >= QUANTIZED_BF16_MIN_KEYS {
        QUERY_BLOCK
    } else {
        QUERIES
    };
    query
        .par_chunks(block_width)
        .zip(out.par_chunks_mut(block_width))
        .for_each(|(q, dst)| {
            let mut tiles = Vec::with_capacity(q.len().div_ceil(QUERIES));
            let mut scores = [[0.0; KEY_BLOCK]; QUERIES];
            let mut probs = [[0; QUERIES]; KEY_BLOCK];
            for head in 0..heads {
                let blocks = &packed[head / (heads / groups)];
                tiles.clear();
                tiles.extend(q.chunks(QUERIES).map(|q| QueryTile::pack::<BF16>(q, head)));
                // Reuse each packed key/value block while it fits in cache.
                // Each query still visits its key blocks in their original order.
                for block in blocks {
                    for tile in &mut tiles {
                        // SAFETY: AVX-512 BF16 is available and packed blocks
                        // pad their tails to the fixed product dimensions.
                        unsafe { tile.update::<BF16>(block, &mut scores, &mut probs) };
                    }
                }
                for ((q, dst), tile) in q.chunks(QUERIES).zip(dst.chunks_mut(QUERIES)).zip(&tiles) {
                    tile.finish(q, dst, head, round_output);
                }
            }
        });
}

struct QueryTile {
    count: usize,
    packed: [[u32; QUERIES]; HEAD],
    output: [[f32; HEAD]; QUERIES],
    maxima: [f32; QUERIES],
    sums: [f32; QUERIES],
}

impl QueryTile {
    fn pack<const BF16: bool>(query: &[Vec<f32>], head: usize) -> Self {
        let stride = if BF16 { 2 } else { 1 };
        let mut packed = [[0; QUERIES]; HEAD];
        for (i, row) in query.iter().enumerate() {
            let query = &row[2 * head * HEAD..(2 * head + 1) * HEAD];
            for (d, channels) in query.chunks_exact(stride).enumerate() {
                packed[d][i] = if BF16 {
                    pack_pair(channels[0], channels[1])
                } else {
                    channels[0].to_bits()
                };
            }
        }
        Self {
            count: query.len(),
            packed,
            output: [[0.0; HEAD]; QUERIES],
            maxima: [f32::NEG_INFINITY; QUERIES],
            sums: [0.0; QUERIES],
        }
    }

    #[target_feature(enable = "avx512f,avx512bf16")]
    unsafe fn update<const BF16: bool>(
        &mut self,
        block: &KeyBlock,
        scores: &mut [[f32; KEY_BLOCK]; QUERIES],
        probs: &mut [[u32; QUERIES]; KEY_BLOCK],
    ) {
        let stride = if BF16 { 2 } else { 1 };
        let inner = HEAD / stride;
        for (tile, k) in block.keys.chunks_exact(inner).enumerate() {
            let dots = product::<BF16>(&self.packed[..inner], k);
            let start = tile * LANES;
            for (row, dot) in scores.iter_mut().zip(dots) {
                row[start..start + LANES].copy_from_slice(&dot);
            }
        }
        let value_inner = block.count.div_ceil(stride);
        for (i, row) in scores.iter_mut().take(self.count).enumerate() {
            let row = &mut row[..block.count];
            let old_max = self.maxima[i];
            // The model head dimension is 256, so its scale is exactly 1/16.
            let block_max = score_max(row) * (1.0 / 16.0);
            self.maxima[i] = old_max.max(block_max);
            let rescale = (old_max - self.maxima[i]).exp();
            self.sums[i] = self.sums[i] * rescale + exp_sum(row, self.maxima[i]);
            for x in &mut self.output[i] {
                *x *= rescale;
            }
            if BF16 {
                let (blocks, tail) = row.as_chunks::<16>();
                // exp_sum writes normal FP32 values or zero in its vector
                // lanes. The scalar tail may also contain denormals.
                for (block, values) in blocks.iter().enumerate() {
                    let bf16 = _mm512_cvtneps_pbh(_mm512_loadu_ps(values.as_ptr()));
                    let words = transmute::<__m256bh, [u32; 8]>(bf16);
                    for (j, word) in words.into_iter().enumerate() {
                        probs[block * 8 + j][i] = word;
                    }
                }
                for (j, values) in tail.chunks(2).enumerate() {
                    probs[blocks.len() * 8 + j][i] =
                        pack_pair(values[0], values.get(1).copied().unwrap_or(0.0));
                }
            } else {
                for (j, &value) in row.iter().enumerate() {
                    probs[j][i] = value.to_bits();
                }
            }
        }
        for (tile, v) in block.values.chunks_exact(value_inner).enumerate() {
            let dots = product::<BF16>(&probs[..value_inner], v);
            let start = tile * LANES;
            for (row, dot) in self.output.iter_mut().zip(dots) {
                for (dst, value) in row[start..start + LANES].iter_mut().zip(dot) {
                    *dst += value;
                }
            }
        }
    }

    fn finish(&self, query: &[Vec<f32>], dst: &mut [Vec<f32>], head: usize, round_output: bool) {
        for (i, (row, out)) in query.iter().zip(dst).enumerate() {
            let inv = self.sums[i].recip();
            let gates = &row[(2 * head + 1) * HEAD..(2 * head + 2) * HEAD];
            for ((dst, &v), &gate) in out[head * HEAD..(head + 1) * HEAD]
                .iter_mut()
                .zip(&self.output[i])
                .zip(gates)
            {
                *dst = if round_output {
                    bf16::round(bf16::round(v * inv) * bf16::round(sigmoid(gate)))
                } else {
                    (v * inv) * sigmoid(gate)
                };
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn product<const BF16: bool>(
    a: &[[u32; QUERIES]],
    b: &[[u32; LANES]],
) -> [[f32; LANES]; QUERIES] {
    let mut sums = [[_mm512_setzero_ps(); 2]; QUERIES];
    for (input, weights) in a.iter().zip(b) {
        let w0 = _mm512_loadu_si512(weights.as_ptr().cast());
        let w1 = _mm512_loadu_si512(weights.as_ptr().add(16).cast());
        for (sum, &word) in sums.iter_mut().zip(input) {
            let x = _mm512_set1_epi32(word as i32);
            if BF16 {
                let x = transmute::<__m512i, __m512bh>(x);
                sum[0] = _mm512_dpbf16_ps(sum[0], x, transmute::<__m512i, __m512bh>(w0));
                sum[1] = _mm512_dpbf16_ps(sum[1], x, transmute::<__m512i, __m512bh>(w1));
            } else {
                let x = _mm512_castsi512_ps(x);
                sum[0] = _mm512_fmadd_ps(x, _mm512_castsi512_ps(w0), sum[0]);
                sum[1] = _mm512_fmadd_ps(x, _mm512_castsi512_ps(w1), sum[1]);
            }
        }
    }
    let mut result = [[0.0; LANES]; QUERIES];
    for (row, sum) in result.iter_mut().zip(sums) {
        _mm512_storeu_ps(row.as_mut_ptr(), sum[0]);
        _mm512_storeu_ps(row.as_mut_ptr().add(16), sum[1]);
    }
    result
}

#[target_feature(enable = "avx512f")]
unsafe fn score_max(scores: &[f32]) -> f32 {
    let (vectors, tail) = scores.as_chunks::<16>();
    let mut max = _mm512_set1_ps(f32::NEG_INFINITY);
    for row in vectors {
        max = _mm512_max_ps(max, _mm512_loadu_ps(row.as_ptr()));
    }
    tail.iter()
        .copied()
        .fold(_mm512_reduce_max_ps(max), f32::max)
}

#[target_feature(enable = "avx512f")]
unsafe fn exp_sum(scores: &mut [f32], max: f32) -> f32 {
    let (vectors, tail) = scores.as_chunks_mut::<16>();
    let maximum = _mm512_set1_ps(max);
    let scale = _mm512_set1_ps(1.0 / 16.0);
    let mut sum = _mm512_setzero_ps();
    for row in vectors {
        let x = _mm512_sub_ps(_mm512_mul_ps(_mm512_loadu_ps(row.as_ptr()), scale), maximum);
        // Range-reduce to [-ln(2)/2, ln(2)/2] and evaluate degree-six exp.
        // Values below the FP32 normal range have negligible softmax mass.
        let n = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            _mm512_mul_ps(x, _mm512_set1_ps(std::f32::consts::LOG2_E)),
        );
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(std::f32::consts::LN_2), x);
        let mut poly = _mm512_set1_ps(1.0 / 720.0);
        for coefficient in [1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
            poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(coefficient));
        }
        let normal = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(x, _mm512_set1_ps(-87.3));
        let p = _mm512_maskz_scalef_ps(normal, poly, n);
        _mm512_storeu_ps(row.as_mut_ptr(), p);
        sum = _mm512_add_ps(sum, p);
    }
    let mut total = _mm512_reduce_add_ps(sum);
    for x in tail {
        *x = (*x * (1.0 / 16.0) - max).exp();
        total += *x;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_attention_keeps_heads_and_block_tails() {
        if !is_x86_feature_detected!("avx512bf16") {
            return;
        }
        let len = KEY_BLOCK + 27;
        let query = vec![vec![0.0; 8 * HEAD]; QUERIES + 1];
        let key = vec![vec![0.0; 2 * HEAD]; len];
        let value: Vec<_> = (0..len)
            .map(|row| {
                (0..2 * HEAD)
                    .map(|column| (row % 5 + column % 7 + 8 * (column / HEAD)) as f32)
                    .collect()
            })
            .collect();
        for bf16 in [false, true] {
            let mut output = vec![vec![0.0; 4 * HEAD]; query.len()];
            assert!(attend(&query, &key, &value, &mut output, 3, bf16));
            for row in output {
                for (column, got) in row.into_iter().enumerate() {
                    let vcol = column % HEAD + column / (2 * HEAD) * HEAD;
                    let mean = value.iter().map(|row| row[vcol]).sum::<f32>() / len as f32;
                    let expected = if bf16 { bf16::round(mean) } else { mean } * 0.5;
                    assert!((got - expected).abs() < if bf16 { 0.016 } else { 3e-6 });
                }
            }
        }
    }
}
