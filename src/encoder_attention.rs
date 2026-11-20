//! Bidirectional ModernBERT attention with bounded softmax scratch space.

use crate::encoder::cast;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::ops::Range;

const KEY_BLOCK: usize = 512;
const HEAD: usize = 64;
#[cfg(target_arch = "x86_64")]
const QUANTIZED_BF16_MIN_KEYS: usize = 512;

pub(crate) fn attend<const BF16: bool>(
    qkv: &[Vec<f32>],
    queries: Range<usize>,
    radius: Option<usize>,
    left_padding: usize,
    hidden: usize,
) -> Vec<Vec<f32>> {
    let mut output = vec![vec![0.0; hidden]; queries.len()];
    let first = queries.start;
    #[cfg(target_arch = "x86_64")]
    if queries.len() >= 24 && std::arch::is_x86_feature_detected!("avx512bf16") {
        if BF16 || qkv.len() >= QUANTIZED_BF16_MIN_KEYS {
            x86::attend::<true, BF16>(qkv, &mut output, first, radius, left_padding, hidden);
        } else {
            x86::attend::<false, false>(qkv, &mut output, first, radius, left_padding, hidden);
        }
        return output;
    }
    attend_scalar::<BF16>(qkv, &mut output, first, radius, left_padding, hidden);
    output
}

fn attend_scalar<const BF16: bool>(
    qkv: &[Vec<f32>],
    output: &mut [Vec<f32>],
    first: usize,
    radius: Option<usize>,
    left_padding: usize,
    hidden: usize,
) {
    let run = |(i, row): (usize, &mut Vec<f32>)| {
        let p = i + first;
        let keys = key_range(p..p + 1, radius, qkv.len());
        for (head, dst) in row.chunks_exact_mut(HEAD).enumerate() {
            let start = head * HEAD;
            let channels = start..start + HEAD;
            let query = &qkv[p][channels.clone()];
            let mut scores = [0.0; KEY_BLOCK];
            let mut max = f32::NEG_INFINITY;
            let mut sum = 0.0;
            let mut k = keys.start;
            while k < keys.end {
                // Align to the reference's padded KV blocks. The pad
                // positions have no probability mass and need no storage.
                let end = (k + KEY_BLOCK - ((left_padding + k) % KEY_BLOCK)).min(keys.end);
                let block = &qkv[k..end];
                let scores = &mut scores[..block.len()];
                let mut new_max = max;
                for (score, row) in scores.iter_mut().zip(block) {
                    let key = &row[hidden + channels.start..hidden + channels.end];
                    *score = dot(query, key) * 0.125;
                    new_max = new_max.max(*score);
                }
                let rescale = (max - new_max).exp();
                sum *= rescale;
                for out in dst.iter_mut() {
                    *out *= rescale;
                }
                let mut weighted = [0.0; HEAD];
                for (&score, row) in scores.iter().zip(block) {
                    let prob = (score - new_max).exp();
                    sum += prob;
                    let prob = cast::<BF16>(prob);
                    let value = &row[2 * hidden + channels.start..2 * hidden + channels.end];
                    for (acc, &v) in weighted.iter_mut().zip(value) {
                        *acc += v * prob;
                    }
                }
                for (out, acc) in dst.iter_mut().zip(weighted) {
                    *out += acc;
                }
                max = new_max;
                k = end;
            }
            let norm = sum.recip();
            for out in dst {
                *out = cast::<BF16>(*out * norm);
            }
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    if output.len() >= 16 {
        output.par_iter_mut().enumerate().for_each(run);
        return;
    }
    output.iter_mut().enumerate().for_each(run);
}

fn key_range(queries: Range<usize>, radius: Option<usize>, len: usize) -> Range<usize> {
    radius.map_or(0..len, |r| {
        queries.start.saturating_sub(r)..queries.end.saturating_add(r).min(len)
    })
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.0; 8];
    for (a, b) in a.as_chunks::<8>().0.iter().zip(b.as_chunks::<8>().0) {
        for (i, sum) in sums.iter_mut().enumerate() {
            *sum += a[i] * b[i];
        }
    }
    sums.into_iter().sum()
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn tiled_attention_matches_scalar_across_a_padded_boundary() {
        if !std::arch::is_x86_feature_detected!("avx512bf16") {
            return;
        }
        let hidden = 2 * HEAD;
        let qkv: Vec<Vec<f32>> = (0..533)
            .map(|i| {
                (0..3 * hidden)
                    .map(|j| (((i * 13 + j * 17 + i * j) % 51) as f32 - 25.0) / 32.0)
                    .collect()
            })
            .collect();
        let first = 405;
        let mut scalar = vec![vec![0.0; hidden]; 79];
        let mut tiled = scalar.clone();
        for radius in [None, Some(64)] {
            scalar.iter_mut().for_each(|row| row.fill(0.0));
            attend_scalar::<true>(&qkv, &mut scalar, first, radius, 71, hidden);
            x86::attend::<true, true>(&qkv, &mut tiled, first, radius, 71, hidden);
            for (a, b) in scalar.iter().flatten().zip(tiled.iter().flatten()) {
                assert!((a - b).abs() <= 0.001, "scalar {a}, tiled {b}");
            }
            scalar.iter_mut().for_each(|row| row.fill(0.0));
            attend_scalar::<false>(&qkv, &mut scalar, first, radius, 71, hidden);
            x86::attend::<false, false>(&qkv, &mut tiled, first, radius, 71, hidden);
            for (a, b) in scalar.iter().flatten().zip(tiled.iter().flatten()) {
                assert!((a - b).abs() <= 1e-6, "scalar {a}, tiled {b}");
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{cast, key_range, HEAD, KEY_BLOCK};
    use crate::attention::{
        exp_sum, pack_pair, pack_probs, product, score_max, KeyBlock, LANES, QUERIES,
    };
    use rayon::prelude::*;
    use std::ops::Range;

    const QUERY_BLOCK: usize = 8 * QUERIES;

    pub(super) fn attend<const BF16: bool, const ROUND_OUTPUT: bool>(
        qkv: &[Vec<f32>],
        output: &mut [Vec<f32>],
        first: usize,
        radius: Option<usize>,
        left_padding: usize,
        hidden: usize,
    ) {
        let first_end = KEY_BLOCK - left_padding % KEY_BLOCK;
        let starts: Vec<_> = std::iter::once(0)
            .chain((first_end..qkv.len()).step_by(KEY_BLOCK))
            .collect();
        let packed: Vec<Vec<KeyBlock>> = (0..hidden / HEAD)
            .into_par_iter()
            .map(|head| {
                let col = head * HEAD;
                starts
                    .iter()
                    .enumerate()
                    .map(|(i, &start)| {
                        let end = starts.get(i + 1).copied().unwrap_or(qkv.len());
                        KeyBlock::pack_slices::<BF16, HEAD>(
                            end - start,
                            |k| &qkv[start + k][hidden + col..hidden + col + HEAD],
                            |k| &qkv[start + k][2 * hidden + col..2 * hidden + col + HEAD],
                        )
                    })
                    .collect()
            })
            .collect();

        output
            .par_chunks_mut(QUERY_BLOCK)
            .enumerate()
            .for_each(|(chunk, dst)| {
                let first = first + chunk * QUERY_BLOCK;
                let mut scratch = Scratch {
                    scores: [[0.0; KEY_BLOCK]; QUERIES],
                    probs: [[0; QUERIES]; KEY_BLOCK],
                };
                for (head, blocks) in packed.iter().enumerate() {
                    let channels = head * HEAD..(head + 1) * HEAD;
                    for (tile, rows) in dst.chunks_mut(QUERIES).enumerate() {
                        let lo = first + tile * QUERIES;
                        let queries = lo..lo + rows.len();
                        let visible = key_range(queries.clone(), radius, qkv.len());
                        let first_block = starts.partition_point(|&p| p <= visible.start) - 1;
                        let mut batch =
                            QueryTile::new::<BF16>(&qkv[queries.clone()], channels.clone());
                        for (i, &offset) in starts.iter().enumerate().skip(first_block) {
                            if offset >= visible.end {
                                break;
                            }
                            // SAFETY: The caller checked AVX-512 BF16 support.
                            // KeyBlock pads its packed rows to complete dot tiles.
                            unsafe {
                                batch.update::<BF16>(
                                    &blocks[i],
                                    offset,
                                    queries.clone(),
                                    &visible,
                                    radius,
                                    &mut scratch,
                                )
                            }
                        }
                        for (i, row) in rows.iter_mut().enumerate() {
                            let norm = batch.sums[i].recip();
                            for (dst, &value) in
                                row[channels.clone()].iter_mut().zip(&batch.output[i])
                            {
                                *dst = cast::<ROUND_OUTPUT>(value * norm);
                            }
                        }
                    }
                }
            });
    }

    struct Scratch {
        scores: [[f32; KEY_BLOCK]; QUERIES],
        probs: [[u32; QUERIES]; KEY_BLOCK],
    }

    struct QueryTile {
        packed: [[u32; QUERIES]; HEAD],
        output: [[f32; HEAD]; QUERIES],
        maxima: [f32; QUERIES],
        sums: [f32; QUERIES],
    }

    impl QueryTile {
        fn new<const BF16: bool>(rows: &[Vec<f32>], channels: Range<usize>) -> Self {
            let stride = if BF16 { 2 } else { 1 };
            let mut packed = [[0; QUERIES]; HEAD];
            for (i, row) in rows.iter().enumerate() {
                for (d, values) in row[channels.clone()].chunks_exact(stride).enumerate() {
                    packed[d][i] = if BF16 {
                        pack_pair(values[0], values[1])
                    } else {
                        values[0].to_bits()
                    };
                }
            }
            Self {
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
            offset: usize,
            queries: Range<usize>,
            visible: &Range<usize>,
            radius: Option<usize>,
            scratch: &mut Scratch,
        ) {
            let stride = if BF16 { 2 } else { 1 };
            let first_tile = visible.start.saturating_sub(offset) / LANES;
            let end_tile = (visible.end - offset).min(block.count).div_ceil(LANES);
            let inner = HEAD / stride;
            for tile in first_tile..end_tile {
                let k = &block.keys[tile * inner..(tile + 1) * inner];
                let dots = product::<BF16>(&self.packed[..inner], k);
                let start = tile * LANES;
                for (row, dot) in scratch.scores.iter_mut().zip(dots) {
                    row[start..start + LANES].copy_from_slice(&dot);
                }
            }
            let first_key = first_tile * LANES;
            let end_key = (end_tile * LANES).min(block.count);
            for (i, pos) in queries.enumerate() {
                let range = key_range(pos..pos + 1, radius, offset + end_key);
                let start = range.start.saturating_sub(offset).clamp(first_key, end_key);
                let end = range.end.saturating_sub(offset).max(start);
                let row = &mut scratch.scores[i];
                row[first_key..start].fill(0.0);
                row[end..end_key].fill(0.0);
                if start < end {
                    let scores = &mut row[start..end];
                    let old_max = self.maxima[i];
                    let max = old_max.max(score_max(scores) * 0.125);
                    self.maxima[i] = max;
                    let rescale = (old_max - max).exp();
                    self.sums[i] = self.sums[i] * rescale + exp_sum(scores, max, 0.125);
                    for out in &mut self.output[i] {
                        *out *= rescale;
                    }
                }
                pack_probs::<BF16>(&row[first_key..end_key], &mut scratch.probs, i);
            }

            let inner = block.count.div_ceil(stride);
            let start = first_key / stride;
            let end = end_key.div_ceil(stride);
            for (tile, values) in block.values.chunks_exact(inner).enumerate() {
                let dots = product::<BF16>(&scratch.probs[..end - start], &values[start..end]);
                let col = tile * LANES;
                for (row, dot) in self.output.iter_mut().zip(dots) {
                    for (out, value) in row[col..col + LANES].iter_mut().zip(dot) {
                        *out += value;
                    }
                }
            }
        }
    }
}
