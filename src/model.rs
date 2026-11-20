//! Reference-oriented Qwen3.5/GLiDE-2B CPU forward pass.
//! Full attention is intentionally NONCAUSAL; Gated DeltaNet remains causal.
//! No LM head, autoregressive decode, or full vocabulary logits.
use crate::{bf16, delta, profile::Profile, weights::Weights};
use anyhow::{bail, Context, Result};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use serde::Deserialize;
use std::ops::Range;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::OnceLock;

const PROJECTION_TOKENS: usize = 4096;
const MLP_TOKENS: usize = 2048;

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    #[serde(default = "default_rotary")]
    pub partial_rotary_factor: f32,
}
fn default_rotary() -> f32 {
    0.25
}
#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub text_config: TextConfig,
}
#[derive(Clone, Debug, Deserialize)]
pub struct DecisionConfig {
    #[serde(default = "default_temp")]
    pub temperature: f32,
    #[serde(default)]
    pub attention_mode: String,
    pub model_name: Option<String>,
}
fn default_temp() -> f32 {
    1.0
}

pub struct GlideModel {
    pub cfg: TextConfig,
    pub decision: DecisionConfig,
    weights: Weights,
    bf16: bool,
}

struct BatchLayout {
    spans: Vec<Range<usize>>,
    prompt_for_token: Vec<usize>,
    max_len: usize,
}

impl BatchLayout {
    fn span_for(&self, token: usize) -> &Range<usize> {
        &self.spans[self.prompt_for_token[token]]
    }

    fn position(&self, token: usize) -> usize {
        token - self.span_for(token).start
    }
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        let z = x.exp();
        z / (1. + z)
    }
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn softplus(x: f32) -> f32 {
    if x > 20. {
        x
    } else {
        x.exp().ln_1p()
    }
}
fn l2_norm(x: &mut [f32], bf16: bool) {
    let squares = if bf16 {
        reference_square_sum(x)
    } else {
        x.iter().map(|v| v * v).sum()
    };
    let scale = (squares + 1e-6).sqrt().recip();
    for v in x {
        *v *= scale;
    }
}

// Match the reference CPU reduction order for GLiDE's 128-, 256-, and
// 2048-component norm rows. The reduction groups four eight-lane accumulators.
fn reference_square_sum(x: &[f32]) -> f32 {
    let mut sums = [0.0; 32];
    for block in x.chunks(512) {
        let mut partial = [0.0; 32];
        for row in block.as_chunks::<32>().0 {
            for (sum, v) in partial.iter_mut().zip(row) {
                *sum += v * v;
            }
        }
        for (sum, add) in sums.iter_mut().zip(partial) {
            *sum += add;
        }
    }
    (0..8)
        .map(|i| ((sums[i] + sums[i + 8]) + sums[i + 16]) + sums[i + 24])
        .sum()
}

fn bf16_silu_table() -> &'static [f32; 65536] {
    static SILU: OnceLock<Box<[f32; 65536]>> = OnceLock::new();
    SILU.get_or_init(|| {
        (0..=u16::MAX)
            .map(|bits| {
                let value = half::bf16::from_bits(bits).to_f32();
                value / (1.0 + (-value).exp())
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .try_into()
            .unwrap()
    })
}

fn rotary_table(len: usize, width: usize, fraction: f32, bf16: bool) -> Vec<Vec<(f32, f32)>> {
    // Qwen3.5 uses rotate_half on the first (head_dim * partial_rotary_factor) dims.
    // All three MRoPE position axes coincide for text-only inputs.
    let dims = ((width as f32 * fraction) as usize).min(width) & !1;
    let frequencies: Vec<_> = (0..dims / 2)
        .map(|i| {
            let exponent = (2 * i) as f32 / dims as f32;
            if bf16 {
                10_000_000.0_f32.powf(exponent).recip()
            } else {
                10_000_000.0_f32.powf(-exponent)
            }
        })
        .collect();
    (0..len)
        .map(|pos| {
            frequencies
                .iter()
                .map(|freq| {
                    let (sin, cos) = (pos as f32 * freq).sin_cos();
                    if bf16 {
                        (bf16::round(sin), bf16::round(cos))
                    } else {
                        (sin, cos)
                    }
                })
                .collect()
        })
        .collect()
}

fn ropify(v: &mut [f32], coefficients: &[(f32, f32)], bf16: bool) {
    let (a, b) = v[..2 * coefficients.len()].split_at_mut(coefficients.len());
    for ((a, b), &(sin, cos)) in a.iter_mut().zip(b).zip(coefficients) {
        let (x, y) = (*a, *b);
        if bf16 {
            *a = bf16::round(bf16::round(x * cos) - bf16::round(y * sin));
            *b = bf16::round(bf16::round(y * cos) + bf16::round(x * sin));
        } else {
            *a = x * cos - y * sin;
            *b = y * cos + x * sin;
        }
    }
}

fn attention_dot(query: &[f32], key: &[f32]) -> f32 {
    // Head dimensions are multiples of 8. Independent accumulators let the
    // compiler vectorize without reassociating an entire floating-point sum.
    let mut sums = [0.0; 8];
    for (q, k) in query.as_chunks::<8>().0.iter().zip(key.as_chunks::<8>().0) {
        for i in 0..sums.len() {
            sums[i] += q[i] * k[i];
        }
    }
    sums.into_iter().sum()
}

impl GlideModel {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(dir: &Path) -> Result<Self> {
        let cfg: ModelConfig = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let decision: DecisionConfig =
            serde_json::from_slice(&std::fs::read(dir.join("decision_config.json"))?)?;
        Self::new(cfg, decision, Weights::open(dir)?)
    }
    pub fn from_parts(
        cfg_json: &str,
        decision_json: &str,
        index_json: &str,
        weights: Vec<u8>,
    ) -> Result<Self> {
        Self::new(
            serde_json::from_str(cfg_json)?,
            serde_json::from_str(decision_json)?,
            Weights::from_bytes(index_json, weights)?,
        )
    }
    pub(crate) fn new(cfg: ModelConfig, decision: DecisionConfig, weights: Weights) -> Result<Self> {
        let c = &cfg.text_config;
        if c.hidden_size != 2048
            || c.num_hidden_layers != 24
            || c.intermediate_size != 6144
            || c.num_attention_heads != 8
            || c.num_key_value_heads != 2
            || c.head_dim != 256
            || c.linear_num_key_heads != 16
            || c.linear_num_value_heads != 16
            || c.linear_key_head_dim != 128
            || c.linear_value_head_dim != 128
            || c.linear_conv_kernel_dim != 4
        {
            bail!("model dimensions are not the supported GLiDE-2B/Qwen3.5-2B text backbone");
        }
        if decision.attention_mode != "noncausal_full_attention" {
            bail!("refusing non-GLiDE attention mode");
        }
        if decision.temperature <= 0. {
            bail!("invalid temperature");
        }
        let needed = ["embed_tokens.weight", "norm.weight", "readout.weight"];
        for name in needed {
            if !weights.has(name) {
                bail!("missing required tensor {name}");
            }
        }
        let bf16 = weights.get("readout.weight")?.dtype() == "bf16x16";
        Ok(Self {
            cfg: c.clone(),
            decision,
            weights,
            bf16,
        })
    }
    fn v(&self, key: &str) -> Result<Vec<f32>> {
        Ok(self.weights.get(key)?.vector())
    }
    fn dense(&self, name: &str, x: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let weight = self.weights.get(name)?;
        if x.len() <= PROJECTION_TOKENS {
            return weight.matmul_tokens(x);
        }
        // Keep the matrix multiplication's packing and transpose buffers
        // bounded. Each token row is independent in a projection.
        let mut output = Vec::with_capacity(x.len());
        for tile in x.chunks(PROJECTION_TOKENS) {
            output.extend(weight.matmul_tokens(tile)?);
        }
        Ok(output)
    }
    #[inline]
    fn cast(&self, value: f32) -> f32 {
        if self.bf16 {
            bf16::round(value)
        } else {
            value
        }
    }
    fn rms(&self, row: &mut [f32], weight: &[f32]) {
        let square_sum = if self.bf16 {
            reference_square_sum(row)
        } else {
            row.iter().map(|v| v * v).sum()
        };
        let scale = (square_sum / row.len() as f32 + self.cfg.rms_norm_eps)
            .sqrt()
            .recip();
        if self.bf16 {
            for (value, w) in row.iter_mut().zip(weight) {
                *value = bf16::round((*value * scale) * (1.0 + w));
            }
        } else {
            for (value, w) in row.iter_mut().zip(weight) {
                *value *= scale * (1.0 + w);
            }
        }
    }
    fn norm_all(&self, x: &mut [Vec<f32>], name: &str) -> Result<()> {
        let w = self.v(name)?;
        #[cfg(not(target_arch = "wasm32"))]
        if x.len() >= 128 {
            x.par_iter_mut().for_each(|row| self.rms(row, &w));
            return Ok(());
        }
        for row in x {
            self.rms(row, &w);
        }
        Ok(())
    }
    fn linear_delta_layer(
        &self,
        layer: usize,
        x: &[Vec<f32>],
        layout: &BatchLayout,
        profile: &mut Profile,
    ) -> Result<Vec<Vec<f32>>> {
        let p = format!("layers.{layer}.linear_attn");
        let keydim = self.cfg.linear_num_key_heads * self.cfg.linear_key_head_dim;
        let vdim = self.cfg.linear_num_value_heads * self.cfg.linear_value_head_dim;
        let kvdim = keydim * 2 + vdim;
        let kernel = self.cfg.linear_conv_kernel_dim;
        let kd = self.cfg.linear_key_head_dim;
        let vd = self.cfg.linear_value_head_dim;
        let heads = self.cfg.linear_num_value_heads;
        let group = heads / self.cfg.linear_num_key_heads;

        let start = profile.start();
        let mut qkv = self.dense(&format!("{p}.in_proj_qkv.weight"), x)?;
        let z = self.dense(&format!("{p}.in_proj_z.weight"), x)?;
        let b = self.dense(&format!("{p}.in_proj_b.weight"), x)?;
        let a = self.dense(&format!("{p}.in_proj_a.weight"), x)?;
        profile.end("delta_input_projections", start);
        let start = profile.start();
        let conv = self.v(&format!("{p}.conv1d.weight"))?;
        let alog = self.v(&format!("{p}.A_log"))?;
        let dt = self.v(&format!("{p}.dt_bias"))?;
        let normw = self.v(&format!("{p}.norm.weight"))?;
        if conv.len() != kvdim * kernel || normw.len() != vd {
            bail!("bad delta layer shapes {layer}");
        }

        let taps: Vec<Vec<_>> = (0..kernel)
            .map(|k| {
                conv.chunks_exact(kernel)
                    .map(|channel| channel[k])
                    .collect()
            })
            .collect();
        let silu_table = self.bf16.then(bf16_silu_table);
        let tile_len = x.len().div_ceil(8).clamp(1, 256);
        // Within a tile, backwards traversal keeps the earlier input rows
        // intact. Copy the three history rows needed across tile boundaries.
        let histories: Vec<_> = (0..x.len())
            .step_by(tile_len)
            .map(|start| qkv[start.saturating_sub(kernel - 1)..start].to_vec())
            .collect();
        let convolve = |(tile, (rows, history)): (usize, (&mut [Vec<f32>], &Vec<Vec<f32>>))| {
            let mut dst = vec![0.0; kvdim];
            for local in (0..rows.len()).rev() {
                dst.fill(0.0);
                let width = kernel.min(layout.position(tile * tile_len + local) + 1);
                let first = history.len() + local + 1 - width;
                let mut accumulate = |tap: usize| {
                    let index = first + tap;
                    let input = if index < history.len() {
                        &history[index]
                    } else {
                        &rows[index - history.len()]
                    };
                    for ((out, value), weight) in
                        dst.iter_mut().zip(input).zip(&taps[kernel - width + tap])
                    {
                        *out += value * weight;
                    }
                };
                if self.bf16 {
                    (0..width).for_each(&mut accumulate);
                } else {
                    (0..width).rev().for_each(&mut accumulate);
                }
                if let Some(table) = silu_table {
                    bf16::activate_silu(&mut dst, None, table);
                } else {
                    for value in &mut dst {
                        *value = silu(*value);
                    }
                }
                std::mem::swap(&mut dst, &mut rows[local]);
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        qkv.par_chunks_mut(tile_len)
            .zip(&histories)
            .enumerate()
            .for_each(convolve);
        #[cfg(target_arch = "wasm32")]
        qkv.chunks_mut(tile_len)
            .zip(&histories)
            .enumerate()
            .for_each(convolve);
        drop(histories);
        let convolved = qkv;
        profile.end("delta_depthwise_convolution", start);
        let start = profile.start();

        // Each head has its own recurrent state. Keep that state hot while
        // scanning the tokens in order, and run separate heads in parallel.
        let scan_head = |head: usize| {
            let kh = head / group;
            let mut state = vec![0.0; kd * vd];
            let mut q = vec![0.0; kd];
            let mut k = vec![0.0; kd];
            let mut delta = vec![0.0; vd];
            let mut mixed = vec![0.0; x.len() * vd];
            let query_scale = (kd as f32).sqrt();
            let decay_rate = alog[head].exp();
            for (t, (qkv, output)) in convolved.iter().zip(mixed.chunks_exact_mut(vd)).enumerate() {
                if layout.position(t) == 0 {
                    state.fill(0.0);
                }
                q.copy_from_slice(&qkv[kh * kd..(kh + 1) * kd]);
                k.copy_from_slice(&qkv[keydim + kh * kd..keydim + (kh + 1) * kd]);
                delta.copy_from_slice(&qkv[keydim * 2 + head * vd..keydim * 2 + (head + 1) * vd]);
                l2_norm(&mut q, self.bf16);
                l2_norm(&mut k, self.bf16);
                if self.bf16 {
                    let reciprocal = query_scale.recip();
                    for value in &mut q {
                        *value *= reciprocal;
                    }
                } else {
                    for value in &mut q {
                        *value /= query_scale;
                    }
                }
                let beta = self.cast(sigmoid(b[t][head]));
                let x = a[t][head] + dt[head];
                let alpha = if self.bf16 || x > 20.0 || x < -20.0 {
                    softplus(x)
                } else {
                    (1.0 + x.exp()).ln()
                };
                let decay = (-decay_rate * alpha).exp();
                delta::update(&mut state, &k, &q, &mut delta, output, decay, beta);
                if self.bf16 {
                    for out in output.iter_mut() {
                        *out = bf16::round(*out);
                    }
                }
                let squares = if self.bf16 {
                    reference_square_sum(output)
                } else {
                    output.iter().map(|v| v * v).sum()
                };
                let scale = (squares / vd as f32 + self.cfg.rms_norm_eps).sqrt().recip();
                let gate = &z[t][head * vd..(head + 1) * vd];
                if let Some(table) = silu_table {
                    for ((out, weight), gate) in output.iter_mut().zip(&normw).zip(gate) {
                        let norm = bf16::round(bf16::round(*out * scale) * weight);
                        *out = bf16::round(norm * bf16::silu(*gate, table));
                    }
                } else {
                    for ((out, weight), gate) in output.iter_mut().zip(&normw).zip(gate) {
                        *out *= scale * weight * silu(*gate);
                    }
                }
            }
            mixed
        };
        #[cfg(not(target_arch = "wasm32"))]
        let head_outputs: Vec<_> = (0..heads).into_par_iter().map(scan_head).collect();
        #[cfg(target_arch = "wasm32")]
        let head_outputs: Vec<_> = (0..heads).map(scan_head).collect();
        drop(convolved);
        drop(z);
        drop(b);
        drop(a);
        let mut mixed = vec![vec![0.0; vdim]; x.len()];
        for (head, values) in head_outputs.into_iter().enumerate() {
            for (out, row) in mixed.iter_mut().zip(values.chunks_exact(vd)) {
                out[head * vd..(head + 1) * vd].copy_from_slice(row);
            }
        }
        profile.end("delta_recurrence_and_gate", start);
        let start = profile.start();
        let output = self.dense(&format!("{p}.out_proj.weight"), &mixed)?;
        profile.end("delta_output_projection", start);
        Ok(output)
    }
    fn full_attn_layer(
        &self,
        layer: usize,
        x: &[Vec<f32>],
        layout: &BatchLayout,
        rotary: &[Vec<(f32, f32)>],
        last_only: bool,
        profile: &mut Profile,
    ) -> Result<Vec<Vec<f32>>> {
        let p = format!("layers.{layer}.self_attn");
        let hd = self.cfg.head_dim;
        let heads = self.cfg.num_attention_heads;
        let kvheads = self.cfg.num_key_value_heads;
        let last_queries;
        let query_input = if last_only {
            last_queries = layout
                .spans
                .iter()
                .map(|span| x[span.end - 1].clone())
                .collect::<Vec<_>>();
            &last_queries
        } else {
            x
        };
        let start = profile.start();
        let mut queries = self.dense(&format!("{p}.q_proj.weight"), query_input)?;
        let mut keys = self.dense(&format!("{p}.k_proj.weight"), x)?;
        let values = self.dense(&format!("{p}.v_proj.weight"), x)?;
        profile.end("attention_qkv_projections", start);
        let start = profile.start();
        let qw = self.v(&format!("{p}.q_norm.weight"))?;
        let kw = self.v(&format!("{p}.k_norm.weight"))?;
        let token_for_query = |index: usize| {
            if last_only {
                layout.spans[index].end - 1
            } else {
                index
            }
        };
        let normalize_queries = |(index, row): (usize, &mut Vec<f32>)| {
            let position = layout.position(token_for_query(index));
            for head in row.chunks_exact_mut(2 * hd) {
                let query = &mut head[..hd];
                self.rms(query, &qw);
                ropify(query, &rotary[position], self.bf16);
            }
        };
        let normalize_keys = |(t, row): (usize, &mut Vec<f32>)| {
            for key in row.chunks_exact_mut(hd) {
                self.rms(key, &kw);
                ropify(key, &rotary[layout.position(t)], self.bf16);
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            queries
                .par_iter_mut()
                .enumerate()
                .for_each(normalize_queries);
            keys.par_iter_mut().enumerate().for_each(normalize_keys);
        }
        #[cfg(target_arch = "wasm32")]
        {
            queries.iter_mut().enumerate().for_each(normalize_queries);
            keys.iter_mut().enumerate().for_each(normalize_keys);
        }

        profile.end("attention_norm_and_rope", start);
        let start = profile.start();
        let scale = (hd as f32).sqrt().recip();
        let mut out = vec![vec![0.0; heads * hd]; queries.len()];
        #[cfg(target_arch = "x86_64")]
        let tiled: Vec<_> = if last_only {
            // One query per prompt has no query tile to reuse.
            vec![false; layout.spans.len()]
        } else {
            let mut remaining = out.as_mut_slice();
            let mut tiled = Vec::with_capacity(layout.spans.len());
            for span in &layout.spans {
                let (dst, rest) = remaining.split_at_mut(span.len());
                remaining = rest;
                tiled.push(crate::attention::attend(
                    &queries[span.clone()],
                    &keys[span.clone()],
                    &values[span.clone()],
                    dst,
                    layout.max_len - span.len(),
                    self.bf16,
                ));
            }
            tiled
        };
        // Full attention sees every key in the query's own prompt.
        let attend = |(index, (query, dst)): (usize, (&Vec<f32>, &mut Vec<f32>))| {
            let token = token_for_query(index);
            #[cfg(target_arch = "x86_64")]
            if tiled[layout.prompt_for_token[token]] {
                return;
            }
            let span = layout.span_for(token);
            let prompt_keys = &keys[span.clone()];
            let prompt_values = &values[span.clone()];
            // The reference CPU SDPA accumulates FP32 sums and rounds the
            // unnormalized weights to BF16 inside each 512-key block.
            let mut scores = vec![
                0.0;
                if self.bf16 {
                    span.len().min(512)
                } else {
                    span.len()
                }
            ];
            let padding = layout.max_len - span.len();
            for (head, (qg, acc)) in query
                .chunks_exact(2 * hd)
                .zip(dst.chunks_exact_mut(hd))
                .enumerate()
            {
                let (q, gates) = qg.split_at(hd);
                let group = head / (heads / kvheads);
                let key_range = group * hd..(group + 1) * hd;
                let mut max = f32::NEG_INFINITY;
                let mut denominator = 0.0;
                let mut start = 0;
                while start < span.len() {
                    let end = if self.bf16 {
                        (start + 512 - (start + padding) % 512).min(span.len())
                    } else {
                        span.len()
                    };
                    let previous_max = max;
                    let scores = &mut scores[..end - start];
                    for (score, key) in scores.iter_mut().zip(&prompt_keys[start..end]) {
                        *score = attention_dot(q, &key[key_range.clone()]) * scale;
                        max = max.max(*score);
                    }
                    if start != 0 {
                        let rescale = (previous_max - max).exp();
                        denominator *= rescale;
                        for sum in acc.iter_mut() {
                            *sum *= rescale;
                        }
                    }
                    for (score, value) in scores.iter().zip(&prompt_values[start..end]) {
                        let prob = (*score - max).exp();
                        denominator += prob;
                        let weight = self.cast(prob);
                        for (sum, v) in acc.iter_mut().zip(&value[key_range.clone()]) {
                            *sum += weight * v;
                        }
                    }
                    start = end;
                }
                let inverse = denominator.recip();
                for (sum, gate) in acc.iter_mut().zip(gates) {
                    if self.bf16 {
                        *sum =
                            bf16::round(bf16::round(*sum * inverse) * bf16::round(sigmoid(*gate)));
                    } else {
                        *sum = (*sum / denominator) * sigmoid(*gate);
                    }
                }
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        queries
            .par_iter()
            .zip(&mut out)
            .enumerate()
            .for_each(attend);
        #[cfg(target_arch = "wasm32")]
        queries.iter().zip(&mut out).enumerate().for_each(attend);
        drop(queries);
        drop(keys);
        drop(values);
        profile.end("attention_scores_and_values", start);
        let start = profile.start();
        let output = self.dense(&format!("{p}.o_proj.weight"), &out)?;
        profile.end("attention_output_projection", start);
        Ok(output)
    }
    fn mlp(&self, layer: usize, x: &[Vec<f32>], profile: &mut Profile) -> Result<Vec<Vec<f32>>> {
        let p = format!("layers.{layer}.mlp");
        let start = profile.start();
        let gate = self.dense(&format!("{p}.gate_proj.weight"), x)?;
        let mut up = self.dense(&format!("{p}.up_proj.weight"), x)?;
        profile.end("mlp_input_projections", start);
        let start = profile.start();
        let silu_table = self.bf16.then(bf16_silu_table);
        let activate = |(up_row, gate_row): (&mut Vec<f32>, &Vec<f32>)| {
            if let Some(table) = silu_table {
                bf16::activate_silu(up_row, Some(gate_row), table);
            } else {
                for (a, b) in up_row.iter_mut().zip(gate_row) {
                    *a *= silu(*b);
                }
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        up.par_iter_mut().zip(&gate).for_each(activate);
        #[cfg(target_arch = "wasm32")]
        up.iter_mut().zip(&gate).for_each(activate);
        drop(gate);
        profile.end("mlp_gate", start);
        let start = profile.start();
        let output = self.dense(&format!("{p}.down_proj.weight"), &up)?;
        profile.end("mlp_output_projection", start);
        Ok(output)
    }
    pub fn predict_tokens(&self, ids: &[u32]) -> Result<Vec<f32>> {
        self.predict_batch(&[ids])?
            .pop()
            .context("empty inference result")
    }

    /// Return the 255 raw decision logits for each prompt, in input order.
    pub fn predict_batch(&self, prompts: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        // Protect CPU/browser hosts from huge activation allocations.
        // This is an execution limit, NOT the 40k-token model context capacity.
        #[cfg(not(target_arch = "wasm32"))]
        let limit = std::env::var("GLIDE_CPU_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2048)
            .min(40000);
        #[cfg(target_arch = "wasm32")]
        let limit = 512usize;
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        let mut profile = Profile::new();
        let start = profile.start();
        let embed = self.weights.get("embed_tokens.weight")?;
        if embed.shape() != [self.cfg.vocab_size, self.cfg.hidden_size] {
            bail!("unexpected embedding shape");
        }
        let mut h = Vec::new();
        let mut layout = BatchLayout {
            spans: Vec::with_capacity(prompts.len()),
            prompt_for_token: Vec::new(),
            max_len: 0,
        };
        for (index, ids) in prompts.iter().enumerate() {
            if ids.is_empty() || ids.len() > limit {
                bail!(
                    "prompt {index} has {} tokens; expected 1..={limit}",
                    ids.len()
                );
            }
            let start = h.len();
            for &id in *ids {
                if id as usize >= self.cfg.vocab_size {
                    bail!("token ID out of vocabulary in prompt {index}");
                }
                h.push(embed.row(id as usize));
            }
            layout.spans.push(start..h.len());
            layout.max_len = layout.max_len.max(ids.len());
            layout
                .prompt_for_token
                .extend(std::iter::repeat_n(index, ids.len()));
        }
        let rotary = rotary_table(
            layout.max_len,
            self.cfg.head_dim,
            self.cfg.partial_rotary_factor,
            self.bf16,
        );
        profile.end("embedding_and_rotary_table", start);
        let mut normalized = Vec::new();
        for i in 0..self.cfg.num_hidden_layers {
            let p = format!("layers.{i}");
            let start = profile.start();
            normalized.clone_from(&h);
            self.norm_all(&mut normalized, &format!("{p}.input_layernorm.weight"))?;
            profile.end("residual_norms", start);
            let last = i + 1 == self.cfg.num_hidden_layers;
            let attn = if i % 4 == 3 {
                self.full_attn_layer(i, &normalized, &layout, &rotary, last, &mut profile)?
            } else {
                self.linear_delta_layer(i, &normalized, &layout, &mut profile)?
            };
            let start = profile.start();
            if last {
                // The remaining operations are row-wise. Retain one row per prompt.
                h = layout
                    .spans
                    .iter()
                    .map(|span| std::mem::take(&mut h[span.end - 1]))
                    .collect();
            }
            for (a, b) in h.iter_mut().zip(attn) {
                for (v, y) in a.iter_mut().zip(b) {
                    *v = self.cast(*v + y);
                }
            }
            normalized.truncate(h.len());
            profile.end("residual_norms", start);
            let norm = format!("{p}.post_attention_layernorm.weight");
            // Feed-forward rows need no cross-token state. Consume both
            // expansion matrices and apply the residual inside the tile.
            for (state, input) in h
                .chunks_mut(MLP_TOKENS)
                .zip(normalized.chunks_mut(MLP_TOKENS))
            {
                let start = profile.start();
                input.clone_from_slice(state);
                self.norm_all(input, &norm)?;
                profile.end("residual_norms", start);
                let mlp = self.mlp(i, input, &mut profile)?;
                let start = profile.start();
                for (a, b) in state.iter_mut().zip(mlp) {
                    for (v, y) in a.iter_mut().zip(b) {
                        *v = self.cast(*v + y);
                    }
                }
                profile.end("residual_norms", start);
            }
        }
        let start = profile.start();
        self.norm_all(&mut h, "norm.weight")?;
        let readout = self.weights.get("readout.weight")?;
        if readout.shape() != [255, self.cfg.hidden_size] {
            bail!("readout must be [255, 2048]");
        }
        let mut logits = readout.matmul_tokens(&h)?;
        if self.weights.has("readout.bias") {
            let bias = self.v("readout.bias")?;
            for row in &mut logits {
                for (logit, b) in row.iter_mut().zip(&bias) {
                    *logit += b;
                }
            }
        }
        profile.end("readout", start);
        profile.report(prompts);
        Ok(logits)
    }
    pub fn probabilities(&self, ids: &[u32], count: usize) -> Result<Vec<f32>> {
        if count == 0 || count > 255 {
            bail!("number of choices must be 1..255");
        }
        self.probabilities_from_logits(&self.predict_tokens(ids)?, count)
    }

    pub fn probabilities_from_logits(&self, logits: &[f32], count: usize) -> Result<Vec<f32>> {
        probabilities(logits, count, self.decision.temperature)
    }
}

pub(crate) fn probabilities(logits: &[f32], count: usize, temperature: f32) -> Result<Vec<f32>> {
    if count == 0 || count > 255 || count > logits.len() {
        bail!("expected 1 to 255 choices with at least one logit per choice");
    }
    let mut logits = logits[..count].to_vec();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for x in &mut logits {
        *x = ((*x - max) / temperature).exp();
        sum += *x;
    }
    if !sum.is_finite() || sum == 0. {
        bail!("invalid probability normalization");
    }
    for x in &mut logits {
        *x /= sum;
    }
    Ok(logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotary_preserves_length() {
        let mut a = vec![0f32; 256];
        a[0] = 3.;
        a[32] = 4.;
        let old = a.iter().map(|x| x * x).sum::<f32>();
        ropify(&mut a, &rotary_table(68, 256, 0.25, false)[67], false);
        assert!((a.iter().map(|x| x * x).sum::<f32>() - old).abs() < 1e-4);
    }
    #[test]
    fn nonlinear_functions() {
        assert!((sigmoid(0.) - 0.5).abs() < 1e-6);
        assert!((softplus(0.) - 2f32.ln()).abs() < 1e-6);
    }
}
