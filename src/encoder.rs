//! Text-only ModernBERT inference for the GLiDE-0.4B decision checkpoint.

use crate::{bf16, encoder_attention, model::DecisionConfig, profile::Profile, weights::Weights};
use anyhow::{ensure, Result};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use serde::Deserialize;
use std::{ops::Range, sync::OnceLock};

pub(crate) const ATTENTION_MODE: &str = "noncausal_bidirectional_encoder";
const EMBEDDINGS: &str = "embeddings.tok_embeddings.weight";
const MLP_TOKENS: usize = 1024;
const PROJECTION_TOKENS: usize = 2048;

#[derive(Deserialize)]
pub(crate) struct EncoderConfig {
    model_type: String,
    pub(crate) hidden_size: usize,
    pub(crate) num_hidden_layers: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    vocab_size: usize,
    norm_eps: f32,
    max_position_embeddings: usize,
    cls_token_id: u32,
    sep_token_id: u32,
    global_attn_every_n_layers: usize,
    global_rope_theta: f32,
    local_attention: usize,
    local_rope_theta: f32,
    hidden_activation: String,
    attention_bias: bool,
    mlp_bias: bool,
    norm_bias: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Pooling {
    Last,
    Cls,
}

#[derive(Deserialize)]
struct EncoderDecision {
    #[serde(flatten)]
    common: DecisionConfig,
    backbone: String,
    pooling: Pooling,
    max_len: usize,
}

pub(crate) struct EncoderModel {
    pub(crate) cfg: EncoderConfig,
    pub(crate) decision: DecisionConfig,
    pub(crate) max_len: usize,
    pooling: Pooling,
    weights: Weights,
    bf16: bool,
}

impl EncoderModel {
    pub(crate) fn new(config: &str, decision: &str, weights: Weights) -> Result<Self> {
        let cfg: EncoderConfig = serde_json::from_str(config)?;
        let mut decision: EncoderDecision = serde_json::from_str(decision)?;
        ensure!(
            cfg.model_type == "modernbert"
                && cfg.hidden_size == 1024
                && cfg.intermediate_size == 2624
                && cfg.num_hidden_layers == 28
                && cfg.num_attention_heads == 16
                && cfg.vocab_size == 50368
                && cfg.hidden_activation == "gelu"
                && !cfg.attention_bias
                && !cfg.mlp_bias
                && !cfg.norm_bias,
            "unsupported GLiDE-0.4B encoder configuration"
        );
        ensure!(
            decision.backbone == "encoder"
                && decision.common.attention_mode == ATTENTION_MODE
                && decision.common.temperature.is_finite()
                && decision.common.temperature > 0.0
                && (2..=7999.min(cfg.max_position_embeddings)).contains(&decision.max_len)
                && cfg.norm_eps.is_finite()
                && cfg.norm_eps > 0.0
                && cfg.global_attn_every_n_layers > 0
                && cfg.global_rope_theta.is_finite()
                && cfg.global_rope_theta > 0.0
                && cfg.local_rope_theta.is_finite()
                && cfg.local_rope_theta > 0.0,
            "invalid GLiDE encoder attention or decision settings"
        );

        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let check = |name: &str, shape: &[usize]| -> Result<()> {
            let tensor = weights.get(name)?;
            ensure!(
                tensor.shape() == shape,
                "invalid {name} shape, expected {shape:?}"
            );
            Ok(())
        };
        check(EMBEDDINGS, &[cfg.vocab_size, hidden])?;
        check("embeddings.norm.weight", &[hidden])?;
        check("final_norm.weight", &[hidden])?;
        check("readout.weight", &[255, hidden])?;
        for layer in 0..cfg.num_hidden_layers {
            let prefix = format!("layers.{layer}");
            if layer != 0 {
                check(&format!("{prefix}.attn_norm.weight"), &[hidden])?;
            }
            check(&format!("{prefix}.attn.Wqkv.weight"), &[3 * hidden, hidden])?;
            check(&format!("{prefix}.attn.Wo.weight"), &[hidden, hidden])?;
            check(&format!("{prefix}.mlp_norm.weight"), &[hidden])?;
            check(
                &format!("{prefix}.mlp.Wi.weight"),
                &[2 * intermediate, hidden],
            )?;
            check(&format!("{prefix}.mlp.Wo.weight"), &[hidden, intermediate])?;
        }
        let bf16 = weights.get("layers.0.attn.Wqkv.weight")?.dtype() == "bf16x16";
        decision
            .common
            .model_name
            .get_or_insert_with(|| "glide-0.4b".into());
        Ok(Self {
            cfg,
            decision: decision.common,
            max_len: decision.max_len,
            pooling: decision.pooling,
            weights,
            bf16,
        })
    }

    pub(crate) fn predict_batch(&self, prompts: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        for (i, ids) in prompts.iter().enumerate() {
            ensure!(
                (2..=self.max_len).contains(&ids.len()),
                "prompt {i} has {} tokens; expected 2..={}",
                ids.len(),
                self.max_len
            );
            ensure!(
                ids.first() == Some(&self.cfg.cls_token_id)
                    && ids.last() == Some(&self.cfg.sep_token_id),
                "prompt {i} must start with [CLS] and end with [SEP]"
            );
            ensure!(
                ids.iter().all(|id| (*id as usize) < self.cfg.vocab_size),
                "token ID out of vocabulary in prompt {i}"
            );
        }
        if self.bf16 {
            self.forward::<true>(prompts)
        } else {
            self.forward::<false>(prompts)
        }
    }

    fn forward<const BF16: bool>(&self, prompts: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        let mut profile = Profile::new();
        let start = profile.start();
        let embed = self.weights.get(EMBEDDINGS)?;
        let mut hidden: Vec<_> = prompts
            .iter()
            .flat_map(|ids| ids.iter().map(|&id| embed.row(id as usize)))
            .collect();
        self.normalize::<BF16>(&mut hidden, "embeddings.norm.weight")?;

        let mut offset = 0;
        let spans: Vec<Range<usize>> = prompts
            .iter()
            .map(|ids| {
                let start = offset;
                offset += ids.len();
                start..offset
            })
            .collect();
        let max_len = prompts.iter().map(|ids| ids.len()).max().unwrap();
        let head_dim = self.cfg.hidden_size / self.cfg.num_attention_heads;
        let global_rotary = rotary_table(max_len, head_dim, self.cfg.global_rope_theta);
        let local_rotary = if self.cfg.local_rope_theta == self.cfg.global_rope_theta {
            None
        } else {
            Some(rotary_table(max_len, head_dim, self.cfg.local_rope_theta))
        };
        profile.end("embedding_and_rotary_table", start);

        let mut normalized = Vec::new();
        for layer in 0..self.cfg.num_hidden_layers {
            let prefix = format!("layers.{layer}");
            let final_layer = layer + 1 == self.cfg.num_hidden_layers;
            let global = layer % self.cfg.global_attn_every_n_layers == 0;
            let (radius, rotary) = if global {
                (None, &global_rotary)
            } else {
                (
                    Some(self.cfg.local_attention / 2),
                    local_rotary.as_ref().unwrap_or(&global_rotary),
                )
            };

            let start = profile.start();
            let input = if layer == 0 {
                &hidden
            } else {
                normalized.clone_from(&hidden);
                self.normalize::<BF16>(&mut normalized, &format!("{prefix}.attn_norm.weight"))?;
                &normalized
            };
            profile.end("residual_norms", start);
            let start = profile.start();
            let mut qkv = self.dense(&format!("{prefix}.attn.Wqkv.weight"), input)?;
            profile.end("attention_qkv", start);

            let mut attended = Vec::with_capacity(if final_layer {
                spans.len()
            } else {
                hidden.len()
            });
            for span in &spans {
                let query = if final_layer {
                    let row = self.pool_row(span.len());
                    row..row + 1
                } else {
                    0..span.len()
                };
                let start = profile.start();
                let qkv = &mut qkv[span.clone()];
                apply_rotary::<BF16>(qkv, rotary, head_dim, self.cfg.hidden_size, &query);
                profile.end("rotary", start);
                let start = profile.start();
                attended.extend(encoder_attention::attend::<BF16>(
                    qkv,
                    query,
                    radius,
                    max_len - span.len(),
                    self.cfg.hidden_size,
                ));
                profile.end(
                    if global {
                        "global_attention"
                    } else {
                        "local_attention"
                    },
                    start,
                );
            }
            drop(qkv);
            let start = profile.start();
            let attention = self.dense(&format!("{prefix}.attn.Wo.weight"), &attended)?;
            drop(attended);
            profile.end("attention_output", start);

            let start = profile.start();
            if final_layer {
                // A position-wise output projection and MLP follow the last
                // attention. Only the row seen by the decision head is needed.
                hidden = spans
                    .iter()
                    .map(|span| std::mem::take(&mut hidden[span.start + self.pool_row(span.len())]))
                    .collect();
            }
            residual::<BF16>(&mut hidden, attention);
            normalized.truncate(hidden.len());
            normalized.resize_with(hidden.len(), Vec::new);
            profile.end("residual_norms", start);

            for (state, scratch) in hidden
                .chunks_mut(MLP_TOKENS)
                .zip(normalized.chunks_mut(MLP_TOKENS))
            {
                let start = profile.start();
                scratch.clone_from_slice(state);
                self.normalize::<BF16>(scratch, &format!("{prefix}.mlp_norm.weight"))?;
                profile.end("residual_norms", start);

                let start = profile.start();
                let mut expansion = self.dense(&format!("{prefix}.mlp.Wi.weight"), scratch)?;
                profile.end("mlp_input", start);
                let start = profile.start();
                let lookup = BF16.then(bf16_gelu_table);
                let float_lookup = (!BF16).then(float_gelu_table);
                rows_mut(&mut expansion, |row| {
                    let (input, gate) = row.split_at_mut(self.cfg.intermediate_size);
                    if let Some(table) = lookup {
                        bf16::activate_gelu_gate(input, gate, table);
                    } else if let Some(table) = float_lookup {
                        for (x, &g) in input.iter_mut().zip(gate.iter()) {
                            *x = interpolate_gelu(*x, table) * g;
                        }
                    }
                    row.truncate(self.cfg.intermediate_size);
                });
                profile.end("geglu", start);

                let start = profile.start();
                let output = self.dense(&format!("{prefix}.mlp.Wo.weight"), &expansion)?;
                profile.end("mlp_output", start);
                let start = profile.start();
                residual::<BF16>(state, output);
                profile.end("residual_norms", start);
            }
        }
        let start = profile.start();
        self.normalize::<BF16>(&mut hidden, "final_norm.weight")?;
        let logits = self.dense("readout.weight", &hidden)?;
        profile.end("readout", start);
        profile.report(prompts);
        Ok(logits)
    }

    fn pool_row(&self, len: usize) -> usize {
        match self.pooling {
            Pooling::Last => len - 2,
            Pooling::Cls => 0,
        }
    }

    fn dense(&self, name: &str, rows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let weight = self.weights.get(name)?;
        if rows.len() <= PROJECTION_TOKENS {
            return weight.matmul_tokens(rows);
        }
        let mut output = Vec::with_capacity(rows.len());
        for chunk in rows.chunks(PROJECTION_TOKENS) {
            output.extend(weight.matmul_tokens(chunk)?);
        }
        Ok(output)
    }

    fn normalize<const BF16: bool>(&self, rows: &mut [Vec<f32>], name: &str) -> Result<()> {
        let weight = self.weights.get(name)?.vector();
        let inv_width = (self.cfg.hidden_size as f32).recip();
        rows_mut(rows, |row| {
            let mean = sum_lanes(row, |x| x) * inv_width;
            let variance = sum_lanes(row, |x| (x - mean).powi(2)) * inv_width;
            let inv_sd = (variance + self.cfg.norm_eps).sqrt().recip();
            for (x, &w) in row.iter_mut().zip(&weight) {
                *x = cast::<BF16>((*x - mean) * inv_sd * w);
            }
        });
        Ok(())
    }
}

fn rows_mut<F: Fn(&mut Vec<f32>) + Send + Sync>(rows: &mut [Vec<f32>], f: F) {
    #[cfg(not(target_arch = "wasm32"))]
    if rows.len() >= 64 {
        rows.par_iter_mut().for_each(f);
        return;
    }
    rows.iter_mut().for_each(f);
}

fn sum_lanes(row: &[f32], map: impl Fn(f32) -> f32) -> f32 {
    let mut lanes = [0.0; 16];
    for values in row.as_chunks::<16>().0 {
        for (sum, &x) in lanes.iter_mut().zip(values) {
            *sum += map(x);
        }
    }
    lanes.into_iter().sum()
}

#[inline]
pub(super) fn cast<const BF16: bool>(value: f32) -> f32 {
    if BF16 {
        bf16::round(value)
    } else {
        value
    }
}

fn residual<const BF16: bool>(state: &mut [Vec<f32>], update: Vec<Vec<f32>>) {
    let apply = |(row, add): (&mut Vec<f32>, Vec<f32>)| {
        for (x, y) in row.iter_mut().zip(add) {
            *x = cast::<BF16>(*x + y);
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    if state.len() >= 64 {
        state.par_iter_mut().zip(update).for_each(apply);
        return;
    }
    state.iter_mut().zip(update).for_each(apply);
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn bf16_gelu_table() -> &'static [f32; 65536] {
    static TABLE: OnceLock<Box<[f32; 65536]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=u16::MAX)
            .map(|bits| bf16::round(gelu(half::bf16::from_bits(bits).to_f32())))
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .try_into()
            .unwrap()
    })
}

const GELU_STEPS: usize = 512;
const GELU_LIMIT: usize = 8;

fn float_gelu_table() -> &'static [f32] {
    static TABLE: OnceLock<Box<[f32]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=2 * GELU_LIMIT * GELU_STEPS)
            .map(|i| gelu(i as f32 / GELU_STEPS as f32 - GELU_LIMIT as f32))
            .collect()
    })
}

fn interpolate_gelu(x: f32, table: &[f32]) -> f32 {
    let limit = GELU_LIMIT as f32;
    if x <= -limit {
        return 0.0;
    }
    let point = x * GELU_STEPS as f32 + (GELU_LIMIT * GELU_STEPS) as f32;
    let index = point as usize;
    if index >= table.len() - 1 {
        return x;
    }
    table[index] + (table[index + 1] - table[index]) * (point - index as f32)
}

fn rotary_table(len: usize, width: usize, theta: f32) -> Vec<Vec<(f32, f32)>> {
    // The upstream CPU BF16 loader casts ModernBERT's inverse-frequency
    // buffers to BF16, then uses FP32 for the angles. Preserve those phases
    // in every weight format so long prompts use the same positions.
    let freq: Vec<_> = (0..width / 2)
        .map(|i| bf16::round(theta.powf((2 * i) as f32 / width as f32).recip()))
        .collect();
    (0..len)
        .map(|pos| {
            freq.iter()
                .map(|f| {
                    let (sin, cos) = (pos as f32 * f).sin_cos();
                    (bf16::round(sin), bf16::round(cos))
                })
                .collect()
        })
        .collect()
}

fn rotate<const BF16: bool>(head: &mut [f32], coefficients: &[(f32, f32)]) {
    let (a, b) = head.split_at_mut(coefficients.len());
    for ((a, b), &(sin, cos)) in a.iter_mut().zip(b).zip(coefficients) {
        let (x, y) = (*a, *b);
        *a = cast::<BF16>(x * cos - y * sin);
        *b = cast::<BF16>(y * cos + x * sin);
    }
}

fn apply_rotary<const BF16: bool>(
    qkv: &mut [Vec<f32>],
    rotary: &[Vec<(f32, f32)>],
    head_dim: usize,
    hidden: usize,
    queries: &Range<usize>,
) {
    let rotary = &rotary[..qkv.len()];
    let apply = |(i, (row, coeffs)): (usize, (&mut Vec<f32>, &Vec<(f32, f32)>))| {
        let first = if queries.contains(&i) { 0 } else { hidden };
        for head in row[first..2 * hidden].chunks_exact_mut(head_dim) {
            rotate::<BF16>(head, coeffs);
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    if qkv.len() >= 64 {
        qkv.par_iter_mut().zip(rotary).enumerate().for_each(apply);
        return;
    }
    qkv.iter_mut().zip(rotary).enumerate().for_each(apply);
}
