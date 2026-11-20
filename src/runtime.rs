use crate::{encoder, model, weights::Weights, GlideModel};
use anyhow::{Context, Result};
#[cfg(not(target_arch = "wasm32"))]
use std::{fs, path::Path};

enum Backbone {
    Decoder(GlideModel),
    Encoder(encoder::EncoderModel),
}

/// A packed Fastino GLiDE-2B or GLiDE-0.4B decision model.
pub struct DecisionModel(Backbone);

impl DecisionModel {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(dir: &Path) -> Result<Self> {
        let config = fs::read_to_string(dir.join("config.json"))?;
        let decision = fs::read_to_string(dir.join("decision_config.json"))?;
        Self::new(&config, &decision, Weights::open(dir)?)
    }

    pub fn from_parts(config: &str, decision: &str, index: &str, bytes: Vec<u8>) -> Result<Self> {
        Self::new(config, decision, Weights::from_bytes(index, bytes)?)
    }

    fn new(config: &str, decision_json: &str, weights: Weights) -> Result<Self> {
        let decision: model::DecisionConfig = serde_json::from_str(decision_json)?;
        let backbone = if decision.attention_mode == encoder::ATTENTION_MODE {
            Backbone::Encoder(encoder::EncoderModel::new(config, decision_json, weights)?)
        } else {
            Backbone::Decoder(GlideModel::new(
                serde_json::from_str(config)?,
                decision,
                weights,
            )?)
        };
        Ok(Self(backbone))
    }

    pub fn decision(&self) -> &model::DecisionConfig {
        match &self.0 {
            Backbone::Decoder(m) => &m.decision,
            Backbone::Encoder(m) => &m.decision,
        }
    }

    pub fn dimensions(&self) -> (usize, usize) {
        match &self.0 {
            Backbone::Decoder(m) => (m.cfg.num_hidden_layers, m.cfg.hidden_size),
            Backbone::Encoder(m) => (m.cfg.num_hidden_layers, m.cfg.hidden_size),
        }
    }

    pub fn encoder_context(&self) -> Option<usize> {
        match &self.0 {
            Backbone::Decoder(_) => None,
            Backbone::Encoder(m) => Some(m.max_len),
        }
    }

    /// Return all 255 raw decision logits for each unpadded prompt.
    pub fn predict_batch(&self, prompts: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        match &self.0 {
            Backbone::Decoder(m) => m.predict_batch(prompts),
            Backbone::Encoder(m) => m.predict_batch(prompts),
        }
    }

    pub fn probabilities_from_logits(&self, logits: &[f32], count: usize) -> Result<Vec<f32>> {
        model::probabilities(logits, count, self.decision().temperature)
    }

    pub fn predict_tokens(&self, ids: &[u32]) -> Result<Vec<f32>> {
        self.predict_batch(&[ids])?
            .pop()
            .context("empty inference result")
    }

    pub fn probabilities(&self, ids: &[u32], count: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(
            (1..=255).contains(&count),
            "number of choices must be 1..255"
        );
        let row = self.predict_tokens(ids)?;
        self.probabilities_from_logits(&row, count)
    }
}
