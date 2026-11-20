use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use glide_cpu::GlideModel;
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{fs, path::Path, time::Instant};

#[derive(Parser)]
#[command(about = "Experimental Fastino GLiDE-2B quantized CPU decision runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Check metadata without executing a forward pass
    Inspect {
        #[arg(long)]
        model: String,
    },
    /// Decide on pre-tokenized questions. Input JSON must supply official prompt token IDs.
    Decide {
        #[arg(long)]
        model: String,
        #[arg(long)]
        request: String,
        /// Maximum number of prompts per forward pass
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        /// Include the 255 raw readout logits per prompt
        #[arg(long)]
        include_logits: bool,
        /// Report per-batch forward times with the decisions
        #[arg(long)]
        timings: bool,
    },
    /// Benchmark repeated forward passes on one prompt or a tokenized request
    Bench {
        #[arg(long)]
        model: String,
        #[arg(long, conflicts_with = "request", required_unless_present = "request")]
        tokens: Option<String>,
        #[arg(long, conflicts_with = "tokens")]
        request: Option<String>,
        /// Prompts per batch when benchmarking a tokenized request
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value_t = 2)]
        options: usize,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value_t = 5)]
        iters: usize,
        /// Output JSON when benchmarking a single token array
        #[arg(long)]
        json: bool,
    },
}
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum QuestionType {
    Choice,
    Noul,
    Score,
}

#[derive(Deserialize)]
struct Question {
    input_ids: Vec<u32>,
    labels: Vec<String>,
    #[serde(rename = "type")]
    kind: Option<QuestionType>,
}

#[derive(Deserialize)]
struct Request {
    questions: IndexMap<String, Question>,
}

fn answer(question: &Question, scores: Vec<f32>, latency_ms: f64) -> Result<Value> {
    let mut probabilities = IndexMap::new();
    for (label, p) in question.labels.iter().zip(scores) {
        if probabilities.insert(label.as_str(), p).is_some() {
            bail!("duplicate option label {label}");
        }
    }
    let mut ranked = probabilities.iter().collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.total_cmp(a.1));
    let (winner, probability) = ranked[0];
    let confidence = *probability - ranked.get(1).map_or(0.0, |(_, p)| **p);

    Ok(match question.kind {
        Some(QuestionType::Choice) => json!({
            "type": "choice", "choice": winner, "confidence": confidence,
            "probabilities": probabilities,
        }),
        Some(QuestionType::Noul) => {
            if probabilities.len() != 2 || !probabilities.contains_key("false") {
                bail!("yes/no questions need true and false labels");
            }
            let p = probabilities.get("true").context("missing true label")?;
            json!({"type": "noul", "noul": p, "confidence": (2.0 * p - 1.0).abs()})
        }
        Some(QuestionType::Score) => {
            let score = winner.parse::<u8>().context("invalid score label")?;
            let mut expected_level = 0.0;
            for (key, p) in &probabilities {
                let level = key.parse::<u8>().context("invalid score label")?;
                expected_level += f64::from(level) * f64::from(*p);
            }
            json!({
                "type": "score", "score": score, "expected_level": expected_level,
                "confidence": confidence, "probabilities": probabilities,
            })
        }
        None => json!({
            "choice": winner, "probabilities": probabilities, "confidence": confidence,
            "input_tokens": question.input_ids.len(), "latency_ms": latency_ms,
        }),
    })
}

fn decide(
    engine: &GlideModel,
    req: Request,
    batch_size: usize,
    include_logits: bool,
    include_timings: bool,
) -> Result<Value> {
    if batch_size == 0 {
        bail!("batch size must be positive");
    }
    let questions = req.questions.iter().collect::<Vec<_>>();
    let mut answers = IndexMap::new();
    let mut raw_logits = IndexMap::new();
    let mut timings = Vec::new();
    let mut input_tokens = 0;
    let mut has_typed_question = questions.is_empty();
    for batch in questions.chunks(batch_size) {
        let prompts = batch
            .iter()
            .map(|(_, q)| q.input_ids.as_slice())
            .collect::<Vec<_>>();
        let start = Instant::now();
        let logits = engine
            .predict_batch(&prompts)
            .with_context(|| format!("batch starting at question {}", batch[0].0))?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        if include_timings {
            let lengths: Vec<_> = prompts.iter().map(|p| p.len()).collect();
            let tokens = lengths.iter().sum::<usize>();
            timings.push(json!({
                "batch_size": lengths.len(), "input_tokens": tokens, "prompt_lengths": lengths,
                "samples_ms": [latency_ms], "median_ms": latency_ms, "p95_ms": latency_ms,
                "tokens_per_second": tokens as f64 * 1000.0 / latency_ms,
            }));
        }
        for ((name, question), row) in batch.iter().copied().zip(logits) {
            let scores = engine.probabilities_from_logits(&row, question.labels.len())?;
            let result =
                answer(question, scores, latency_ms).with_context(|| format!("question {name}"))?;
            answers.insert(name, result);
            if include_logits {
                raw_logits.insert(name, row);
            }
            input_tokens += question.input_ids.len();
            has_typed_question |= question.kind.is_some();
        }
    }
    let mut response = json!({"answers": answers});
    if has_typed_question {
        response["model"] = json!(engine.decision.model_name.as_deref().unwrap_or("glide-2b"));
        response["usage"] = json!({"input_tokens": input_tokens, "output_tokens": 0});
    }
    if include_logits {
        response["logits"] = json!(raw_logits);
    }
    if include_timings {
        response["batches"] = json!(timings);
    }
    Ok(response)
}

fn measure(
    engine: &GlideModel,
    prompts: &[&[u32]],
    options: &[usize],
    warmup: usize,
    iters: usize,
) -> Result<Value> {
    let run = || -> Result<()> {
        let raw = engine.predict_batch(prompts)?;
        for (row, &count) in raw.iter().zip(options) {
            std::hint::black_box(engine.probabilities_from_logits(row, count)?);
        }
        Ok(())
    };
    for _ in 0..warmup {
        run()?;
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        run()?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let mut ordered = samples.clone();
    ordered.sort_by(f64::total_cmp);
    let median = (ordered[(iters - 1) / 2] + ordered[iters / 2]) * 0.5;
    let tokens: usize = prompts.iter().map(|p| p.len()).sum();
    Ok(json!({
        "batch_size": prompts.len(),
        "input_tokens": tokens,
        "prompt_lengths": prompts.iter().map(|p| p.len()).collect::<Vec<_>>(),
        "samples_ms": samples,
        "median_ms": median,
        "mean_ms": samples.iter().sum::<f64>() / iters as f64,
        "p95_ms": ordered[(iters * 95).div_ceil(100) - 1],
        "prompts_per_second": prompts.len() as f64 * 1000.0 / median,
        "tokens_per_second": tokens as f64 * 1000.0 / median,
    }))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect { model } => {
            let dir = Path::new(&model);
            let index: serde_json::Value =
                serde_json::from_slice(&fs::read(dir.join("index.json"))?)?;
            let filesize = fs::metadata(dir.join("weights.bin"))?.len();
            println!(
                "File: {:.3} GB; tensors: {}",
                filesize as f64 / 1e9,
                index["tensors"].as_object().map_or(0, |m| m.len())
            );
            let model = GlideModel::open(dir)?;
            println!(
                "Backbone: {} layers, {} hidden; temperature={}",
                model.cfg.num_hidden_layers, model.cfg.hidden_size, model.decision.temperature
            );
        }
        Command::Decide {
            model,
            request,
            batch_size,
            include_logits,
            timings,
        } => {
            let engine = GlideModel::open(Path::new(&model))?;
            let req: Request = serde_json::from_slice(&fs::read(request)?)?;
            let response = decide(&engine, req, batch_size, include_logits, timings)?;
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        Command::Bench {
            model,
            tokens,
            request,
            batch_size,
            options,
            warmup,
            iters,
            json,
        } => {
            if iters == 0 || batch_size == 0 {
                bail!("iterations and batch size must be positive");
            }
            let opening = Instant::now();
            let engine = GlideModel::open(Path::new(&model))?;
            let load_ms = opening.elapsed().as_secs_f64() * 1000.0;
            if let Some(tokens) = tokens {
                let data: Value = serde_json::from_slice(&fs::read(tokens)?)?;
                let ids: Vec<u32> =
                    serde_json::from_value(data.get("input_ids").cloned().unwrap_or(data))?;
                let stats = measure(&engine, &[&ids], &[options], warmup, iters)?;
                if json {
                    println!("{}", json!({"model_open_ms": load_ms, "batches": [stats]}));
                } else {
                    println!(
                        "tokens={} runs={} median_ms={:.1} mean_ms={:.1}",
                        ids.len(),
                        iters,
                        stats["median_ms"].as_f64().unwrap(),
                        stats["mean_ms"].as_f64().unwrap(),
                    );
                }
            } else {
                let path = request.context("expected --tokens or --request")?;
                let req: Request = serde_json::from_slice(&fs::read(path)?)?;
                let questions: Vec<_> = req.questions.values().collect();
                if questions.is_empty() {
                    bail!("the benchmark request contains no questions");
                }
                let mut batches = Vec::new();
                for batch in questions.chunks(batch_size) {
                    let prompts: Vec<_> = batch.iter().map(|q| q.input_ids.as_slice()).collect();
                    let options: Vec<_> = batch.iter().map(|q| q.labels.len()).collect();
                    batches.push(measure(&engine, &prompts, &options, warmup, iters)?);
                }
                println!("{}", json!({"model_open_ms": load_ms, "batches": batches}));
            }
        }
    }
    Ok(())
}
