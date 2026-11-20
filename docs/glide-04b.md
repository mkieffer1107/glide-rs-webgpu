# GLiDE-0.4B on CPU

The Rust runtime loads Fastino's text-only ModernBERT checkpoint. It produces
all 255 decision logits and the same choice, yes/no, and score response fields
as the reference. GLiDE-0.4B uses bidirectional full attention every third
layer and a bidirectional 64-token radius in the other layers. The readout
pools the token before `[SEP]`.

The encoder context is 7,999 tokens including `[CLS]` and `[SEP]`. The
tokenization helper uses Fastino's chat template and encoder tokenizer. For
long text it keeps the first 1,024 and last 6,973 content tokens. The Rust API
expects those wrapped, unpadded IDs. Mixed batches share projections and
keep attention and positions independent for each prompt.

## Build and run

Use Rust stable and uv. The Python environment includes the CPU PyTorch wheel.
If you have the packaged Q8 weights, extract them as `model-04b-q8` and skip
the checkpoint download and conversion. It includes the assets required for
`make_request.py`.

```bash
uv sync --locked
cargo build --release --locked
uv run --locked hf download fastino/GLiDE-0.4B \
  --revision 6bda7ab91aeb36dcad6d27828a8871e548c7fc71 --local-dir GLiDE-0.4B
uv run --locked tools/convert.py --source GLiDE-0.4B --output model-04b-q8 \
  --layout tile16 --weight-bits 8 --embeddings f16

mkdir -p .bench
uv run --locked tools/make_request.py --model model-04b-q8 \
  --request examples/request.json --output .bench/04b-input.json
RAYON_NUM_THREADS=4 target/release/glide-cpu decide --model model-04b-q8 \
  --request .bench/04b-input.json --batch-size 4 --include-logits
```

Use `--weight-bits 16` in a fresh output directory for BF16. The packed
projections use AVX-512 BF16 or VNNI when available. The encoder allocates only
one block of attention scores per worker, skips keys outside the local window,
and computes the final output projection and MLP only for each pooled row.

## Reproduce the validation

```bash
uv run --locked tools/context_suite.py --model GLiDE-0.4B --output .bench/04b
uv run --locked tools/bench_reference.py --model GLiDE-0.4B \
  --request .bench/04b/topics.json --threads 4 --batch-size 4 \
  --warmup 0 --iters 1 --include-logits > .bench/04b/torch.json
GLIDE_PROFILE=rss RAYON_NUM_THREADS=4 target/release/glide-cpu decide \
  --model model-04b-q8 --request .bench/04b/topics.json --batch-size 4 \
  --include-logits --timings > .bench/04b/rust.json
uv run --locked tools/compare.py .bench/04b/torch.json .bench/04b/rust.json \
  --expected .bench/04b/labels.json
```

The suite includes 24 varied text and JSON questions and eight retrieval
prompts of 128, 129, 256, 512, 1,024, 2,048, 4,096, and 7,999 tokens. Run the
topic comparisons with batch sizes 1, 2, 4, and 8. Use `sweep.json` with batch
size 1 for the context sweep, or `mixed.json` with batch size 4 to benchmark
unequal lengths. The comparator checks the decision fields, probabilities,
and every raw logit. Use `glide-cpu bench` for warm runs and measure peak RSS
from process start through inference. On Linux, `/usr/bin/time -v` reports RSS
in KiB. `GLIDE_PROFILE=rss` also reports attention, projection, normalization,
and MLP stage times from Rust.

## CPU results

These runs used four pinned logical CPUs on an AMD EPYC 9V74. The reference
was the pinned Fastino wrapper with PyTorch 2.14.0+cpu BF16 and Transformers
5.17.0. Each warm measurement used one warmup and five timed forwards.
The context rows use one forward per length in the sweep. Measurements
exclude tokenization and model loading. Peak RSS includes both model loading
and inference in separate processes.

| Input | PyTorch BF16 | Rust BF16 | Rust Q8 + FP16 embeddings |
|---|---:|---:|---:|
| 128 tokens, warm median | 225 ms | 249 ms | 229 ms |
| 512 tokens, warm median | 817 ms | 847 ms | 666 ms |
| 1,024 tokens, warm median | 1.94 s | 1.76 s | 1.49 s |
| Mixed batch of 128, 256, 512, and 1,024 tokens, warm median | 8.46 s | 3.81 s | 2.80 s |
| 4,096 tokens, one forward | 14.60 s | 7.50 s | 6.13 s |
| 7,999 tokens, one forward | 41.16 s | 16.12 s | 13.96 s |

| Input | Metric | PyTorch BF16 | Rust BF16 | Rust Q8 + FP16 embeddings |
|---|---|---:|---:|---:|
| Mixed batch | Actual tokens/s | 227 | 504 | 686 |
| Mixed batch | p95 latency | 9.29 s | 4.15 s | 3.11 s |
| Mixed batch | Peak RSS | 2.63 GiB | 0.72 GiB | 0.42 GiB |
| Context sweep | Peak RSS | 2.63 GiB | 0.87 GiB | 0.57 GiB |

Both Rust formats matched all eight PyTorch decisions in the context sweep
and all 24 decisions in each topic batch-size run. The reference and both
Rust formats answered 20 of 24 topic questions correctly. At 7,999 tokens
the selected Q8 build read all 255 logits with mean absolute error 0.0139
and maximum error 0.1045. The native API rejects a wrapped 8,000-token
array. The Python preparation path was also checked for text truncation.

| Topic batch size | BF16 raw logit MAE / max | Q8 + FP16 raw logit MAE / max |
|---:|---:|---:|
| 1 | 0.0050 / 0.1250 | 0.0079 / 0.1834 |
| 2 | 0.0048 / 0.1250 | 0.0079 / 0.1176 |
| 4 | 0.0049 / 0.1250 | 0.0079 / 0.1123 |
| 8 | 0.0048 / 0.1250 | 0.0080 / 0.1020 |

In the initial Rust A/B run, tiling attention cut the 1,024-token BF16 warm
median from 2.81 to 1.68 s. The 1,024-token global attention profile dropped
from 923 to 116 ms. With Q8 weights and Q8 embeddings, BF16 attention operands
at 7,999 tokens reduced total forward time from 19.52 to 16.08 s and mean raw
logit error from 0.0155 to 0.0149. Below 512 keys, the native quantized
attention path uses FP32 operands.

### Quantized weight formats

Q8 projections with FP16 embeddings gave the smallest raw-logit error of
the tested compressed formats. The table uses the 24-topic, batch-size-four
comparison. Weight sizes exclude tokenizers and metadata.

| Weights and embeddings | Packed weights | Matching decisions | Raw logit MAE / max |
|---|---:|---:|---:|
| BF16 | 790 MB | 24/24 | 0.0049 / 0.1250 |
| Q8 + FP16 | 469 MB | 24/24 | 0.0079 / 0.1123 |
| Q8 + Q8 | 421 MB | 24/24 | 0.0087 / 0.2127 |
| Q8 + Q4 | 395 MB | 24/24 | 0.0148 / 0.2535 |
| Q6 + Q8 | 335 MB | 24/24 | 0.0163 / 0.2843 |
| Q4 MSE with Q8 attention + Q4 | 282 MB | 23/24 | 0.0684 / 0.9091 |
| Q4 MSE + Q4 | 223 MB | 22/24 | 0.0889 / 1.3341 |

## Browser export

```bash
uv run --locked tools/webgpu_manifest.py --model model-04b-q8
rustup target add wasm32-unknown-unknown
RUSTFLAGS='-C target-feature=+simd128' cargo build --locked --release \
  --target wasm32-unknown-unknown --features wasm --lib
wasm-bindgen --target web --out-dir pkg \
  target/wasm32-unknown-unknown/release/glide_cpu.wasm
```

Use a `wasm-bindgen` CLI matching the version in `Cargo.lock`. `WasmGlide`
accepts the packed model and pre-tokenized input. Its `predictLogits` method
returns the full readout. The WASM Q8 fallback uses FP32 activations. WebGPU
projections use quantized activations and get the tensor ranges from
`webgpu.json`. The JavaScript modules cover buffer streaming and projections.
The browser's WASM CPU path evaluates the whole encoder.

A 128-token Node.js WASM smoke test matched the PyTorch choice. Across all 255
logits, its mean absolute error was 0.0074 and its maximum error was 0.0328.
The single-threaded forward took 20.1 s in Node.js. In Chromium with the
SwiftShader adapter, the WebGPU test streamed weight ranges and checked
real MLP and QKV projections against an independent calculation. The maximum
projection error was 0.0000006. WebGPU validation covers individual
projections. The `pkg/` browser build generated above runs complete inference
on the CPU.

The model files are distributed under the Fastino Community License and
include Ettin's original MIT license. [Fastino checkpoint](https://huggingface.co/fastino/GLiDE-0.4B/tree/6bda7ab91aeb36dcad6d27828a8871e548c7fc71)
