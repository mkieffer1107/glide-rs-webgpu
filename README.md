# GLiDE-CPU

Native Rust CPU inference for Fastino GLiDE-2B / Qwen3.5-2B. The runtime reads
decision probabilities from the checkpoint's 255-way readout and supports
choice, yes/no (`noul`), and score responses for text and JSON states.

## Run the compact Q8 download

Extract the model ZIP next to this directory. Its `model-q8-compact` folder
contains the packed checkpoint, tokenizer, Fastino prompt renderer, license
notices, and the browser manifest.

```bash
cargo build --release --locked
./target/release/glide-cpu decide --model ../model-q8-compact \
  --request examples/request.tokens.json --include-logits
```

To prepare your own text request from `examples/request.json`, use the
tokenizer and prompt renderer in the model archive.

```bash
uv sync --locked
uv run --locked tools/make_request.py --model ../model-q8-compact \
  --request examples/request.json --output request.tokens.json
```

The [WebGPU projection tools](#browser-weight-export-and-webgpu-projection)
use the included browser manifest.

## Build and run on CPU

Use a recent stable Rust toolchain and uv. The locked Python 3.12 environment
includes the PyTorch CPU wheel for conversion and comparison. The Rust binary
loads the converted weights directly.

In a workspace with a read-only home directory, set `UV_CACHE_DIR` to a
writable directory before running uv.

```bash
uv sync --locked
uv run --locked hf download fastino/GLiDE-2B \
  --revision 00f122c120cdeed82284106cb77d9bab5a50ad16 --local-dir ./GLiDE-2B

uv run --locked tools/convert.py --source ./GLiDE-2B --output ./model-bf16 \
  --layout tile16 --weight-bits 16

cargo test --release --locked
RUSTFLAGS='-C target-cpu=native' cargo build --release --locked
./target/release/glide-cpu inspect --model ./model-bf16

mkdir -p .bench
uv run --locked tools/make_request.py --model ./GLiDE-2B \
  --request examples/request.json --output .bench/ids.json
RAYON_NUM_THREADS=4 ./target/release/glide-cpu decide --model ./model-bf16 \
  --request .bench/ids.json
```

The tokenization helper uses the Fastino renderer, decision codes, and Qwen chat
template from the local snapshot. The CLI accepts pre-tokenized requests from
other callers as well. The CLI defaults to one question per forward pass.
Set `--batch-size` to share the matrix multiplications across several prompts.
Each prompt keeps its own positions, attention, and recurrent state.

BF16 preserves the checkpoint's weight bits, uses FP32 projection partial
sums, and rounds intermediate results at the reference's BF16 boundaries.
The tiled BF16 kernel uses AVX-512 BF16 where available.

For a smaller and faster model, convert to a fresh directory with
`--layout tile16 --weight-bits 8`. The Q8 kernel uses AVX-512 VNNI and retains
FP16 embeddings by default. `--layout row --weight-bits 4` produces the
original portable Q4 format. For native Q4 with SIMD, use `--layout tile16
--weight-bits 4`. Add `--embeddings q4` for a smaller embedding table, or
`--q8-attention` to retain 8-bit attention projections in a tiled Q4 model.
Use `--layout tile16 --weight-bits 6` for packed Q6 backbone projections.
`--q4-mse` searches for an FP16 scale that reduces squared reconstruction
error within each group of 32 uniformly quantized Q4 weights. `--q4-codebook`
uses nonuniform codes for the tiled Q4 backbone at the same storage size.
Each layout has a scalar fallback.
Use a fresh output directory when a process has already mapped a model.

## Compare with upstream PyTorch

```bash
uv run --locked tools/bench_reference.py --model ./GLiDE-2B \
  --request .bench/ids.json --threads 4 --batch-size 2 --include-logits > .bench/reference.json
RAYON_NUM_THREADS=4 ./target/release/glide-cpu decide --model ./model-bf16 \
  --request .bench/ids.json --batch-size 2 --include-logits > .bench/rust.json
uv run --locked tools/compare.py .bench/reference.json .bench/rust.json

RAYON_NUM_THREADS=4 ./target/release/glide-cpu bench --model ./model-bf16 \
  --request .bench/ids.json --batch-size 2 --warmup 1 --iters 5
uv run --locked tools/bench_reference.py --model ./GLiDE-2B \
  --request .bench/ids.json --batch-size 2 --threads 4 --warmup 1 --iters 5
```

The comparator checks the typed responses, token usage, selected answers,
probabilities, and all 255 raw logits before temperature scaling. It reports
errors across the full readout and across the active option slots. Optional
thresholds are `--min-agreement`, `--max-probability-error`, and
`--max-logit-error`.

The benchmark output includes every timed sample, median and p95 latency, and
prompt and actual-token throughput. For matched measurements, pin both
processes to the same CPUs, use the same token IDs and batch sizes, warm up each
shape, and measure peak RSS from process start through inference. On Linux,
`/usr/bin/time -v` reports that peak. The results below use separate processes
and reverse their launch order between the single-prompt and mixed-batch runs.

The CPU benchmark and profiling methods are based on the shared-input,
same-CPU comparisons and per-operation profiling in these projects.
[Blinkfast](https://github.com/thepiglab/blinkfast#method) ·
[Flashcluster](https://github.com/thepiglab/flashcluster/blob/main/docs/BENCHMARKS.md) ·
[Clipfast](https://github.com/thepiglab/clipfast) ·
[Flashembed](https://github.com/thepiglab/flashembed/blob/main/docs/PROFILING.md)

## Earlier CPU validation

The Rust runtime was compared with upstream BF16 PyTorch 2.14.0+cpu and
Transformers 5.17.0 on an AMD EPYC 9V74 VM. Both processes used four pinned CPU
threads. The 35 text questions included 24 MMLU-Pro test questions and 11
examples and edge cases covering choice, yes/no, and score. Prompts ranged
from 78 to 666 tokens and had 1 to 32 options. The larger batches reused the
same mixed-length subset of 16 questions.

| Batch size | BF16 matching decisions | Q8 raw logit MAE / max | BF16 raw logit MAE / max | BF16 max probability error |
|---:|---:|---:|---:|---:|
| 1 | 35/35 | 0.0139 / 0.2879 | 0.0071 / 0.1250 | 4.02 percentage points |
| 2 | 16/16 | 0.0157 / 0.2814 | 0.0064 / 0.1250 | 4.02 percentage points |
| 4 | 16/16 | 0.0151 / 0.2410 | 0.0062 / 0.0938 | 1.30 percentage points |
| 8 | 16/16 | 0.0158 / 0.2410 | 0.0056 / 0.0781 | 1.71 percentage points |

Each row measures all 255 raw logits per question. BF16 peak process memory
was 2.69–2.85 GiB across the batch tests, and PyTorch peaked at 7.52 GiB.
The BF16 and Q8 weight files take 3.77 GB and 2.48 GB on disk, respectively.

BF16 follows the reference CPU attention kernel's 512-key blocks. The left
padding implied by a mixed batch can change where a prompt's blocks fall.
In this test PyTorch's own logits differed by up to 0.156 between batch
shapes. The optimized Q8 path retained its previous logits and decisions on
all 35 single-prompt cases.

### Latency, throughput, and memory

The workload below uses one warmup per shape, five timed single-prompt runs,
and four timed mixed-batch runs. Q8 was measured again on the same CPU set
after tuning the VNNI tile.

| Workload | PyTorch median | Rust BF16 median | Rust Q8 median |
|---|---:|---:|---:|
| 78 tokens | 440 ms | 522 ms | 356 ms |
| 95 tokens | 507 ms | 728 ms | 451 ms |
| 492 tokens | 2,642 ms | 3,266 ms | 2,308 ms |
| Four prompts of 78, 95, 103, and 344 tokens | 9,127 ms | 3,566 ms | 2,560 ms |

| Mixed-batch engine | Actual tokens/s | p95 latency | Peak RSS |
|---|---:|---:|---:|
| PyTorch BF16 | 67.9 | 9,951 ms | 7.51 GiB |
| Rust BF16 | 173.9 | 3,653 ms | 2.66 GiB |
| Rust Q8 | 242.1 | 2,805 ms | 1.46 GiB |

Rust processes the 620 actual tokens without padding each prompt to 344.
Both Rust formats matched all four decisions on that timed workload. BF16
mean and maximum raw-logit errors were 0.0043 and 0.0469, respectively.

In a paired Q8 run, the 12-token VNNI tile cut the 492-token median from
2,610 to 2,308 ms. All 255 logits remained bit-identical in the 35-question
single-prompt suite.

[MMLU-Pro test data](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)

### Context-length and topic suite

Fastino's default service cap is 40,000 tokens and it reserves eight tokens
per rendered question, leaving 39,992 for text. The Qwen text configuration
specifies 262,144 positions. A 20,000-token prompt is just over half of the
default service capacity. [Fastino model and code](https://huggingface.co/fastino/GLiDE-2B/tree/00f122c120cdeed82284106cb77d9bab5a50ad16)

The sweep renders 96, 256, 512, 1,024, 2,048, 4,096, 8,192, 12,000, and
20,000 tokens using the Fastino chat template. Each record retrieval prompt
has a labeled answer, and the target occurs at early, middle, or late
positions. The companion set has 24 authored questions over 57–352 tokens
and includes arithmetic, program tracing, contract reading, document
synthesis, Spanish and Japanese reading, choice, yes/no, and rubric scores.
The generated `mixed.json` groups four prompts with 96–1,024 tokens.

```bash
uv run --locked tools/context_suite.py --model ./GLiDE-2B --output .bench/context
uv run --locked tools/bench_reference.py --model ./GLiDE-2B \
  --request .bench/context/sweep.json --threads 4 --batch-size 1 \
  --warmup 0 --iters 1 --include-logits > .bench/context/torch.json
GLIDE_CPU_MAX_TOKENS=39992 RAYON_NUM_THREADS=4 \
  ./target/release/glide-cpu decide --model ./model-bf16 \
  --request .bench/context/sweep.json --batch-size 1 --include-logits --timings \
  > .bench/context/rust.json
uv run --locked tools/compare.py .bench/context/torch.json .bench/context/rust.json \
  --expected .bench/context/labels.json
```

The current runtime was tested on the same four pinned CPU threads. The
20,000-token results are single forward passes with separate peak RSS
measurements. The expected tag was `violet`, which all three engines chose.

| Engine at 20,000 tokens | Forward time | Actual tokens/s | Peak RSS | Raw logit MAE / max |
|---|---:|---:|---:|---:|
| PyTorch BF16 | 202.13 s | 99.0 | 7.50 GiB | Reference |
| Rust BF16 | 153.60 s | 130.2 | 4.47 GiB | 0.0080 / 0.0625 |
| Rust Q8 | 168.20 s | 118.9 | 3.22 GiB | 0.0147 / 0.0527 |

The BF16 sweep matched all nine decisions and all nine expected labels.
Across 2,295 raw logits, mean absolute error was 0.0085 and maximum error
was 0.125. The largest probability difference was 0.0000086.

| Rendered tokens | PyTorch forward | Rust BF16 forward |
|---:|---:|---:|
| 96 | 1.00 s | 0.62 s |
| 256 | 1.62 s | 1.58 s |
| 512 | 2.77 s | 2.86 s |
| 1,024 | 6.95 s | 5.45 s |
| 2,048 | 13.14 s | 11.42 s |
| 4,096 | 31.42 s | 22.91 s |
| 8,192 | 67.94 s | 51.99 s |
| 12,000 | 109.55 s | 85.50 s |
| 20,000 | 202.13 s | 153.60 s |

At 4,096 tokens the previous Rust BF16 build took 36.60 s. The new build
took 21.59 s in a paired run. Peak RSS fell from 2.99 to 2.88 GiB. Mean
raw-logit error against PyTorch went from 0.00561 to 0.00529 and the maximum
was 0.02344 for both builds. At that length, Q8 with FP32 attention took
20.48 s and peaked at 1.67 GiB. Q8 at 20,000 tokens took 210.77 s with FP32
attention operands and 168.20 s with BF16 operands. Its logit MAE rose from
0.01082 to 0.01475 with that precision change.

The short timings below use one warmup and five timed runs per length.

| Prompt tokens | PyTorch median / p95 | Rust BF16 median / p95 | Rust Q8 median / p95 |
|---:|---:|---:|---:|
| 96 | 649 / 668 ms | 547 / 664 ms | 442 / 533 ms |
| 256 | 1,363 / 1,834 ms | 1,566 / 1,668 ms | 1,064 / 1,175 ms |
| 512 | 3,911 / 5,074 ms | 3,209 / 3,367 ms | 2,404 / 2,554 ms |

For the four-prompt batch with 96, 256, 512, and 1,024 tokens, all engines
chose all four expected tags. Rust passes 1,888 real tokens to the backbone.
PyTorch left-pads each sequence to 1,024 in this batch.

| Engine, batch size 4 | Forward time | Actual tokens/s | Peak RSS | Raw logit MAE / max |
|---|---:|---:|---:|---:|
| PyTorch BF16 | 29.65 s | 63.7 | 7.50 GiB | Reference |
| Rust BF16 | 12.82 s | 147.3 | 2.64 GiB | 0.0069 / 0.0625 |
| Rust Q8 | 9.79 s | 192.8 | 1.48 GiB | 0.0150 / 0.0747 |

On the eight mixed-topic prompts, BF16 matched all decisions at batch sizes
1, 2, and 8. Raw logit MAE was 0.0060, 0.0062, and 0.0062, respectively.
In the 24-question suite at batch size 4, every tested format matched all
PyTorch choices. PyTorch and Rust each answered 23 of 24 correctly. Both
chose the borrower on the umbrella ownership question. BF16 also matched all
35 decisions on the earlier suite and brought logit MAE from 0.00707 to
0.00685. On the 24 topics, BF16 logit MAE moved from 0.00570 to 0.00572,
with the same maximum error of 0.0625.

| Format | Weight file | Peak RSS, 24 topics | Raw logit MAE / max, 24 topics |
|---|---:|---:|---:|
| BF16 | 3.76 GB | 2.68 GiB | 0.0057 / 0.0625 |
| Q8, FP16 embeddings | 2.48 GB | 1.49 GiB | 0.0114 / 0.1129 |
| Q8, Q4 embeddings | 1.75 GB | 1.49 GiB | 0.0160 / 0.1592 |
| Q4, FP16 embeddings | 1.79 GB | 0.85 GiB | 0.1044 / 0.8310 |
| Q4, Q8 attention, Q4 embeddings | 1.30 GB | 1.07 GiB | 0.0856 / 0.7880 |
| Q4, Q4 embeddings | 1.06 GB | 0.86 GiB | 0.1079 / 0.8137 |

On the earlier 35-question workload, Q8 with Q4 embeddings matched 34
PyTorch decisions, mixed Q4 with Q8 attention matched 32, and both Q4 formats
with 4-bit backbone projections matched 29. The compact Q8 format uses
`--layout tile16 --weight-bits 8 --embeddings q4`. The compact Q4 build
took 22.54 s at 4,096 tokens and peaked at 1.09 GiB. The mixed Q4 build took
21.72 s and peaked at 1.30 GiB.

### Full context and CPU profiling

Use the boundary prompt and one token beyond it to check the configured
39,992-token service limit. Set `GLIDE_PROFILE=1` for wall time by stage or
`GLIDE_PROFILE=rss` to include Linux memory samples. The runtime writes a
`GLIDE_PROFILE` JSON record to stderr for each forward. Stage RSS is sampled
after the stage. Process high-water marks include temporary allocations.

```bash
uv run --locked tools/context_suite.py --model ./GLiDE-2B \
  --output .bench/full-context --lengths 39992 39993
GLIDE_CPU_MAX_TOKENS=39992 GLIDE_PROFILE=rss RAYON_NUM_THREADS=4 \
  taskset -c 0-3 ./target/release/glide-cpu decide --model ./model-bf16 \
  --request .bench/full-context/length-39992.json --include-logits --timings
```

The CPU audit uses frozen binaries and inputs, alternating candidate order,
and exact raw-logit comparisons for changes that preserve arithmetic. The
parity check rejects a changed decision and a changed inactive logit. The
isolated kernel runs include uneven query counts, key tails, and left
padding. [Kernel optimization method](https://github.com/gauravjain14/kernel-autoresearch-plugin/blob/64d53f37009e46ae9a05c8c319d44a90f9740bb3/kar/agents/prompts/method.md)

The 39,992-token case ran on four pinned vCPUs on the same AMD EPYC 9V74.
The runs below are single forwards. Peak RSS covers the whole process.

| Engine | Forward time | Actual tokens/s | Peak RSS | Raw logit MAE / max vs PyTorch |
|---|---:|---:|---:|---:|
| PyTorch BF16 | 619.82 s | 64.5 | 8.66 GiB | Reference |
| Previous Rust BF16 | 465.28 s | 86.0 | 5.82 GiB | 0.00625 / 0.06250 |
| Optimized Rust BF16 | 307.96 s | 129.9 | 4.88 GiB | 0.00625 / 0.06250 |
| Rust Q6, Q4 embeddings | 295.15 s | 135.5 | 3.49 GiB | 0.04610 / 0.23456 |
| Rust Q8, FP16 embeddings | 265.76 s | 150.5 | 3.78 GiB | 0.01438 / 0.05260 |
| Rust Q8, Q4 embeddings | 267.67 s | 149.4 | 3.79 GiB | 0.03684 / 0.13958 |

Both Rust BF16 runs produced identical values for all 255 logits. Every
engine in the table selected `cobalt`, the expected answer. The configured
Rust limit rejected 39,993 tokens.

At 4,096 tokens, feed-forward projections took 46% of the previous BF16
forward time. At the full context limit, noncausal attention took 227.10 s
of the previous run and 104.20 s after optimization. In the optimized run,
feed-forward projections took 101.23 s. The four-thread streaming copy probe
measured 59.7 GB/s counting reads and writes. Hardware performance counters
were unavailable on this host, so the audit used stage timings, process RSS,
paired kernel tests, and native disassembly.

For a representative block of 1,513 queries over 39,992 keys, reusing the key
and value blocks across eight query tiles cut the median from 1.686 s to
1.205 s. A separate paired test cut the grouped kernel from 1.179 s to
0.822 s by packing BF16 probabilities with SIMD. The sampled attention
outputs were bit-identical. The 64-lane tile was within 0.2% of the 32-lane
tile on the long sample. Moving the BF16 projection panels outside the token
loop showed no consistent speed gain across the tested shapes.

The projection-window A/B at 4,097 tokens preserved every logit. Its median
was 24.33 s versus 24.37 s for the previous projection layout, with median
peak RSS of 2.77 versus 2.86 GiB. The integrated build also matched every
logit at 4,097 tokens and on the mixed-length batch.

Short-run BF16 medians in alternating order were 522 to 567 ms at 96 tokens,
1,490 to 1,281 ms at 256 tokens, and 2,795 to 2,809 ms at 512 tokens. A
follow-up 96-token run with stage profiling measured 620 to 566 ms. Full
context accounts for the largest measured speed gain.

### Compact quantization audit

All five compact formats used Q4 embeddings. The 35-query suite includes
the 24 labeled MMLU-Pro questions. PyTorch answered 14 of those 24 correctly.

| Format | Weight file | PyTorch decisions matched | MMLU-Pro correct | Raw logit MAE / max |
|---|---:|---:|---:|---:|
| Q4 | 1.06 GB | 29/35 | 14/24 | 0.1291 / 1.6642 |
| Q4 with fitted scales | 1.06 GB | 32/35 | 15/24 | 0.1360 / 1.2905 |
| Q4 with nonuniform codebook | 1.06 GB | 33/35 | 14/24 | 0.0965 / 1.4859 |
| Q6 | 1.41 GB | 33/35 | 15/24 | 0.0371 / 0.7081 |
| Q8 | 1.75 GB | 34/35 | 13/24 | 0.0211 / 0.5786 |

Fitted Q4 scales improved decision agreement and reduced the worst logit
error on this suite. Mean logit error increased. The nonuniform codebook
reduced mean logit error by 25% against uniform Q4 on the same suite. It
took 18.33 s at 4,097 tokens, where uniform Q4 took 18.03 s.

On the 24 topic questions, all five formats matched every PyTorch decision.
Mean logit error was 0.1079 for uniform Q4, 0.1221 for fitted Q4, 0.0835 for
codebook Q4, 0.0310 for Q6, and 0.0160 for Q8. Codebook Q4's largest
probability difference on the 24 topics was 39.8 percentage points.

Compact codebook Q4 uses `--layout tile16 --weight-bits 4 --embeddings q4
--q4-codebook`. Compact Q6 uses `--layout tile16 --weight-bits 6 --embeddings
q4` and stores about 20% fewer bytes than compact Q8.

## Runtime details

- Memory-mapped storage pages in projection weights and the embedding rows
  used by the prompt. BF16 keeps the original checkpoint weights and uses
  FP32 for the recurrent state and normalization reductions.
- Native inference parallelizes projections and the independent Gated DeltaNet
  heads. The AVX-512 recurrent update keeps the scalar operation order.
  On AVX-512 BF16 hosts, noncausal attention reuses packed keys and values
  across six query rows and applies online softmax over 512-key blocks.
  Long prompts reuse each key/value block across eight query tiles before
  advancing. Score scratch is bounded by the block sizes. The portable path
  uses one score row per worker.
- Tiled attention uses BF16 dot products for BF16 weights and for quantized
  prompts of 16,384 tokens or more. It keeps the normalization and
  accumulators in FP32 and packs the weights for the value product as BF16.
  Shorter quantized prompts keep FP32 attention activations.
- Native BF16 vectorizes SiLU and reuses the normalization buffer across
  layers. Activations are released before the next projection needs scratch.
- Projection scratch is tiled in chunks of at most 4,096 token rows. The
  feed-forward block consumes its expansion matrices in 2,048-row chunks.
  Gated DeltaNet reuses its QKV projection rows for the depthwise convolution
  and copies the small history needed at each chunk boundary.
- The final layer evaluates each prompt's last query against all of its keys.
  Last-token pooling then reads the same position as Fastino.
- The native default limit is 2,048 tokens. Set `GLIDE_CPU_MAX_TOKENS` to
  increase it up to 40,000. Execution time grows with context length.

## WebAssembly interface

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build --release --target web --features wasm --lib
```

The `WasmGlide` constructor accepts `config.json`, `decision_config.json`,
`index.json`, and the weight bytes. Call `predict(token_ids, option_count)` to
get probabilities. The CPU WASM limit is 512 tokens. The native and portable
paths read the same tiled formats.
The release WASM binary validates in Node and exposes the constructor
and prediction bindings.

### Browser weight export and WebGPU projection

Start the WebGPU export with **compact Q8**. Both Q8 builds have 8-bit
backbone projections. Q4 embeddings reduce the weight file from 2.48 GB to
1.75 GB. On the 35-question suite, compact Q8 matched 34 PyTorch decisions
with mean logit error of 0.0211. Standard Q8 gave closer logits at the full
CPU context limit, with nearly the same peak RSS. Standard Q8 also matched
all 35 decisions. Use compact Q6 as the 1.41 GB option when the download
budget takes priority. It matched 33 decisions with mean logit error of
0.0371.

```bash
uv run --locked tools/convert.py --source ./GLiDE-2B --output ./model-q8-browser \
  --layout tile16 --weight-bits 8 --embeddings q4
uv run --locked tools/webgpu_manifest.py --model ./model-q8-browser
```

The manifest describes the existing packed bytes with no conversion pass or
duplicate weight file. Serve it next to the model assets with HTTP byte-range
support. The loader issues bounded requests at row or tile boundaries and
streams the response into a mapped WebGPU buffer. Each buffer is capped at
64 MiB or the device limit, whichever is smaller. The embedding table splits
into five buffers for compact Q8 and 16 for standard Q8 at that cap. For
cross-origin hosting, expose the `Content-Range` response header. [WebGPU limits](https://www.w3.org/TR/webgpu/#limits)

`webgpu/projection.mjs` packs activations once for projections that share a
hidden state. Its specialized shaders read Q4, codebook Q4, Q6, and Q8 tiles
directly and use packed integer dot products when the browser exposes them.
Chromium with SwiftShader validated 20 projection cases using real checkpoint
weights, with maximum absolute error of 0.00000036 against an independent CPU
oracle. The test exercises both dot-product paths and streams a full-size
embedding shard for each format.

```bash
node tools/test_webgpu.mjs ./model-q8-browser
```

The test uses Node.js 24, Playwright, and Chromium. Set
`GLIDE_WEBGPU_HARDWARE=1` for a physical adapter and `GLIDE_CHROMIUM` for a
different Chromium executable. The WASM CPU interface still accepts the
whole checkpoint. The WebGPU code covers weight loading and projections.
Attention, recurrence, and full-model execution on a physical GPU remain
unverified.

The Rust source is Apache-2.0. The downloaded checkpoint and converted weights
are governed by Fastino's Community License.

[Fastino GLiDE-2B checkpoint and license](https://huggingface.co/fastino/GLiDE-2B)
