# GLiDE-0.4B in the browser

The WebGPU runtime reads the existing GLiDE-0.4B export with tiled Q8
projections and FP16 embeddings. It runs all 28 encoder layers and returns
the 255 raw decision logits per prompt.

## Run the example

Extract the Q8 model archive as `model-04b-q8/` in this checkout.
The model archive already supplies
`webgpu.json`. For a newly converted checkpoint, run
`uv run --locked tools/webgpu_manifest.py --model model-04b-q8`. [Build the model export](glide-04b.md#build-and-run)

```bash
node tools/serve_webgpu.mjs model-04b-q8
```

Open [the local example](http://127.0.0.1:8080/), click **Load Q8 model**,
then **Run decision**. It uses a two-question sample, with an option to open
a tokenized JSON file. Use
`uv run --locked tools/make_request.py --model model-04b-q8 --request INPUT.json --output TOKENS.json`
to prepare a request from text or JSON.

The server binds to loopback and serves byte ranges for the packed weights.
For a deployed app, serve the page over HTTPS and expose the
`Content-Range` header if the weights are on a different origin.

## JavaScript API

```js
import { Glide04bWebGPU } from "./webgpu/encoder.mjs";

const model = await Glide04bWebGPU.load("/model/webgpu.json", {
    onProgress(loadedBytes, totalBytes) {
        console.log(loadedBytes / totalBytes);
    },
});

try {
    const response = await fetch("/examples/glide-04b.tokens.json");
    const { questions } = await response.json();
    const requests = Object.values(questions);
    const logits = await model.predictLogitsBatch(requests.map(q => q.input_ids));
    const choices = logits.map((row, i) => model.probabilities(row, requests[i].labels.length));
    console.log(choices);
} finally {
    await model.destroy();
}
```

Pass unpadded GLiDE-0.4B encoder IDs with `[CLS]` and `[SEP]`. The model
context is 7,999 tokens per prompt. Batches share projections while keeping
the attention and positions separate. The runtime splits batches when
their total length exceeds the active device's token budget. Call
`predictLogits(ids)` for a single prompt or `predict(ids, optionCount)`
for its option probabilities.

`load` also accepts an existing `device`, an optional `adapter`, an
`AbortSignal` as `signal` for loading, and `maxBatchTokens` to tune scratch
memory and throughput. Pass the adapter alongside a supplied device to use
its software/hardware information in automatic kernel selection.

The runtime keeps packed weights and intermediate activations on the GPU.
Coalesced range requests bound upload staging to one 32 MiB range at a
time. Projection tiles reuse packed weights and activations, GEGLU is fused
with activation packing, and the final layer computes attention output and
the MLP only for pooled rows. Local attention skips keys outside the
64-token radius. The online softmax uses bounded scratch even in global
layers. The forward pass reads back only the logits.

On hardware that advertises packed dot products, the automatic path uses
tiled integer dot products. For software adapters and devices without the
packed-dot feature, the runtime dequantizes tiles into shared memory for
FP32 accumulation. Both paths keep the same Q8 weights resident. Override
the selection with `projection: "int8"` or `projection: "fp32"` when
benchmarking.

## Check parity and benchmark

The browser tests use Node.js, Playwright and Chromium. Set `GLIDE_CHROMIUM`
to the path of your Chromium executable if it differs from
`/usr/bin/chromium`.

```bash
cargo build --release --locked
target/release/glide-cpu decide --model model-04b-q8 \
  --request examples/glide-04b.tokens.json --include-logits > cpu.json
node tools/test_encoder_webgpu.mjs model-04b-q8 examples/glide-04b.tokens.json cpu.json
node tools/test_encoder_attention.mjs
```

Pass additional CPU or PyTorch reference JSON files to compare all 255
logits. Add `--hardware --warmup 1 --runs 5` to measure a real GPU and
`--wasm pkg` to also time the WASM build in the same Chromium process.
The attention check compares ragged sequences, the sliding window, and
queries over the full 7,999-key context against a separate CPU calculation.

## Measured results

Chromium 156 used SwiftShader on four pinned AMD EPYC 9V74 logical CPUs.
The 128-token comparison used one warmup and two timed forwards per backend
in the same browser. Times cover dispatch through logit readback and exclude
initialization and loading.

| 128-token prompt | Forward median |
|---|---:|
| WebGPU, FP32 projection tiles | 37.4 s |
| WASM CPU, SIMD128 | 20.2 s |

The browser had no hardware WebGPU adapter. These timings measure CPU
execution through the software adapter, and hardware GPU speedup remains
unmeasured. The runtime allocated 475.3 MB of GPU buffers for this prompt,
including 469.0 MB of weights.

On the same adapter, a standalone 128-token projection through a real
5,248-by-1,024 MLP matrix took 539 ms with the FP32 tile and 883 ms with the
packed integer tile. Those medians use one warmup and three timed runs.

| Inference comparison | Matching decisions | Raw logit MAE | Max absolute error |
|---|---:|---:|---:|
| 128 tokens vs PyTorch BF16 | 1/1 | 0.0103 | 0.0467 |
| 128 tokens vs Rust Q8 | 1/1 | 0.0094 | 0.0547 |
| 107- and 124-token prompts in one batch vs Rust Q8 | 2/2 | 0.0027 | 0.0209 |

The tiled attention test used individual queries at positions 0, 3,999,
and 7,997 with 7,999 keys. Its maximum error was 0.0000016 against the
independent CPU calculation. The full-model comparisons above used the
shorter prompts.
