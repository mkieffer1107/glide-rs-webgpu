// End-to-end logits and timing on the existing GLiDE-0.4B Q8 export.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { parseArgs } from "node:util";
import { createGlideServer } from "./serve_webgpu.mjs";

const { chromium } = createRequire(import.meta.url)("playwright");
const {positionals, values} = parseArgs({
    allowPositionals: true,
    options: {
        hardware: {type: "boolean"},
        projection: {type: "string", default: "auto"},
        question: {type: "string"},
        "compare-packed": {type: "boolean"},
        "max-batch-tokens": {type: "string"},
        wasm: {type: "string"},
        runs: {type: "string", default: "1"},
        warmup: {type: "string", default: "0"},
        "max-error": {type: "string", default: "0.25"},
    },
});
const [modelDir, requestFile, ...referenceFiles] = positionals;
const runs = Number(values.runs);
const warmup = Number(values.warmup);
const tolerance = Number(values["max-error"]);
const maxBatchTokens = values["max-batch-tokens"] === undefined ? undefined : Number(values["max-batch-tokens"]);
if (!modelDir || !requestFile || !referenceFiles.length || !Number.isSafeInteger(runs) || runs < 1 ||
    !Number.isSafeInteger(warmup) || warmup < 0 || !Number.isFinite(tolerance) || tolerance < 0 ||
    (maxBatchTokens !== undefined && (!Number.isInteger(maxBatchTokens) || maxBatchTokens < 2))) {
    throw new Error("Usage: node tools/test_encoder_webgpu.mjs MODEL REQUEST REFERENCE [REFERENCES...] " +
                    "[--question NAME] [--runs 3] [--warmup 1] [--wasm pkg] [--hardware] " +
                    "[--projection auto|int8|int8-vector|fp32] [--compare-packed] " +
                    "[--max-batch-tokens 768] [--max-error 0.25]");
}
const json = path => JSON.parse(readFileSync(path, "utf8"));
const questions = Object.entries(json(requestFile).questions)
    .filter(([name]) => values.question === undefined || name === values.question);
const input = questions.map(([, q]) => q.input_ids);
assert(input.length > 0, values.question === undefined
    ? "The request needs at least one question"
    : `Question ${values.question} was not found in ${requestFile}`);

const server = createGlideServer(modelDir, values.wasm);
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let browser;
try {
    const software = !values.hardware;
    browser = await chromium.launch({
        executablePath: process.env.GLIDE_CHROMIUM || "/usr/bin/chromium",
        headless: true,
        args: ["--no-sandbox", "--disable-dev-shm-usage", "--enable-unsafe-webgpu",
            ...(software ? ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] : [])],
    });
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async ({
        software, projection, withWasm, input, runs, warmup, comparePacked, maxBatchTokens,
    }) => {
        const {Glide04bWebGPU} = await import("/webgpu/encoder.mjs");
        const adapter = await navigator.gpu?.requestAdapter({
            forceFallbackAdapter: software, powerPreference: "high-performance",
        });
        if (!adapter || (!software && adapter.info.isFallbackAdapter)) {
            throw new Error(software ? "No software WebGPU adapter" : "No hardware WebGPU adapter");
        }
        const adapterInfo = {
            vendor: adapter.info.vendor, architecture: adapter.info.architecture,
            description: adapter.info.description, fallback: adapter.info.isFallbackAdapter,
        };
        const device = await adapter.requestDevice();
        const uncaptured = [];
        device.addEventListener("uncapturederror", event => uncaptured.push(event.error.message));
        const opened = performance.now();
        const model = await Glide04bWebGPU.load("/model/webgpu.json", {adapter, device, projection, maxBatchTokens});
        const gpuLoadMs = performance.now() - opened;
        let logits;
        let baseline;
        let comparison;
        const gpuMs = [];
        const baselineMs = [];
        let gpuBufferBytes;
        try {
            await model.predictLogits([0]).then(
                () => { throw new Error("Expected a short input to fail"); },
                error => { if (!(error instanceof RangeError)) throw error; },
            );
            await model.predictLogits(new Uint32Array(model.maxContext + 1)).then(
                () => { throw new Error("Expected an over-context input to fail"); },
                error => { if (!(error instanceof RangeError)) throw error; },
            );
            if (comparePacked) {
                baseline = await Glide04bWebGPU.load("/model/webgpu.json",
                    {adapter, device, projection: "int8", maxBatchTokens});
            }
            const engines = baseline ? [[model, gpuMs], [baseline, baselineMs]] : [[model, gpuMs]];
            for (let run = -warmup; run < runs; run++) {
                let baselineLogits;
                const order = run % 2 ? engines.slice().reverse() : engines;
                for (const [engine, samples] of order) {
                    const start = performance.now();
                    const rows = await engine.predictLogitsBatch(input);
                    if (run >= 0) samples.push(performance.now() - start);
                    if (engine === model) logits = rows;
                    else baselineLogits = rows;
                }
                if (baselineLogits) {
                    for (const [i, row] of logits.entries()) {
                        if (row.some((value, j) => !Object.is(value, baselineLogits[i][j]))) {
                            throw new Error(`Logits differ from the packed INT8 baseline for prompt ${i + 1}`);
                        }
                    }
                }
            }
            gpuBufferBytes = model.gpuBufferBytes;
            if (baseline) {
                comparison = {projection: baseline.projection, gpuMs: baselineMs,
                    gpuBufferBytes: baseline.gpuBufferBytes, bitExact: true, logitsCompared: input.length * 255};
            }
        } finally {
            await baseline?.destroy();
            await model.destroy();
            device.destroy();
        }
        if (uncaptured.length) throw new Error(uncaptured.join("\n"));
        const wasmMs = [];
        let wasmLoadMs;
        if (withWasm) {
            const start = performance.now();
            const wasm = await import("/pkg/glide_cpu.js");
            await wasm.default();
            async function get(name, binary) {
                const response = await fetch(`/model/${name}`);
                if (!response.ok) throw new Error(`Cannot load ${name}`);
                return binary ? new Uint8Array(await response.arrayBuffer()) : response.text();
            }
            const [cfg, decision, manifest, weights] = await Promise.all([
                get("config.json"), get("decision_config.json"), get("webgpu.json"), get("weights.bin", true),
            ]);
            const engine = new wasm.WasmGlide(cfg, decision, manifest, weights);
            wasmLoadMs = performance.now() - start;
            try {
                for (let run = -warmup; run < runs; run++) {
                    const before = performance.now();
                    for (const ids of input) {
                        const row = engine.predictLogits(Uint32Array.from(ids));
                        if (row.length !== 255 || row.some(x => !Number.isFinite(x))) {
                            throw new Error("Invalid WASM logits");
                        }
                    }
                    if (run >= 0) wasmMs.push(performance.now() - before);
                }
            } finally {
                engine.free();
            }
        }
        return {adapter: adapterInfo, projection: model.projection, projectionTiles: model.projectionTiles,
            maxContext: model.maxContext, gpuLoadMs, gpuMs, wasmLoadMs, wasmMs, gpuBufferBytes,
            comparison, logits: logits.map(row => Array.from(row))};
    }, {software, projection: values.projection, withWasm: Boolean(values.wasm), input, runs, warmup,
        comparePacked: Boolean(values["compare-packed"]), maxBatchTokens});

    const argmax = row => row.indexOf(Math.max(...row));
    const checks = referenceFiles.map(file => {
        const expected = json(file).logits;
        let sum = 0, max = 0;
        const cases = questions.map(([name, question], i) => {
            const reference = expected[name];
            const actual = result.logits[i];
            assert.equal(actual.length, 255);
            assert.equal(reference?.length, 255, `Missing 255-logit reference for ${name}`);
            const errors = actual.map((x, j) => Math.abs(x - reference[j]));
            const largest = Math.max(...errors);
            assert(largest <= tolerance, `${name} max absolute error ${largest} exceeds ${tolerance} vs ${file}`);
            const count = question.labels.length;
            const expectedTop = argmax(reference.slice(0, count));
            const actualTop = argmax(actual.slice(0, count));
            assert.equal(actualTop, expectedTop, `${name} selected a different answer vs ${file}`);
            sum += errors.reduce((a, b) => a + b, 0);
            max = Math.max(largest, max);
            return {name, tokens: question.input_ids.length, decision: question.labels[actualTop], maxError: largest};
        });
        return {file, meanError: sum / (questions.length * 255), maxError: max, cases};
    });
    const {logits, ...stats} = result;
    console.log(JSON.stringify({...stats, checks}, null, 2));
} finally {
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
}
