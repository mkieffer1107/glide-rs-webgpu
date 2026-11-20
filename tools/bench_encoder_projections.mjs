// Compare two GLiDE-0.4B projection kernels with the same Q8 tiles on one device.
import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";
import { createGlideServer } from "./serve_webgpu.mjs";

const {chromium} = createRequire(import.meta.url)("playwright");
const {positionals, values} = parseArgs({
    allowPositionals: true,
    options: {
        baseline: {type: "string"},
        tokens: {type: "string", default: "124"},
        runs: {type: "string", default: "5"},
        hardware: {type: "boolean"},
        vector: {type: "boolean"},
        panel: {type: "boolean"},
    },
});
const [modelDir] = positionals;
const counts = values.tokens.split(",").map(Number);
const runs = Number(values.runs);
if (positionals.length !== 1 || ![1, 4].includes(counts.length) ||
    !counts.every(n => Number.isSafeInteger(n) && n >= 1 && n <= 7999) ||
    !Number.isSafeInteger(runs) || runs < 1 || (values.vector && values.panel)) {
    throw new Error("Usage: node tools/bench_encoder_projections.mjs MODEL [--baseline 8178386] " +
                    "[--tokens 124|665,672,750,539] [--runs 5] [--vector|--panel] [--hardware]");
}
const gitOptions = {cwd: fileURLToPath(new URL("..", import.meta.url)), encoding: "utf8"};
const baselineRef = values.baseline ?? (values.panel ? "e6fa42e" : "8178386");
const revision = execFileSync("git", ["rev-parse", "--verify", "--end-of-options", `${baselineRef}^{commit}`],
    gitOptions).trim();
const baseline = execFileSync("git", ["show", `${revision}:webgpu/encoder-kernels.mjs`], gitOptions);

const server = createGlideServer(modelDir);
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
    const result = await page.evaluate(async ({baseline, revision, counts, runs, software, vector, panel}) => {
        const adapter = await navigator.gpu?.requestAdapter({forceFallbackAdapter: software,
            powerPreference: "high-performance"});
        if (!adapter || (!software && adapter.info.isFallbackAdapter)) {
            throw new Error(software ? "No software WebGPU adapter" : "No hardware WebGPU adapter");
        }
        const timestamps = adapter.features.has("timestamp-query");
        const packedDot = navigator.gpu.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product") ?? false;
        const device = await adapter.requestDevice({requiredFeatures: timestamps ? ["timestamp-query"] : []});
        const uncaptured = [];
        device.addEventListener("uncapturederror", event => uncaptured.push(event.error.message));
        const baselineURL = URL.createObjectURL(new Blob([baseline], {type: "text/javascript"}));
        try {
            const before = await import(baselineURL);
            const after = await import("/webgpu/encoder-kernels.mjs");
            const {readWeightRange} = await import("/webgpu/weights.mjs");
            const manifestURL = new URL("/model/webgpu.json", location.href);
            const manifest = await (await fetch(manifestURL)).json();
            const old = await before.createEncoderKernels(device, packedDot);
            const packedBaseline = {
                name: revision, pipeline: old.project, rows: before.PROJECTION_ROWS, tokens: before.PROJECTION_TOKENS,
            };
            let baselines;
            let candidates;
            if (panel) {
                if (typeof before.createVectorProjection !== "function") {
                    throw new Error("--panel needs a baseline revision with vector FMA projections");
                }
                baselines = await Promise.all([16, 32].map(n => before.createVectorProjection(device, n)));
                candidates = await Promise.all([2, 4].map(n => after.createPanelProjection(device, n)));
            } else if (vector) {
                candidates = await Promise.all([16, 32].map(n => after.createVectorProjection(device, n)));
            } else {
                candidates = [{
                    name: "current", pipeline: (await after.createEncoderKernels(device, packedDot)).project,
                    rows: after.PROJECTION_ROWS, tokens: after.PROJECTION_TOKENS,
                }];
            }
            const storage = GPUBufferUsage.STORAGE;
            const write = GPUBufferUsage.COPY_DST;
            const copy = GPUBufferUsage.COPY_SRC;
            const map = GPUBufferUsage.MAP_READ;
            const buffer = (size, usage) => device.createBuffer({size, usage});
            const queries = timestamps ? device.createQuerySet({type: "timestamp", count: 2}) : null;
            const resolved = timestamps ? buffer(16, GPUBufferUsage.QUERY_RESOLVE | copy) : null;
            const clock = timestamps ? buffer(16, map | write) : null;
            function params(words) {
                const out = device.createBuffer({size: 32, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true});
                new Uint32Array(out.getMappedRange()).set(words);
                out.unmap();
                return out;
            }
            function bindings(pipeline, buffers) {
                return device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
                    entries: buffers.map((buffer, binding) => ({binding, resource: {buffer}}))});
            }

            const cases = [];
            for (const [i, name] of ["mlp.Wi", "attn.Wqkv", "mlp.Wo", "attn.Wo"].entries()) {
                const tokens = counts[i] ?? counts[0];
                const residual = i >= 2;
                // The M1 Max profile used 16 tokens for attention output and 32 for the other projections.
                const original = panel ? baselines[i === 3 ? 0 : 1] : packedBaseline;
                const tensor = manifest.tensors[`layers.0.${name}.weight`];
                if (tensor?.dtype !== "q8x16") throw new Error(`Expected Q8 weights for ${name}`);
                const bytes = await readWeightRange(manifest, manifestURL,
                    {offset: tensor.offset, byteLength: tensor.nbytes});
                const [width, columns] = tensor.shape;
                const groups = columns / 32;
                // Exercise offset and residual stores as well as complete and partial token tiles.
                const first = residual ? 5 : 0;
                const outputBytes = (first + tokens + 2) * width * 4;
                const input = buffer(tokens * columns * 4, storage | write);
                const acts = buffer(tokens * groups * 36, storage);
                const weights = buffer(bytes.byteLength, storage | write);
                const output = buffer(outputBytes, storage | copy | write);
                const readback = buffer(outputBytes, map | write);
                device.queue.writeBuffer(input, 0, Float32Array.from({length: tokens * columns},
                    (_, i) => 0.87 * Math.cos(i * (i % 97 + 1))));
                device.queue.writeBuffer(weights, 0, bytes);
                const packParams = params([columns, groups, tokens, 0]);
                const encoder = device.createCommandEncoder();
                const pack = encoder.beginComputePass();
                pack.setPipeline(old.pack);
                pack.setBindGroup(0, bindings(old.pack, [input, acts, packParams]));
                pack.dispatchWorkgroups(Math.ceil(tokens * groups / 64));
                pack.end();
                device.queue.submit([encoder.finish()]);
                await device.queue.onSubmittedWorkDone();
                packParams.destroy();

                const dims = params([groups, tokens, width, first, +residual]);
                const initial = Float32Array.from({length: outputBytes / 4}, (_, i) => 1.3 * Math.sin(i + 0.123));
                const variants = [original, ...candidates].map(tile => ({
                    name: tile.name, pipeline: tile.pipeline, binding: bindings(tile.pipeline, [weights, acts, output, dims]),
                    x: Math.ceil(width / tile.rows), y: Math.ceil(tokens / tile.tokens),
                    ms: [], bitExact: true, maxError: 0,
                }));
                let expected;
                // Check the projection and the rows outside its output slice during warm-up.
                for (let run = -1; run < runs; run++) {
                    const order = run % 2 === 0 ? variants.slice().reverse() : variants;
                    for (const variant of order) {
                        device.queue.writeBuffer(output, 0, initial);
                        const cmd = device.createCommandEncoder();
                        const pass = cmd.beginComputePass(timestamps ? {timestampWrites: {
                            querySet: queries, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
                        }} : {});
                        pass.setPipeline(variant.pipeline);
                        pass.setBindGroup(0, variant.binding);
                        pass.dispatchWorkgroups(variant.x, variant.y);
                        pass.end();
                        if (timestamps) {
                            cmd.resolveQuerySet(queries, 0, 2, resolved, 0);
                            cmd.copyBufferToBuffer(resolved, 0, clock, 0, 16);
                        }
                        if (run < 0) cmd.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
                        if (!timestamps) await device.queue.onSubmittedWorkDone();
                        const start = performance.now();
                        device.queue.submit([cmd.finish()]);
                        let ms;
                        if (timestamps) {
                            await clock.mapAsync(GPUMapMode.READ);
                            const [begin, end] = new BigUint64Array(clock.getMappedRange());
                            ms = Number(end - begin) / 1e6;
                            clock.unmap();
                        } else {
                            await device.queue.onSubmittedWorkDone();
                            ms = performance.now() - start;
                        }
                        if (run >= 0) variant.ms.push(ms);
                        if (run < 0) {
                            await readback.mapAsync(GPUMapMode.READ);
                            const data = readback.getMappedRange();
                            const bits = new Uint32Array(data);
                            const values = new Float32Array(data);
                            if (values.some(value => !Number.isFinite(value))) {
                                throw new Error(`${name} produced non-finite values`);
                            }
                            if (variant === variants[0]) {
                                expected = bits.slice();
                            } else {
                                const reference = new Float32Array(expected.buffer);
                                for (let j = 0; j < values.length; j++) {
                                    if (bits[j] === expected[j]) continue;
                                    variant.bitExact = false;
                                    const error = Math.abs(values[j] - reference[j]);
                                    variant.maxError = Math.max(variant.maxError, error);
                                    if (vector || panel || error > 3e-5 + 2e-6 * Math.abs(reference[j])) {
                                        throw new Error(`${name} ${variant.name} differs from baseline at element ${j}`);
                                    }
                                }
                            }
                            readback.unmap();
                        }
                    }
                }
                const median = values => {
                    const sorted = values.slice().sort((a,b) => a-b);
                    return (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2;
                };
                const [reference, ...results] = variants.map(({name, ms, bitExact, maxError}) => ({
                    name, ms, bitExact, maxError, medianMs: median(ms),
                }));
                const result = vector || panel ? {baselineKernel: reference.name, candidates: results} : {
                    bitExact: results[0].bitExact, maxError: results[0].maxError,
                    tunedMs: results[0].ms, tunedMedianMs: results[0].medianMs,
                };
                cases.push({stage: name, width, columns, tokens,
                    baselineMs: reference.ms, baselineMedianMs: reference.medianMs, ...result});
                for (const b of [input, acts, weights, output, readback, dims]) b.destroy();
            }
            if (uncaptured.length) throw new Error(uncaptured.join("\n"));
            return {revision, adapter: adapter.info.architecture, fallback: adapter.info.isFallbackAdapter,
                packedDot, timing: timestamps ? "gpu-timestamps" : "host", cases};
        } finally {
            URL.revokeObjectURL(baselineURL);
            device.destroy();
        }
    }, {baseline, revision, counts, runs, software, vector: Boolean(values.vector), panel: Boolean(values.panel)});
    console.log(JSON.stringify(result, null, 2));
} finally {
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
}
