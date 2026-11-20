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
        baseline: {type: "string", default: "8178386"},
        tokens: {type: "string", default: "124"},
        runs: {type: "string", default: "5"},
        hardware: {type: "boolean"},
    },
});
const [modelDir] = positionals;
const tokens = Number(values.tokens);
const runs = Number(values.runs);
if (positionals.length !== 1 || !Number.isSafeInteger(tokens) || tokens < 1 || tokens > 7999 ||
    !Number.isSafeInteger(runs) || runs < 1) {
    throw new Error("Usage: node tools/bench_encoder_projections.mjs MODEL [--baseline 8178386] " +
                    "[--tokens 124] [--runs 5] [--hardware]");
}
const gitOptions = {cwd: fileURLToPath(new URL("..", import.meta.url)), encoding: "utf8"};
const revision = execFileSync("git", ["rev-parse", "--verify", "--end-of-options", `${values.baseline}^{commit}`],
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
    const result = await page.evaluate(async ({baseline, revision, tokens, runs, software}) => {
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
            const [old, next] = await Promise.all([before, after].map(m => m.createEncoderKernels(device, packedDot)));
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
            for (const [name, residual] of [["mlp.Wi", false], ["attn.Wqkv", false],
                                           ["mlp.Wo", true], ["attn.Wo", true]]) {
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
                const variants = [[before, old.project], [after, next.project]].map(([module, pipeline]) => ({
                    pipeline, binding: bindings(pipeline, [weights, acts, output, dims]),
                    x: Math.ceil(width / module.PROJECTION_ROWS), y: Math.ceil(tokens / module.PROJECTION_TOKENS),
                    ms: [],
                }));
                let expected;
                let bitExact = true;
                let maxError = 0;
                // Check the projection and the rows outside its output slice during warm-up.
                for (let run = -1; run < runs; run++) {
                    for (const i of run % 2 === 0 ? [1, 0] : [0, 1]) {
                        const variant = variants[i];
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
                            if (i === 0) {
                                expected = bits.slice();
                            } else {
                                const reference = new Float32Array(expected.buffer);
                                for (let j = 0; j < values.length; j++) {
                                    if (bits[j] === expected[j]) continue;
                                    bitExact = false;
                                    const error = Math.abs(values[j] - reference[j]);
                                    maxError = Math.max(maxError, error);
                                    if (error > 3e-5 + 2e-6 * Math.abs(reference[j])) {
                                        throw new Error(`${name} differs from the baseline projection at element ${j}`);
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
                cases.push({stage: name, width, columns, tokens, bitExact, maxError,
                    baselineMs: variants[0].ms, tunedMs: variants[1].ms,
                    baselineMedianMs: median(variants[0].ms), tunedMedianMs: median(variants[1].ms)});
                for (const b of [input, acts, weights, output, readback, dims]) b.destroy();
            }
            if (uncaptured.length) throw new Error(uncaptured.join("\n"));
            return {revision, adapter: adapter.info.architecture, fallback: adapter.info.isFallbackAdapter,
                packedDot, timing: timestamps ? "gpu-timestamps" : "host", cases};
        } finally {
            URL.revokeObjectURL(baselineURL);
            device.destroy();
        }
    }, {baseline, revision, tokens, runs, software});
    console.log(JSON.stringify(result, null, 2));
} finally {
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
}
