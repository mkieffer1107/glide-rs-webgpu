// Run with Node and Playwright. Uses real row tiles from converted checkpoints.
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { createRequire } from "node:module";
import { weightRanges } from "../webgpu/weights.mjs";

const { chromium } = createRequire(import.meta.url)("playwright");
const models = process.argv.slice(2).map(dir => ({
    dir: path.resolve(dir),
    manifest: JSON.parse(fs.readFileSync(path.join(dir, "webgpu.json"))),
}));
if (!models.length) throw new Error("Pass one or more model directories containing webgpu.json");

const stages = [
    ["layers.0.mlp.gate_proj.weight", 16, 64, 3],
    ["layers.3.self_attn.q_proj.weight", 32, 32, 17],
];
const levels = [-127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127];

function fixture(model, tensorName, firstRow, rows, tokens) {
    const t = model.manifest.tensors[tensorName];
    const columns = t.shape[1], groups = Math.ceil(columns / 32);
    const [range] = weightRanges(t, model.manifest.chunkBytes, firstRow, rows);
    const weights = Buffer.alloc(range.byteLength);
    const fd = fs.openSync(path.join(model.dir, "weights.bin"), "r");
    try {
        assert.equal(fs.readSync(fd, weights, 0, weights.length, range.offset), weights.length);
    } finally {
        fs.closeSync(fd);
    }
    const view = new DataView(weights.buffer, weights.byteOffset, weights.length);
    const input = Float32Array.from({length: tokens * columns}, (_, i) =>
        i < 32 ? 0 : ((i * 47 + 13 * (i % 29)) % 501 - 250) / 256);
    const expected = new Float32Array(tokens * rows);
    const f32 = Math.fround;
    const even = x => {
        const low = Math.floor(x), rest = x - low;
        return low + Number(rest > 0.5 || (rest === 0.5 && low % 2 !== 0));
    };
    const blockBytes = t.bytesPerUnit / groups;
    function signedCode(base, row, col) {
        if (t.dtype === "q8x16") return weights[base + 32 + Math.floor(col / 4) * 64 + row * 4 + col % 4] - 128;
        const part = base + 32 + Math.floor(col / 8) * (t.dtype === "q6x16" ? 96 : 64);
        const half = Math.floor(col % 8 / 4);
        const low = (weights[part + row * 4 + col % 4] >> (half * 4)) & 15;
        if (t.dtype === "q4nx16") return levels[low];
        if (t.dtype === "q4x16") return low - 8;
        assert.equal(t.dtype, "q6x16");
        const high = (weights[part + 64 + half * 16 + row] >> (2 * (col % 4))) & 3;
        return low + 16 * high - 32;
    }
    for (let token = 0; token < tokens; token++) {
        for (let group = 0; group < groups; group++) {
            const values = input.subarray(token * columns + 32 * group,
                                          token * columns + Math.min(columns, 32 * (group + 1)));
            const peak = Math.max(...Array.from(values, Math.abs));
            const inverse = peak === 0 ? 0 : f32(127 / peak);
            const a = Array.from(values, x => even(f32(x * inverse)));
            for (let row = 0; row < rows; row++) {
                const lane = row % 16;
                const base = (Math.floor(row / 16) * groups + group) * blockBytes;
                const scale = f32(f32(peak / 127) * view.getFloat16(base + lane * 2, true));
                let sum = 0;
                for (let c = 0; c < a.length; c++) sum += signedCode(base, lane, c) * a[c];
                const k = token * rows + row;
                expected[k] = f32(expected[k] + sum * scale);
            }
        }
    }
    return { tensorName, firstRow, rows, tokens, columns, dtype: t.dtype, input: [...input], expected: [...expected] };
}

const assets = new Map([
    ["/weights.mjs", {file: new URL("../webgpu/weights.mjs", import.meta.url), mime: "text/javascript"}],
    ["/projection.mjs", {file: new URL("../webgpu/projection.mjs", import.meta.url), mime: "text/javascript"}],
]);
for (const [i, model] of models.entries()) {
    assets.set(`/${i}/webgpu.json`, {file: path.join(model.dir, "webgpu.json"), mime: "application/json"});
    assets.set(`/${i}/weights.bin`, {file: path.join(model.dir, "weights.bin"), weightBytes: model.manifest.weightBytes});
}
const server = http.createServer((req, res) => {
    if (req.url === "/") return res.end("GLiDE WebGPU projection test");
    if (req.url === "/whole-buffer.bin") return res.end(Buffer.alloc(4));
    const asset = assets.get(req.url);
    if (!asset) return res.writeHead(404).end();
    if (!asset.weightBytes) {
        res.setHeader("Content-Type", asset.mime);
        return fs.createReadStream(asset.file).pipe(res);
    }
    const match = /^bytes=(\d+)-(\d+)$/.exec(req.headers.range ?? "");
    if (!match) return res.writeHead(416).end();
    const start = Number(match[1]), end = Number(match[2]);
    if (end < start || end >= asset.weightBytes) return res.writeHead(416).end();
    res.writeHead(206, {
        "Content-Range": `bytes ${start}-${end}/${asset.weightBytes}`,
        "Content-Length": end - start + 1,
    });
    return fs.createReadStream(asset.file, {start, end}).pipe(res);
});

await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let browser;
try {
    const chromiumPath = process.env.GLIDE_CHROMIUM || "/usr/bin/chromium";
    const software = process.env.GLIDE_WEBGPU_HARDWARE !== "1";
    browser = await chromium.launch({
        executablePath: chromiumPath, headless: true,
        args: ["--no-sandbox", "--disable-dev-shm-usage", "--enable-unsafe-webgpu",
               ...(software ? ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] : [])],
    });
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const results = await page.evaluate(async ({ software, cases }) => {
        const { weightRanges, readWeightRange, uploadTensor } = await import("/weights.mjs");
        const { createProjectionKernels } = await import("/projection.mjs");
        const adapter = await navigator.gpu.requestAdapter({forceFallbackAdapter: software});
        if (!adapter) throw new Error("No requested WebGPU adapter");
        if (!software && adapter.info.isFallbackAdapter) throw new Error("A hardware WebGPU adapter is required");
        const device = await adapter.requestDevice();
        const adapterInfo = {vendor: adapter.info.vendor, architecture: adapter.info.architecture,
                             fallback: adapter.info.isFallbackAdapter};
        const errors = [];
        device.addEventListener("uncapturederror", event => errors.push(event.error.message));
        const out = [];
        const kernels = [createProjectionKernels(device), createProjectionKernels(device, {packedDot: false})];
        for (const [i, entry] of cases.entries()) {
            const manifestURL = new URL(`/${i}/webgpu.json`, location.href);
            const manifest = await (await fetch(manifestURL)).json();
            const cap = Math.min(manifest.chunkBytes, device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
            let totalRanges = 0;
            let largestBuffer = 0;
            for (const tensor of Object.values(manifest.tensors)) {
                const ranges = weightRanges(tensor, cap);
                let offset = tensor.offset;
                for (const range of ranges) {
                    if (range.offset !== offset || range.bufferBytes > cap) {
                        throw new Error("Invalid shard placement");
                    }
                    offset += range.byteLength;
                    largestBuffer = Math.max(largestBuffer, range.bufferBytes);
                }
                if (offset !== tensor.offset + tensor.nbytes) throw new Error("Incomplete tensor ranges");
                totalRanges += ranges.length;
            }
            const embedding = manifest.tensors["embed_tokens.weight"];
            const embeddingRanges = weightRanges(embedding, cap);
            if (embeddingRanges.reduce((sum, range) => sum + range.byteLength, 0) !== embedding.nbytes) {
                throw new Error("Embedding shard byte count disagrees with the index");
            }
            const firstEmbedding = embeddingRanges[0];
            for await (const chunk of uploadTensor(device, manifest, manifestURL, "embed_tokens.weight", {
                firstRow: 0, rowCount: firstEmbedding.rowCount,
            })) {
                if (chunk.buffer.size !== firstEmbedding.bufferBytes) throw new Error("Incorrect embedding upload size");
                chunk.buffer.destroy();
            }
            // Reproduce a slice starting at the junction of two embedding shards.
            if (embeddingRanges.length < 2) throw new Error("Expected split embedding weights");
            const nearBoundary = weightRanges(embedding, embedding.bytesPerUnit,
                                               embeddingRanges[1].firstRow - 1, 2);
            for (const range of nearBoundary) {
                await readWeightRange(manifest, manifestURL, range);
            }
            // Guard against a server that ignores Range and returns the entire file.
            let rejected = false;
            try {
                await readWeightRange({...manifest, weights: "/whole-buffer.bin"}, manifestURL, nearBoundary[0]);
            } catch (err) {
                rejected = err.message.includes("HTTP byte range");
            }
            if (!rejected) throw new Error("HTTP 200 should have been rejected");

            for (const sample of entry.samples) {
                const received = [];
                for await (const chunk of uploadTensor(device, manifest, manifestURL, sample.tensorName, {
                    firstRow: sample.firstRow, rowCount: sample.rows,
                })) received.push(chunk);
                if (received.length !== 1) throw new Error("Expected one projection test buffer");
                const weights = received[0].buffer;
                const input = device.createBuffer({size: sample.input.length * 4,
                    usage: GPUBufferUsage.STORAGE, mappedAtCreation: true});
                new Float32Array(input.getMappedRange()).set(sample.input);
                input.unmap();
                for (const kernel of kernels) {
                    device.pushErrorScope("validation");
                    const encoder = device.createCommandEncoder();
                    const acts = kernel.pack(encoder, input, sample.tokens, sample.columns);
                    const y = kernel.project(encoder, weights, sample.dtype, sample.rows, acts);
                    const readback = device.createBuffer({size: y.buffer.size,
                        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
                    encoder.copyBufferToBuffer(y.buffer, 0, readback, 0, readback.size);
                    device.queue.submit([encoder.finish()]);
                    const validation = await device.popErrorScope();
                    if (validation) throw new Error(validation.message);
                    await readback.mapAsync(GPUMapMode.READ);
                    const result = new Float32Array(readback.getMappedRange());
                    let maxError = 0;
                    for (let j = 0; j < result.length; j++) {
                        const diff = Math.abs(result[j] - sample.expected[j]);
                        if (!Number.isFinite(diff) || diff > 3e-5 + 2e-6 * Math.abs(sample.expected[j])) {
                            throw new Error(`${sample.dtype} projection mismatch at ${j}: ${result[j]} vs ${sample.expected[j]}`);
                        }
                        maxError = Math.max(maxError, diff);
                    }
                    out.push({model: entry.name, dtype: sample.dtype, rows: sample.rows, tokens: sample.tokens,
                        columns: sample.columns, tensor: sample.tensorName, packedDot: kernel.packedDot, maxError,
                        weightBytes: manifest.weightBytes, embeddingRanges: embeddingRanges.length,
                        totalRanges, largestBuffer, embeddingUploadBytes: firstEmbedding.bufferBytes});
                    readback.unmap();
                    for (const buffer of [acts.buffer, acts.params, y.buffer, y.params, readback]) buffer.destroy();
                }
                weights.destroy();
                input.destroy();
            }
        }
        device.destroy();
        if (errors.length) throw new Error(errors.join("\n"));
        return {adapter: adapterInfo, bindingLimit: device.limits.maxStorageBufferBindingSize, cases: out};
    }, {
        software,
        cases: models.map(model => ({name: path.basename(model.dir), samples: stages.map(s => fixture(model, ...s))})),
    });
    console.log(JSON.stringify(results, null, 2));
} finally {
    await browser?.close();
    server.close();
}
