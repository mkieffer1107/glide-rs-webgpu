// Exercise the WebGPU attention shader at context boundaries without running all 28 layers.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { createGlideServer } from "./serve_webgpu.mjs";

const {chromium} = createRequire(import.meta.url)("playwright");
const server = createGlideServer(".");
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let browser;
try {
    browser = await chromium.launch({
        executablePath: process.env.GLIDE_CHROMIUM || "/usr/bin/chromium",
        headless: true,
        args: ["--no-sandbox", "--disable-dev-shm-usage", "--enable-unsafe-webgpu",
               "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    });
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async () => {
        const {createEncoderKernels, QUERY_TILE} = await import("/webgpu/encoder-kernels.mjs");
        const adapter = await navigator.gpu?.requestAdapter({forceFallbackAdapter: true});
        if (!adapter) throw new Error("No software WebGPU adapter");
        const device = await adapter.requestDevice();
        const {attention} = await createEncoderKernels(device, false, true);
        const width = 64;
        const cases = [
            {name: "ragged full", length: 17, radius: 7999},
            {name: "ragged full tail", length: 33, radius: 7999},
            {name: "sliding window", length: 131, radius: 64},
            {name: "full context keys", length: 7999, radius: 7999, queries: [0, 3999, 7997]},
        ];
        const tokens = cases.reduce((sum, c) => sum + c.length, 0);
        const qkv = Float32Array.from({length: tokens * width * 3}, (_, i) =>
            Math.sin(i * 0.0317 + (i % 117) * 0.0793));
        const tiles = [];
        const checks = [];
        let firstRow = 0;
        let outRow = 0;
        for (const {name, length, queries, radius} of cases) {
            const firstTile = tiles.length / 5;
            const pos = queries ?? Array.from({length}, (_, i) => i);
            for (let i = 0; i < pos.length; i += queries ? 1 : QUERY_TILE) {
                const count = queries ? 1 : Math.min(QUERY_TILE, pos.length - i);
                tiles.push(firstRow, length, pos[i], outRow + i, count);
            }
            checks.push({name, firstRow, firstTile, length, radius, pos, outRow,
                tileCount: tiles.length / 5 - firstTile});
            firstRow += length;
            outRow += pos.length;
        }

        const st = GPUBufferUsage.STORAGE;
        const buffer = (data, usage) => {
            const buf = device.createBuffer({size: data.byteLength, usage, mappedAtCreation: true});
            new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer));
            buf.unmap();
            return buf;
        };
        const input = buffer(qkv, st);
        const layout = buffer(Uint32Array.from(tiles), st);
        const size = outRow * width * 4;
        const out = device.createBuffer({size, usage: st | GPUBufferUsage.COPY_SRC});
        const readback = device.createBuffer({size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
        device.pushErrorScope("validation");
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(attention);
        const params = [];
        for (const {firstTile, radius, tileCount} of checks) {
            const p = buffer(Uint32Array.of(width, firstTile, radius), GPUBufferUsage.UNIFORM);
            params.push(p);
            pass.setBindGroup(0, device.createBindGroup({
                layout: attention.getBindGroupLayout(0),
                entries: [input, layout, out, p].map((buf, binding) => ({binding, resource: {buffer: buf}})),
            }));
            pass.dispatchWorkgroups(tileCount, 1);
        }
        pass.end();
        encoder.copyBufferToBuffer(out, 0, readback, 0, size);
        device.queue.submit([encoder.finish()]);
        const validation = await device.popErrorScope();
        if (validation) throw new Error(validation.message);
        await readback.mapAsync(GPUMapMode.READ);
        const actual = new Float32Array(readback.getMappedRange());
        const report = checks.map(({name, firstRow, length, radius, pos, outRow}) => {
            let maxError = 0;
            for (const [r, query] of pos.entries()) {
                const q = (firstRow + query) * 3 * width;
                const lo = Math.max(0, query - radius);
                const hi = Math.min(length, query + radius + 1);
                const scores = new Float64Array(hi - lo);
                for (let key = lo; key < hi; key++) {
                    const k = ((firstRow + key) * 3 + 1) * width;
                    let sum = 0;
                    for (let d = 0; d < width; d++) sum += qkv[q + d] * qkv[k + d];
                    scores[key - lo] = sum * 0.125;
                }
                const max = Math.max(...scores);
                let denom = 0;
                const expected = new Float64Array(width);
                for (let key = lo; key < hi; key++) {
                    const prob = Math.exp(scores[key - lo] - max);
                    const v = ((firstRow + key) * 3 + 2) * width;
                    denom += prob;
                    for (let d = 0; d < width; d++) expected[d] += prob * qkv[v + d];
                }
                for (let d = 0; d < width; d++) {
                    maxError = Math.max(maxError, Math.abs(actual[(outRow + r) * width + d] - expected[d] / denom));
                }
            }
            return {name, length, radius, queries: pos.length, maxError};
        });
        readback.unmap();
        for (const b of [input, layout, out, readback, ...params]) b.destroy();
        device.destroy();
        return report;
    });
    for (const row of result) assert(row.maxError <= 3e-5, JSON.stringify(row));
    console.log(JSON.stringify(result, null, 2));
} finally {
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
}
