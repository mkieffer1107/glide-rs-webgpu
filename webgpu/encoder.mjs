import { uploadWeightSet } from "./weights.mjs";
import {
    createEncoderKernels, createPanelProjection, createVectorProjection,
    PACKED_GROUP_BYTES, PROJECTION_ROWS, PROJECTION_TOKENS, QUERY_TILE,
    FLOAT_PROJECTION_ROWS, FLOAT_PROJECTION_TOKENS,
} from "./encoder-kernels.mjs";

const LOGITS = 255;
const MLP_ROWS = 1024;
const PARAM_BYTES = 32;
const SHORT_PROJECTION_BATCH = 256;
const embed = "embeddings.tok_embeddings.weight";

async function fetchJSON(url, signal) {
    const response = await fetch(url, {signal});
    if (!response.ok) throw new Error(`Cannot load ${url}: HTTP ${response.status}`);
    return response.json();
}

function loadConfig(manifest, c, d) {
    if (manifest.version !== 1 || c.model_type !== "modernbert" || c.hidden_size !== 1024 ||
        c.num_attention_heads !== 16 || c.num_hidden_layers !== 28 || c.intermediate_size !== 2624 ||
        c.vocab_size !== 50368 || c.hidden_activation !== "gelu" || c.attention_bias || c.mlp_bias ||
        c.norm_bias || c.is_causal || c.causal_mask || d.backbone !== "encoder" ||
        d.attention_mode !== "noncausal_bidirectional_encoder") {
        throw new Error("Expected the GLiDE-0.4B ModernBERT encoder");
    }
    const positive = value => Number.isFinite(value) && value > 0;
    const temperature = d.temperature ?? 1;
    if (!Number.isInteger(d.max_len) || !Number.isInteger(c.max_position_embeddings) ||
        d.max_len < 2 || d.max_len > Math.min(7999, c.max_position_embeddings) ||
        (d.pooling !== "last" && d.pooling !== "cls") ||
        !Number.isSafeInteger(c.global_attn_every_n_layers) || c.global_attn_every_n_layers < 1 ||
        !Number.isSafeInteger(c.local_attention) || c.local_attention < 0 || c.local_attention > 2 * 7999 ||
        ![c.cls_token_id, c.sep_token_id].every(id => Number.isInteger(id) && id >= 0 && id < c.vocab_size) ||
        ![c.norm_eps, c.global_rope_theta, c.local_rope_theta, temperature].every(positive)) {
        throw new Error("Invalid GLiDE encoder or decision settings");
    }

    const hidden = c.hidden_size;
    const intermediate = c.intermediate_size;
    const names = [];
    function expect(name, dtype, shape) {
        const t = manifest.tensors?.[name];
        const rowsPerUnit = dtype === "q8x16" ? 16 : 1;
        const columns = shape[1] ?? 1;
        const bytesPerUnit = dtype === "q8x16" ? columns / 32 * 544 : columns * (dtype === "f16" ? 2 : 4);
        if (t?.dtype !== dtype || !Array.isArray(t.shape) || t.shape.length !== shape.length ||
            !shape.every((n, i) => t.shape[i] === n) ||
            t.rowsPerUnit !== rowsPerUnit || t.bytesPerUnit !== bytesPerUnit ||
            t.nbytes !== shape[0] / rowsPerUnit * bytesPerUnit) {
            throw new Error(`Expected ${name} as ${dtype} [${shape.join(", ")}]`);
        }
        names.push(name);
    }
    expect(embed, "f16", [c.vocab_size, hidden]);
    expect("embeddings.norm.weight", "f32", [hidden]);
    expect("final_norm.weight", "f32", [hidden]);
    expect("readout.weight", "f32", [LOGITS, hidden]);
    for (let layer = 0; layer < c.num_hidden_layers; layer++) {
        const prefix = `layers.${layer}`;
        if (layer > 0) expect(`${prefix}.attn_norm.weight`, "f32", [hidden]);
        expect(`${prefix}.attn.Wqkv.weight`, "q8x16", [3 * hidden, hidden]);
        expect(`${prefix}.attn.Wo.weight`, "q8x16", [hidden, hidden]);
        expect(`${prefix}.mlp_norm.weight`, "f32", [hidden]);
        expect(`${prefix}.mlp.Wi.weight`, "q8x16", [2 * intermediate, hidden]);
        expect(`${prefix}.mlp.Wo.weight`, "q8x16", [hidden, intermediate]);
    }
    return {
        names, hidden, intermediate, temperature, vocab: c.vocab_size,
        layers: c.num_hidden_layers, heads: c.num_attention_heads, maxContext: d.max_len,
        cls: c.cls_token_id, sep: c.sep_token_id, poolLast: d.pooling === "last",
        fullEvery: c.global_attn_every_n_layers, localRadius: Math.floor(c.local_attention / 2),
        globalTheta: c.global_rope_theta, localTheta: c.local_rope_theta, epsilon: c.norm_eps,
    };
}

function rotaryTable(length, theta) {
    const f32 = new Float32Array(1);
    const bits = new Uint32Array(f32.buffer);
    const bf16 = value => {
        f32[0] = value;
        bits[0] = (bits[0] + 0x7fff + ((bits[0] >>> 16) & 1)) & 0xffff0000;
        return f32[0];
    };
    // The upstream BF16 model rounds the inverse frequencies and the final
    // sine/cosine values. Preserve its phase even for quantized projections.
    const frequency = Array.from({length: 32}, (_, i) =>
        bf16(1 / Math.fround(Math.pow(theta, i / 32))));
    const table = new Float32Array(length * 64);
    for (let pos = 0; pos < length; pos++) {
        for (let i = 0; i < 32; i++) {
            const angle = Math.fround(pos * frequency[i]);
            table[pos * 64 + i * 2] = bf16(Math.sin(angle));
            table[pos * 64 + i * 2 + 1] = bf16(Math.cos(angle));
        }
    }
    return table;
}

async function checked(device, encode) {
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    try {
        return encode();
    } finally {
        const errors = await Promise.all([device.popErrorScope(), device.popErrorScope()]);
        const error = errors.find(Boolean);
        if (error) throw new Error(error.message, {cause: error});
    }
}

function gpuArray(device, label, array) {
    const buffer = device.createBuffer({
        label, size: array.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
    buffer.unmap();
    return buffer;
}

const projectionKey = (columns, width) => `${columns}x${width}`;

function optionalTuningFailure(error) {
    const cause = error.cause ?? error;
    return cause instanceof GPUOutOfMemoryError || cause instanceof GPUValidationError ||
        cause instanceof GPUPipelineError || cause.name === "OperationError";
}

async function tuneProjections(device, weights, config, cap, baseline, candidates, signal) {
    const {hidden, intermediate} = config;
    const shapes = [
        ["attn.Wqkv", hidden, 3 * hidden, false], ["attn.Wo", hidden, hidden, true],
        ["mlp.Wi", hidden, 2 * intermediate, false], ["mlp.Wo", intermediate, hidden, true],
    ];
    const batches = [Math.min(124, cap)];
    if (cap > SHORT_PROJECTION_BATCH) batches.push(Math.min(672, cap));
    const capacity = batches.at(-1);
    const buffers = [];
    const allocate = (label, size, usage) => {
        const buffer = device.createBuffer({label: `GLiDE projection tuning ${label}`, size, usage});
        buffers.push(buffer);
        return buffer;
    };
    const selections = [];
    try {
        const {acts, output, readback, params} = await checked(device, () => {
            const storage = GPUBufferUsage.STORAGE;
            const write = GPUBufferUsage.COPY_DST;
            const bytes = capacity * Math.max(3 * hidden, 2 * intermediate) * 4;
            const acts = allocate("activations", capacity * intermediate / 32 * PACKED_GROUP_BYTES, storage | write);
            // Deterministic signed bytes with a nonzero scale per 32 channels.
            const words = new Uint32Array(acts.size / 4);
            let seed = 15937;
            for (let i = 0; i < words.length; i += 9) {
                words[i] = 0x3c800000;
                for (let j = 1; j < 9; j++) {
                    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
                    words[i + j] = seed;
                }
            }
            device.queue.writeBuffer(acts, 0, words);
            return {
                acts, output: allocate("output", bytes, storage | write | GPUBufferUsage.COPY_SRC),
                readback: allocate("readback", bytes, write | GPUBufferUsage.MAP_READ),
                params: allocate("parameters", PARAM_BYTES, write | GPUBufferUsage.UNIFORM),
            };
        });
        for (const tokens of batches) {
            const choices = new Map();
            for (const [name, columns, width, residual] of shapes) {
                signal?.throwIfAborted();
                const size = tokens * width * 4;
                device.queue.writeBuffer(params, 0, Uint32Array.of(columns / 32, tokens, width, 0, +residual));
                const matrix = weights.get(`layers.0.${name}.weight`);
                const variants = await checked(device, () => [baseline, ...candidates].map(tile => ({
                    tile, times: [],
                    binding: device.createBindGroup({
                        layout: tile.pipeline.getBindGroupLayout(0),
                        entries: [matrix, acts, output, params].map((buffer, binding) => ({
                            binding, resource: {buffer},
                        })),
                    }),
                })));
                const dispatch = ({tile, binding}, warmup) => {
                    const encoder = device.createCommandEncoder();
                    if (warmup) encoder.clearBuffer(output, 0, size);
                    const pass = encoder.beginComputePass();
                    pass.setPipeline(tile.pipeline);
                    pass.setBindGroup(0, binding);
                    pass.dispatchWorkgroups(Math.ceil(width / tile.rows), Math.ceil(tokens / tile.tokens));
                    pass.end();
                    if (warmup) encoder.copyBufferToBuffer(output, 0, readback, 0, size);
                    return encoder.finish();
                };
                let expected;
                for (const variant of variants) {
                    await checked(device, () => device.queue.submit([dispatch(variant, true)]));
                    await readback.mapAsync(GPUMapMode.READ, 0, size);
                    try {
                        const bits = new Uint32Array(readback.getMappedRange(0, size));
                        if (!expected) expected = bits.slice();
                        variant.exact = bits.every((word, i) => word === expected[i]);
                    } finally {
                        readback.unmap();
                    }
                }
                const eligible = variants.filter(variant => variant.exact);
                for (let run = 0; run < 3; run++) {
                    signal?.throwIfAborted();
                    const order = run % 2 ? eligible.slice().reverse() : eligible;
                    for (const variant of order) {
                        const command = dispatch(variant, false);
                        const started = performance.now();
                        device.queue.submit([command]);
                        await device.queue.onSubmittedWorkDone();
                        variant.times.push(performance.now() - started);
                    }
                }
                const original = variants[0];
                let best = original;
                let bestRatio = 0.95;
                for (const variant of eligible.slice(1)) {
                    // Compare panels directly to the best of the previous choices.
                    const reference = variant.tile.groups ? best : original;
                    const threshold = variant.tile.groups ? 0.95 : bestRatio;
                    const ratios = variant.times.map((ms, i) => ms / reference.times[i]).sort((a, b) => a - b);
                    if (ratios[1] < threshold) {
                        best = variant;
                        bestRatio = ratios[1];
                    }
                }
                choices.set(projectionKey(columns, width), best.tile);
            }
            selections.push(choices);
        }
        return selections;
    } finally {
        for (const buffer of buffers) buffer.destroy();
    }
}

function prepareBatch(prompts, poolLast, rows) {
    const tokens = new Uint32Array(rows * 2);
    const tiles = [];
    const pooled = [];
    let first = 0;
    for (const ids of prompts) {
        for (let pos = 0; pos < ids.length; pos++) {
            tokens[(first + pos) * 2] = ids[pos];
            tokens[(first + pos) * 2 + 1] = pos;
        }
        for (let q = 0; q < ids.length; q += QUERY_TILE) {
            tiles.push(first, ids.length, q, first + q, Math.min(QUERY_TILE, ids.length - q));
        }
        pooled.push(first, ids.length, poolLast ? ids.length - 2 : 0, pooled.length / 5, 1);
        first += ids.length;
    }
    const firstPooled = tiles.length / 5;
    return {tokens, tiles: Uint32Array.from(tiles.concat(pooled)), firstPooled};
}

class Commands {
    #dispatches = [];
    #splits = [];
    #stride;

    constructor(alignment) {
        this.#stride = Math.ceil(PARAM_BYTES / alignment) * alignment;
    }

    dispatch(pipeline, buffers, params, x, y = 1) {
        this.#dispatches.push({pipeline, buffers, params, x, y});
    }

    endLayer() {
        this.#splits.push(this.#dispatches.length);
    }

    get paramSize() { return this.#dispatches.length * this.#stride; }

    submit(device, parameterBuffer, output, readback, outputBytes) {
        const words = new Uint32Array(this.paramSize / 4);
        for (const [i, command] of this.#dispatches.entries()) {
            words.set(command.params, i * this.#stride / 4);
        }
        device.queue.writeBuffer(parameterBuffer, 0, words);
        const encoded = [];
        let start = 0;
        for (const end of [...this.#splits, this.#dispatches.length]) {
            const encoder = device.createCommandEncoder({label: "GLiDE forward"});
            const pass = encoder.beginComputePass();
            for (let i = start; i < end; i++) {
                const {pipeline, buffers, x, y} = this.#dispatches[i];
                const entries = buffers.map((buffer, binding) => ({binding, resource: {buffer}}));
                entries.push({
                    binding: buffers.length,
                    resource: {buffer: parameterBuffer, offset: i * this.#stride, size: PARAM_BYTES},
                });
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries}));
                pass.dispatchWorkgroups(x, y);
            }
            pass.end();
            if (end === this.#dispatches.length) {
                encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
            }
            encoded.push(encoder.finish());
            start = end;
        }
        device.queue.submit(encoded);
    }
}

/** WebGPU inference on the GLiDE-0.4B Q8 + FP16 embedding checkpoint. */
export class Glide04bWebGPU {
    #device;
    #ownsDevice;
    #config;
    #kernels;
    #weights;
    #rotary;
    #workspace = new Map();
    #maxBatchTokens;
    #projection;
    #closed = false;
    #lost;
    #pending = Promise.resolve();

    static async load(url, {adapter, device, signal, onProgress, maxBatchTokens, projection = "auto"} = {}) {
        if (!["auto", "int8", "int8-vector", "int8-panel", "fp32"].includes(projection)) {
            throw new Error("Unknown projection kernel");
        }
        const manifestURL = new URL(url, globalThis.location?.href);
        const manifest = await fetchJSON(manifestURL, signal);
        const [c, d] = await Promise.all([
            fetchJSON(new URL(manifest.config, manifestURL), signal),
            fetchJSON(new URL(manifest.decisionConfig, manifestURL), signal),
        ]);
        const config = loadConfig(manifest, c, d);
        const ownsDevice = !device;
        if (ownsDevice) {
            const gpu = globalThis.navigator?.gpu;
            if (!gpu) throw new Error("WebGPU requires a supported browser and a secure origin");
            adapter ??= await gpu.requestAdapter({powerPreference: "high-performance"});
            if (!adapter) throw new Error("No WebGPU adapter is available");
            device = await adapter.requestDevice();
        }
        let weights;
        const rotary = [];
        try {
            const maxBuffer = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
            const maxRows = Math.min(Math.floor(maxBuffer / (3 * config.hidden * 4)),
                device.limits.maxComputeWorkgroupsPerDimension);
            const cap = maxBatchTokens ?? Math.min(config.maxContext, maxRows);
            if (!Number.isSafeInteger(cap) || cap < 2 || cap > maxRows) {
                throw new RangeError(`This device can batch at most ${maxRows} tokens`);
            }
            const packedDot = globalThis.navigator?.gpu?.wgslLanguageFeatures
                ?.has("packed_4x8_integer_dot_product") ?? false;
            const software = adapter?.info.isFallbackAdapter ?? device.adapterInfo?.isFallbackAdapter;
            const useFloat = projection === "fp32" || (projection === "auto" && (software || !packedDot));
            const kernels = await createEncoderKernels(device, packedDot, useFloat);
            const primary = {
                pipeline: kernels.project,
                rows: useFloat ? FLOAT_PROJECTION_ROWS : PROJECTION_ROWS,
                tokens: useFloat ? FLOAT_PROJECTION_TOKENS : PROJECTION_TOKENS,
                name: useFloat ? "fp32" : "packed INT8",
            };
            const plan = {mode: useFloat ? "fp32" : "int8", primary};
            if (projection === "int8-vector") {
                plan.mode = projection;
                plan.primary = await createVectorProjection(device, 32);
            }
            if (projection === "int8-panel") {
                plan.mode = projection;
                plan.primary = await createPanelProjection(device, 4);
            }
            weights = await uploadWeightSet(device, manifest, manifestURL, config.names, {signal, onProgress});
            await checked(device, () => {
                rotary[0] = gpuArray(device, "GLiDE full RoPE", rotaryTable(config.maxContext, config.globalTheta));
                rotary[1] = config.localTheta === config.globalTheta ? rotary[0]
                    : gpuArray(device, "GLiDE local RoPE", rotaryTable(config.maxContext, config.localTheta));
            });
            if (projection === "auto" && packedDot && software === false && cap > 32) {
                try {
                    const candidates = await Promise.all([16, 32].map(tokens => createVectorProjection(device, tokens)));
                    const panels = await Promise.allSettled([2, 4].map(groups => createPanelProjection(device, groups)));
                    for (const panel of panels) {
                        if (panel.status === "fulfilled") candidates.push(panel.value);
                        else if (!optionalTuningFailure(panel.reason)) throw panel.reason;
                    }
                    plan.tuned = await tuneProjections(device, weights, config, cap, primary, candidates, signal);
                } catch (error) {
                    signal?.throwIfAborted();
                    if (!optionalTuningFailure(error)) throw error;
                }
            }
            signal?.throwIfAborted();
            return new Glide04bWebGPU(device, ownsDevice, config, kernels, weights, rotary, cap, plan);
        } catch (error) {
            for (const buffer of weights?.values() ?? []) buffer.destroy();
            for (const buffer of new Set(rotary)) buffer.destroy();
            if (ownsDevice) device.destroy();
            throw error;
        }
    }

    constructor(device, ownsDevice, config, kernels, weights, rotary, maxBatchTokens, projection) {
        this.#device = device;
        this.#ownsDevice = ownsDevice;
        this.#config = config;
        this.#kernels = kernels;
        this.#weights = weights;
        this.#rotary = rotary;
        this.#maxBatchTokens = maxBatchTokens;
        this.#projection = projection;
        device.lost.then(info => {
            this.#lost = new Error(`WebGPU device lost: ${info.message || info.reason}`);
        });
    }

    get maxContext() { return Math.min(this.#config.maxContext, this.#maxBatchTokens); }
    get maxBatchTokens() { return this.#maxBatchTokens; }
    get projection() { return this.#projection.mode; }
    get projectionTiles() {
        return this.#projection.tuned?.map(choices =>
            Object.fromEntries([...choices].map(([shape, tile]) => [shape, tile.name])));
    }
    get gpuBufferBytes() {
        let bytes = 0;
        for (const buffer of this.#weights.values()) bytes += buffer.size;
        for (const buffer of new Set(this.#rotary)) bytes += buffer.size;
        for (const buffer of this.#workspace.values()) bytes += buffer.size;
        return bytes;
    }

    /** Preserve input order. Each prompt is unpadded and already includes [CLS] and [SEP]. */
    async predictLogitsBatch(prompts) {
        if (this.#closed) throw new Error("The GLiDE WebGPU runtime is closed");
        if (!Array.isArray(prompts)) throw new TypeError("Expected an array of token sequences");
        const {cls, sep, vocab} = this.#config;
        const input = Array.from(prompts, (ids, i) => {
            if (!(Array.isArray(ids) || ids instanceof Uint32Array) || ids.length < 2 || ids.length > this.maxContext) {
                throw new RangeError(`Prompt ${i} must contain 2 to ${this.maxContext} tokens`);
            }
            if (ids[0] !== cls || ids[ids.length - 1] !== sep) {
                throw new Error(`Prompt ${i} must start with [CLS] and end with [SEP]`);
            }
            for (const id of ids) {
                if (!Number.isInteger(id) || id < 0 || id >= vocab) {
                    throw new RangeError(`Prompt ${i} contains an invalid token ID`);
                }
            }
            return Uint32Array.from(ids);
        });

        // Scratch buffers are reused. Queue concurrent calls so a later
        // request cannot overwrite an earlier request's inputs or outputs.
        const work = this.#pending.then(async () => {
            if (this.#lost) throw this.#lost;
            const output = [];
            let start = 0;
            while (start < input.length) {
                let end = start;
                let rows = 0;
                while (end < input.length && rows + input[end].length <= this.#maxBatchTokens) {
                    rows += input[end++].length;
                }
                try {
                    for (const logits of await this.#forward(input.slice(start, end), rows)) output.push(logits);
                } catch (error) {
                    this.#clearWorkspace();
                    throw error;
                }
                start = end;
            }
            return output;
        });
        this.#pending = work.catch(() => {});
        return work;
    }

    async predictLogits(ids) {
        const [logits] = await this.predictLogitsBatch([ids]);
        return logits;
    }

    probabilities(logits, count) {
        if (!Number.isInteger(count) || count < 1 || count > LOGITS || count > logits.length) {
            throw new RangeError("Expected 1 to 255 decision choices");
        }
        const row = Array.from(logits.slice(0, count));
        const max = Math.max(...row);
        const exps = row.map(x => Math.exp((x - max) / this.#config.temperature));
        const sum = exps.reduce((total, p) => total + p, 0);
        if (!Number.isFinite(sum) || sum <= 0) throw new Error("Invalid GLiDE logits");
        return exps.map(p => p / sum);
    }

    async predict(ids, count) {
        return this.probabilities(await this.predictLogits(ids), count);
    }

    destroy() {
        if (!this.#closed) {
            this.#closed = true;
            this.#pending = this.#pending.then(() => {
                this.#clearWorkspace();
                for (const buffer of this.#weights.values()) buffer.destroy();
                this.#weights.clear();
                for (const buffer of new Set(this.#rotary)) buffer.destroy();
                this.#rotary = [];
                if (this.#ownsDevice) this.#device.destroy();
            });
        }
        return this.#pending;
    }

    #clearWorkspace() {
        for (const buffer of this.#workspace.values()) buffer.destroy();
        this.#workspace.clear();
    }

    #buffer(name, size, usage) {
        const existing = this.#workspace.get(name);
        if (existing?.size >= size) return existing;
        existing?.destroy();
        const buffer = this.#device.createBuffer({label: `GLiDE ${name}`, size, usage});
        this.#workspace.set(name, buffer);
        return buffer;
    }

    async #forward(prompts, rows) {
        const device = this.#device;
        const c = this.#config;
        const batch = prompts.length;
        const {tokens, tiles, firstPooled} = prepareBatch(prompts, c.poolLast, rows);
        const outputBytes = batch * LOGITS * 4;
        const storage = GPUBufferUsage.STORAGE;
        const upload = storage | GPUBufferUsage.COPY_DST;
        const chunk = Math.min(rows, MLP_ROWS);
        const readback = await checked(device, () => {
            const ids = this.#buffer("tokens", tokens.byteLength, upload);
            const layout = this.#buffer("attention tiles", tiles.byteLength, upload);
            const hidden = this.#buffer("residual", rows * c.hidden * 4, storage);
            const pool = this.#buffer("pooled", batch * c.hidden * 4, storage);
            const scratch = this.#buffer("hidden scratch", rows * c.hidden * 4, storage);
            const wide = this.#buffer("QKV and MLP scratch",
                Math.max(rows * 3 * c.hidden, chunk * 2 * c.intermediate) * 4, storage);
            const acts = this.#buffer("packed activations",
                Math.max(rows * c.hidden, chunk * c.intermediate) / 32 * PACKED_GROUP_BYTES, storage);
            const logits = this.#buffer("logits", outputBytes, storage | GPUBufferUsage.COPY_SRC);
            const host = this.#buffer("logit readback", outputBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
            const plan = new Commands(device.limits.minUniformBufferOffsetAlignment);
            const kernels = this.#kernels;
            const weight = name => this.#weights.get(name);
            const add = (name, buffers, params, x, y) => plan.dispatch(kernels[name], buffers, params, x, y);
            const epsilon = new Uint32Array(Float32Array.of(c.epsilon).buffer)[0];
            const norm = (src, gain, dst, count, first = 0, embedding = false) =>
                add("norm", [src, ids, weight(gain), dst], [c.hidden, first, +embedding, epsilon], count);
            const pack = (src, count, columns, gated = false) =>
                add("pack", [src, acts], [columns, columns / 32, count, +gated],
                    Math.ceil(count * (columns / 32) / 64));
            const project = (name, dst, count, columns, width, first = 0, residual = false) => {
                const few = count <= (this.projection === "fp32" ? 16 : 8);
                const choices = count > 32 ? this.#projection.tuned?.[+(count > SHORT_PROJECTION_BATCH)] : undefined;
                const tile = choices?.get(projectionKey(columns, width)) ?? this.#projection.primary;
                plan.dispatch(few ? kernels.projectFew : tile.pipeline, [weight(name), acts, dst],
                    [columns / 32, count, width, first, +residual],
                    Math.ceil(width / (few ? 16 : tile.rows)),
                    Math.ceil(count / (few ? 4 : tile.tokens)));
            };

            norm(weight(embed), "embeddings.norm.weight", hidden, rows, 0, true);
            for (let layer = 0; layer < c.layers; layer++) {
                const prefix = `layers.${layer}`;
                const final = layer === c.layers - 1;
                const local = layer % c.fullEvery !== 0;
                if (layer > 0) norm(hidden, `${prefix}.attn_norm.weight`, scratch, rows);
                pack(layer === 0 ? hidden : scratch, rows, c.hidden);
                project(`${prefix}.attn.Wqkv.weight`, wide, rows, c.hidden, 3 * c.hidden);
                add("rope", [wide, ids, this.#rotary[+local]], [c.hidden], c.hidden / 128, rows);
                add("attention", [wide, layout, scratch],
                    [c.hidden, final ? firstPooled : 0, local ? c.localRadius : c.maxContext],
                    final ? batch : firstPooled, c.heads);

                // After the final attention, later operations are row-local.
                // Keep just the pooled residual for each prompt.
                const count = final ? batch : rows;
                const state = final ? pool : hidden;
                if (final) add("gather", [hidden, layout, pool], [c.hidden / 4, firstPooled],
                    Math.ceil(c.hidden / 256), batch);
                pack(scratch, count, c.hidden);
                project(`${prefix}.attn.Wo.weight`, state, count, c.hidden, c.hidden, 0, true);
                for (let first = 0; first < count; first += MLP_ROWS) {
                    const n = Math.min(MLP_ROWS, count - first);
                    norm(state, `${prefix}.mlp_norm.weight`, scratch, n, first);
                    pack(scratch, n, c.hidden);
                    project(`${prefix}.mlp.Wi.weight`, wide, n, c.hidden, 2 * c.intermediate);
                    pack(wide, n, c.intermediate, true);
                    project(`${prefix}.mlp.Wo.weight`, state, n, c.intermediate, c.hidden, first, true);
                }
                if (rows > 2048 && !final) plan.endLayer();
            }
            norm(pool, "final_norm.weight", scratch, batch);
            add("readout", [scratch, weight("readout.weight"), logits], [c.hidden / 4, LOGITS],
                Math.ceil(LOGITS / 64), batch);
            const parameters = this.#buffer("dispatch parameters", plan.paramSize,
                GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            device.queue.writeBuffer(ids, 0, tokens);
            device.queue.writeBuffer(layout, 0, tiles);
            plan.submit(device, parameters, logits, host, outputBytes);
            return host;
        });

        await readback.mapAsync(GPUMapMode.READ, 0, outputBytes);
        let result;
        try {
            result = new Float32Array(readback.getMappedRange(0, outputBytes).slice(0));
        } finally {
            readback.unmap();
        }
        if (result.some(x => !Number.isFinite(x))) throw new Error("Non-finite GLiDE logits");
        return Array.from({length: batch}, (_, i) => result.subarray(i * LOGITS, (i + 1) * LOGITS));
    }
}
