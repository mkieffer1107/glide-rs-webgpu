// GLiDE-0.4B uses 64-channel attention heads and group-32, tile-16 Q8 weights.
export const QUERY_TILE = 16;
export const PROJECTION_TOKENS = 32;
export const PROJECTION_ROWS = 32;
export const PACKED_GROUP_BYTES = 36;
export const FLOAT_PROJECTION_TOKENS = 32;
export const FLOAT_PROJECTION_ROWS = 64;

const norm = `
struct Params { width: u32, first: u32, embedding: u32, epsilon: f32 }
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read> tokens: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> gain: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
var<workgroup> partial: array<vec2<f32>, 128>;

fn value(row: u32, col: u32) -> f32 {
    if (p.embedding != 0u) {
        let index = tokens[row].x * p.width + col;
        return unpack2x16float(source[index / 2u])[index % 2u];
    }
    return bitcast<f32>(source[row * p.width + col]);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = p.first + group.x;
    var sums = vec2(0.0);
    for (var c = lane; c < p.width; c += 128u) {
        let x = value(row, c);
        sums += vec2(x, x * x);
    }
    partial[lane] = sums;
    for (var stride = 64u; stride > 0u; stride /= 2u) {
        workgroupBarrier();
        if (lane < stride) { partial[lane] += partial[lane + stride]; }
    }
    workgroupBarrier();
    let moments = partial[0] / f32(p.width);
    let scale = inverseSqrt(max(moments.y - moments.x * moments.x, 0.0) + p.epsilon);
    for (var c = lane; c < p.width; c += 128u) {
        output[group.x * p.width + c] = (value(row, c) - moments.x) * scale * gain[c];
    }
}
`;

const pack = `
struct Params { columns: u32, groups: u32, rows: u32, gated: u32 }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

// GELU via an erf approximation (maximum erf error below 1.5e-7).
fn gelu(x: vec4<f32>) -> vec4<f32> {
    let a = abs(x) * 0.7071067811865475;
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
        - 0.284496736) * t + 0.254829592) * t;
    return 0.5 * x * (1.0 + sign(x) * (1.0 - poly * exp(-a * a)));
}

fn roundEven(x: vec4<f32>) -> vec4<i32> {
    let low = floor(x);
    let rest = x - low;
    let integer = vec4<i32>(low);
    let tie = select(vec4<i32>(0), integer & vec4<i32>(1), rest == vec4(0.5));
    return integer + max(tie, select(vec4<i32>(0), vec4<i32>(1), rest > vec4(0.5)));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.rows * p.groups) { return; }
    let row = id.x / p.groups;
    let width = p.columns / 4u;
    let start = row * width * (1u + p.gated) + (id.x % p.groups) * 8u;
    var words: array<vec4<f32>, 8>;
    var peak = vec4(0.0);
    for (var i = 0u; i < 8u; i++) {
        var value = input[start + i];
        if (p.gated != 0u) { value = gelu(value) * input[start + width + i]; }
        words[i] = value;
        peak = max(peak, abs(value));
    }
    let maxValue = max(max(peak.x, peak.y), max(peak.z, peak.w));
    let scale = bitcast<u32>(maxValue / 127.0);
    let inverse = select(0.0, 127.0 / maxValue, maxValue > 0.0);
    let dest = id.x * 9u;
    packed[dest] = scale;
    for (var i = 0u; i < 8u; i++) {
        let v = bitcast<vec4<u32>>(roundEven(words[i] * inverse)) & vec4(255u);
        packed[dest + 1u + i] = v.x | (v.y << 8u) | (v.z << 16u) | (v.w << 24u);
    }
}
`;

const projectionBindings = `
struct Params { groups: u32, tokens: u32, rows: u32, first: u32, residual: u32 }
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> acts: array<u32>;
@group(0) @binding(3) var<uniform> p: Params;
`;

const projectionHeader = projectionBindings + `
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn store(token: u32, row: u32, value: f32) {
    if (token < p.tokens && row < p.rows) {
        let i = (p.first + token) * p.rows + row;
        if (p.residual != 0u) {
            output[i] += value;
        } else {
            output[i] = value;
        }
    }
}
`;

function dotWGSL(packed) {
    if (packed) return "fn idot(a: u32, b: u32) -> i32 { return dot4I8Packed(a, b); }";
    return `
fn signedBytes(word: u32) -> vec4<i32> {
    let bits = (vec4(word) >> vec4(0u, 8u, 16u, 24u)) & vec4(255u);
    return vec4<i32>(bits ^ vec4(128u)) - vec4<i32>(128);
}
fn idot(a: u32, b: u32) -> i32 {
    let v = signedBytes(a) * signedBytes(b);
    return v.x + v.y + v.z + v.w;
}
`;
}

function project(packed) {
    // Generate the four independent token accumulators with constant offsets.
    const tokenOffsets = [0, 8, 16, 24];
    const accumulators = tokenOffsets.map((_, i) => `var acc${i} = vec4(0.0);`).join("\n    ");
    const products = tokenOffsets.map((offset, i) => `{
            let a = block * TOKENS + local.y + ${offset}u;
            let words = Words(aLo[a], aHi[a]);
            let sums = vec4(dotGroup(words, w0), dotGroup(words, w1),
                            dotGroup(words, w2), dotGroup(words, w3));
            acc${i} = fma(vec4<f32>(sums), scales * aScale[a], acc${i});
        }`).join("\n        ");
    const stores = tokenOffsets.map((offset, i) => `store(token + ${offset}u, row, acc${i});`).join("\n    ");
    return projectionBindings + dotWGSL(packed) + `
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

// Each of 64 lanes computes four tokens by four consecutive output rows.
// The 32x32 tile reuses each weight word across 32 tokens. Six input groups
// use 13,824 bytes of shared storage, within the 16 KiB WebGPU minimum.
const TOKENS = ${PROJECTION_TOKENS}u;
const ROWS = ${PROJECTION_ROWS}u;
const K_GROUPS = 6u;
struct Words { lo: vec4<u32>, hi: vec4<u32> }
var<workgroup> wLo: array<vec4<u32>, K_GROUPS * ROWS>;
var<workgroup> wHi: array<vec4<u32>, K_GROUPS * ROWS>;
var<workgroup> aLo: array<vec4<u32>, K_GROUPS * TOKENS>;
var<workgroup> aHi: array<vec4<u32>, K_GROUPS * TOKENS>;
var<workgroup> wScale: array<f32, K_GROUPS * ROWS>;
var<workgroup> aScale: array<f32, K_GROUPS * TOKENS>;

fn dotGroup(a: Words, w: Words) -> i32 {
    return idot(a.lo.x, w.lo.x) + idot(a.lo.y, w.lo.y)
         + idot(a.lo.z, w.lo.z) + idot(a.lo.w, w.lo.w)
         + idot(a.hi.x, w.hi.x) + idot(a.hi.y, w.hi.y)
         + idot(a.hi.z, w.hi.z) + idot(a.hi.w, w.hi.w);
}

fn store(token: u32, row: u32, value: vec4<f32>) {
    if (token < p.tokens && row < p.rows) {
        let index = ((p.first + token) * p.rows + row) / 4u;
        if (p.residual != 0u) {
            output[index] += value;
        } else {
            output[index] = value;
        }
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) gid: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let firstRow = gid.x * ROWS;
    let firstToken = gid.y * TOKENS;
    ${accumulators}
    for (var start = 0u; start < p.groups; start += K_GROUPS) {
        let count = min(K_GROUPS, p.groups - start);
        for (var i = lane; i < count * ROWS; i += 64u) {
            let col = i % ROWS;
            let row = firstRow + col;
            var words: Words;
            var scale = 0.0;
            if (row < p.rows) {
                let base = (row / 16u * p.groups + start + i / ROWS) * 136u;
                let word = base + 8u + row % 16u;
                let sign = vec4(0x80808080u);
                words.lo = vec4(weights[word], weights[word + 16u], weights[word + 32u], weights[word + 48u]) ^ sign;
                words.hi = vec4(weights[word + 64u], weights[word + 80u], weights[word + 96u], weights[word + 112u]) ^ sign;
                scale = unpack2x16float(weights[base + (row % 16u) / 2u])[row % 2u];
            }
            // Keep adjacent lanes' vector loads contiguous for each of the four output channels.
            let dst = i - col + (col % 4u) * 8u + col / 4u;
            wLo[dst] = words.lo;
            wHi[dst] = words.hi;
            wScale[dst] = scale;
        }
        for (var i = lane; i < count * TOKENS; i += 64u) {
            let token = firstToken + i % TOKENS;
            var words: Words;
            var scale = 0.0;
            if (token < p.tokens) {
                let word = (token * p.groups + start + i / TOKENS) * 9u;
                words.lo = vec4(acts[word + 1u], acts[word + 2u], acts[word + 3u], acts[word + 4u]);
                words.hi = vec4(acts[word + 5u], acts[word + 6u], acts[word + 7u], acts[word + 8u]);
                scale = bitcast<f32>(acts[word]);
            }
            aLo[i] = words.lo;
            aHi[i] = words.hi;
            aScale[i] = scale;
        }
        workgroupBarrier();
        for (var block = 0u; block < count; block++) {
            let w = block * ROWS + local.x;
            let w0 = Words(wLo[w], wHi[w]);
            let w1 = Words(wLo[w + 8u], wHi[w + 8u]);
            let w2 = Words(wLo[w + 16u], wHi[w + 16u]);
            let w3 = Words(wLo[w + 24u], wHi[w + 24u]);
            let scales = vec4(wScale[w], wScale[w + 8u], wScale[w + 16u], wScale[w + 24u]);
            ${products}
        }
        workgroupBarrier();
    }
    let token = firstToken + local.y;
    let row = firstRow + local.x * 4u;
    ${stores}
}
`;
}

function projectFew(packed) {
    return projectionHeader + dotWGSL(packed) + `
@compute @workgroup_size(16, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.rows || id.y >= p.tokens) { return; }
    let lane = id.x % 16u;
    var value = 0.0;
    for (var k = 0u; k < p.groups; k++) {
        let w = (id.x / 16u * p.groups + k) * 136u;
        let a = (id.y * p.groups + k) * 9u;
        var sum = 0;
        for (var part = 0u; part < 8u; part++) {
            sum += idot(acts[a + 1u + part], weights[w + 8u + part * 16u + lane] ^ 0x80808080u);
        }
        let scale = unpack2x16float(weights[w + lane / 2u])[lane % 2u] * bitcast<f32>(acts[a]);
        value = fma(f32(sum), scale, value);
    }
    store(id.y, id.x, value);
}
`;
}

function projectFloat() {
    // Dequantize once per workgroup. The tile then uses vector FMAs on four
    // output channels per invocation, including on adapters without fast i8 dot.
    const tokenBlocks = [0, 1, 2, 3];
    const accum = tokenBlocks.map(i => `var c${i} = vec4(0.0);`).join("\n    ");
    const inner = tokenBlocks.map(i => `
        let a${i} = aTile[(lid.y + ${i * 8}u) * 8u + part];
        c${i} = fma(vec4(a${i}.x), w0, c${i});
        c${i} = fma(vec4(a${i}.y), w1, c${i});
        c${i} = fma(vec4(a${i}.z), w2, c${i});
        c${i} = fma(vec4(a${i}.w), w3, c${i});
    `).join("");
    const write = tokenBlocks.map(i => `
        let t${i} = firstToken + lid.y + ${i * 8}u;
        store(t${i}, row, c${i}.x);
        store(t${i}, row + 1u, c${i}.y);
        store(t${i}, row + 2u, c${i}.z);
        store(t${i}, row + 3u, c${i}.w);
    `).join("");
    return projectionHeader + `
var<workgroup> wTile: array<vec4<f32>, 32 * 16>;
var<workgroup> aTile: array<vec4<f32>, 32 * 8>;

fn signedBytes(word: u32) -> vec4<f32> {
    let bits = ((vec4(word) >> vec4(0u, 8u, 16u, 24u)) & vec4(255u)) ^ vec4(128u);
    return vec4<f32>(vec4<i32>(bits) - vec4<i32>(128));
}

@compute @workgroup_size(16, 8)
fn main(@builtin(workgroup_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let firstRow = gid.x * 64u;
    let firstToken = gid.y * 32u;
    let row = firstRow + lid.x * 4u;
    ${accum}
    for (var k = 0u; k < p.groups; k++) {
        let base = (row / 16u * p.groups + k) * 136u;
        let w = base + 8u + lid.y * 16u + row % 16u;
        var codes = vec4<u32>(128u) * 0x01010101u;
        var scales = vec4(0.0);
        if (row < p.rows) {
            codes = vec4(weights[w], weights[w + 1u], weights[w + 2u], weights[w + 3u]);
            let s = base + (row % 16u) / 2u;
            scales = vec4(unpack2x16float(weights[s]), unpack2x16float(weights[s + 1u]));
        }
        for (var i = 0u; i < 4u; i++) {
            let decoded = vec4<i32>((codes >> vec4(i * 8u)) & vec4(255u)) - vec4<i32>(128);
            wTile[(lid.y * 4u + i) * 16u + lid.x] = vec4<f32>(decoded) * scales;
        }
        for (var i = lane; i < 256u; i += 128u) {
            let token = firstToken + i / 8u;
            var value = vec4(0.0);
            if (token < p.tokens) {
                let a = (token * p.groups + k) * 9u;
                value = signedBytes(acts[a + 1u + i % 8u]) * bitcast<f32>(acts[a]);
            }
            aTile[i] = value;
        }
        workgroupBarrier();
        for (var part = 0u; part < 8u; part++) {
            let w = part * 64u + lid.x;
            let w0 = wTile[w];
            let w1 = wTile[w + 16u];
            let w2 = wTile[w + 32u];
            let w3 = wTile[w + 48u];
            ${inner}
        }
        workgroupBarrier();
    }
    ${write}
}
`;
}

const rope = `
struct Params { hidden: u32 }
@group(0) @binding(0) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> tokens: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> angles: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.hidden) { return; }
    let coeff = angles[tokens[id.y].y * 32u + id.x % 32u];
    let pixel = id.y * 3u * p.hidden + id.x / 32u * 64u + id.x % 32u;
    let x = qkv[pixel];
    let y = qkv[pixel + 32u];
    qkv[pixel] = x * coeff.y - y * coeff.x;
    qkv[pixel + 32u] = y * coeff.y + x * coeff.x;
}
`;

const tileWGSL = `
struct Tile { first: u32, length: u32, query: u32, output: u32, count: u32 }
`;

const attention = tileWGSL + `
struct Params { hidden: u32, firstTile: u32, radius: u32 }
@group(0) @binding(0) var<storage, read> qkv: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tiles: array<Tile>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;

const DIM = 16u; // vec4 channels per head
const Q = ${QUERY_TILE}u;
const K = 16u;
const MASKED = -1e30;
var<workgroup> queryTile: array<vec4<f32>, Q * DIM>;
var<workgroup> keyTile: array<vec4<f32>, K * DIM>;
var<workgroup> valTile: array<vec4<f32>, K * DIM>;
var<workgroup> probs: array<f32, Q * K>;
var<workgroup> rowMax: array<f32, Q>;
var<workgroup> rowSum: array<f32, Q>;
var<workgroup> rescale: array<f32, Q>;

@compute @workgroup_size(16, ${QUERY_TILE})
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_id) lane: vec3<u32>) {
    let tile = tiles[p.firstTile + id.x];
    let width = p.hidden / 4u;
    let col = id.y * DIM + lane.x;
    let q = tile.query + lane.y;
    let valid = lane.y < tile.count;
    if (valid) {
        queryTile[lane.y * DIM + lane.x] = qkv[(tile.first + q) * 3u * width + col];
    }
    if (lane.x == 0u) {
        rowMax[lane.y] = MASKED;
        rowSum[lane.y] = 0.0;
    }
    var acc = vec4(0.0);
    let first = tile.query - min(tile.query, p.radius);
    let end = min(tile.query + tile.count + p.radius, tile.length);
    for (var start = first / K * K; start < end; start += K) {
        let key = start + lane.y;
        if (key < tile.length) {
            let row = (tile.first + key) * 3u * width + col;
            keyTile[lane.y * DIM + lane.x] = qkv[row + width];
            valTile[lane.y * DIM + lane.x] = qkv[row + 2u * width];
        } else {
            keyTile[lane.y * DIM + lane.x] = vec4(0.0);
            valTile[lane.y * DIM + lane.x] = vec4(0.0);
        }
        workgroupBarrier();
        let k = start + lane.x;
        var score = MASKED;
        if (valid && k < tile.length && max(q, k) - min(q, k) <= p.radius) {
            score = 0.0;
            for (var d = 0u; d < DIM; d++) {
                score += dot(queryTile[lane.y * DIM + d], keyTile[lane.x * DIM + d]);
            }
            score *= 0.125;
        }
        probs[lane.y * K + lane.x] = score;
        workgroupBarrier();

        // Online softmax: the numerator and denominator share the running max.
        if (lane.x == 0u && valid) {
            let base = lane.y * K;
            let prev = rowMax[lane.y];
            var high = prev;
            for (var i = 0u; i < K; i++) { high = max(high, probs[base + i]); }
            let scale = exp(prev - high);
            rowMax[lane.y] = high;
            rescale[lane.y] = scale;
            var sum = rowSum[lane.y] * scale;
            for (var i = 0u; i < K; i++) {
                let s = probs[base + i];
                let prob = select(0.0, exp(s - high), s != MASKED);
                probs[base + i] = prob;
                sum += prob;
            }
            rowSum[lane.y] = sum;
        }
        workgroupBarrier();
        if (valid) {
            acc *= rescale[lane.y];
            for (var i = 0u; i < K; i++) {
                acc = fma(vec4(probs[lane.y * K + i]), valTile[i * DIM + lane.x], acc);
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        output[(tile.output + lane.y) * width + col] = acc / rowSum[lane.y];
    }
}
`;

const gather = tileWGSL + `
struct Params { width: u32, firstTile: u32 }
@group(0) @binding(0) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tiles: array<Tile>;
@group(0) @binding(2) var<storage, read_write> pooled: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.width) { return; }
    let tile = tiles[p.firstTile + id.y];
    pooled[id.y * p.width + id.x] = state[(tile.first + tile.query) * p.width + id.x];
}
`;

const readout = `
struct Params { width: u32, choices: u32 }
@group(0) @binding(0) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.choices) { return; }
    var sum = vec4(0.0);
    for (var i = 0u; i < p.width; i++) {
        sum = fma(state[id.y * p.width + i], weights[id.x * p.width + i], sum);
    }
    logits[id.y * p.choices + id.x] = dot(sum, vec4(1.0));
}
`;

export async function createEncoderKernels(device, packedDot, useFloat = false) {
    const shaders = {norm, pack, project: useFloat ? projectFloat() : project(packedDot),
                     projectFew: projectFew(packedDot),
                     rope, attention, gather, readout};
    const entries = await Promise.all(Object.entries(shaders).map(async ([name, code]) => {
        const module = device.createShaderModule({label: `GLiDE ${name}`, code});
        try {
            const pipeline = await device.createComputePipelineAsync({
                label: module.label, layout: "auto", compute: {module, entryPoint: "main"},
            });
            return [name, pipeline];
        } catch (error) {
            const info = await module.getCompilationInfo();
            const messages = info.messages.filter(message => message.type === "error")
                .map(message => `${message.lineNum}:${message.linePos} ${message.message}`);
            throw new Error(`GLiDE ${name}: ${messages.join("\n") || error.message}`, {cause: error});
        }
    }));
    return Object.fromEntries(entries);
}
