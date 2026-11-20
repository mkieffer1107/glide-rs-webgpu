const paramsWGSL = `
struct Params { columns: u32, groups: u32, tokens: u32, rows: u32 }
`;

const packActivationsWGSL = paramsWGSL + `
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@group(0) @binding(2) var<uniform> dims: Params;

fn roundEven(x: f32) -> i32 {
    let low = floor(x);
    let remainder = x - low;
    return i32(low) + i32(remainder > 0.5 || (remainder == 0.5 && (i32(low) & 1) != 0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= dims.tokens * dims.groups) { return; }
    let row = id.x / dims.groups;
    let start = (id.x % dims.groups) * 32u;
    let input = row * dims.columns + start;
    let count = min(32u, dims.columns - start);
    var maximum = 0.0;
    for (var i = 0u; i < count; i++) {
        maximum = max(maximum, abs(values[input + i]));
    }
    var inverse = 0.0;
    if (maximum > 0.0) { inverse = 127.0 / maximum; }
    let dest = id.x * 10u;
    packed[dest] = bitcast<u32>(maximum / 127.0);
    var sum = 0;
    for (var word = 0u; word < 8u; word++) {
        var bits = 0u;
        for (var lane = 0u; lane < 4u; lane++) {
            let i = word * 4u + lane;
            if (i < count) {
                let code = roundEven(values[input + i] * inverse);
                sum += code;
                bits |= (bitcast<u32>(code) & 255u) << (lane * 8u);
            }
        }
        packed[dest + 2u + word] = bits;
    }
    packed[dest + 1u] = bitcast<u32>(sum);
}
`;

const unpackers = {
    q8x16: {
        words: 136, bias: 0,
        body: "return weights[base + 8u + part * 16u + lane] ^ 0x80808080u;",
    },
    q4x16: {
        words: 72, bias: 8,
        body: `
            let bits = weights[base + 8u + (part / 2u) * 16u + lane];
            return (bits >> ((part % 2u) * 4u)) & 0x0f0f0f0fu;
        `,
    },
    q6x16: {
        words: 104, bias: 32,
        body: `
            let block = base + 8u + (part / 2u) * 24u;
            let low = (weights[block + lane] >> ((part % 2u) * 4u)) & 0x0f0f0f0fu;
            let highWord = weights[block + 16u + (part % 2u) * 4u + lane / 4u];
            let hi = (highWord >> (8u * (lane % 4u))) & 255u;
            return low | ((hi & 3u) << 4u) | ((hi & 12u) << 10u)
                       | ((hi & 48u) << 16u) | ((hi & 192u) << 22u);
        `,
    },
    q4nx16: {
        words: 72, bias: 0,
        body: `
            let levels = array<i32, 16>(-127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127);
            let bits = weights[base + 8u + (part / 2u) * 16u + lane] >> ((part % 2u) * 4u);
            var word = 0u;
            for (var shift = 0u; shift < 32u; shift += 8u) {
                word |= (bitcast<u32>(levels[(bits >> shift) & 15u]) & 255u) << shift;
            }
            return word;
        `,
    },
};

function projectionWGSL({ words, bias, body }, packedDot) {
    const dot = packedDot ? "return dot4I8Packed(a, b);" : `
        var sum = 0;
        for (var shift = 0u; shift < 32u; shift += 8u) {
            let left = bitcast<i32>(a << (24u - shift)) >> 24u;
            let right = bitcast<i32>(b << (24u - shift)) >> 24u;
            sum += left * right;
        }
        return sum;
    `;
    return paramsWGSL + `
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> acts: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dims: Params;

fn weightWord(base: u32, part: u32, lane: u32) -> u32 { ${body} }

fn dot(a: u32, b: u32) -> i32 { ${dot} }

@compute @workgroup_size(16, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= dims.rows || id.y >= dims.tokens) { return; }
    let lane = id.x % 16u;
    let row = id.x / 16u;
    var value = 0.0;
    for (var group = 0u; group < dims.groups; group++) {
        let w = (row * dims.groups + group) * ${words}u;
        let a = (id.y * dims.groups + group) * 10u;
        let scales = unpack2x16float(weights[w + lane / 2u]);
        let scale = scales[lane % 2u] * bitcast<f32>(acts[a]);
        var sum = -${bias} * bitcast<i32>(acts[a + 1u]);
        for (var part = 0u; part < 8u; part++) {
            sum += dot(acts[a + 2u + part], weightWord(w, part, lane));
        }
        value = fma(f32(sum), scale, value);
    }
    output[id.y * dims.rows + id.x] = value;
}
`;
}

function parameters(device, shape) {
    const buffer = device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true});
    new Uint32Array(buffer.getMappedRange()).set(shape);
    buffer.unmap();
    return buffer;
}

function encode(device, encoder, pipeline, buffers, x, y = 1) {
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({binding, resource: {buffer}})),
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x, y);
    pass.end();
}

// A packed activation buffer can be used for several projections of the same input.
// Keep returned buffers alive through submission. The caller owns their lifetime.
export function createProjectionKernels(device, { packedDot = true } = {}) {
    const usePackedDot = packedDot && (navigator.gpu.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product") ?? false);
    const pipeline = code => device.createComputePipeline({
        layout: "auto", compute: {module: device.createShaderModule({code}), entryPoint: "main"},
    });
    const pack = pipeline(packActivationsWGSL);
    const projections = new Map();

    return {
        packedDot: usePackedDot,
        pack(encoder, input, tokens, columns) {
            const groups = Math.ceil(columns / 32);
            const buffer = device.createBuffer({size: tokens * groups * 40, usage: GPUBufferUsage.STORAGE});
            const params = parameters(device, [columns, groups, tokens, 0]);
            encode(device, encoder, pack, [input, buffer, params], Math.ceil(tokens * groups / 64));
            return {buffer, params, tokens, columns, groups};
        },
        project(encoder, weights, dtype, rows, acts) {
            const layout = unpackers[dtype];
            if (!layout) throw new Error(`Unsupported WebGPU projection dtype ${dtype}`);
            if (rows % 16 !== 0 || weights.size !== rows / 16 * acts.groups * layout.words * 4) {
                throw new RangeError("Projection weights must hold complete 16-row tiles with the activation width");
            }
            if (!projections.has(dtype)) {
                projections.set(dtype, pipeline(projectionWGSL(layout, usePackedDot)));
            }
            const buffer = device.createBuffer({
                size: acts.tokens * rows * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
            const params = parameters(device, [acts.columns, acts.groups, acts.tokens, rows]);
            encode(device, encoder, projections.get(dtype), [weights, acts.buffer, buffer, params],
                rows / 16, Math.ceil(acts.tokens / 4));
            return {buffer, params};
        },
    };
}
