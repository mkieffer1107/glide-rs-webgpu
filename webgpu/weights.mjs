// Packed CPU and GPU checkpoints share the same tensor bytes.
const align4 = bytes => Math.ceil(bytes / 4) * 4;

export function weightRanges(tensor, maxBufferBytes, firstRow = 0, rowCount = tensor.shape[0] - firstRow) {
    const { offset, rowsPerUnit, bytesPerUnit, shape } = tensor;
    const endRow = firstRow + rowCount;
    if (![offset, rowsPerUnit, bytesPerUnit, maxBufferBytes, firstRow, rowCount, endRow].every(Number.isSafeInteger) ||
        offset < 0 || rowsPerUnit < 1 || bytesPerUnit < 1 || firstRow < 0 || rowCount < 1 ||
        endRow > shape[0] || firstRow % rowsPerUnit !== 0 ||
        (endRow % rowsPerUnit !== 0 && endRow !== shape[0])) {
        throw new RangeError("Invalid or unaligned tensor row range");
    }

    const unitsPerBuffer = Math.floor(Math.floor(maxBufferBytes / 4) * 4 / bytesPerUnit);
    if (unitsPerBuffer < 1) throw new RangeError("A tensor row tile exceeds the device buffer limit");

    const ranges = [];
    const lastUnit = Math.ceil(endRow / rowsPerUnit);
    for (let unit = firstRow / rowsPerUnit; unit < lastUnit; unit += unitsPerBuffer) {
        const units = Math.min(unitsPerBuffer, lastUnit - unit);
        const byteLength = units * bytesPerUnit;
        ranges.push({
            offset: offset + unit * bytesPerUnit,
            byteLength,
            bufferBytes: align4(byteLength),
            firstRow: unit * rowsPerUnit,
            rowCount: Math.min(endRow - unit * rowsPerUnit, units * rowsPerUnit),
        });
    }
    return ranges;
}

async function fetchRange(manifest, manifestURL, range, signal) {
    const start = range.offset;
    const end = start + range.byteLength - 1;
    if (![start, end, manifest.weightBytes].every(Number.isSafeInteger) ||
        start < 0 || end < start || end >= manifest.weightBytes) {
        throw new RangeError("Weight range exceeds the checkpoint");
    }
    const response = await fetch(new URL(manifest.weights, manifestURL), {
        headers: { Range: `bytes=${start}-${end}` }, signal,
    });
    if (response.status !== 206 ||
        response.headers.get("Content-Range") !== `bytes ${start}-${end}/${manifest.weightBytes}`) {
        await response.body?.cancel();
        throw new Error("The checkpoint server must return the exact requested HTTP byte range");
    }
    return response;
}

export async function readWeightRange(manifest, manifestURL, range, signal) {
    const response = await fetchRange(manifest, manifestURL, range, signal);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength !== range.byteLength) throw new Error("Incomplete weight range");
    return bytes;
}

// Fill the resident tensors with coalesced range requests. A tensor can span
// several requests while remaining a single bindable GPU buffer.
export async function uploadWeightSet(device, manifest, manifestURL, names, {signal, onProgress} = {}) {
    const tensors = names.map(name => ({name, ...manifest.tensors[name]}))
        .sort((a, b) => a.offset - b.offset);
    const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    const chunk = Math.floor(Math.min(manifest.chunkBytes, 32 * 1024 ** 2) / 4) * 4;
    const total = manifest.weightBytes;
    if (!Number.isSafeInteger(total) || total < 4 || total % 4 ||
        !Number.isSafeInteger(chunk) || chunk < 4) {
        throw new Error("Invalid weight stream length");
    }
    const buffers = new Map();
    let previousEnd = 0;
    for (const {name, offset, nbytes} of tensors) {
        if (!Number.isSafeInteger(offset) || offset % 4 || offset < previousEnd ||
            !Number.isSafeInteger(nbytes) || nbytes < 4 || nbytes % 4 || offset + nbytes > total) {
            throw new Error(`Invalid byte range for ${name}`);
        }
        if (nbytes > limit) throw new Error(`${name} exceeds this device's storage buffer limit`);
        previousEnd = offset + nbytes;
    }
    try {
        device.pushErrorScope("out-of-memory");
        try {
            for (const {name, nbytes} of tensors) {
                buffers.set(name, device.createBuffer({
                    label: name, size: nbytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                }));
            }
        } finally {
            const error = await device.popErrorScope();
            if (error) throw new Error(`Cannot allocate the GLiDE weights: ${error.message}`);
        }
        let first = 0;
        for (let offset = 0; offset < total; offset += chunk) {
            const byteLength = Math.min(chunk, total - offset);
            const end = offset + byteLength;
            const bytes = await readWeightRange(manifest, manifestURL, {offset, byteLength}, signal);
            for (let i = first; i < tensors.length && tensors[i].offset < end; i++) {
                const tensor = tensors[i];
                const lo = Math.max(offset, tensor.offset);
                const hi = Math.min(end, tensor.offset + tensor.nbytes);
                device.queue.writeBuffer(buffers.get(tensor.name), lo - tensor.offset, bytes, lo - offset, hi - lo);
                if (hi === tensor.offset + tensor.nbytes) first = i + 1;
            }
            // Bound the upload queue's temporary memory to one range at a time.
            await device.queue.onSubmittedWorkDone();
            onProgress?.(end, total);
        }
        return buffers;
    } catch (error) {
        for (const buffer of buffers.values()) buffer.destroy();
        throw error;
    }
}

// Consumers own each yielded GPU buffer and can release it after its last dispatch.
export async function* uploadTensor(device, manifest, manifestURL, name, options = {}) {
    const tensor = manifest.tensors[name];
    if (!tensor) throw new Error(`Unknown tensor ${name}`);
    const maxBytes = Math.min(manifest.chunkBytes, device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    for (const range of weightRanges(tensor, maxBytes, options.firstRow, options.rowCount)) {
        const response = await fetchRange(manifest, manifestURL, range, options.signal);
        const buffer = device.createBuffer({
            label: name, size: range.bufferBytes, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true,
        });
        try {
            // Stream into the mapped staging buffer without a full-size JS copy.
            const destination = new Uint8Array(buffer.getMappedRange());
            let received = 0;
            for await (const chunk of response.body) {
                if (received + chunk.byteLength > range.byteLength) throw new Error("Weight range is oversized");
                destination.set(chunk, received);
                received += chunk.byteLength;
            }
            if (received !== range.byteLength) throw new Error("Incomplete weight range");
            buffer.unmap();
        } catch (error) {
            buffer.destroy();
            if (!response.body?.locked) await response.body?.cancel().catch(() => {});
            throw error;
        }
        yield { ...range, buffer };
    }
}
