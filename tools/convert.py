#!/usr/bin/env python3
"""Stream a Fastino GLiDE checkpoint to memory-mappable packed weights.

One row group at a time is copied from safetensors. Requires only Python, numpy,
PyTorch (for BF16 -> float32) and safetensors. No GPU required.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open


def canonical(name):
    """Strip the Qwen multimodal/backbone prefixes, discard vision and LM head."""
    if name.startswith(('visual.', 'vision_tower.', 'lm_head.', 'model.visual.', 'model.vision_tower.')):
        return None
    for needle in ('layers.', 'embed_tokens.weight'):
        at = name.find(needle)
        if at >= 0 and (at == 0 or name[at - 1] == '.'):
            return name[at:]
    if name == 'norm.weight' or name.endswith('.norm.weight') and '.layers.' not in name and not name.startswith('layers.'):
        return 'norm.weight'
    return None


def f32(x):
    # NumPy does not reliably support the PyTorch bfloat16 dtype.
    return x.detach().float().cpu().numpy().astype(np.float32, copy=False)


def get_chunks(handle, name, batch=32):
    sl = handle.get_slice(name)
    shape = list(sl.get_shape())
    if len(shape) == 0:
        yield np.array([float(handle.get_tensor(name))], dtype=np.float32)
    else:
        for pos in range(0, shape[0], batch):
            yield f32(sl[pos : pos + batch])


def fit_q4_scale(groups, scale):
    """Choose a representable FP16 scale by minimizing each group's squared error."""
    def error(candidate):
        step = candidate.astype(np.float32)[:, None]
        codes = np.clip(np.rint(groups / step), -8, 7)
        return np.square(groups - step * codes).sum(axis=1)

    subnormal = np.finfo(np.float16).smallest_subnormal
    nominal = scale.astype(np.float32)
    best_error = error(scale)
    for ratio in (0.96875, 0.9375, 0.875, 0.8125, 0.75, 0.6875):
        candidate = np.maximum(nominal * ratio, subnormal).astype('<f2')
        candidate_error = error(candidate)
        use = candidate_error < best_error
        scale[use] = candidate[use]
        best_error[use] = candidate_error[use]
    return scale


Q4_CODEBOOK = np.array(
    [-127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127],
    dtype=np.int16,
)
Q4_NEAREST = np.abs(np.arange(-127, 128)[:, None] - Q4_CODEBOOK).argmin(axis=1).astype(np.uint8)


def quantize(arr, bits=4, tiled=False, q4_mse=False, q4_codebook=False):
    """Quantize groups of 32 to FP16 scales and biased integer values."""
    if bits not in (4, 6, 8) or (bits != 4 and not tiled):
        raise ValueError('row packing requires Q4, and tiled packing supports Q4, Q6, and Q8')
    if q4_codebook and (bits != 4 or not tiled):
        raise ValueError('the nonuniform codebook uses tiled Q4')
    arr = np.asarray(arr, dtype=np.float32)
    n, c = arr.shape
    if tiled and n % 16:
        raise ValueError('tiled quantization requires a multiple of 16 rows')
    padded = ((c + 31) // 32) * 32
    if padded != c:
        arr = np.pad(arr, ((0, 0), (0, padded - c)))
    groups = arr.reshape(-1, 32)
    bias = 1 << (bits - 1)
    scale = np.max(np.abs(groups), axis=1) / (127 if q4_codebook else bias - 1)
    scale = np.maximum(scale, np.finfo(np.float16).smallest_subnormal).astype('<f2')
    if bits == 4 and q4_mse and not q4_codebook:
        scale = fit_q4_scale(groups, scale)
    real_scale = scale.astype(np.float32)
    rounded = np.rint(groups / real_scale[:, None])
    if q4_codebook:
        q = Q4_NEAREST[np.clip(rounded, -127, 127).astype(np.int16) + 127]
    else:
        q = np.clip(rounded + bias, 0, 2 * bias - 1).astype(np.uint8)
    if tiled:
        # Store 16 scales, then groups of four consecutive values across 16 rows.
        tiles, blocks = n // 16, padded // 32
        scales = scale.view(np.uint8).reshape(tiles, 16, blocks, 2).transpose(0, 2, 1, 3)
        output = np.empty((tiles, blocks, 32 + bits * 64), dtype=np.uint8)
        output[..., :32] = scales.reshape(tiles, blocks, 32)
        if bits == 6:
            # Each 96-byte part holds 64 packed low nibbles and two
            # 16-byte arrays of high bits, one byte per row in each array.
            q = q.reshape(tiles, 16, blocks, 4, 8).transpose(0, 2, 3, 1, 4)
            first, second = q[..., :4], q[..., 4:]
            dst = output[..., 32:].reshape(tiles, blocks, 4, 96)
            dst[..., :64] = ((first & 15) | ((second & 15) << 4)).reshape(tiles, blocks, 4, 64)
            shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
            dst[..., 64:80] = np.bitwise_or.reduce((first >> 4) << shifts, axis=-1)
            dst[..., 80:96] = np.bitwise_or.reduce((second >> 4) << shifts, axis=-1)
            return output.tobytes()
        if bits == 4:
            # Low and high nibbles hold two separate groups of four.
            q = q.reshape(tiles, 16, blocks, 4, 8)
            q = q[..., :4] | (q[..., 4:] << 4)
        else:
            q = q.reshape(tiles, 16, blocks, 8, 4)
        packed = q.transpose(0, 2, 3, 1, 4)
        output[..., 32:] = packed.reshape(tiles, blocks, bits * 64)
        return output.tobytes()
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    rows = np.empty((len(groups), 18), dtype=np.uint8)
    rows[:, :2] = scale.view(np.uint8).reshape(-1, 2)
    rows[:, 2:] = packed
    return rows.tobytes()


def bf16_bits(values):
    """Store FP32 values as little-endian BF16 with round-to-nearest-even."""
    bits = values.view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype('<u2')


def pack_bf16(values):
    bits = bf16_bits(values)
    rows, cols = bits.shape
    if rows % 16 or cols % 2:
        bits = np.pad(bits, ((0, -rows % 16), (0, -cols % 2)))
    return bits.reshape(-1, 16, bits.shape[1] // 2, 2).transpose(0, 2, 1, 3).tobytes()


def convert(source, output, embeddings='f16', chunk=32, layout='row', weight_bits=4,
            q8_attention=False, q4_mse=False, q4_codebook=False):
    if chunk < 1 or (layout == 'tile16' and chunk % 16):
        raise ValueError('chunk must be positive and a multiple of 16 for tile16')
    if weight_bits in (6, 8, 16) and layout != 'tile16':
        raise ValueError('Q6, Q8, and BF16 weights use the tile16 layout')
    if q8_attention and (weight_bits != 4 or layout != 'tile16'):
        raise ValueError('--q8-attention requires --layout tile16 --weight-bits 4')
    if q4_codebook and (weight_bits != 4 or layout != 'tile16'):
        raise ValueError('--q4-codebook requires --layout tile16 --weight-bits 4')
    if embeddings == 'q8' and layout != 'tile16':
        raise ValueError('Q8 embeddings require --layout tile16')
    if weight_bits == 16:
        embeddings = 'bf16'
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for filename in ('config.json', 'decision_config.json', 'tokenizer.json', 'tokenizer_config.json',
                     'special_tokens_map.json', 'LICENSE', 'NOTICE'):
        if (source / filename).exists():
            shutil.copyfile(source / filename, output / filename)
    if not (output / 'config.json').exists() or not (output / 'decision_config.json').exists():
        raise SystemExit('Missing config.json or decision_config.json; download the GLiDE checkpoint first.')

    decision = json.loads((source / 'decision_config.json').read_text())
    encoder = decision.get('backbone') == 'encoder'
    if encoder:
        config = json.loads((source / 'config.json').read_text())
        if config.get('model_type') != 'modernbert':
            raise ValueError('Unsupported GLiDE encoder architecture')
        hidden = config['hidden_size']
        embedding_name = 'embeddings.tok_embeddings.weight'
        model_id = 'fastino/GLiDE-0.4B'
        mandatory = (embedding_name, 'embeddings.norm.weight', 'final_norm.weight',
                     'readout.weight', 'layers.0.attn.Wqkv.weight')
        for directory in ('render_tokenizer', 'licenses', 'glide'):
            if (source / directory).is_dir():
                shutil.copytree(source / directory, output / directory, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    else:
        hidden = 2048
        embedding_name = 'embed_tokens.weight'
        model_id = 'fastino/GLiDE-2B'
        mandatory = (embedding_name, 'norm.weight', 'readout.weight',
                     'layers.0.linear_attn.in_proj_qkv.weight', 'layers.23.self_attn.q_proj.weight')

    paths = sorted(source.glob('*.safetensors'))
    if not paths:
        raise SystemExit(f'No safetensors found. Use: uv run hf download {model_id}')
    tensors = {}
    ignored = []
    with (output / 'weights.bin').open('wb') as dest:
        for path in paths:
            with safe_open(str(path), framework='pt', device='cpu') as src:
                for oldkey in sorted(src.keys()):
                    name = oldkey.removeprefix('model.') if encoder else canonical(oldkey)
                    if path.name == 'readout.safetensors':
                        shape = list(src.get_slice(oldkey).get_shape())
                        if shape == [255, hidden]:
                            name = 'readout.weight'
                        elif shape == [255]:
                            name = 'readout.bias'
                        else:
                            name = None
                    elif encoder and not oldkey.startswith(('model.embeddings.', 'model.layers.', 'model.final_norm.')):
                        name = None
                    if name is None:
                        ignored.append(oldkey)
                        continue
                    if name in tensors:
                        raise ValueError(f'duplicate normalized tensor {name}: {oldkey}')
                    shape = list(src.get_slice(oldkey).get_shape())
                    if weight_bits == 16:
                        dtype = 'bf16x16' if len(shape) == 2 and name != embedding_name else 'bf16'
                    elif name.startswith('readout.') or len(shape) != 2:
                        dtype = 'f32'
                    elif name == embedding_name:
                        dtype = 'q8x16' if embeddings == 'q8' else embeddings
                    elif layout == 'tile16' and shape[0] % 16 == 0:
                        attention = any(part in name for part in ('.linear_attn.', '.self_attn.', '.attn.'))
                        bits = 8 if q8_attention and attention else weight_bits
                        dtype = 'q4nx16' if q4_codebook and bits == 4 else f'q{bits}x16'
                    else:
                        dtype = 'f16' if weight_bits in (6, 8) else 'q4'
                    dest.write(b'\x00' * (-dest.tell() % 64))
                    start = dest.tell()
                    for values in get_chunks(src, oldkey, batch=chunk):
                        if dtype in ('q4', 'q4x16', 'q4nx16', 'q6x16', 'q8x16'):
                            dest.write(quantize(values, bits=int(dtype[1]), tiled=dtype.endswith('x16'),
                                                q4_mse=q4_mse, q4_codebook=dtype == 'q4nx16'))
                        elif dtype == 'f16':
                            dest.write(values.astype('<f2').tobytes())
                        elif dtype == 'f32':
                            dest.write(values.astype('<f4').tobytes())
                        elif dtype == 'bf16':
                            dest.write(bf16_bits(values).tobytes())
                        elif dtype == 'bf16x16':
                            dest.write(pack_bf16(values))
                        else:
                            raise AssertionError(dtype)
                    tensors[name] = {'shape': shape, 'dtype': dtype, 'offset': start, 'nbytes': dest.tell() - start}
                    print(f'{name:67} {dtype:>3} {tensors[name]["nbytes"] / 1e6:8.1f} MB', flush=True)
    absent = [name for name in mandatory if name not in tensors]
    if absent:
        raise SystemExit(f'Conversion has missing required model tensors {absent}. Available first keys: {list(tensors)[:10]}')
    index = {'version': 1, 'source': model_id, 'embedding_dtype': tensors[embedding_name]['dtype'],
             'tensors': tensors}
    (output / 'index.json').write_text(json.dumps(index, indent=2))
    print(f'\nWrote {len(tensors)} tensors, {(output / "weights.bin").stat().st_size / 1e9:.3f} GB total')
    print(f'Skipped {len(ignored)} unused/vision/LM-head tensors')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Downloaded GLiDE checkpoint directory')
    parser.add_argument('--output', required=True, help='Destination directory')
    parser.add_argument('--embeddings', choices=['f16', 'q8', 'q4'], default='f16',
                        help='Quantize embeddings to reduce model size')
    parser.add_argument('--layout', choices=['row', 'tile16'], default='row',
                        help='Use tile16 for SIMD-packed matrices or row for the portable Q4 layout')
    parser.add_argument('--weight-bits', choices=[4, 6, 8, 16], type=int, default=4,
                        help='Use 16 with tile16 to retain the BF16 checkpoint precision')
    parser.add_argument('--q8-attention', action='store_true',
                        help='Keep attention projections at Q8 in a tile16 Q4 checkpoint')
    parser.add_argument('--q4-mse', action='store_true',
                        help='Fit a minimum-squared-error scale for each group of Q4 weights')
    parser.add_argument('--q4-codebook', action='store_true',
                        help='Use nonuniform Q4 codes for tiled backbone projections')
    parser.add_argument('--chunk', type=int, default=32, help='Rows converted at once')
    args = parser.parse_args()
    convert(args.source, args.output, args.embeddings, args.chunk, args.layout, args.weight_bits,
            args.q8_attention, args.q4_mse, args.q4_codebook)
