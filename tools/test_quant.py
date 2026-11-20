#!/usr/bin/env python3
"""Numerical checks for checkpoint weight packing."""
import numpy as np
import torch
from convert import Q4_CODEBOOK, pack_bf16, quantize

with np.errstate(divide='raise', invalid='raise'):
    packed_zeros = quantize(np.zeros((1, 32), dtype=np.float32))
assert packed_zeros[2:] == b'\x88' * 16

rng = np.random.default_rng(123)
for rows,cols in [(1,32),(9,2048),(6,47),(3,256)]:
    source = rng.normal(0,0.7,(rows,cols)).astype(np.float32)
    packed=quantize(source)
    decoded=np.zeros_like(source)
    stride=((cols+31)//32)*18
    for row in range(rows):
        for col in range(cols):
            base=row*stride+(col//32)*18
            scale=np.frombuffer(packed[base:base+2],dtype='<f2')[0].astype(np.float32)
            bits=packed[base+2+(col%32)//2]
            q=(bits&15) if col%2==0 else (bits>>4)
            decoded[row,col]=(int(q)-8)*scale
    mae=np.abs(source-decoded).mean()
    assert np.isfinite(decoded).all() and mae < 0.14, (rows,cols,mae)
    a=rng.normal(size=cols).astype('float32')
    actual=decoded@a
    check=sum(decoded[:,i]*a[i] for i in range(cols))
    np.testing.assert_allclose(actual,check,rtol=3e-4,atol=1e-3)
    print(f'PASS {rows}x{cols} Q4 round-trip: MAE={mae:.5f}')

source = rng.normal(size=(32, 47)).astype(np.float32)
row = quantize(source)
tiled = quantize(source, tiled=True)
assert len(row) == len(tiled)
for i in range(32):
    for j in range(47):
        row_block = (i * 2 + j // 32) * 18
        tile_block = (i // 16 * 2 + j // 32) * 288
        scale_offset = tile_block + (i % 16) * 2
        assert row[row_block:row_block + 2] == tiled[scale_offset:scale_offset + 2]
        packed_row = row[row_block + 2 + j % 32 // 2]
        packed_tile = tiled[tile_block + 32 + (j % 32 // 8) * 64 + (i % 16) * 4 + j % 4]
        assert (packed_row >> (j % 2 * 4)) & 15 == (packed_tile >> (j % 8 // 4 * 4)) & 15
print('PASS tile16 preserves Q4 values across row and column tails')

codebook_rows = np.tile(Q4_CODEBOOK, (16, 4)).astype(np.float32)
codebook_packed = quantize(codebook_rows, tiled=True, q4_codebook=True)
for i in range(16):
    for j in range(64):
        block = j // 32 * 288
        assert codebook_packed[block + 2 * i:block + 2 * i + 2] == b'\x00\x3c'
        packed = codebook_packed[block + 32 + (j % 32 // 8) * 64 + i * 4 + j % 4]
        code = (packed >> (4 * (j % 8 // 4))) & 15
        assert code == j % 16
print('PASS tiled Q4 codebook packs every code and the group scales')

tiled = quantize(source, bits=6, tiled=True)
decoded = np.empty_like(source)
for i in range(32):
    for j in range(47):
        block = (i // 16 * 2 + j // 32) * 416
        scale_offset = block + (i % 16) * 2
        scale = np.frombuffer(tiled[scale_offset:scale_offset + 2], dtype='<f2')[0]
        part = block + 32 + (j % 32 // 8) * 96
        half = j % 8 // 4
        low = (tiled[part + i % 16 * 4 + j % 4] >> (half * 4)) & 15
        high = (tiled[part + 64 + half * 16 + i % 16] >> (2 * (j % 4))) & 3
        decoded[i, j] = ((high * 16 + low) - 32) * np.float32(scale)
assert np.max(np.abs(decoded - source)) < 0.06
print('PASS tile16 Q6 round-trip across row and column tails')

groups = rng.normal(size=(64, 32)).astype(np.float32)
errors = []
for fit in (False, True):
    packed = np.frombuffer(quantize(groups, q4_mse=fit), dtype=np.uint8).reshape(-1, 18)
    scale = packed[:, :2].copy().view('<f2').astype(np.float32)
    codes = np.stack((packed[:, 2:] & 15, packed[:, 2:] >> 4), axis=-1).reshape(-1, 32)
    reconstructed = (codes.astype(np.float32) - 8) * scale
    errors.append(np.square(groups - reconstructed).sum(axis=1))
assert np.all(errors[1] <= errors[0]) and np.any(errors[1] < errors[0])
print('PASS fitted Q4 scales reduce squared weight error without increasing any group error')

tiled = quantize(source, bits=8, tiled=True)
decoded = np.empty_like(source)
for i in range(32):
    for j in range(47):
        block = (i // 16 * 2 + j // 32) * 544
        scale_offset = block + (i % 16) * 2
        scale = np.frombuffer(tiled[scale_offset:scale_offset + 2], dtype='<f2')[0]
        value = tiled[block + 32 + (j % 32 // 4) * 64 + (i % 16) * 4 + j % 4]
        decoded[i, j] = (value - 128) * np.float32(scale)
assert np.max(np.abs(decoded - source)) < 0.02
print('PASS tile16 Q8 round-trip across row and column tails')

source = rng.normal(size=(17, 35)).astype(np.float32)
expected = torch.from_numpy(source).bfloat16().view(torch.int16).numpy().view(np.uint16)
packed = np.frombuffer(pack_bf16(source), dtype='<u2').reshape(2, 18, 16, 2)
unpacked = packed.transpose(0, 2, 1, 3).reshape(32, 36)
np.testing.assert_array_equal(unpacked[:17, :35], expected)
assert not unpacked[17:, :].any() and not unpacked[:, 35].any()
print('PASS tiled BF16 packing exactly matches PyTorch, including padding')
