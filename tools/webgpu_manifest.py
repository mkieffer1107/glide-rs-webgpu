#!/usr/bin/env python3
"""Export browser buffer geometry for an existing packed GLiDE-2B checkpoint."""

import argparse
import json
import os
from math import prod
from pathlib import Path

TILE_BYTES = {"q4x16": 288, "q4nx16": 288, "q6x16": 416, "q8x16": 544}
SCALAR_BYTES = {"f32": 4, "f16": 2, "bf16": 2}


def tensor_units(shape, dtype):
    if not shape or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError(f"invalid tensor shape {shape}")
    if dtype in SCALAR_BYTES:
        return 1, prod(shape[1:]) * SCALAR_BYTES[dtype]
    if len(shape) == 2:
        if dtype == "q4":
            return 1, (shape[1] + 31) // 32 * 18
        if dtype in TILE_BYTES and shape[0] % 16 == 0:
            return 16, (shape[1] + 31) // 32 * TILE_BYTES[dtype]
        if dtype == "bf16x16":
            return 16, (shape[1] + 1) // 2 * 64
    raise ValueError(f"unsupported shape {shape} for {dtype}")


def export_manifest(model, output):
    index = json.loads((model / "index.json").read_text())
    if index["version"] != 1:
        raise ValueError("unsupported weight index version")
    weights = model / "weights.bin"
    total_bytes = weights.stat().st_size
    tensors = {}
    end = 0
    for name, item in sorted(index["tensors"].items(), key=lambda entry: entry[1]["offset"]):
        offset, size = item["offset"], item["nbytes"]
        shape, dtype = item["shape"], item["dtype"]
        rows, size_per_unit = tensor_units(shape, dtype)
        expected = (shape[0] + rows - 1) // rows * size_per_unit
        if offset < end or offset + size > total_bytes or size != expected:
            raise ValueError(f"inconsistent weight span for {name}")
        tensors[name] = {
            **item,
            "rowsPerUnit": rows,
            "bytesPerUnit": size_per_unit,
        }
        end = offset + size

    def asset(name):
        return Path(os.path.relpath(model / name, output.parent)).as_posix()

    result = {
        "version": 1,
        "weights": asset("weights.bin"),
        "config": asset("config.json"),
        "decisionConfig": asset("decision_config.json"),
        "weightBytes": total_bytes,
        "chunkBytes": 64 * 1024 * 1024,
        "tensors": tensors,
    }
    if (model / "tokenizer.json").is_file():
        result["tokenizer"] = asset("tokenizer.json")
    output.write_text(json.dumps(result, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export_manifest(args.model, args.output or args.model / "webgpu.json")


if __name__ == "__main__":
    main()
