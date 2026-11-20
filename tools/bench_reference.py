#!/usr/bin/env python3
"""Benchmark upstream GLiDE on CPU using the same token IDs as Rust."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from make_request import use_upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Downloaded Fastino GLiDE snapshot")
    parser.add_argument("--request", required=True, type=Path, help="Tokenized JSON from make_request.py")
    parser.add_argument("--question", help="Benchmark a single question by name")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--include-logits", action="store_true", help="Include 255 raw readout logits per prompt")
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1 or args.threads < 1:
        parser.error("expected warmup >= 0, iters >= 1 and threads >= 1")
    if args.batch_size < 1:
        parser.error("batch size must be positive")

    questions = json.loads(args.request.read_text(encoding="utf-8"))["questions"]
    if args.question is not None:
        if args.question not in questions:
            parser.error(f"question {args.question!r} was not found")
        questions = {args.question: questions[args.question]}

    model_dir = use_upstream(args.model)
    from glide import Glide
    from glide.model import _answer, _softmax

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    max_prompt = max((len(q["input_ids"]) for q in questions.values()), default=1)
    start = time.perf_counter()
    model = Glide.from_pretrained(
        str(model_dir), device="cpu", dtype=torch.bfloat16, vision=False,
        token_budget=args.batch_size * max_prompt,
    )
    load_ms = (time.perf_counter() - start) * 1000

    answers = {}
    timings = {}
    raw_logits = {}
    batches = []
    items = list(questions.items())
    for i in range(0, len(items), args.batch_size):
        batch = items[i : i + args.batch_size]
        prompts = [q["input_ids"] for _, q in batch]
        text_only = [None] * len(batch)
        for _ in range(args.warmup):
            model._logits(prompts, text_only)
        durations = []
        for _ in range(args.iters):
            start = time.perf_counter()
            logits = model._logits(prompts, text_only)
            for (name, question), row in zip(batch, logits, strict=True):
                labels = question["labels"]
                probs = _softmax(row, len(labels), model.temperature)
                answers[name] = _answer(question.get("type", "choice"), labels, probs)
            durations.append((time.perf_counter() - start) * 1000)
        for (name, question), row in zip(batch, logits, strict=True):
            raw_logits[name] = row
            timings[name] = {
                "input_tokens": len(question["input_ids"]),
                "batch_size": len(batch),
                "median_ms": statistics.median(durations),
                "mean_ms": statistics.mean(durations),
            }
        median = statistics.median(durations)
        tokens = sum(map(len, prompts))
        batches.append({
            "batch_size": len(batch),
            "input_tokens": tokens,
            "prompt_lengths": list(map(len, prompts)),
            "samples_ms": durations,
            "median_ms": median,
            "mean_ms": statistics.mean(durations),
            "p95_ms": sorted(durations)[math.ceil(0.95 * len(durations)) - 1],
            "prompts_per_second": len(batch) * 1000 / median,
            "tokens_per_second": tokens * 1000 / median,
        })
    usage = {"input_tokens": sum(t["input_tokens"] for t in timings.values()), "output_tokens": 0}
    response = {
        "model": model.name, "answers": answers, "usage": usage,
        "model_load_ms": load_ms, "timings": timings, "batches": batches,
    }
    if args.include_logits:
        response["logits"] = raw_logits
    print(json.dumps(response, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
