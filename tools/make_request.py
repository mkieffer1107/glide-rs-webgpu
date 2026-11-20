#!/usr/bin/env python3
"""Prepare Rust input using the prompt renderer from the GLiDE-2B snapshot."""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def use_upstream(model_dir):
    model_dir = Path(model_dir).resolve()
    if not (model_dir / "glide" / "prompt.py").is_file():
        raise ValueError(f"{model_dir} needs the upstream glide package from the Hub snapshot")
    sys.path.insert(0, str(model_dir))
    return model_dir


def load_prompter(model_dir):
    from glide.prompt import Prompter

    config = json.loads((model_dir / "decision_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    return Prompter(tokenizer, config["codes"])


def prepare(prompter, request):
    from glide.prompt import MAX_OPTIONS, RequestError, options_for, split_state

    if not isinstance(request, dict):
        raise RequestError("request must be an object")
    questions = request.get("questions")
    if not isinstance(questions, dict):
        raise RequestError("questions must be an object mapping keys to questions")
    state, images = split_state(request.get("state"))
    if images or request.get("images"):
        raise RequestError("images are not supported by this runtime")

    prepared = {}
    trace = {}
    for name, question in questions.items():
        options = options_for(question)
        if not 1 <= len(options) <= MAX_OPTIONS:
            raise RequestError(f"expected 1 to {MAX_OPTIONS} options for {name}")
        prompt = prompter.text(state, question, options)
        ids = prompter.tokenizer.encode(prompt, add_special_tokens=False)
        prepared[name] = {
            "type": question["type"],
            "input_ids": ids,
            "labels": [key for key, _ in options],
        }
        trace[name] = {"rendered_prompt": prompt, "token_count": len(ids)}
    return {"questions": prepared}, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Downloaded Fastino GLiDE-2B snapshot")
    parser.add_argument("--request", required=True, type=Path, help="System One request JSON")
    parser.add_argument("--output", required=True, type=Path, help="Output tokenized Rust request")
    parser.add_argument("--trace", action="store_true", help="Save prompts and token counts")
    args = parser.parse_args()

    model_dir = use_upstream(args.model)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    prepared, trace = prepare(load_prompter(model_dir), request)
    args.output.write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.trace:
        args.output.with_suffix(args.output.suffix + ".debug.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
