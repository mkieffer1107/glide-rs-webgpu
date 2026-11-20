#!/usr/bin/env python3
"""Render deterministic text decision tasks and a labeled context-length sweep."""

import argparse
import json
import random
from pathlib import Path

from make_request import load_prompter, prepare, use_upstream

LENGTHS = (20000, 12000, 8192, 4096, 2048, 1024, 512, 256, 96)
POSITIONS = ("late", "early", "middle")


def choice(instructions, labels):
    return {"type": "choice", "instructions": instructions, "criteria": labels}


def context_case(prompter, length, position):
    """Find a rendered length, including labels and the Fastino chat template."""
    rng = random.Random(length)
    outcomes = ["cedar", "amber", "cobalt", "violet"]
    rng.shuffle(outcomes)
    expected = rng.choice(outcomes)
    question = choice("What is the routing tag for ticket NORTHSTAR?", dict.fromkeys(outcomes, ""))
    locations = ["Austin", "Oslo", "Lima", "Accra", "Kyoto", "Seville", "Taipei"]
    topics = [
        "library delivery", "bridge inspection", "bus schedule", "water meter", "park permit", "road survey",
    ]
    distractions = [
        f"Ticket {i + 1000} in {rng.choice(locations)} concerns a {rng.choice(topics)}. "
        f"The routing tag is {rng.choice(outcomes)}.\n"
        for i in range(length // 12 + 1)
    ]
    target = f"Ticket NORTHSTAR in Halifax concerns a ferry reservation. The routing tag is {expected}.\n"

    def render(rows, extra=""):
        split = {"early": 0, "middle": rows // 2, "late": rows}[position]
        state = "".join(distractions[:split]) + target + "".join(distractions[split:rows]) + extra
        data, _ = prepare(prompter, {"state": state, "questions": {"case": question}})
        return data["questions"]["case"]

    low, high = 0, len(distractions)
    while low + 1 < high:
        mid = (low + high) // 2
        if len(render(mid)["input_ids"]) <= length:
            low = mid
        else:
            high = mid
    size = len(render(low)["input_ids"])
    if size > length:
        raise ValueError(f"The required text exceeds {length} tokens")
    gap = length - size
    for n in range(max(0, gap - 6), gap + 1):
        data = render(low, "Index:" + " entry" * n + "." if n else "")
        if len(data["input_ids"]) == length:
            return data, expected
    for n in range(max(0, gap - 2), gap + 4):
        data = render(low, " Index" * n)
        if len(data["input_ids"]) == length:
            return data, expected
    raise ValueError(f"Could not render {length} tokens")


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def topic_cases(prompter):
    path = Path(__file__).with_name("context_questions.json")
    cases = json.loads(path.read_text(encoding="utf-8"))
    questions, labels = {}, {}
    for case in cases:
        name = case["name"]
        data, _ = prepare(prompter, {"state": case["state"], "questions": {name: case["question"]}})
        question = data["questions"][name]
        if name in questions or case["expected"] not in question["labels"]:
            raise ValueError(f"Invalid case {name}")
        questions[name] = question
        labels[name] = {"expected": case["expected"], "topic": case["topic"]}
    return questions, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Downloaded Fastino GLiDE-2B snapshot")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    args = parser.parse_args()
    if len(args.lengths) != len(set(args.lengths)) or any(n < 96 for n in args.lengths):
        parser.error("lengths must be unique and at least 96 tokens")
    args.output.mkdir(parents=True, exist_ok=True)
    prompter = load_prompter(use_upstream(args.model))
    sweep, labels = {}, {}
    for length in args.lengths:
        if length in LENGTHS:
            position = POSITIONS[LENGTHS.index(length) % len(POSITIONS)]
        else:
            position = random.Random(length).choice(POSITIONS)
        name = f"routing-{length}-{position}"
        sweep[name], expected = context_case(prompter, length, position)
        labels[name] = {"expected": expected, "topic": "ticket retrieval", "record_position": position}
        write_json(args.output / f"length-{length}.json", {"questions": {name: sweep[name]}})
    write_json(args.output / "sweep.json", {"questions": sweep})
    by_length = {len(q["input_ids"]): (name, q) for name, q in sweep.items()}
    mixed = (1024, 512, 256, 96)
    if all(n in by_length for n in mixed):
        write_json(args.output / "mixed.json", {"questions": dict(by_length[n] for n in mixed)})
    topics, topic_labels = topic_cases(prompter)
    write_json(args.output / "topics.json", {"questions": topics})
    labels.update(topic_labels)
    write_json(args.output / "labels.json", labels)


if __name__ == "__main__":
    main()
