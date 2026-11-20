#!/usr/bin/env python3
"""Compare official GLiDE System One response against this Rust binary's answer JSON."""
import argparse
import json
import math
import statistics
from pathlib import Path


def response_from(path):
    raw = json.loads(Path(path).read_text())
    if 'answers' in raw:
        return raw
    if 'data' in raw and isinstance(raw['data'], dict) and 'answers' in raw['data']:
        return raw['data']
    raise ValueError(f'{path} must contain top-level answers or data.answers')


def probabilities(answer):
    if answer.get('type') == 'noul':
        yes = float(answer['noul'])
        return {'true': yes, 'false': 1 - yes}
    return answer['probabilities']


def selected(answer, distribution):
    if 'score' in answer:
        return str(answer['score'])
    if 'choice' in answer:
        return answer['choice']
    return max(distribution, key=distribution.get)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('reference', help='Saved official GLiDE server or Python response')
    p.add_argument('rust', help='Saved glide-cpu decide response')
    p.add_argument('--max-probability-error', type=float)
    p.add_argument('--min-agreement', type=float, help='Minimum top-choice agreement, from 0 to 1')
    p.add_argument('--max-logit-error', type=float, help='Maximum absolute error across all 255 raw logits')
    p.add_argument('--expected', type=Path, help='Expected labels from context_suite.py')
    args = p.parse_args()
    reference, rust = response_from(args.reference), response_from(args.rust)
    if 'usage' in reference and reference['usage'] != rust.get('usage'):
        raise ValueError('Usage differs between reference and Rust')
    if 'model' in reference and reference['model'] != rust.get('model'):
        raise ValueError('Model names differ between reference and Rust')
    ref, got = reference['answers'], rust['answers']
    if set(ref) != set(got):
        raise ValueError('Question keys differ between reference and Rust')
    questions = sorted(ref)
    if not questions:
        raise SystemExit('No questions to compare')
    compare_logits = 'logits' in reference or 'logits' in rust
    if compare_logits and any(set(r.get('logits', {})) != set(questions) for r in (reference, rust)):
        raise ValueError('Both responses must provide logits for every question')
    if args.max_logit_error is not None and not compare_logits:
        raise ValueError('Generate both responses with --include-logits to check logit error')
    differences = []
    disagreements = []
    before_errors, after_errors = [], []
    expected = json.loads(args.expected.read_text(encoding='utf-8')) if args.expected else None
    confidence_differences = []
    level_differences = []
    logit_errors = []
    active_logit_errors = []
    for name in questions:
        r, g = ref[name], got[name]
        if r.get('type') != g.get('type'):
            raise ValueError(f'Answer type differs for {name}')
        if 'type' in r and r.keys() != g.keys():
            raise ValueError(f'Answer fields differ for {name}')
        rp, gp = probabilities(r), probabilities(g)
        if set(rp) != set(gp):
            raise ValueError(f'Label set mismatch for {name}')
        for distribution in (rp, gp):
            if not all(0 <= p <= 1 for p in distribution.values()) or not math.isclose(
                sum(distribution.values()), 1, abs_tol=1e-6
            ):
                raise ValueError(f'Invalid probability distribution for {name}')
        differences.extend(abs(float(rp[k]) - float(gp[k])) for k in rp)
        if compare_logits:
            reference_row, rust_row = reference['logits'][name], rust['logits'][name]
            if any(len(row) != 255 or not all(map(math.isfinite, row)) for row in (reference_row, rust_row)):
                raise ValueError(f'Expected 255 finite logits for {name}')
            errors = [abs(a - b) for a, b in zip(reference_row, rust_row, strict=True)]
            logit_errors.extend(errors)
            active_logit_errors.extend(errors[:len(rp)])
        if 'confidence' in r and 'confidence' in g:
            confidence_differences.append(abs(r['confidence'] - g['confidence']))
        if 'expected_level' in r:
            level_differences.append(abs(r['expected_level'] - g['expected_level']))
        r_choice, g_choice = selected(r, rp), selected(g, gp)
        if expected is not None:
            label = expected[name]['expected']
            if label not in rp:
                raise ValueError(f'Expected label is absent from question {name}')
            if r_choice != label:
                before_errors.append(name)
            if g_choice != label:
                after_errors.append(name)
        if r_choice != g_choice:
            disagreements.append((name, r_choice, g_choice))
    max_error = max(differences)
    agreement = 1 - len(disagreements) / len(questions)
    print(f'Questions: {len(questions)}')
    print(f'Top-1 agreement: {agreement:.1%} ({len(questions) - len(disagreements)}/{len(questions)})')
    if expected is not None:
        print(f'Correct against expected labels (reference / Rust): {len(questions) - len(before_errors)} / {len(questions) - len(after_errors)} of {len(questions)}')
        for name in sorted(set(before_errors) | set(after_errors)):
            print(f'  {name}: expected={expected[name]["expected"]} '
                  f'reference={selected(ref[name], probabilities(ref[name]))} '
                  f'rust={selected(got[name], probabilities(got[name]))}')
    print(f'Mean absolute probability difference: {statistics.mean(differences):.7f}')
    print(f'Max absolute probability difference: {max_error:.7f}')
    if confidence_differences:
        print(f'Max absolute confidence difference: {max(confidence_differences):.7f}')
    if level_differences:
        print(f'Max absolute expected-level difference: {max(level_differences):.7f}')
    if compare_logits:
        print(f'Raw logits MAE / max: {statistics.mean(logit_errors):.7f} / {max(logit_errors):.7f}')
        print(f'Active logits MAE / max: {statistics.mean(active_logit_errors):.7f} / {max(active_logit_errors):.7f}')
    for name, r, g in disagreements[:20]:
        print(f'  {name}: reference={r} rust={g}')
    if args.max_probability_error is not None and max_error > args.max_probability_error:
        raise SystemExit('Maximum probability error exceeds the requested tolerance')
    if args.min_agreement is not None and agreement < args.min_agreement:
        raise SystemExit('Top-choice agreement falls below the requested minimum')
    if args.max_logit_error is not None and max(logit_errors) > args.max_logit_error:
        raise SystemExit('Maximum logit error exceeds the requested tolerance')
