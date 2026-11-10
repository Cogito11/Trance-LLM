#!/usr/bin/env python3
"""Prepare graded, fictional fact-selection diagnostics without downloading data.

Uses the same family-disjoint splits as the grounding experiment. The first
eight validation families select checkpoints; the other 117 are reserved for
post-training evaluation. These were already validation data in the earlier
grounding run, so they are not a new external benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

from prepare_trance12 import render
from prepare_trance12_grounded import synthetic

FACT = re.compile(r'([A-Z][a-z]+) owns (?:a|an) (\w+) (\w+) and keeps it ([^.]+)\.')
QUESTION = re.compile(r'(?:What does \w+ own|Where is the \w+|What color is the \w+|Who owns the \w+)\?')


def simplify(turns, count, rng):
    prompt, answer = turns[0][1], turns[-1][1]
    facts = list(FACT.finditer(prompt))
    assert len(facts) == 3 and count in (1, 2)
    question = QUESTION.search(prompt).group()
    ownership = question.startswith('What does')
    target = question.split()[2] if ownership else question.split()[-1][:-1]
    # Exact field equality matters: 'hat' is a substring of 'What'.
    relevant = [f for f in facts if f.group(1 if ownership else 3) == target]
    assert len(relevant) == 1
    chosen = {relevant[0].start()}
    if count == 2:
        chosen.add(rng.choice([f for f in facts if f != relevant[0]]).start())
    context = ' '.join(f.group() for f in facts if f.start() in chosen)
    assert answer in context
    prompt = prompt[:facts[0].start()] + context + prompt[facts[-1].end():]
    return [('user', prompt), ('assistant', answer)]


def presence_pair(turns, rng):
    """Same question with one relevant fact, then one irrelevant fact."""
    prompt = turns[0][1]
    facts = list(FACT.finditer(prompt))
    question = QUESTION.search(prompt).group()
    owner = question.startswith('What does')
    target = question.split()[2] if owner else question.split()[-1][:-1]
    other = rng.choice([f for f in facts if f.group(1 if owner else 3) != target])
    negative = prompt[:facts[0].start()] + other.group() + prompt[facts[-1].end():]
    return [simplify(turns, 1, rng), [('user', negative), ('assistant', 'Not stated.')]]


def prompt_key(record):
    return ' '.join(record.split('<assistant>', 1)[0].casefold().split())


def novel_pairs(text, seen):
    rows = text.split('<eot>\n')[:-1]
    assert len(rows) % 2 == 0
    kept = [r for i in range(0, len(rows), 2)
            if all(prompt_key(x) not in seen for x in rows[i:i+2]) for r in rows[i:i+2]]
    return ''.join(r+'<eot>\n' for r in kept), len(rows)-len(kept)


def build(output):
    if output.exists():
        raise FileExistsError(f'Refusing to replace {output}')
    generated, families = synthetic()
    splits = {'train': generated['train'], 'dev': generated['validation'][:128],
              'test': generated['validation'][128:]}
    rng = random.Random(4111)
    files = {}
    for split, rows in splits.items():
        files[f'facts_{split}.txt'] = ''.join(map(render, rows))
        for count in (1, 2):
            files[f'facts{count}_{split}.txt'] = ''.join(render(simplify(t, count, rng)) for t in rows)
    # 80% two-fact examples, 20% one-fact examples; all answers are checked
    # fictional attributes, never authored substitutes for real-world facts.
    rng = random.Random(4112)
    one = files['facts1_train.txt'].split('<eot>\n')[:-1]
    two = files['facts2_train.txt'].split('<eot>\n')[:-1]
    mix = two + rng.sample(one, 15000)
    rng.shuffle(mix)
    files['progressive_train.txt'] = ''.join(r+'<eot>\n' for r in mix)
    rng = random.Random(4114)
    for split, rows in splits.items():
        files[f'presence_{split}.txt'] = ''.join(render(r) for t in rows for r in presence_pair(t, rng))
    rng = random.Random(4115)
    presence = files['presence_train.txt'].split('<eot>\n')[:-1]
    # Preserve positive/negative pairs when selecting the matching-task replay.
    anchors = [presence[2*i+j] for i in rng.sample(range(len(presence)//2), 15000) for j in (0, 1)]
    selection = files['facts_train.txt'].split('<eot>\n')[:-1] + rng.sample(two, 30000) + anchors
    rng.shuffle(selection)
    files['selection_train.txt'] = ''.join(r+'<eot>\n' for r in selection)
    # Removing distractors can collapse otherwise disjoint three-fact families
    # into the same one-fact prompt. Remove entire evaluation pairs on overlap.
    seen = {prompt_key(r) for name, text in files.items() if name.endswith('_train.txt')
            for r in text.split('<eot>\n')[:-1]}
    removed = {}
    for kind in ('facts', 'facts1', 'facts2', 'presence'):
        for split in ('dev', 'test'):
            name = f'{kind}_{split}.txt'
            files['novel_'+name], removed[name] = novel_pairs(files[name], seen)
    output.mkdir(parents=True)
    hashes = {}
    for name, text in files.items():
        content = text.encode()
        (output/name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    manifest = {'source': 'prepare_trance12_grounded.synthetic(seed=4109)',
                'simplification_seed': 4111, 'mix_seed': 4112,
                'counts': {s: len(rows) for s, rows in splits.items()},
                'progressive_train_records': len(mix), 'sha256': hashes,
                'presence_seed': 4114, 'selection_seed': 4115,
                'selection_train_records': len(selection),
                'evaluation_rows_removed_with_partner_for_prompt_overlap': removed,
                'original_family_hashes': families,
                'evaluation_limit': 'Same templates and vocabulary as training; new combinations, not broad chat capability.'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
