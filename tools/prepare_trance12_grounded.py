#!/usr/bin/env python3
"""Prepare a deterministic grounding experiment from local data and checked facts.

Synthetic examples describe fictional possessions, not real-world facts.
Every fact family and its changed-answer counterparts stay in one split.
Existing corpora and models are never replaced. No network or GPU required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from prepare_trance12 import heldout_prompt_key, parse, raw_token_bound, render
from prepare_trance12_replay import publish, records, sha
from prepare_trance12_curriculum import key

ROOT = Path(__file__).resolve().parents[1]
SEED = 4109
NAMES = 'Ada Ben Cora Dean Elena Felix Hana Ivan June Kira Leon Mina Noah Omar Pia Quinn Rosa Seth Tara Vera Willa Yuri Zane'.split()
OBJECTS = 'book cup bag scarf hat coat pencil notebook bottle bowl plate spoon fork towel blanket basket lamp clock ribbon brush'.split()
COLORS = 'red blue green yellow purple orange pink white black brown gray silver'.split()
PLACES = ['on the shelf', 'under the desk', 'beside the door', 'in the drawer',
          'on the table', 'under the chair', 'beside the window', 'in the cupboard',
          'on the bench', 'inside the box', 'near the sofa', 'by the bed']


def family_key(names, objects, colors, places):
    # All assignments/permutations of the same vocabulary family share a split.
    return key(json.dumps([sorted(x) for x in (names, objects, colors, places)]))


def fact_pairs(names, objects, colors, places, order, style):
    """Eight questions, each paired with changed facts and a changed answer."""
    questions = []
    for i in (0, 1):
        questions.extend([(f'What does {names[i]} own?', 'object', names[i]),
                          (f'Where is the {objects[i]}?', 'place', objects[i]),
                          (f'What color is the {objects[i]}?', 'color', objects[i]),
                          (f'Who owns the {objects[i]}?', 'owner', objects[i])])
    versions = []
    for assigned in (objects, objects[1:] + objects[:1]):
        facts = {name: dict(object=assigned[i], color=colors[i], place=places[i])
                 for i, name in enumerate(names)}
        context = ' '.join(f'{names[i]} owns {"an" if colors[i][0] in "aeiou" else "a"} {colors[i]} {assigned[i]} and keeps it {places[i]}.'
                           for i in order)
        outputs = []
        for question, kind, target in questions:
            if kind == 'object':
                answer = facts[target]['object']
            else:
                owner = next(name for name, fact in facts.items() if fact['object'] == target)
                answer = owner if kind == 'owner' else facts[owner][kind]
            prompt = (
                f'Read the passage and answer briefly.\nPassage: {context}\nQuestion: {question}',
                f'{context}\n\n{question} Answer with only the answer.',
                f'Question: {question}\nInformation: {context}\nGive a short answer using the information.'
            )[style]
            turns = [('user', prompt), ('assistant', answer)]
            assert answer in context and raw_token_bound(turns) <= 1024
            outputs.append(turns)
        versions.append(outputs)
    assert all(a[-1][1] != b[-1][1] for a, b in zip(*versions))
    return [turns for pair in zip(*versions) for turns in pair]


def synthetic(train_families=3750, validation_families=125, seed=SEED):
    rng = random.Random(seed)
    result = {'train': [], 'validation': []}
    families = {'train': set(), 'validation': set()}
    limits = {'train': train_families, 'validation': validation_families}
    while any(len(families[s]) < limits[s] for s in limits):
        values = [rng.sample(pool, 3) for pool in (NAMES, OBJECTS, COLORS, PLACES)]
        identity = family_key(*values)
        split = 'validation' if int(identity[:8], 16) % 20 == 0 else 'train'
        if len(families[split]) >= limits[split] or identity in families[split]:
            continue
        families[split].add(identity)
        result[split].extend(fact_pairs(*values, rng.sample(range(3), 3), rng.randrange(3)))
    assert not families['train'] & families['validation']
    return result, {s: sorted(v) for s, v in families.items()}


def evaluation_rows(fact_records):
    """Six paired questions from six distinct held-out families and varied tasks."""
    selected = [fact_records[16*i + 2*(i % 8) + offset] for i in range(6) for offset in (0, 1)]
    turns = [parse(record.rsplit('<eot>', 1)[0]) for record in selected]
    return [{'category': 'heldout_checked_facts', 'prompt': t[0][1], 'expected_answer': t[-1][1],
             'pair_id': i//2, 'check': 'Return the correct short answer from the supplied facts.'}
            for i, t in enumerate(turns)]


def build(root=ROOT):
    root = Path(root)
    source = root/'data/trance12_curriculum'
    inputs = {}

    def read(path):
        inputs[str(path.relative_to(root))] = sha(path)
        return list(records(path))

    generated, families = synthetic()
    generated = {s: [render(t) for t in rows] for s, rows in generated.items()}
    full = read(source/'train.txt')
    reading = sorted({r for r in full if r.startswith('<user>\nRead the passage and answer briefly.\nPassage: ')})
    assert len(reading) == 63192, 'Unexpected upstream reading corpus'
    reading_validation = read(source/'reading_validation.txt')
    instruction = {'train': [], 'validation': []}
    for name in ('no_robots', 'everyday_conversations', 'oasst1_en', 'smol_constraints', 'smol_rewrite'):
        for split in instruction:
            rows = read(root/f'data/trance12/{name}/{split}.txt')
            # Favor the smaller conversation sources without dropping task data.
            repeat = 3 if split == 'train' and name in ('no_robots', 'everyday_conversations', 'oasst1_en') else 1
            instruction[split].extend(rows * repeat)

    reserved_records, reserved_prompts = set(), set()
    # Reserve ALL existing validation, not just the selected new validation subset.
    for record in read(root/'data/trance12/validation.txt') + read(source/'validation.txt') + generated['validation']:
        reserved_records.add(key(record))
        turns = parse(record.rsplit('<eot>', 1)[0])
        for role, text in turns:
            if role == 'user':
                identity = heldout_prompt_key(text)
                if identity:
                    reserved_prompts.add(identity)
    evaluation_prompts = set()
    for name in ('trance_generalization_prompts.json', 'trance_curriculum_prompts.json'):
        p = root/'tools'/name
        inputs[str(p.relative_to(root))] = sha(p)
        evaluation_prompts.update(key(r['prompt']) for r in json.loads(p.read_text()))

    def allowed(record):
        if key(record) in reserved_records:
            return False
        turns = parse(record.rsplit('<eot>', 1)[0])
        return all(heldout_prompt_key(text) not in reserved_prompts and key(text) not in evaluation_prompts
                   for role, text in turns if role == 'user')

    def select(rows, count, label):
        return random.Random(f'{SEED}:{label}').sample(rows, count)

    components = {
        'checked_facts': [r for r in generated['train'] if allowed(r)],
        'squad': select([r for r in reading if allowed(r)], 20000, 'reading'),
        'instructions': select([r for r in instruction['train'] if allowed(r)], 20000, 'instructions'),
        'curriculum_replay': select([r for r in full if allowed(r)], 20000, 'replay'),
    }
    assert len(components['checked_facts']) == 60000
    valid = {'facts': generated['validation'],
             'reading': select(reading_validation, 500, 'validation-reading'),
             'instructions': select(instruction['validation'], 500, 'validation-instructions')}
    train = [r for rows in components.values() for r in rows]
    validation = [r for rows in valid.values() for r in rows]
    random.Random(SEED).shuffle(train)
    random.Random(SEED + 1).shuffle(validation)
    assert not {key(r) for r in train} & {key(r) for r in validation}
    outputs = {'train.txt': ''.join(train).encode(), 'validation.txt': ''.join(validation).encode()}
    for name, rows in valid.items():
        outputs[name+'_validation.txt'] = ''.join(rows).encode()
    # Twelve paired questions from held-out fact families for generated-answer testing.
    outputs['eval_prompts.json'] = (json.dumps(evaluation_rows(valid['facts']), indent=2)+'\n').encode()
    manifest = {
        'seed': SEED, 'training_records': len(train), 'training_unique': len({key(r) for r in train}),
        'validation_records': len(validation), 'component_records': {k: len(v) for k,v in components.items()},
        'validation_component_records': {k: len(v) for k,v in valid.items()},
        'synthetic_provenance': 'Programmatically authored fictional ownership/color/location facts. Answers derived from structured records; each question has a counterpart with changed facts and a different checked answer. No LLM-generated factual answers.',
        'source_provenance': 'Existing pinned SQuAD and instruction/curriculum data; see data/trance12_curriculum/manifest.json and data/trance12/manifest.json for source revisions/licenses.',
        'split_protection': 'Fact vocabulary families and all their permutations stay in one hash-assigned split. All original held-out records and substantive user prompts reserved. Fixed evaluation prompts excluded. Existing upstream splits retained. No semantic near-duplicate audit.',
        'limits': 'Synthetic templates are shared across splits: held-out fact accuracy tests new fact combinations, not arbitrary natural-language generalization. Use original reading/generalization suites and retention losses too. Record proportions are not token proportions.',
        'fact_families': families, 'inputs_sha256': inputs,
        'files': {k: {'bytes': len(v), 'sha256': hashlib.sha256(v).hexdigest()} for k,v in outputs.items()},
        'training_launched': False,
    }
    outputs['manifest.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    out = root/'data/trance12_grounded'
    for name, data in outputs.items():
        if (out/name).exists() and sha(out/name) != hashlib.sha256(data).hexdigest():
            raise ValueError(f'Existing output differs: {out/name}')
    for name, data in outputs.items():
        publish(out/name, data)
    return {k: v for k,v in manifest.items() if k not in ('fact_families', 'inputs_sha256')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.project), indent=2))
