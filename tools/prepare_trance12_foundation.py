#!/usr/bin/env python3
"""Extract the existing, source-identified TinyStories subset for a fresh pilot.

No downloads or newly authored story text. Preserve the previous train/validation
split and complete story requests/continuations. This is a language-foundation
stage; the resulting model is not trained as a general assistant.
"""
import hashlib
import json
from pathlib import Path
import random

from prepare_trance12 import parse
from prepare_trance12_curriculum import key
from prepare_trance12_replay import records, sha

ROOT = Path(__file__).resolve().parents[1]
SEED = 4116
PREFIXES = ('Continue this story:\n', 'Finish this short story:\n',
            'What happens next?\n', 'Continue the story below:\n')
REQUESTS = {'Tell me a short story.', 'Write a simple story.',
            'Tell a short story.', 'Please tell a short story.'}


def story_identity(record):
    turns = parse(record.rsplit('<eot>', 1)[0])
    assert turns and len(turns) == 2 and [t[0] for t in turns] == ['user', 'assistant']
    prompt, answer = turns[0][1], turns[1][1]
    for prefix in PREFIXES:
        if prompt.startswith(prefix):
            return key(prompt[len(prefix):] + ' ' + answer)
    assert prompt in REQUESTS, f'Unexpected story source prompt: {prompt[:80]}'
    return key(answer)


def build(root=ROOT):
    root = Path(root)
    output = root/'data/trance12_foundation'
    if output.exists():
        raise FileExistsError(f'Refusing to replace {output}')
    curriculum = root/'data/trance12_curriculum/train.txt'
    legacy = root/'data/trance12/train.txt'
    validation = root/'data/trance12_curriculum/stories_validation.txt'
    legacy_keys = {key(r) for r in records(legacy)}
    stories = []
    for record in records(curriculum):
        if record.startswith('<user>\nRead the passage and answer briefly.\nPassage: '):
            continue
        if key(record) in legacy_keys:
            continue
        story_identity(record)  # Reject unidentified source records.
        stories.append(record)
    assert len(stories) == 200000, f'Unexpected story subset size: {len(stories)}'
    heldout = list(records(validation))
    train_keys = {story_identity(r) for r in stories}
    valid_keys = {story_identity(r) for r in heldout}
    assert len(train_keys) == len(stories) and len(valid_keys) == 2500
    assert not train_keys & valid_keys, 'Story content overlaps validation'
    random.Random(SEED).shuffle(stories)
    content = ''.join(stories).encode()
    previous = json.loads((root/'data/trance12_curriculum/manifest.json').read_text())
    manifest = {
        'purpose': 'Fresh 30M language-foundation pilot, before general assistant fine-tuning',
        'seed': SEED, 'newly_authored_examples': 0, 'new_downloads': 0,
        'train_records': len(stories), 'validation_records': len(heldout),
        'train_bytes': len(content), 'source': previous['sources']['stories'],
        'input_sha256': {str(p.relative_to(root)): sha(p) for p in (curriculum, legacy, validation)},
        'output_sha256': {'train.txt': hashlib.sha256(content).hexdigest()},
        'validation_data': str(validation.relative_to(root)),
        'checks': {'identified_story_prompts_only': True, 'unique_full_stories': True,
                   'full_story_train_validation_disjoint': True},
        'limitations': 'Synthetic fictional stories. This stage targets coherent language, not factual QA, instruction following or release readiness.'}
    output.mkdir(parents=True)
    (output/'train.txt').write_bytes(content)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ('train_records', 'validation_records', 'train_bytes')}))


if __name__ == '__main__':
    build()
