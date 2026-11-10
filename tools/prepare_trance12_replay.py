#!/usr/bin/env python3
"""Prepare a shuffled 75% original-data / 25% focused-record pilot mix.

No downloads, training or replacement of existing corpora. Repeated records
are intentional sampling weights. Existing outputs must match exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

from prepare_trance12 import iter_text, parse, raw_token_bound, render

ROOT=Path(__file__).resolve().parents[1]
FOCUS=("no_robots","everyday_conversations","oasst1_en")


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def records(path):
    for index,text in enumerate(iter_text(path),1):
        turns=parse(text)
        if turns is None or raw_token_bound(turns)>1024:
            raise ValueError(f"Invalid or over-length record {index}: {path}")
        yield render(turns)


def publish(path,contents):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        temp=Path(f.name);f.write(contents);f.flush();os.fsync(f.fileno())
    try:
        try:os.link(temp,path)
        except FileExistsError:
            if sha(path)!=hashlib.sha256(contents).hexdigest():
                raise ValueError(f"Existing output differs; preserved: {path}")
    finally:temp.unlink(missing_ok=True)


def build(root=ROOT,seed=4004):
    root=Path(root);source=root/'data/trance12';out=root/'data/trance12_replay'
    focused=[];seen=set();human=set();inputs={}
    for name in FOCUS:
        path=source/name/'train.txt';inputs[str(path.relative_to(root))]=sha(path)
        for record in records(path):
            key=' '.join(record.split()).casefold()
            if key in seen:raise ValueError("Focused records are not globally unique")
            seen.add(key);focused.append(record)
            if name!='everyday_conversations':human.add(key)
    if not focused:raise ValueError("Focused corpus is empty")
    wanted=3*len(focused);replay=[];count=0;rng=random.Random(f'replay:{seed}')
    full=source/'train.txt';inputs[str(full.relative_to(root))]=sha(full)
    for count,record in enumerate(records(full),1):
        if count<=wanted:replay.append(record)
        else:
            index=rng.randrange(count)
            if index<wanted:replay[index]=record
    if count<wanted:raise ValueError("Original corpus is too small for the replay sample")
    # Exact duplicates across the two components are intentional weights.
    mixed=focused+replay;random.Random(seed).shuffle(mixed)
    text=''.join(mixed).encode('utf-8')
    manifest={'seed':seed,'context_raw_bound':1024,'focused_records':len(focused),
              'replay_records':len(replay),'original_weighted_records':count,'total_records':len(mixed),
              'focused_component_share':.25,'replay_component_share':.75,
              'unique_records':len({' '.join(t.split()).casefold() for t in mixed}),
              'human_record_share':sum(' '.join(t.split()).casefold() in human for t in mixed)/len(mixed),
              'duplicates':'intentional sampling weights; original repetition retained in reservoir sampling',
              'input_sha256':inputs,'train_bytes':len(text),'train_sha256':hashlib.sha256(text).hexdigest(),
              'validation':'unchanged original validation; evaluate focused validation separately',
              'training_started':False}
    manifest_bytes=(json.dumps(manifest,indent=2)+'\n').encode()
    # Preflight both destinations before creating anything.
    for path,contents in ((out/'train.txt',text),(out/'manifest.json',manifest_bytes)):
        if path.exists() and sha(path)!=hashlib.sha256(contents).hexdigest():
            raise ValueError(f"Existing output differs; preserved: {path}")
    publish(out/'train.txt',text);publish(out/'manifest.json',manifest_bytes)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=ROOT)
    args=parser.parse_args();print(json.dumps(build(args.project),indent=2))
