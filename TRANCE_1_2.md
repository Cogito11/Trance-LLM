# Trance 1.2 — Training Record

This document records the completed training stages incorporated into Trance 1.2. The accepted lineage currently ends at **71,000 optimizer steps**, as of October 14, 2026. Append a stage only after its resulting checkpoint has been accepted for inclusion in the model, and update the artifact and evaluation sections to match.

## Model and training method

Trance 1.2 is a custom decoder-only Transformer trained from random initialization using the project's native CUDA/C++ engine and cuBLAS on an **NVIDIA RTX 4060 with 8 GB VRAM**. It does not inherit Trance 1.1 weights.

| Setting | Value throughout the accepted lineage |
|---|---:|
| Parameters | 30,349,440 |
| Transformer layers | 12 |
| Embedding width | 448 |
| Attention heads | 8 |
| Feed-forward width | 1,792 |
| Vocabulary | 1,024 tokens |
| Context length | 1,024 tokens |
| Batch size | 4 |
| Global gradient clipping | 1.0 |
| Optimizer | Adam, beta1 0.9, beta2 0.999, epsilon 1e-8 |

The architecture uses causal self-attention, LayerNorm and GELU feed-forward blocks. A byte-based BPE tokenizer was learned from the initial corpus and retained unchanged through every continuation. Training examples use user/assistant role markers and end-of-turn markers. Cross-entropy is restricted to assistant targets; the engine averages supervised-token loss within each sequence and then across the batch.

Learning rates are constant within each stage. Continuations restore model weights, Adam state and the cumulative step count from optimizer checkpoints. Sampling seeds are recorded separately because the sampler's random state is reset on resume. There is no configured learning-rate schedule, gradient accumulation or weight decay.

## Accepted training lineage

| Stage | Step range | Updates included | Learning rate | Sampling seed | Evaluation / checkpoint interval |
|---|---|---:|---:|---:|---|
| Initial instruction training | 0 → 40,000 | 40,000 | 0.00020 | 4004 | 1,000 / 10,000 |
| Balanced replay | 40,000 → 40,500 | 500 | 0.00001 | 4004 | 100 / 250 |
| Balanced replay continuation | 40,500 → 41,000 | 500 | 0.00001 | 4005 | 100 / 250 |
| Language and reading curriculum | 41,000 → 71,000 | 30,000 | 0.00003 | 4107 | 1,000 / 1,000 |

### Initial instruction training: 0 → 40,000

Training started from random weights with a newly learned tokenizer. The accepted output of this stage is **models/trance1.2_30m.bin.step40000.bin**. The original config, configs/trance1.2_30m.json, allocated a 140,000-step budget; the lineage documented here uses its **40,000-step checkpoint**.

The prepared instruction corpus contained **100,928 unique conversations**, expanded through source weighting to **118,972 records**, totaling **90,321,874 bytes**. Validation contained **11,933 unique records**, totaling **8,643,265 bytes**.

| Training source | Unique conversations | Repetition |
|---|---:|---:|
| Smol-Magpie-Ultra | 35,465 | 1× |
| UltraChat | 15,000 | 1× |
| No Robots | 5,069 | 3× |
| Everyday conversations | 2,006 | 3× |
| SmolTalk constraints | 15,000 | 1× |
| SmolTalk rewriting | 6,441 | 1× |
| SmolTalk summarization | 20,000 | 1× |
| OpenAssistant English | 1,947 | 3× |

Sources combine human-written and synthetic content. Preparation retained complete user/assistant exchanges, filtered malformed text and reserved markers, deduplicated across sources, and excluded substantive held-out first prompts from training. Long multi-turn examples retained complete prefixes; an over-length first exchange was rejected. Accepted records fit a conservative 1,024-token bound calculated from UTF-8 bytes and reserved markers. The trainer can pack following records into the remaining window, so a packed tail may still be cut.

Large sources were sampled using seeded uniform reservoir sampling, and the final corpus was shuffled with seed 4004. The existing local splits were preserved for legacy inputs; source manifests record their hashes. Split filtering checks exact normalized overlap, not all semantic near-duplicates. Structural filtering does not establish factual accuracy.

### Balanced replay: 40,000 → 41,000

Two completed 500-step stages continued the initial checkpoint using **data/trance12_replay/train.txt**. The corpus contained **36,088 weighted records**, **30,023 unique conversations** and **26,363,238 bytes**:

- 27,066 records sampled from the original weighted instruction mix: 75% of records.
- 9,022 records from No Robots, everyday conversations and OpenAssistant: 25% of records.

The sampling overlap and repeated records provide intentional weighting. These shares describe records, not tokens. Validation remained the original instruction validation file.

The first stage produced **models/trance1.2_replay.bin.step40500.bin**, which was resumed by the second stage. The second produced **models/trance1.2_replay_extended.bin.step41000.bin**, the starting checkpoint for the curriculum. Recorded runtimes were approximately **168 seconds** and **169 seconds**, respectively.

These continuations used the corrected Beta 19 engine. Its training sampler preserves conversation starts near the end of the corpus and masks padded targets. Its chat path ends the generation prefix at the atomic assistant marker, lets the model generate the formatting newline, and normalizes separators when rebuilding history. Model and tokenizer formats remain compatible.

### Language and reading curriculum: 41,000 → 71,000

The completed curriculum stage used **configs/trance1.2_curriculum_41k.json** and **data/trance12_curriculum/train.txt**, resuming the 41,000-step optimizer checkpoint. It added simpler narrative language and passage-based question answering while retaining all original weighted instruction records.

| Component | Unique examples | Weighted records | Record share |
|---|---:|---:|---:|
| TinyStories V2, GPT-4-only | 200,000 | 200,000 | 44.9% |
| SQuAD passage/question/answer | 63,192 | 126,384 | 28.4% |
| Original instruction corpus | 100,928 | 118,972 | 26.7% |

The combined corpus contained **364,120 unique examples**, **445,356 weighted records** and **339,622,205 bytes**. Record proportions differ from supervised-token proportions. Each selected SQuAD example appears twice; original instruction repetition weights are retained.

TinyStories content came from the source's GPT-4-generated V2 files. Uniform sampling selected 200,000 eligible stories, formatted as story requests or continuations with supervised assistant text. Filtering retained complete endings and rejected overly long or obviously malformed stories. These are synthetic fictional stories, not factual reference material.

SQuAD supplied human-authored questions and annotated answer spans. Preparation retained eligible full passages, checked answers against their annotated source offsets, and selected short answers. It did not author replacement factual answers. Official validation stories, reading passages and questions were reserved before selecting training examples. Exact normalized conversations and substantive prompts were also checked for overlap, and the existing generalization prompts were excluded.

The new mixed validation file contains **5,000 records**: 2,500 stories, 1,000 reading examples and 1,500 original instruction examples. Separate component files support evaluation. All prepared records meet the conservative 1,024-token bound.

The stage completed on **October 14, 2026**, in **2 hours 43 minutes**. Its final and lowest logged mixed validation loss was **1.5264** at step 71,000. The retained artifact is **models/trance1.2_curriculum.bin.step71000.bin**, usable for both inference and training resume. The duplicate inference-only export was removed during model cleanup; the accepted weights are unchanged.

## Evaluation of the accepted checkpoints

The following are read-only, assistant-masked cross-entropy measurements using the same tokenizer and 64 fixed sampled windows per dataset, evaluated with batch size one. Lower is better. They are sampled prediction losses, not accuracy scores; absolute losses from different validation datasets should not be compared as if they measured identical tasks.

| Original held-out set | 40,000 | 40,500 | 41,000 | 71,000 |
|---|---:|---:|---:|---:|
| Full instruction validation | 1.9301 | 1.8987 | 1.8873 | 1.8826 |
| No Robots + everyday + OpenAssistant | 2.5747 | 2.5480 | 2.5391 | 2.5138 |

| Curriculum held-out set | Before curriculum, 41,000 | After curriculum, 71,000 |
|---|---:|---:|
| Mixed curriculum | 2.5283 | 1.5264 |
| Stories | 2.7348 | 1.3653 |
| Reading | 2.4063 | 1.1394 |
| Selected original instructions | 1.8922 | 1.8893 |

At 71,000, generated story sentences were more readable than at 41,000, but character consistency, logical progression and repetition remained problems. On eight fixed held-out reading questions, manual review found **1/8 correct**, compared with **0/8** before the curriculum. The separate 20-prompt generalization screen found no response that fully satisfied its prompt's requirements. Factual answering, arithmetic and instruction following remained unreliable. These small screens establish limitations of the accepted checkpoint, not comprehensive benchmark scores.

Both response screens used fresh contexts, top K 1, top P 1, temperature 0.5 and seed 4004. Curriculum replies had a 160-token cap; generalization replies had a 192-token cap. Three of four curriculum stories reached the cap, so their truncated endings alone do not demonstrate inability to finish a story.

## Data provenance and reproducibility

The preparation tools are **tools/prepare_trance12.py**, **tools/prepare_trance12_replay.py** and **tools/prepare_trance12_curriculum.py**. Dependency versions are recorded in **tools/requirements-trance12.lock.txt**. Dataset manifests contain input/output SHA-256 hashes, counts, filtering details and source provenance:

- data/trance12_sources/manifest.json
- data/trance12/manifest.json
- data/trance12_replay/manifest.json
- data/trance12_curriculum/manifest.json

Downloaded additions used the following pinned dataset revisions. Source cards are retained with the prepared data; the license column records their declared dataset licenses.

| Dataset repository | Revision | Declared license |
|---|---|---|
| HuggingFaceTB/smoltalk | 5feaf2fd3ffca7c237fc38d1861bc30365d48ffa | Apache-2.0 |
| OpenAssistant/oasst1 | fdf72ae0827c1cda404aff25b6603abec9e3399b | Apache-2.0 |
| roneneldan/TinyStories | f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 | CDLA-Sharing-1.0 |
| rajpurkar/squad | 7b6d24c440a36b6815f21b70d25016731768db1f | CC-BY-SA-4.0 |

Legacy source files are identified by their hashes in the original corpus manifest; pinned upstream revisions are not available for every legacy input.

Exact replay run configurations are preserved in data/gui/runs/20261014-090331-44b62bc0/config.json and data/gui/runs/20261014-091726-a53bea1b/config.json. The completed curriculum's configuration, events and timing are preserved in data/gui/runs/20261014-144934-5c1d4015/. Evaluation tools and fixed prompts are in tools/evaluate_checkpoint.cu, tools/compare_checkpoints.py, tools/trance_curriculum_prompts.json and tools/trance_generalization_prompts.json. The accepted 71,000-step losses and raw responses are in data/gui/evaluations/trance12-curriculum-71000-{heldout,replies,generalization}.json.

## Accepted artifact identity

Keep the model and its matching tokenizer together. The retained optimizer checkpoint supports both inference and continued training. The following SHA-256 hashes identify the currently accepted 71,000-step files:

```text
7ed4ddabe5f0321df561272e44d9a8e0671053e11b0e9f06188136a0f4240b52  trance1.2_curriculum.bin.step71000.bin
096af86814980bd1d9b269bd2075ffd9a80193b6b774c6973e4db28dbb225423  trance1.2_curriculum.bin.step71000.bin.tok
```

When a further completed stage is accepted, append its actual training settings, data provenance, parent/output checkpoints and measured results, then update the accepted step count and artifact hashes. This record can accompany the eventual model release; its current endpoint identifies the accepted development checkpoint.
