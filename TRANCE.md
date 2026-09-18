# How Trance Works

This document walks through `trance.cu` end to end: the model architecture,
the tokenizer, the training pipeline, the CUDA/cuBLAS implementation, and
the CLI. It's meant to orient someone reading the source for the first
time, and to explain *why* things are built the way they are, not just
*what* the code does.

Trance is a from-scratch, single-file, GPU-trained decoder-only Transformer
(a small GPT). Everything, the tokenizer, model, training loop, and CLI,
lives in one `.cu` file and compiles with `nvcc` plus cuBLAS.

```
nvcc -O3 -o trance trance.cu -lcublas
# run this before trusting a build
./trance test
```

---

## 1. The model

Trance is a standard decoder-only Transformer: token and positional
embeddings, a stack of identical blocks (causal self-attention, residual,
LayerNorm, feed-forward, residual, LayerNorm), a final LayerNorm, and a
linear head that projects to vocabulary logits.

Each layer does, in order:

1. **LayerNorm**, then **Q/K/V projections**, then **causal self-attention**,
   then **output projection**, then **residual add**
2. **LayerNorm**, then **feed-forward (linear, GELU, linear)**, then
   **residual add**

This is the "pre-norm" Transformer layout (LayerNorm before each sub-block,
not after). Most modern small Transformers use this layout because it
trains more stably than a post-norm design.

A `Config` (loaded from a JSON file, see [§6](#6-configuration)) fully
specifies the architecture: vocabulary size, context length, embedding
dimension, number of layers, number of attention heads, and feed-forward
width. Attention heads must evenly divide the embedding dimension, since
each head gets `embedding_dim / attention_heads` dimensions to itself.

### Parameters

Every learnable tensor is a `Tensor { float *w, *g, *m, *v; size_t n; }`,
meaning weights, gradients, and the two Adam moment buffers, all as flat
device arrays. A `Model` holds the embedding tables, the final
LayerNorm/output head, and an array of per-layer tensors (16 tensors per
layer: two LayerNorm gain/bias pairs, Q/K/V/output-projection weight and
bias, and the two feed-forward weight and bias pairs).

---

## 2. The tokenizer

Trance uses byte-level BPE (byte-pair encoding), the same family of
tokenizer GPT-2 and GPT-3 use, implemented from scratch.

### Base vocabulary and special tokens

- IDs `0-255`: the 256 raw byte values. Every possible byte is a valid
  starting token, so *anything* can be encoded, even text the tokenizer has
  never seen.
- ID `256`: `EOT`, end-of-text / conversation boundary.
- ID `257`: a reserved `<user>` token.
- ID `258`: a reserved `<assistant>` token.
- IDs `259` and up: BPE merges, learned from training data.

`<user>`, `<assistant>`, and `<eot>` are **not** ordinary text that happens
to get BPE-merged. They're reserved, atomic token IDs from the start. This
matters for two things: the training loss mask (§4) needs to know exactly
where an assistant turn begins and ends, and generation needs to stop
exactly on a role boundary. Both become simple integer comparisons instead
of pattern-matching decoded text.

### Training the merges

`bpe_train()` is the classic BPE algorithm: repeatedly find the most
frequent adjacent pair of tokens in the corpus and merge it into a new
token, until the vocabulary reaches its target size. Special tokens are
excluded from merge candidacy. They can never be absorbed into a larger
token or have anything merged onto them.

Training the tokenizer on a huge corpus is expensive, since every merge
iteration rescans the whole corpus, so `bpe_train()` only ever runs on a
bounded random **sample** of the training data (see
`sample_for_tokenizer_training()` in §4). A representative ~8MB sample
learns essentially the same merge rules as the full corpus would, at a
fraction of the cost. The *full* dataset is still encoded and trained on
afterward. Only merge-rule discovery is sampled.

### Encoding: the trie

Once trained, encoding text means repeatedly finding the *longest* known
token starting at each position (greedy longest-match). The naive way to do
that is to test every vocabulary entry at every position. That's what an
early version of this code did, and it scales as `O(vocabulary_size)` per
character, which becomes the dominant cost on any dataset larger than a few
tens of megabytes.

Instead, `bpe_encode_raw()` builds a **trie** (prefix tree) over the
vocabulary once, then walks it byte by byte from each position, following
the actual input bytes. That's `O(longest_token_length)` per character,
independent of vocabulary size. The trie is cached process-wide
(`cached_trie()`) and only rebuilt when a tokenizer is actually retrained or
reloaded (tracked with a generation counter, not pointer identity; see the
comment on `g_tok_generation` for why pointer identity isn't safe here: the
REPL's `load` command can repopulate the same `Tokenizer` variable with a
different model's vocabulary).

### Token storage

Every token ID, whether a raw byte, a special token, or a merge, fits in 16
bits (the vocabulary is capped at `MAX_VOCAB = 1024`). `Data.x` is
`uint16_t*` (aliased as `tok_t`), not `int*`. On a multi-hundred-megabyte
dataset this cuts the memory footprint of the single largest allocation in
the program in half.

---

## 3. Attention

Causal self-attention: each position can attend to itself and everything
before it, never anything after. For a batch of `B` sequences, each of
length `T`, with `H` heads of dimension `dh = embedding_dim / H`:

1. **Scores**: `scale · Q · Kᵗ` per head, where `scale = 1/√dh`.
2. **Softmax**, row-wise, causally masked (only columns `0..t` count for
   row `t`).
3. **Weighted sum**: `softmax_scores · V`.

The two matrix multiplies (steps 1 and 3) run through **cuBLAS batched
GEMM** (`cublasSgemmStridedBatched`), one call per attention head covering
every sequence in the batch at once. Only the softmax itself, the part that
isn't a plain matrix multiply, is a small custom kernel (`k_softmax_causal`
/ `k_softmax_back`).

One correctness subtlety worth knowing if you're reading the backward pass:
a GEMM has no concept of "only the lower-triangular half of this matrix
matters." `k_softmax_causal` explicitly zeroes the non-causal (upper
triangular) part of the score matrix after computing the causal softmax.
Once that's zero, every downstream gradient GEMM (for `dV`, `dQ`, `dK`)
automatically produces the correct causally-masked gradient for free.
Multiplying by zero contributes nothing, no matter what's on the other side
of the multiplication. That's why the backward pass doesn't need any extra
masking logic of its own.

---

## 4. Training data and the loss

### Conversation format

Training text is plain UTF-8 with three literal markers:

```
<user>
What's the capital of France?

<assistant>
The capital of France is Paris.

<eot>
```

`raw_file_list()` reads this and converts each marker to its reserved token
ID (§2) while building a **loss mask** in lockstep: `1` from the
`<assistant>` marker through the end of that turn (including the
terminating `<eot>`), `0` everywhere else. The mask survives BPE encoding
(`bpe_encode_raw()` propagates it token by token) and ends up as
`Data.mask`, the same length as `Data.x`.

### Why mask the loss at all

Without masking, the model spends its (limited) capacity learning to
predict the *user's* side of the conversation too, the question text, which
is wasted effort and actively pulls training away from the actual task of
generating good responses. Masking makes the objective "predict the
assistant's turn, given everything before it," which is what you actually
want from an instruction-following model.

`k_xent` (the cross-entropy kernel) takes a per-row mask and a per-row
**normalizer**. The normalizer isn't a fixed `1/context_length`. It's
`1/(number of assistant-masked tokens in that sequence's window)`, computed
per sequence. This means a window that's mostly user text with a short
reply isn't penalized or rewarded differently than a window that's mostly
assistant text. Both are averaged over *their own* relevant token count.

### Training-window sampling

Rather than picking a uniformly random byte offset in the corpus (which
would disproportionately sample from wherever conversations happen to be
positioned, and can land mid-token or mid-turn), `train()` first scans the
tokenized corpus once for conversation boundaries (every position right
after an `EOT`). Each training step then:

1. Picks a random **conversation** (not a random byte offset).
2. If the conversation fits in the context window, uses it whole (starting
   exactly at its first token; if it's short, the window naturally spills
   into the next conversation, which is harmless since `EOT` and the loss
   mask still draw the correct boundaries).
3. If it's longer than the context window, picks a random valid window
   *within* that conversation.

This gives every conversation roughly equal sampling weight regardless of
length, and never falls back to a fixed offset (like position 0) when
something doesn't fit. That fallback was a bug in an earlier version, and
it would have silently over-trained on whatever happened to be at the start
of the file.

### Batching

`train()` builds one flat array of `batch_size × context_length` tokens
(`batch_size` independent conversation windows, back to back) and calls
`forward()`/`backward()` **once** per training step, not once per sequence
in the batch. This is real architectural batching: every linear layer and
LayerNorm operates on all `batch_size × context_length` rows in a single
cuBLAS call, instead of looping over the batch with separate small calls.
Attention still processes one sequence at a time internally (per-head
batched GEMM across the whole batch, per §3), but nothing else does.

---

## 5. The CUDA / cuBLAS implementation

The linear layers, meaning every attention projection and every
feed-forward layer, are the dominant cost in the whole model. They run
through **cuBLAS SGEMM**, not a hand-rolled kernel. A hand-written
one-thread-per-output-element kernel (which is where this code started)
has no shared-memory reuse and needs `atomicAdd` in its backward pass
wherever multiple threads write to the same gradient location, both of
which are slow on real GPU hardware. cuBLAS uses tiled, shared-memory
matmul algorithms and its own reduction strategy instead of atomics.

Passing row-major C data to cuBLAS (which is column-major) uses the
standard reinterpretation trick: a row-major matrix `M[p,q]` occupies
exactly the same bytes as a column-major matrix `Mᵗ[q,p]`, so nothing is
ever physically transposed. Only the `transa`/`transb` flags and dimensions
passed to cuBLAS change. See the comments on `linear()` / `linear_back()`
for the exact derivation if you need to touch this code.

`cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH)` is enabled unconditionally
at startup. On Ampere-or-newer GPUs this routes FP32 GEMMs through tensor
cores at a small, generally negligible precision cost. On older GPUs it's
simply ignored.

Everything that *isn't* a big matrix multiply, meaning embeddings,
LayerNorm, GELU, Adam, the softmax kernels, and cross-entropy, is a small,
straightforward CUDA kernel, one thread per element or per row.

---

## 6. Configuration

Training is driven by a JSON config file:

| JSON key | Meaning |
|---|---|
| `vocab_size` | Target BPE vocabulary size (>= 259, <= 1024) |
| `context_length` | Tokens per training window (<= 1024) |
| `embedding_dim` | Model width; must be divisible by `attention_heads` |
| `layers` | Number of Transformer blocks |
| `attention_heads` | Number of attention heads |
| `feed_forward_dim` | Width of the feed-forward hidden layer |
| `batch_size` | Sequences per training step |
| `training_steps` | Total optimizer steps |
| `eval_interval` | Steps between validation loss checks |
| `checkpoint_interval` | Steps between checkpoint saves |
| `learning_rate`, `gradient_clip` | Adam / clipping hyperparameters |
| `seed` | RNG seed |
| `train_data`, `validation_data` | Comma-separated file path(s) |
| `output_model` | Where to save the final model |
| `resume_model` | Optional: continue training from a checkpoint |

`config_load()` validates all of this up front (positive/finite learning
rate, heads dividing embedding dimension, non-negative intervals, and so
on), so a malformed config fails immediately with a clear message instead
of crashing deep inside training.

---

## 7. Files on disk

Two file formats, both custom binary:

- **Model files** (`MAGIC = "TAIGPT1"`, `VERSION = 2`): the architecture
  (vocab/ctx/embedding_dim/layers/heads/ff), the step count, and every
  tensor's weights, plus, for checkpoints, the Adam moment buffers too, so
  training can resume exactly where it left off.
- **Tokenizer files** (`magic = "TAITOK3"`, saved alongside the model as
  `MODEL.tok`): the vocabulary size and every token's byte sequence and
  length.

The version/magic strings get bumped whenever the on-disk format changes in
a way that would otherwise be silently misread (the vocabulary reserving
`USER_TOK`/`ASST_TOK` was one such change). Old files are cleanly rejected
instead of being loaded and misinterpreted.

---

## 8. Generation

`generate()` builds a fixed-size context window from the conversation
history so far (real tokens first, `EOT` padding after; the padding never
affects the tokens actually used for sampling, since causal attention can't
look at positions after the one being predicted), runs `forward()`, and
samples the next token from the logits at the last real position using
temperature, top-k, and top-p sampling. Generation stops the moment it
samples *any* special token (`EOT`, `<user>`, or `<assistant>`), an exact
check now that these are real token IDs, rather than the string-search over
decoded text an earlier version relied on.

`chat_loop()` wraps this in a REPL: each message is framed as
`<user>\n{message}\n\n<assistant>\n` before being handed to `generate()`.

Note: every generated token currently recomputes the full forward pass over
the whole context so far. There's no KV-cache. For this model's size that's
not a practical problem, but it's the main remaining inference-speed lever
if longer generations ever matter.

---

## 9. Testing

`./trance test` (`run_tests()`) is the thing to run after touching anything
in this file, especially the CUDA/cuBLAS code. It checks, in order:

- BPE merge training and encoding
- Tokenizer save/load round-trip
- Model initialization
- Forward pass: finite outputs, correct causal masking
- **Finite-difference gradient checks**: perturbs an actual weight by
  ±0.001, and compares the resulting loss change against the analytic
  gradient from backprop. Done for the output layer, token embeddings, and
  a Q projection weight (attention-specific, since that's the part most
  recently rewritten). This is the check that would catch a subtly wrong
  CUDA/cuBLAS kernel: it would still produce *a* number, just the wrong
  one, and this test compares against ground truth instead of just
  checking whether it crashed.
- **Batch consistency**: running two sequences together (`B=2`) must give
  the same total loss and gradients as running them separately (`B=1`
  twice) and summing. This directly verifies that batching doesn't change
  the math, only the performance.
- Loss-mask parsing, tokenizer round-trip, and special-token handling
- Serialization (model and checkpoint save/load)
- Generation (runs without crashing)

If you change anything about the linear layers, attention, batching, or the
tokenizer, the finite-difference and batch-consistency checks are the ones
that actually verify correctness, rather than just confirming it compiled
and didn't crash. Extending them to cover more parameters (K/V/output
projections, feed-forward weights, LayerNorm gains) is a reasonable next
step if you're making further changes in those areas.

---

## 10. Known limitations

- **No KV-cache** for generation (§8). Fine for this model's size, but
  would matter more for longer contexts or bigger models.
- **No context packing**. A training window samples one conversation (or a
  slice of a long one); short conversations leave the rest of the context
  window unused rather than packing in a second, unrelated conversation.
- **Special-token detection during data loading is a literal string
  match** on `<user>`/`<assistant>`/`<eot>`/`<|endoftext|>` in the raw
  text. This is exact once the text is being parsed, but relies on those
  exact strings appearing in your source data. A different marker
  convention would need a corresponding change in `raw_file_list()`.
- **Validation window count is fixed at 64** (though now batched, see §4,
  so raising it is cheap to do if you want a less noisy validation curve).
