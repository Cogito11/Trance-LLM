# Causal attention and heads

Attention lets each token combine information from relevant positions in its context. In “The dog was tired because it had run”, useful context helps connect “it” with “dog”. This example describes the goal of attention, not a guarantee that Trance will interpret it correctly.

## Queries, keys and values

Each block transforms its incoming vectors into three sets of vectors:

- **Queries** describe what each position is looking for.
- **Keys** provide features that queries can match.
- **Values** hold information to combine.

Trance compares queries with keys, scales the scores, applies a softmax and uses the resulting weights to mix values. Learned projections transform that mixture back into the model width.

## The causal mask

While predicting the next token, a position can use itself and earlier positions, but not future ones. This **causal mask** prevents the model from seeing the answer during prediction.

Multiple **heads** perform attention in parallel on different portions of the width. They can learn different relationships; each head is not assigned a fixed human-defined job.

Trance materializes attention matrices. Their memory cost grows roughly with the square of context length, so doubling context can be much more expensive than doubling a simple text buffer.

See [Transformer blocks](article://blocks). Further reading: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), the original Transformer paper. Trance uses a decoder-only variant rather than the paper's full encoder-decoder model.
