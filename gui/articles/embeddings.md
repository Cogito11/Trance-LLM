# Embeddings and positions

A token ID is an index, not a numerical measure of a word's meaning. ID 200 is not “twice as meaningful” as ID 100.

Trance looks up each ID in a learned **embedding table**. The result is a vector: a list of floating-point numbers. The configuration's **embedding_dim** is the length of this vector, also called the model width.

## Giving the sequence an order

Attention needs to distinguish where tokens occur. “The dog chased the cat” and “The cat chased the dog” use similar words in different positions.

Trance adds a learned **position embedding** to each token vector. There is one position vector per position in the configured context window. These are learned absolute positions; this implementation does not use rotary position embeddings.

The combined vectors enter the Transformer blocks. As they move through the blocks, their values become dependent on the surrounding text rather than being fixed dictionary entries.

## Changing the width

A wider model has more parameters and larger working buffers. Width must divide evenly among the attention heads. Increasing width or context changes the model architecture, so it cannot be applied to an existing checkpoint as a simple settings adjustment.

Continue with [causal attention](article://attention) or [the output head](article://output).
