# Feed-forward, residuals and LayerNorm

A Trance Transformer block applies two main operations: attention combines information across positions, and a feed-forward network transforms each position's vector.

## One block, in order

1. Apply **LayerNorm** to the incoming vectors.
2. Compute causal attention and its output projection.
3. Add that result to the original vectors: a **residual connection**.
4. Apply a second LayerNorm.
5. Apply a linear layer, **GELU**, and another linear layer.
6. Add a second residual connection.

Trance uses this **pre-norm** layout, with normalization before each main operation.

## What the pieces contribute

LayerNorm adjusts the scale of a vector using learned gain and bias parameters. It helps keep numerical behavior manageable, but it does not replace data validation or gradient clipping.

The feed-forward network temporarily expands the width to **feed_forward_dim**, uses GELU as a nonlinear transformation, then projects back down. A linear-only stack would be much less expressive.

Residual connections retain a direct route for information and gradients through a deep model. They add vectors rather than discarding the previous representation.

The **layers** setting controls how many of these blocks are stacked. More layers increase capacity, computation and memory use. Bigger is not automatically better on a small dataset.

The final representation goes to [the output head](article://output).
