# How training learns

Training repeatedly asks the model to predict text from examples and changes its weights to reduce prediction error.

## A training step

1. Sample a batch of conversation windows.
2. Run a **forward pass** to produce logits.
3. Compute cross-entropy on the target assistant tokens.
4. Run **backpropagation** to compute gradients.
5. Clip the overall gradient magnitude if necessary.
6. Use **Adam** to update the parameters.

The learning rate controls update size. Batch size controls how many sequences contribute to each step. Neither setting directly measures how much the model understands.

## Assistant-only loss

Trance's examples contain user and assistant turns. User text provides context, but the loss is masked to focus learning on assistant targets and boundaries. Trance normalizes by the relevant token count within each window, then averages over the batch.

The training loss printed at a step describes that sampled batch. Different examples can have different difficulty, so individual training-loss values can jump around.

Trance currently uses constant-rate Adam with fixed internal moment settings. Configs do not expose a learning-rate schedule, dropout or working weight decay. Do not add unsupported JSON keys expecting those features to take effect.

Learning can eventually become [overfitting](article://losses), so monitor held-out validation loss and actual generated answers, not just the training number.
