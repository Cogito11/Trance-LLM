# The output head and probabilities

After the Transformer blocks, Trance applies a final LayerNorm. A learned linear **output head** then maps each position's vector to one score for every vocabulary token.

These scores are called **logits**. They are not probabilities yet. A softmax converts them into a distribution that sums to one.

For a toy vocabulary, the model might assign 0.6 probability to one token, 0.3 to another and 0.1 to a third. That distribution expresses the model's learned prediction; it is not a confidence rating for the truth of an entire answer.

## Training versus generating

During training, the correct next token is known from the example text. Cross-entropy penalizes predictions that give too little probability to that token. Backpropagation computes how the weights should change.

During generation, the answer is not known. Trance uses [sampling settings](article://sampling) to choose a token from the distribution, appends it to the context, and makes another prediction.

Trance's token embedding table and output projection are separate learned tensors. The vocabulary size affects both their parameter counts.

A reply ends when generation reaches its token limit, a special token boundary or a cancellation request. Producing grammatical text does not mean the model has checked the answer against an external source.
