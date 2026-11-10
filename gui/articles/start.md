# What is a language model?

A language model predicts the next **token**, a small piece of text, from the tokens it has already seen. Repeating that prediction builds a whole reply.

If the prompt is “The capital of France is”, a trained model may assign a high probability to a token representing “Paris”. It is making a learned prediction, not consulting a built-in database of verified facts.

## Where the knowledge lives

Training changes millions of numerical **parameters**, also called weights. Patterns in these weights help the model connect words, follow instructions and produce sentences. The weights are different from the text currently in the conversation: your messages provide temporary context, while training changes the model itself.

Trance is a small **decoder-only Transformer** built in CUDA. It uses learned token and position embeddings, causal attention, feed-forward networks and an output head. Larger language models use related ideas, but their scale and training data can differ greatly.

## What to expect

Fluent output can still be wrong. A low loss does not establish factual accuracy, reliable reasoning or safety. Trance is an educational model; inspect its answers instead of treating confident wording as evidence.

Start with [tokens and BPE](article://tokens), then [embeddings](article://embeddings) and [attention](article://attention). You do not need advanced mathematics to follow these articles.
