# How a reply is generated

The GUI sends conversation text and settings to Trance's native inference engine. It loads the chosen model and its tokenizer, frames the turns, and predicts one token at a time.

## The generation controls

- **Temperature** rescales logits. Lower values favor likely tokens; higher values add variety. Changing it cannot teach the model new knowledge.
- **Top K** restricts candidates to the most likely K tokens. Zero disables this restriction.
- **Top P** keeps leading candidates up to a cumulative probability threshold. It works together with Top K.
- **Max tokens** caps the generated reply length. Tokens are not words.
- **Seed** controls random sampling. Keep the model, prompt, settings and seed identical when comparing repeatability.

## Context has a limit

The GUI supplies completed prior exchanges. The native engine counts tokens using the selected tokenizer and drops the oldest whole exchanges when necessary to reserve room for the reply. A latest message that still cannot fit is rejected.

An incomplete, cancelled or failed assistant exchange is excluded from later context, although its text can remain visible in saved history.

Trance has no KV cache: it recomputes a full forward pass for each generated token. Loading a model happens on demand; opening a chat does not itself start generation.

See [tokens](article://tokens) and [the output head](article://output) for the underlying pieces.
