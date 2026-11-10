# Tokens and the BPE tokenizer

Computers need numbers rather than raw text. A **tokenizer** converts text into token IDs; decoding converts those IDs back into bytes and text.

A token may represent a whole word, a word fragment, punctuation or a single byte. Token counts are not the same as word counts. With Trance's small vocabulary, many words take several tokens.

## Trance's base vocabulary

- IDs 0–255 represent every possible byte.
- ID 256 is the conversation boundary, EOT.
- IDs 257 and 258 mark user and assistant turns.
- IDs 259 and above hold learned byte-pair merges.

**Byte-pair encoding (BPE)** starts with bytes and repeatedly merges common adjacent pairs. Trance learns these merges from a bounded sample of its training text, then encodes the corpus. Its encoder uses longest-known-token matching through a trie. Special role markers remain separate tokens.

A UTF-8 character can contain several bytes, so one byte does not always mean one character. Keeping raw bytes available lets Trance represent unfamiliar text without an unknown-word token.

## Why the tokenizer file matters

Every model has a matching **.tok** companion. The same token ID can mean different text under another tokenizer. Keep both files together and never substitute another model's tokenizer.

Trance currently supports vocabularies from 259 to 1,024 tokens. See [configuration settings](article://configs) for the consequences of that limit.
