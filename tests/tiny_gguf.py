"""Build a tiny random-weight llama-architecture GGUF for backend tests.

Needs the `gguf` and `numpy` packages (test/dev only; the app itself does not
depend on them). The model is gibberish by design: it exists so the real
llama-server binary has something valid to load and stream from.
"""
from __future__ import annotations

from pathlib import Path

CHAT_TEMPLATE = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                 "{% if add_generation_prompt %}assistant:{% endif %}")


def build(path: Path, *, context=256, embed=64, layers=2, heads=4, ff=128, seed=7) -> Path:
    import numpy as np
    from gguf import GGUFWriter, GGMLQuantizationType, TokenType

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pieces = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
    types = [TokenType.UNKNOWN, TokenType.CONTROL, TokenType.CONTROL] + [TokenType.BYTE] * 256
    for word in ["\u2581", "\u2581the", "\u2581a", "\u2581hello", "\u2581world", "e", "t", "a", "o", "n", "s", "r", "i", "l", "d", "u"]:
        pieces.append(word); types.append(TokenType.NORMAL)
    vocab = len(pieces)
    w = GGUFWriter(str(path), "llama")
    w.add_name("tiny-random-test"); w.add_context_length(context); w.add_embedding_length(embed)
    w.add_block_count(layers); w.add_feed_forward_length(ff); w.add_head_count(heads); w.add_head_count_kv(heads)
    w.add_layer_norm_rms_eps(1e-5); w.add_rope_dimension_count(embed // heads); w.add_vocab_size(vocab)
    w.add_file_type(0)
    w.add_tokenizer_model("llama"); w.add_token_list(pieces); w.add_token_scores([0.0] * vocab); w.add_token_types(types)
    w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_unk_token_id(0)
    w.add_chat_template(CHAT_TEMPLATE)

    def t(name, *shape, ones=False):
        data = np.ones(shape, dtype=np.float32) if ones else (rng.standard_normal(shape) * 0.05).astype(np.float32)
        w.add_tensor(name, data, raw_dtype=GGMLQuantizationType.F32)

    t("token_embd.weight", vocab, embed); t("output_norm.weight", embed, ones=True); t("output.weight", vocab, embed)
    for i in range(layers):
        t(f"blk.{i}.attn_norm.weight", embed, ones=True)
        for n in ("q", "k", "v"): t(f"blk.{i}.attn_{n}.weight", embed, embed)
        t(f"blk.{i}.attn_output.weight", embed, embed); t(f"blk.{i}.ffn_norm.weight", embed, ones=True)
        t(f"blk.{i}.ffn_gate.weight", ff, embed); t(f"blk.{i}.ffn_up.weight", ff, embed); t(f"blk.{i}.ffn_down.weight", embed, ff)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return Path(path)


if __name__ == "__main__":
    import sys
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "tiny.gguf")))
