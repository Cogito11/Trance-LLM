"""Project-local history, model discovery, and training configuration checks."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
import tempfile
import uuid

from .runtime import PROJECT
ROOT = PROJECT
STATE = ROOT / "data/gui"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def model_info(path: Path) -> dict:
    path = path.resolve()
    with path.open("rb") as f:
        raw = f.read(56)
    if len(raw) != 56:
        raise ValueError("Incomplete model header")
    magic, version, dtype, vocab, ctx, width, layers, heads, ff, step, optimizer = struct.unpack("<8sII6iQI4x", raw)
    if magic != b"TAIGPT1\0" or version != 2 or dtype != 1:
        raise ValueError("Unsupported model format")
    if not (259 <= vocab <= 1024 and 2 <= ctx <= 1024 and 1 <= width <= 4096 and
            1 <= layers <= 64 and heads > 0 and width % heads == 0 and 1 <= ff <= 16384):
        raise ValueError("Invalid model architecture")
    if not Path(str(path) + ".tok").is_file():
        raise ValueError("The model's .tok file is missing")
    parameters = 2*vocab*width + ctx*width + 2*width + vocab + layers*(4*width*width + 2*width*ff + 9*width + ff)
    if optimizer not in (0, 1) or path.stat().st_size != 56 + parameters*4*(3 if optimizer else 1):
        raise ValueError("Model weights are incomplete or have an invalid size")
    return {"path": str(path), "name": path.name, "parameters": parameters, "context_length": ctx,
            "vocab_size": vocab, "width": width, "layers": layers, "heads": heads, "step": step,
            "optimizer": bool(optimizer), "feed_forward_dim": ff}


def discover_models() -> list[dict]:
    models = []
    for path in sorted((ROOT / "models").glob("*.bin")):
        try:
            models.append(model_info(path))
        except (OSError, ValueError):
            continue
    return models


def new_chat(model: str = "") -> dict:
    return {"id": uuid.uuid4().hex, "title": "New conversation", "model": model, "messages": []}


def completed_context(messages: list[dict]) -> list[dict]:
    """Only complete successful exchanges enter later prompts; partial output stays visible."""
    context = []
    for i in range(0, len(messages)-1, 2):
        user, assistant = messages[i:i+2]
        if (user.get("role") == "user" and assistant.get("role") == "assistant" and
                assistant.get("status") == "complete" and assistant.get("content")):
            context.extend({"role": m["role"], "content": m["content"]} for m in (user, assistant))
    # Native protocol accepts at most 200 historic exchanges plus the current user.
    # Token-aware pruning happens in the engine, using the actual selected tokenizer.
    return context[-400:]


CONFIG_KEYS = {"vocab_size", "context_length", "embedding_dim", "layers", "attention_heads",
               "feed_forward_dim", "batch_size", "training_steps", "eval_interval", "checkpoint_interval",
               "learning_rate", "gradient_clip", "seed", "train_data", "validation_data", "output_model", "resume_model"}


def validate_config(config: dict, *, check_output=True, check_files=True) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError("Unsupported settings: " + ", ".join(sorted(unknown)))
    for key in CONFIG_KEYS - {"train_data", "validation_data", "output_model", "resume_model"}:
        if key not in config:
            raise ValueError(f"Missing setting: {key}")
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if key not in ("learning_rate", "gradient_clip") and (not isinstance(value, int) or not 0 <= value <= 2**31-1):
            raise ValueError(f"{key} must be a nonnegative integer <= 2^31-1")
    if not 259 <= config["vocab_size"] <= 1024 or not 2 <= config["context_length"] <= 1024:
        raise ValueError("Vocabulary must be 259–1024 and context 2–1024")
    if config["attention_heads"] <= 0 or config["embedding_dim"] <= 0 or config["embedding_dim"] % config["attention_heads"]:
        raise ValueError("Attention heads must divide the embedding width")
    if any(config[k] <= 0 for k in ("layers", "feed_forward_dim", "batch_size", "training_steps", "learning_rate", "gradient_clip")):
        raise ValueError("Architecture, batch size, steps, learning rate and gradient clip must be positive")
    for key in ("train_data", "validation_data", "output_model", "resume_model"):
        value = config.get(key, "")
        if not isinstance(value, str) or len(value.encode("utf-8")) >= 1024 or any(c in value for c in '\x00\n\r'):
            raise ValueError(f"Invalid {key} path")
    if not config.get("train_data") or not config.get("output_model"):
        raise ValueError("Training data and output model paths are required")
    for key in ("train_data", "validation_data") if check_files else ():
        for name in filter(None, config.get(key, "").split(",")):
            path = ROOT / name
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing or empty data file: {name}")
    if check_files and config.get("resume_model"):
        info = model_info(ROOT / config["resume_model"])
        if not info["optimizer"]:
            raise ValueError("Resume requires an optimizer checkpoint, not the final inference model")
        for key, meta in (("embedding_dim", "width"), ("layers", "layers"), ("attention_heads", "heads"),
                          ("context_length", "context_length"), ("vocab_size", "vocab_size"), ("feed_forward_dim", "feed_forward_dim")):
            if config[key] != info[meta]:
                raise ValueError(f"Resume checkpoint does not match {key}")
    if check_output and (ROOT / config["output_model"]).exists():
        raise ValueError("Output model already exists. Choose a new output_model to preserve it.")
    return config
