"""Minimal GGUF header reader and model discovery (no third-party packages).

Only the handful of metadata keys the model picker needs are read; reading
stops at the first tokenizer key because tokenizer arrays are large and always
come after the general and architecture keys in files written by llama.cpp.
"""
from __future__ import annotations

import os
import re
import struct
from pathlib import Path

from .storage import ROOT

MAGIC = b"GGUF"
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
              12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 32: "BF16"}
_WANTED = {"general.architecture", "general.name", "general.file_type", "general.size_label"}
_SPLIT_PART = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def _read(f, size: int) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise ValueError("Truncated GGUF file")
    return data


def _string(f, limit: int = 1 << 20) -> str:
    (length,) = struct.unpack("<Q", _read(f, 8))
    if length > limit:
        raise ValueError("Implausible GGUF string length")
    return _read(f, length).decode("utf-8", "replace")


def _value(f, kind: int):
    if kind in _SCALARS:
        fmt = _SCALARS[kind]
        return struct.unpack(fmt, _read(f, struct.calcsize(fmt)))[0]
    if kind == _STRING:
        return _string(f)
    if kind == _ARRAY:
        (inner, count) = struct.unpack("<IQ", _read(f, 12))
        if inner in _SCALARS:
            f.seek(count * struct.calcsize(_SCALARS[inner]), os.SEEK_CUR)
        elif inner == _STRING:
            for _ in range(count):
                _string(f)
        else:
            raise ValueError("Unsupported GGUF array type")
        return None
    raise ValueError(f"Unsupported GGUF value type {kind}")


def read_gguf_info(path: Path) -> dict:
    """Return a model-picker record for a GGUF file; raises ValueError for anything else."""
    path = Path(path).resolve()
    meta = {}
    try:
        with path.open("rb") as f:
            head = _read(f, 24)
            if head[:4] != MAGIC:
                raise ValueError("Not a GGUF file")
            version, _tensors, count = struct.unpack("<IQQ", head[4:])
            if version not in (2, 3) or count > 100_000:
                raise ValueError(f"Unsupported GGUF version {version}")
            for _ in range(count):
                key = _string(f)
                if key.startswith("tokenizer."):
                    break
                (kind,) = struct.unpack("<I", _read(f, 4))
                value = _value(f, kind)
                if key in _WANTED or key.endswith(".context_length"):
                    meta[key] = value
    except (OSError, struct.error) as error:
        raise ValueError(f"Cannot read GGUF file: {error}") from error
    architecture = meta.get("general.architecture") or ""
    if not architecture:
        raise ValueError("The GGUF file does not name an architecture")
    if architecture == "clip":
        raise ValueError("Multimodal projector files are not chat models")
    return {"path": str(path), "name": path.name, "title": meta.get("general.name") or path.stem, "kind": "gguf",
            "architecture": architecture, "context_length": int(meta.get(f"{architecture}.context_length") or 0),
            "quantization": FILE_TYPES.get(meta.get("general.file_type")), "size_label": meta.get("general.size_label"),
            "size_bytes": path.stat().st_size, "parameters": 0}


def default_model_folders() -> list[Path]:
    home = Path.home()
    hub = Path(os.environ.get("HF_HOME", home / ".cache" / "huggingface")) / "hub"
    return [ROOT / "models", home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models", hub]


def discover_gguf(folders: list[Path] | None = None, *, limit: int = 300, max_depth: int = 6) -> list[dict]:
    """Find readable chat-model GGUF files; later parts of split models are skipped."""
    found, seen = [], set()
    for folder in (default_model_folders() if folders is None else folders):
        stack = [(Path(folder), 0)]
        while stack and len(found) < limit:
            directory, depth = stack.pop()
            try:
                entries = sorted(os.scandir(directory), key=lambda e: e.name)
            except OSError:
                continue
            for entry in entries:
                if len(found) >= limit: break
                try:
                    if entry.is_dir() and depth < max_depth:
                        stack.append((Path(entry.path), depth + 1))
                    elif entry.name.lower().endswith(".gguf") and entry.is_file():
                        part = _SPLIT_PART.search(entry.name)
                        real = str(Path(entry.path).resolve())
                        if (part and int(part.group(1)) != 1) or real in seen:
                            continue
                        seen.add(real)
                        found.append(read_gguf_info(Path(entry.path)))
                except (OSError, ValueError):
                    continue
    return sorted(found, key=lambda m: m["title"].lower())
