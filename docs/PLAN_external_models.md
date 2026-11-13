# Plan: chat with standard model formats (GGUF, MLX, OpenAI-compatible servers)

Status: phases 0 and 2 are implemented (backend interface, managed `llama-server`
with detect/download/verify, GGUF in the picker); phase 1's connections UI,
phase 3 (MLX) and phase 4 (export) are not started.

Decisions made: runtime delivery is *detect, then offer a pinned, hash-checked
download* (not bundled); the privacy wording now says the only built-in network
use is that optional download. See `gui/runtimes.py`, `gui/backends/` and
`tools/pin_llama_runtime.py`. The "bundled" slot in the resolver is already in
place, so bundling later is a packaging change.

Findings while implementing: the llama.cpp GitHub API is rate limited for
anonymous callers, so downloads use direct release URLs with a pinned tag and
pinned SHA-256 hashes; current releases include Linux CUDA, Vulkan and ROCm
builds and separate CUDA-runtime archives; `llama-server` enables Jinja chat
templates by default.

## Goal

Let people chat with the models the rest of the ecosystem uses, in the same
app they use to train Trance models, and let them take Trance models out to
those ecosystems. The native CUDA engine stays, and stays focused on what is
unique here: **training** and running Trance's own `.bin` files (it is also the
reference for checking exported models).

## What LM Studio actually runs on

LM Studio is a closed-source app wrapped around two open engines:
**llama.cpp** (GGUF; CPU, CUDA, Vulkan, ROCm, Metal) and **MLX** (Apple
silicon). It exposes them through an OpenAI-compatible local server. So "the
same libraries LM Studio uses" means llama.cpp and MLX, not LM Studio itself.

## Recommended approach

**Run the engines as managed local servers and talk to them over the
OpenAI-compatible HTTP API**, instead of linking them into the app.

- `llama-server` (llama.cpp) for GGUF.
- `mlx_lm.server` (mlx-lm) for MLX models on Apple silicon.
- Any OpenAI-compatible endpoint the user already runs (LM Studio, Ollama,
  llama-server) through one generic client.

Why not in-process bindings (llama-cpp-python, mlx-lm as a library)?
They give finer control (raw logits, KV cache), but every GPU flavor
(CUDA/Vulkan/Metal/ROCm) means a different compile or wheel, versions drift
quickly, and a native crash takes the whole GUI down. A server process is
isolated, ships as prebuilt binaries per platform from the llama.cpp releases,
and matches how this app already works (the CUDA engine is a subprocess).
It also means one wire protocol covers the whole market.

Trade-offs to accept: we manage a process (port, health, shutdown), we must pin
engine versions (llama.cpp moves fast), and per-token control is limited to
what the HTTP API exposes. Both llama.cpp and MLX are MIT-licensed, but
confirm license terms for anything we redistribute.

## Where it plugs in

`gui/engine.py` already has the right shape. `EngineClient` exposes
`generate(model, messages, settings)`, `cancel()`, `release()`, `close()`,
`busy`, and the signals `delta`, `started`, `finished`, `failed`,
`state_changed`, `ready`, `loaded`. A new backend only has to provide the same
surface, so `app.py` barely changes.

## Phases

**0. Backend interface (small).** Extract that surface as a `Backend`
protocol in `gui/backends/`. Wrap the existing client as `TranceBackend`.
Each model gets a kind (`trance`, `gguf`, `mlx`, `remote`) shown as a small
badge in the model picker. Saved chats record the backend kind with the model.

**1. Generic OpenAI-compatible client (highest value per effort).**
Settings gets a Connections section: base URL plus optional key, with presets
for LM Studio (`:1234`), Ollama (`:11434`) and llama-server (`:8080`). List
models from `/v1/models`; stream from `/v1/chat/completions` over SSE using
only the standard library in a worker thread; cancel by closing the stream.
Map temperature, top_p, top_k, max_tokens and seed. The server applies the
model's chat template. This alone opens the whole GGUF/MLX ecosystem with no
binary management.

**2. Managed `llama-server` for local GGUF files.** Detect it on PATH or in
the app data folder; otherwise offer a one-click download of a pinned,
checksummed release matching the GPU (CUDA, Vulkan or Metal). Spawn it on a
free port with the chosen `.gguf`, wait for `/health`, kill it on unload and on
exit. Discover GGUFs in the usual places (LM Studio and Hugging Face caches,
plus Browse) and read their metadata (architecture, context length, quant) for
the picker tooltip. Optional later: download from Hugging Face.

**3. Apple silicon via `mlx_lm.server`.** macOS/arm64 only. Detect an existing
install, or create an app-owned virtual environment with `mlx-lm`. Training is
still CUDA-only, so Mac users get chat, compare and export but not training;
the UI should say so plainly.

**4. Export from Trance.** `tools/export_gguf.py` (Trance `.bin` to GGUF using
the `gguf` Python package) and a Hugging Face safetensors export for MLX and
transformers. Trance is a GPT-2-style model (learned positions, LayerNorm,
GELU), which llama.cpp supports. Known risks, all checked by a verification
harness that compares logits between the native engine and llama.cpp on a
fixed prompt set:
- Tokenizer parity: Trance uses a custom byte-level BPE with its own encoder.
  Merge order and special tokens must reproduce exactly.
- The separate output head with a bias may not map to llama.cpp's GPT-2
  loader, which expects a tied head; check against the pinned version.
- Chat format: embed a Jinja chat template for Trance's `<user>`,
  `<assistant>` and `<eot>` markers so `--jinja` formats conversations right.

**5. Later, separate decisions.** A modern-architecture option in the trainer
(RoPE, RMSNorm, SwiGLU) so exports land on Llama-family formats; bf16 or mixed
precision; fine-tuning imported models (LoRA). These are large and should not
block phases 0 to 4.

## Decisions needed

1. **Runtime delivery:** detect plus one-click download (recommended), bundle
   in releases, or require a user install. Bundling bloats the AppImage and
   multiplies CUDA combinations.
2. **Privacy wording:** the README says the app makes no network requests.
   Remote connections and downloads change that. Proposal: local-only by
   default, and any network use is an explicit user action with clear wording.
3. **Scope of phase 1 first:** ship the generic client alone before any binary
   management, so the ecosystem story lands early.

## Testing

A fake OpenAI-compatible server built on `http.server` covers streaming,
cancel, errors and model listing in CI with no GPU. Export tests use a tiny
random model and compare logits against the native engine within a tolerance.
