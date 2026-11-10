# Trance LLM 

Trance LLM is an educational, compute-heavy, GPU-accelerated decoder-only Transformer written in CUDA and C. It has a full GPU implementation of the standard LLM operations: embeddings, multi-head causal self-attention, LayerNorm, GELU activations, cross-entropy loss, and Adam optimization. It uses cuBLAS for the heavy matrix multiplication (GEMM) work.

Everything else, the tokenizer, configuration parsing, file I/O, sampling, and the interactive CLI/REPL, runs on the host CPU.

The [Trance 1.2 training record](TRANCE_1_2.md) documents the completed stages incorporated into the model, including data sources, training settings and measured results.

---

## Key Features

- **CUDA & cuBLAS Accelerated Architecture**: The core tensor operations (linear projections, attention GEMMs, feed-forward layers) use cuBLAS SGEMM and strided-batched GEMM calls instead of naive CUDA kernels.
- **Batched Training & Forward Pass**: A single forward/backward pass processes a full training batch on-device, which keeps host-GPU synchronization overhead low.
- **Custom BPE Tokenizer**: Built-in byte-pair encoding (BPE) training and serialization (`.tok` files).
- **Interactive REPL & CLI Modes**: An interactive shell (`trance>`), multi-turn interactive chat mode, text generation, evaluation, model inspection, and unit testing.
- **Few Dependencies**: A lightweight C/CUDA implementation that only needs `nvcc`, the standard C library, the CUDA Toolkit, and cuBLAS.

---

## Build Instructions

### Prerequisites
- An NVIDIA GPU with CUDA support.
- The NVIDIA CUDA Toolkit installed (the `nvcc` compiler and the `cuBLAS` library).

### Building

To compile Trance LLM, run:

```bash
nvcc -O3 -o trance trance.cu -lcublas
```

---

## Verification & Testing

Before training or generating text, check that all the CUDA kernels and matrix operations pass numerical correctness and gradient tests on your GPU:

```bash
./trance test
```

This runs a set of numerical tests: gradient checks (finite-difference comparison), causal masking checks, loss calculation checks, batch-consistency checks, and serialization checks.

---

## Usage

### Desktop GUI

Trance LLM provides streaming chat, persistent local conversations, sequential model comparison, and a training dashboard with checkpointed stopping and resume. Setup and usage are described in [gui/README.md](gui/README.md).

```bash
./tools/setup_gui.sh
./trance-gui
```

Rebuild `trance` from the current source first if your binary predates the desktop engine protocol. Opening the GUI does not start training.

The portable Linux AppImage bundles Python, Qt, the native engine and cuBLAS. It uses your existing project folder for models, configs and data. See [packaging/README.md](packaging/README.md) for launch and build instructions.

Rebuild and verify the AppImage with `./tools/build_release.sh`. Add `--compile-engine` to compile the CUDA engine into a separate build directory first. The sidebar groups Chat, Train and Tools. Temporary chats remain in memory; Chat history expands a scrollable list and remembers its state. Right-click a saved conversation to rename or delete it.

### 1. Interactive Console (REPL)
Run without any arguments to open the interactive REPL shell:
```bash
./trance
```

Inside the console (`trance>`), you can run commands like:
- `help`: show available console commands.
- `status`: show details about the currently loaded model.
- `load <path>`: load a model file.
- `chat`: enter interactive multi-turn chat mode (`/exit` returns to the main menu).
- `prompt "MESSAGE"`: send a single conversational prompt.
- `generate "PROMPT"`: generate raw text completions.
- `train <config.json>`: train a model using a JSON config.
- `inspect <model.bin>`: view architecture parameters and the training step count.

### 2. Command Line Interface (CLI)

#### Training
Train a model using a JSON configuration file:
```bash
./trance train --config configs/trance1.2_30m.json
```

#### Generation
Generate text completions from a prompt:
```bash
./trance generate --model models/trance1.1_21m.bin --prompt "Explain a proton." --max-new-tokens 80 --temperature 0.8
```

#### Interactive Chat
Launch directly into interactive chat mode with a model:
```bash
./trance chat --model models/trance1.1_21m.bin --temperature 0.7 --top-k 40 --top-p 0.9
```

#### Model Evaluation
Calculate validation loss on a dataset:
```bash
./trance evaluate --model models/trance1.1_21m.bin --data data/trance12/validation.txt
```

#### Inspect Model Metadata
Display the parameter count, architecture configuration, and training steps:
```bash
./trance inspect --model models/trance1.1_21m.bin
```

---

## Configuration File Format

Training parameters are set with a standard JSON file (for example `configs/trance1.2_30m.json`):

```json
{
  "vocab_size": 1024,
  "context_length": 1024,
  "embedding_dim": 448,
  "layers": 12,
  "attention_heads": 8,
  "feed_forward_dim": 1792,
  "batch_size": 4,
  "training_steps": 140000,
  "eval_interval": 1000,
  "checkpoint_interval": 10000,
  "learning_rate": 0.0002,
  "gradient_clip": 1.0,
  "seed": 4004,
  "train_data": "data/trance12/train.txt",
  "validation_data": "data/trance12/validation.txt",
  "output_model": "models/trance1.2_30m.bin"
}
```

---

## Architecture Details

| Parameter | Description |
|---|---|
| **Model Type** | Decoder-only Transformer |
| **Precision** | FP32 (single precision) |
| **Tokenizer** | Custom byte-pair encoding (BPE) |
| **Position Embeddings** | Learned positional embeddings |
| **Normalisation** | LayerNorm (pre-LN style) |
| **Activation** | GELU |
| **Optimizer** | Adam with gradient clipping |

---

## New here?

If you don't have a background in machine learning and want to understand what this code is actually doing, start with [`LLMS_EXPLAINED.md`](LLMS_EXPLAINED.md). It walks through how LLMs and Transformers work using plain language and no assumed math background. `TRANCE.md` is the technical companion, written for someone reading the source code.
