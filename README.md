# Trance LLM 

Trance LLM is an educational, compute-heavy, GPU-accelerated decoder-only Transformer written in CUDA and C. It has a full GPU implementation of the standard LLM operations: embeddings, multi-head causal self-attention, LayerNorm, GELU activations, cross-entropy loss, and Adam optimization. It uses cuBLAS for the heavy matrix multiplication (GEMM) work.

Everything else, the tokenizer, configuration parsing, file I/O, sampling, and the interactive CLI/REPL, runs on the host CPU.

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

### 1. Interactive Console (REPL)
Run without any arguments to open the interactive REPL shell:
```bash
./trance
```

Inside the console (`trance>`), you can run commands like:
- `help`: show available console commands.
- `status`: show details about the currently loaded model.
- `load <path>`: load a model file (defaults to `models/trance1-stem-3b.bin`).
- `chat`: enter interactive multi-turn chat mode (`/exit` returns to the main menu).
- `prompt "MESSAGE"`: send a single conversational prompt.
- `generate "PROMPT"`: generate raw text completions.
- `train <config.json>`: train a model using a JSON config.
- `inspect <model.bin>`: view architecture parameters and the training step count.

### 2. Command Line Interface (CLI)

#### Training
Train a model using a JSON configuration file:
```bash
./trance train --config configs/trance1_stem_3b.json
```

#### Generation
Generate text completions from a prompt:
```bash
./trance generate --model models/trance1-stem-3b.bin --prompt "Explain a proton." --max-new-tokens 80 --temperature 0.8
```

#### Interactive Chat
Launch directly into interactive chat mode with a model:
```bash
./trance chat --model models/trance1-stem-3b.bin --temperature 0.7 --top-k 40 --top-p 0.9
```

#### Model Evaluation
Calculate validation loss on a dataset:
```bash
./trance evaluate --model models/trance1-stem-3b.bin --data data/trance1/validation.txt
```

#### Inspect Model Metadata
Display the parameter count, architecture configuration, and training steps:
```bash
./trance inspect --model models/trance1-stem-3b.bin
```

---

## Configuration File Format

Training parameters are set with a standard JSON file (for example `configs/trance1_stem_3b.json`):

```json
{
  "vocab_size": 257,
  "context_length": 32,
  "embedding_dim": 48,
  "layers": 2,
  "attention_heads": 4,
  "feed_forward_dim": 192,
  "batch_size": 2,
  "training_steps": 100,
  "eval_interval": 20,
  "checkpoint_interval": 50,
  "learning_rate": 0.002,
  "gradient_clip": 1.0,
  "seed": 42,
  "train_data": "data/train.txt",
  "validation_data": "data/valid.txt",
  "output_model": "models/trance1-stem-3b.bin"
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
