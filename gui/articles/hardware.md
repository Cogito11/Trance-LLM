# CUDA, memory and model size

Trance performs the model's numerical work on an NVIDIA GPU. CUDA kernels implement operations such as LayerNorm, GELU, softmax, loss and Adam. cuBLAS handles the large matrix multiplications.

The current implementation stores weights and optimizer values in **FP32**, using four bytes per number. Some matrix multiplies enable TF32 math through cuBLAS; that does not change the storage format to half precision.

## Where memory goes

- Learned weights and gradients.
- Adam's two moment buffers during training.
- Intermediate activations needed by backpropagation.
- Attention matrices, whose size grows with context squared.
- CUDA/cuBLAS working memory and other applications using the GPU.

The Configs page reports weights-only size. A 116 MiB model file does not imply training uses only 116 MiB of GPU memory. Increasing batch size, context, width or layers can substantially increase peak use.

Use a small experiment when exploring unfamiliar settings. A configuration that passes structural validation can still exceed available GPU memory; the builder does not promise that every allowed architecture fits your card.

Trance does not currently expose mixed-precision training or gradient accumulation through its configs. Its numerical tests check gradients and batch behavior, but do not replace validation on held-out text.

Continue with [training](article://training) or [configuration](article://configs).
