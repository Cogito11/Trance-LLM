# Checkpoints and resuming

A saved model contains architecture information and learned weights. An **optimizer checkpoint** also contains Adam's moment buffers, which are needed to continue the optimization state.

Keep the matching tokenizer beside every model or checkpoint. The naming convention is MODEL.bin and MODEL.bin.tok; scheduled checkpoints include their step number in the filename.

## Stop and resume

**Stop & save checkpoint** requests a stop at a safe step boundary. The native trainer writes an interrupted optimizer checkpoint. The GUI's **Resume checkpoint** action prepares a config with the remaining step budget; it does not begin training until Start training is clicked.

A model saved only for final inference cannot serve as an optimizer checkpoint. The architecture of the checkpoint must match the resume config. The GUI checks both conditions.

The existing format does not serialize the training sampling RNG state. Resume restores weights and Adam state, but is not guaranteed to reproduce the exact trajectory of an uninterrupted run.

## Which file should you use?

Use validation trends and held-out generated responses to choose among checkpoints. A longer-trained checkpoint can be worse if the model overfits. Keep original files while comparing; the GUI does not replace a final model that already exists.

Closing the GUI is different from stopping training: the detached supervisor keeps the native trainer running. See [background processes](article://processes).
