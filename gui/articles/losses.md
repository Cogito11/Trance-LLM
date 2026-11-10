# Loss, validation and overfitting

**Training loss** measures prediction error on examples used to update the model. **Validation loss** measures prediction error on held-out examples that do not update the weights.

Lower loss is usually desirable when comparing the same model/tokenizer and evaluation data. Loss is not a percentage correct, and it does not directly score factual accuracy or conversational quality.

## Read the two curves together

- Training and validation both decrease: learning is helping on held-out examples.
- Training decreases while validation rises over a sustained interval: the model is likely overfitting.
- Both stay high: consider data quality, architecture, learning rate and whether enough learning has happened.

A single spike is not enough to diagnose overfitting. Look for a trend across multiple evaluations.

## Trance's evaluation

Trance evaluates up to 64 deterministically spaced token windows, with the same assistant-target mask used for training. This is a limited sample of the validation corpus, not an exhaustive test. The positions are reused across evaluations within a run.

Different learned tokenizers divide text differently, so token-level losses from separately trained models are not strictly comparable. Use fixed prompts and inspect responses as well.

## Picking a checkpoint

The last checkpoint is not always the best. Compare checkpoints around the lowest validation loss, then test them on prompts excluded from training. Keep some final test material separate from repeated checkpoint selection.

Related: [data splits](article://data) and [checkpoints](article://checkpoints). Further reading: [Google's introduction to overfitting](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting).
