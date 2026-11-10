# Conversation data and held-out splits

Trance reads UTF-8 conversation files with explicit role markers:

    <user>
    Explain what a token is.
    <assistant>
    A token is a piece of text represented by an ID.
    <eot>

The markers are recognized during loading and become reserved token IDs. Arbitrary text inside an answer that resembles these markers can disrupt parsing, so data preparation should protect them.

## Why splits matter

Training examples teach the model. Validation examples help monitor learning and choose checkpoints. A separate test set should be reserved for the final assessment. Duplicated or closely related examples across splits can make evaluation look better than real performance.

Trance samples conversation starts. Its sampling is not weighted simply by file byte size. Repeated records increase their chances of selection, while long conversations can be cropped. A short sampled conversation can spill into later records within the window; the role masks and EOT boundaries remain important.

The prepared Trance 1.2 corpus favors complete, bounded conversations and blends instruction tasks with reviewed human-written dialogue. These filters improve structure; they do not establish that every source answer is true.

More steps revisit a finite corpus more often. Broader, well-filtered data can be more useful than repeatedly fitting the same narrow examples. See [loss and overfitting](article://losses).
