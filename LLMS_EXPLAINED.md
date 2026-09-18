# LLMs and Transformers, explained simply

This document is for anyone who wants to understand what Trance is actually
doing, without needing a machine learning background. If you already know
this material, `TRANCE.md` is the version written for people reading the
source code.

We'll build up the ideas one at a time: what a language model is, what a
token is, what a Transformer does, and how training actually teaches it
anything. No math background is assumed. A few equations show up, but only
where a picture in words would be worse.

---

## 1. What is a language model?

A language model is a program that predicts the next piece of text, given
the text that came before it. That's the whole job. If you type "The
capital of France is", a good language model predicts that the next word
is probably "Paris."

That single skill, predicting what comes next, turns out to be enough to
build something that can hold a conversation, answer questions, and write
code. If a model gets good enough at guessing the next word over and over,
it ends up having to learn grammar, facts about the world, how arguments
are structured, and how conversations flow, because all of that helps it
guess better.

Trance is a small language model. It's not trained on the whole internet
and it won't compete with a huge commercial model, but it uses the same
core ideas as those larger models. Understanding Trance means you
understand the foundation that everything else is built on.

---

## 2. Text isn't fed in as text

Computers work with numbers, not letters, so before any of this can
happen, text needs to be converted into numbers. The pieces it gets broken
into are called **tokens**, and the process of breaking text into tokens is
called **tokenization**.

A token might be a whole word, part of a word, or even a single character,
depending on how the tokenizer was built. For example, the word
"unbelievable" might become three tokens: "un," "believ," and "able."
Common short words like "the" or "is" are often their own single token.

Trance uses a method called **byte-pair encoding**, or BPE. Here's the idea
in plain terms:

1. Start by treating every single byte (essentially, every character) as
   its own token.
2. Look through a big pile of example text and find the pair of tokens
   that appears next to each other most often.
3. Merge that pair into one new, bigger token.
4. Repeat, over and over, each time finding the next most common pair and
   merging it.

After doing this thousands of times, you end up with a vocabulary that
contains single letters, common word fragments, and entire common words,
because the pairs that showed up together constantly got merged into
single units. This is useful because it lets a small, fixed vocabulary
represent literally any text, even words the tokenizer never saw during
training. If a word is unfamiliar, the tokenizer just falls back to
smaller, more familiar pieces, down to individual bytes if it has to.

Once text is tokenized, every token gets assigned a number, its **token
ID**. From this point on, the model only ever works with lists of numbers.
A sentence becomes a list of IDs like `[72, 105, 33]`, and the model's job
is to predict the next number in that list.

---

## 3. Turning numbers into meaning: embeddings

A token ID like `72` doesn't mean anything by itself. It's just an index,
like a page number. So the model's first real step is to convert each
token ID into a list of numbers called a **vector**, using a lookup table
called an **embedding table**.

Think of an embedding as coordinates that place each token somewhere in a
huge, imaginary space. Tokens with similar meanings or similar usage
patterns end up near each other in this space. During training, the model
adjusts these coordinates so that useful patterns emerge naturally. Nobody
tells it "cat" and "dog" should be close together; it works that out on
its own because those words tend to appear in similar contexts.

The model also needs to know *where* in the sentence each token sits,
since "the dog bit the man" and "the man bit the dog" use the same words in
a different order with a very different meaning. To capture this, Trance
adds a second vector to each token, called a **positional embedding**,
which encodes the token's position (first, second, third, and so on) in
the sequence. The token's meaning vector and its position vector are added
together, so the model has both pieces of information available at once.

---

## 4. The Transformer: how the model "thinks"

The Transformer is the architecture, meaning the overall design, that
almost every modern language model is built on. Its key idea is a
mechanism called **attention**, which lets the model look back at earlier
words in a sentence and decide which ones matter most for predicting the
next word.

### Why "attention"?

Imagine reading the sentence: "The trophy didn't fit in the suitcase
because it was too big." What does "it" refer to, the trophy or the
suitcase? You instinctively look back at the earlier words and weigh them:
"trophy" feels more relevant here than "suitcase." That weighing process,
deciding how much each earlier word should influence your understanding of
the current word, is exactly what attention does, except the model does it
mathematically, for every word, every time.

Concretely, for each token, the model asks a version of the question "given
everything I've seen so far, which earlier tokens should I pay the most
attention to right now?" It computes a score for every earlier token,
turns those scores into a set of weights that add up to one hundred
percent, and then blends the earlier tokens together according to those
weights. That blended result becomes part of the token's updated
representation.

This is called **causal** self-attention because a token is only allowed
to look backward, at itself and everything before it, never forward at
tokens that come later. This matters because a language model has to work
purely from what came before; it isn't allowed to cheat by peeking at the
answer.

### Multiple heads

Trance doesn't do this attention calculation just once per layer. It does
it several times in parallel, using what are called **attention heads**.
Each head can end up specializing in a different kind of relationship:
one might learn to track which pronoun refers to which noun, another might
track grammatical structure, and another might track something else
entirely. The results from all the heads are combined afterward. This
gives the model several independent "angles" to look at the same text
from, rather than being limited to just one.

### Feed-forward layers

After the attention step, each token's representation passes through a
small neural network on its own, called a **feed-forward layer**. If
attention is where tokens gather information from each other, the
feed-forward layer is where the model processes that information further,
for each token individually. It works by expanding each token's
representation into a larger space, applying a simple nonlinear function
called **GELU** (this is what lets the model represent more than just
straight-line relationships between numbers), and then shrinking it back
down.

### Stacking layers

One round of attention followed by one feed-forward step is called a
**layer**, or sometimes a **block**. Trance stacks several of these layers
on top of each other. Each layer refines the token representations a bit
further, building on the work of the layer before it. Early layers tend to
pick up on simple local patterns, and later layers combine those into
more abstract ones, similar to how you might first notice individual
brushstrokes in a painting before your eye assembles them into a whole
scene.

### Keeping the numbers well-behaved: LayerNorm

Neural networks can be numerically unstable during training. Values can
grow too large or shrink too close to zero as they pass through many
layers, which makes learning slow or unreliable. **LayerNorm** is a step
that rescales the numbers at various points in the network to keep them in
a reasonable, consistent range. Trance applies it before each attention
step and before each feed-forward step (this ordering is called "pre-norm"
and tends to train more reliably than applying it afterward).

### Turning the final numbers into a prediction

After passing through every layer, each token has a final vector that
represents everything the model has learned about it in context. To turn
that into an actual prediction, the model runs one more linear
transformation that produces a score for every single token in the
vocabulary. These scores are called **logits**. A high logit means the
model thinks that token is a strong candidate for what comes next; a low
one means it thinks that token is unlikely.

---

## 5. Learning from mistakes: training

At the start, a model's parameters (all of its embedding tables and layer
weights) are set to small random numbers. This means an untrained model's
predictions are pure noise. Training is the process of adjusting all of
those numbers so the model's predictions get better.

### Comparing the prediction to reality

Training works by showing the model real text where we already know the
correct next token, and checking how good its prediction was. The logits
described above get converted into probabilities (using a function called
**softmax**, which squashes any set of numbers into something that behaves
like a probability distribution: everything is between zero and one, and
it all adds up to one hundred percent). We then measure how far the
model's predicted probabilities were from the actual correct answer. This
measurement is called the **loss**, and a common way to calculate it is
called **cross-entropy loss**: it comes out low when the model gave a high
probability to the correct token, and high when the model was confident
about the wrong one.

### Backpropagation: figuring out what to change

Once we know how wrong the prediction was, we need to know how to adjust
every single parameter in the model, and there can be millions of them, to
make it a little less wrong next time. This is done with an algorithm
called **backpropagation**. It works backward through the network, layer
by layer, calculating exactly how much each individual parameter
contributed to the error. This produces a **gradient** for every
parameter: a number that says which direction to nudge that parameter, and
by roughly how much, to reduce the loss.

### Adam: actually making the update

With the gradients calculated, the model's parameters get updated. Trance
uses an update rule called **Adam**, which is a popular and effective
method for this. Instead of just subtracting the raw gradient from each
parameter, Adam keeps a running memory of recent gradients for each
parameter and uses that memory to take smarter, more stable steps. This
generally trains faster and more reliably than a plain, naive update
would.

This whole cycle, predict, measure the loss, backpropagate, update with
Adam, is called one **training step**. A model might go through many
thousands of these steps, each time on a different chunk of training text,
gradually getting better at prediction across the board.

### Why only train on the assistant's replies

Trance is trained on conversations that look like this:

```
<user>
What's the capital of France?

<assistant>
The capital of France is Paris.

<eot>
```

If we let the model spend its learning effort trying to predict the
*user's* messages too, it wastes time learning to predict things it will
never actually need to generate itself. Instead, Trance uses a **loss
mask** that only scores the model's predictions during the assistant's
turn. This focuses all of its learning on the one skill we actually
want: producing a good reply, given a conversation so far.

---

## 6. Generating text

Once a model is trained, generating text works one token at a time:

1. Feed in the tokens seen so far (the conversation history).
2. Get the logits for what token should come next.
3. Turn those logits into a probability for every possible token.
4. Pick one token based on those probabilities, this is called
   **sampling**.
5. Add that chosen token to the sequence, and repeat the whole process to
   generate the next one.

If the model always picked the single most likely token, its output would
be repetitive and boring. Instead, generation usually adds a bit of
controlled randomness, using a few common settings:

- **Temperature** controls how "confident" or "random" the choices are. A
  low temperature makes the model stick close to its top choice most of
  the time. A high temperature spreads its choices out more, which can
  make output more varied but also more likely to go off the rails.
- **Top-k** limits the model to picking only from its k most likely
  tokens, ignoring the long tail of unlikely ones entirely.
- **Top-p** (also called nucleus sampling) is similar, but instead of a
  fixed count of tokens, it keeps adding the most likely tokens until
  their combined probability crosses some threshold p, then samples from
  just that group.

Generation keeps going, one token at a time, until it produces a special
stop token (in Trance's case, the end-of-text marker or a role marker) or
hits a maximum length limit.

---

## 7. Putting it all together

Here's the whole pipeline from start to finish, in order:

1. Raw text gets broken into tokens using the trained BPE vocabulary.
2. Each token ID is converted into a vector using the embedding table, and
   a positional vector is added so the model knows where in the sequence
   it sits.
3. Those vectors pass through a stack of Transformer layers. Each layer
   lets tokens gather relevant information from earlier tokens using
   causal self-attention, then processes each token further with a
   feed-forward network.
4. The final vectors are converted into logits, a score for every token in
   the vocabulary.
5. During training, those logits are compared against the real next token
   to compute a loss, and backpropagation plus Adam adjust every parameter
   to reduce that loss.
6. During generation, those logits are turned into probabilities and
   sampled from, one token at a time, to produce new text.

Every large, well-known language model you may have heard of uses this
same basic recipe: tokenize, embed, run through Transformer layers,
predict the next token, and repeat. What differs between a small model
like Trance and a massive commercial one is mostly scale: more parameters,
more layers, more training data, and much more computing power. The
underlying ideas are the same ones described in this document.

---

## Where to look in the code

If you want to trace these ideas into the actual implementation, `TRANCE.md`
maps each concept here to the specific functions and files in `trance.cu`:
tokenization and the BPE trie, the attention and feed-forward kernels, the
training loop and loss masking, and the sampling logic used during
generation.
