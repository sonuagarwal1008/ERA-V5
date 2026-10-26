# Language Model Loss Harness — Assignment Guide



For this experiment: using small randomly initialized causal Transformer and two short example documents. Its goal is to make the language-model training loss pipeline observable and check common implementation mistakes. The model is intentionally small enough to inspect, but the GPT-2 tokenizer still has a vocabulary of 50,257 tokens.


---

## 1. What problem does this solve?

A language model reads a sequence of token IDs and produces a prediction for the next token at every position. The model's hidden states are converted to **logits**—one score for every vocabulary item. Cross-entropy compares those scores with the correct target token and produces a loss.

The basic operation is:

```python
hidden = model(tokens)
logits = output_head(hidden)
loss = cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    tokens[:, 1:].reshape(-1),
)
```

The two slices are essential:

- `logits[:, :-1]` uses predictions at positions `0` through `T-2`.
- `tokens[:, 1:]` uses target tokens at positions `1` through `T-1`.
- Therefore, the model is trained to predict the *next* token, not to copy the token it has already seen.

This  checks seven details in Part 1 and adds a second prediction head in Part 2.

### configuration

The  run reports:

- Tokenizer: GPT-2 byte-pair encoding (`tiktoken`)
- Vocabulary size: **50,257**
- Model width (`D`): **128** for the main experiments
- Transformer layers: **2**
- Attention heads: **4**
- Batch size (`B`): **2** for the padded two-document example
- Sequence length (`T`): **12** after padding
- Device: **CPU** (`torch` output says CPU-only; no CUDA peak-memory measurement was available)

Example documents:

- Document A: `The cat sat on the mat and then the cat slept.`
- Document B: `A robot learns to predict the next token from context.`

The GPT-2 tokenizer divides text into subword tokens. A decoded token may contain a leading space; that space is part of the token and is not an error.

---

## 2. Setup and special-token handling

The sets:

```python
V = enc.n_vocab
PAD_ID = enc.eot_token
```



**Why this matters:** if padding positions contribute to cross-entropy, the model can lower its loss by learning to predict repeated padding tokens. The reported loss then looks better even though the model has not become better at predicting real text.



---

## 3. Part 1.1 — Print and explain tensor shapes

###  output

```text
Tensor shapes (B=batch, T=sequence positions, D=model width, V=vocabulary):
  tokens             (2, 12)              B sequences x T token IDs
  hidden             (2, 12, 128)         B x T x D contextual vectors
  logits             (2, 12, 50257)       B x T x V scores for vocabulary tokens
  logits[:, :-1]     (2, 11, 50257)       B x (T-1) x V; last position has no next target
  tokens[:, 1:]      (2, 11)              B x (T-1); target is shifted one position forward
  flat logits        (22, 50257)          N x V; N=B*(T-1) prediction rows
  flat targets       (22,)                N target IDs, one per prediction row
Masked next-token loss: 10.853086
```

### How to interpret it

| Tensor | Meaning |
|---|---|
| `tokens (2, 12)` | Two sequences, each padded to 12 token positions. Each value is a token ID. |
| `hidden (2, 12, 128)` | Each input position has a 128-number contextual representation from the Transformer. |
| `logits (2, 12, 50257)` | For every position, the output head scores all 50,257 vocabulary tokens. |
| `logits[:, :-1]` | Removes the last prediction row because there is no next token in the input window to use as its target. |
| `tokens[:, 1:]` | Drops the first input token so targets are shifted one position to the right. |
| `flat logits (22, 50257)` | The 2 batches × 11 prediction positions are flattened into 22 prediction rows. Each row still contains scores for the full vocabulary. |
| `flat targets (22,)` | One correct target ID for each prediction row. Padding targets are later ignored by the masked loss. |

**Why does the vocabulary dimension come last?** Cross-entropy expects one row of class scores per prediction. Here each row has `V=50,257` class scores.

**Is the loss of 10.8531 a bug?** Not by itself. It is a randomly initialized model's masked loss on this tiny example. The perplexity check in Section 1.5 is a more explicit sanity check for whether the initial loss is in the expected range.

---

## 4. Part 1.2 — Verify the shift with token strings

### Recorded output

```text
Decoded token alignment (strings, not IDs):
position  input              target t+1         target t+2
       0  The                 cat                sat
       1   cat                sat                on
       2   sat                on                 the
       3   on                 the                mat
       4   the                mat                and
       5   mat                and                then
       6   and                then               the
       7   then               the                cat
       8   the                cat                slept
       9   cat                slept             .
```


Each row should move forward by one token for `t+1`, and by two tokens for `t+2`.

For example:

- Input `The` → target ` cat`
- Input ` cat` → target ` sat`
- Input ` sat` → target ` on`

The leading spaces shown by the tokenizer are part of its decoded token pieces. The important property is the sequence order, not whether the token piece looks like a whole natural-language word.

**What bug does this catch?** If the targets were not shifted, the model would be asked to predict `The` from the position that already represents `The`. That is a copy-the-input objective, not next-token prediction. Printing strings makes this mistake much easier to spot than printing long lists of integer IDs.

The table is printed for the first document. The final token has no valid `t+1` target beyond the sequence, and the last two positions do not both have valid `t+2` targets.

---

## 5. Part 1.3 — Mask padding

###  output

```text
Targets without mask: 22
Targets with mask:    21
Loss without mask:    10.841351
Loss with mask:       10.853086
```

### 

The batch contains 2 × 11 = **22 shifted target positions**. One of those positions is padding, so after masking, only **21** targets contribute to the mean loss.

 uses `IGNORE_INDEX` for padding targets. PyTorch's cross-entropy ignores those positions when computing the loss.

The loss changes from `10.841351` to `10.853086`. This is expected and is not a failure. The exp does **not** require the masked loss to be lower. It requires the number of contributing targets to decrease correctly.

Why can the loss increase after masking? The unmasked mean included a padding prediction that happened to be relatively easy for the randomly initialized model. Removing that easy target increases the average loss over the remaining real targets.


---

## 6. Part 1.4 — Pack two documents and mask their boundary

Packing places multiple documents into one sequence so that less space is wasted on padding. But if the documents are unrelated, the last token of Document A should not be trained to predict the first token of Document B.

###  output

```text
Seam pair: '.' -> 'A'
Before mask: loss=10.842791, count=22
After mask:  loss=10.845577, count=21
```

### 

The boundary is the pair:

```text
'.' -> 'A'
```

The period closes Document A; `A` begins Document B. Without a boundary mask, the model is trained as if the next token after the period were `A`. That creates a false training example.


- Before masking: 22 prediction positions contribute.
- After masking: 21 prediction positions contribute.
- Mean loss changes from `10.842791` to `10.845577`.

Again, the direction of the loss change is not the criterion. It rises slightly because the excluded seam prediction happened to have a slightly lower loss than the average of the retained positions.

**Important scope note:** this experiment masks the loss for the cross-document target. It does not, by itself, create a fully isolated attention context for each document. In a strict packed-document training implementation, you may also need attention masks or position handling that prevent tokens in one document from attending to tokens in a previous document, depending on the intended training semantics.

**Success criterion:** the boundary pair is printed correctly, and the contributing loss count falls by one.

---

## 7. Part 1.5 — Perplexity sanity check

Perplexity is calculated from mean cross-entropy loss:

\[
\mathrm{perplexity} = e^{\mathrm{loss}}
\]

For a model whose predictions are close to uniform across `V` vocabulary choices, the expected loss is approximately:

\[
\mathrm{loss} \approx \ln(V)
\]

and perplexity is approximately `V`.

###  output

```text
V=50,257; ln(V)=10.8249; loss=10.8921; perplexity=53,751.7; perplexity/V=1.070
```

### 

- Vocabulary size: `50,257`
- Uniform-prediction reference loss: `ln(50,257) ≈ 10.8249`
- Observed untrained loss: `10.8921`
- Observed perplexity: `53,751.7`
- Perplexity divided by vocabulary size: `1.070`

The untrained loss is close to `ln(V)`, and the perplexity is about 1.07 times the vocabulary size. This is a reasonable sanity-check result for this randomly initialized model.

It does **not** need to match the vocabulary size exactly. Random initialization, model architecture, and the sample targets can move it above or below that reference.

**What would be suspicious?** A dramatically smaller initial loss—for example, one that is far below `ln(V)` before meaningful training—could indicate target leakage, an accidental unshifted target, or a loss calculation that is unintentionally scoring easy special tokens. Investigate rather than assuming the model is already good.

The exp caps the exponent input at 30 to avoid numerical overflow in unusual cases. For the recorded loss, this cap does not affect the result.

**Do not compare perplexity across different tokenizers without care.** A tokenizer that splits a word into more pieces creates a different prediction task from one that represents it with fewer pieces.

---

## 8. Part 1.6 — Tied versus untied output-head parameters

###  output

```text
Untied parameters: 13,328,128
Tied parameters:   6,895,232
Saved:             6,432,896 (expected V*D=6,432,896)
Parameter-count check: PASS
```

### 

The input embedding table maps each token ID to a `D`-dimensional vector. Its size is `V × D`.

With an **untied output head**, the model has a separate output matrix mapping the hidden vector back to vocabulary scores. With **weight tying**, the output head shares the input embedding matrix instead of storing another independent matrix of the same shape.

For this configuration:

\[
V \times D = 50{,}257 \times 128 = 6{,}432{,}896
\]

The parameter reduction is exactly **6,432,896**, matching the  output.

The total parameter count drops from 13,328,128 to 6,895,232. This is a reduction in unique stored parameters, not a guarantee that model quality or training speed improves in every setup.

**Success criterion:** the measured saving equals `V * D`, and the parameter-count check passes.

---

## 9. Part 1.7 — Ordinary versus chunked cross-entropy memory

Ordinary cross-entropy materializes logits for all prediction positions at once. Chunked cross-entropy processes smaller groups of positions to reduce the size of the temporary logits tensor.

The memory experiment uses a smaller model (`D=64`, one Transformer layer) and a sample sequence of shape `(2, 128)`. The chunk size is 16.

###  output

```text
CPU forward-only run: analytic logits-memory estimate (RSS is allocator-dependent):
  ordinary logits tensor: 48.70 MiB
  chunked logits tensor:  6.13 MiB
  analytic ratio:         7.94x
  losses: ordinary=10.855759, chunked=10.855759, abs diff=0.000e+00
```

### 

- Ordinary logits estimate: **48.70 MiB**
- Chunked logits estimate: **6.13 MiB**
- Estimated reduction factor: **7.94×**
- Ordinary loss: `10.855759`
- Chunked loss: `10.855759`
- Absolute loss difference: `0.000e+00`

The ordinary and chunked losses match exactly at the printed precision. That is the key correctness check: chunking changes how the computation is performed, not the mathematical loss being calculated.

The  ratio is an **analytic logits-tensor memory estimate**, not a measured reduction in total process memory. The run is on CPU  says that process resident-set-size measurements can be affected by memory allocators. It is therefore incorrect to describe `7.94×` as a measured end-to-end RAM or training-memory improvement.

The ratio also depends on the chunk size and number of prediction positions. A chunk size of 16 limits each temporary logits block to about 16 positions at a time, whereas the ordinary calculation holds the logits for all prediction positions in one tensor. There is overhead and other model memory beyond logits, so total memory does not necessarily fall by the same ratio.

**Success criterion:** the losses agree and the estimated chunked logits tensor is smaller. For actual peak allocated GPU memory, run the benchmark on a CUDA-enabled runtime and report the measured GPU values separately. Also note that training with autograd can retain additional tensors, so a forward-only measurement is not a full training-memory benchmark.

---

## 10. Part 2 — Add a second head for token `t+2`

The first output head predicts the next token (`t+1`). The second head predicts the token two positions ahead (`t+2`). we calculates the losses separately and sums them:

```python
loss1, _ = horizon_loss(..., horizon=1)
loss2, _ = horizon_loss(..., horizon=2)
total = loss1 + loss2
```

The model is trained on the same two short documents concatenated together and repeated eight times. Training runs for 80 optimizer updates, printing every 10 steps.

###  output

```text
step=  0 | loss t+1=10.7201 | loss t+2=10.8964 | sum=21.6164
step= 10 | loss t+1=9.5277 | loss t+2=9.5873 | sum=19.1150
step= 20 | loss t+1=8.6713 | loss t+2=8.6819 | sum=17.3533
step= 30 | loss t+1=7.9632 | loss t+2=7.9850 | sum=15.9481
step= 40 | loss t+1=7.3085 | loss t+2=7.3347 | sum=14.6431
step= 50 | loss t+1=6.6275 | loss t+2=6.6604 | sum=13.2879
step= 60 | loss t+1=5.9083 | loss t+2=5.9321 | sum=11.8404
step= 70 | loss t+1=5.1470 | loss t+2=5.1674 | sum=10.3144
step= 80 | loss t+1=4.3652 | loss t+2=4.3827 | sum=8.7479
One-head parameters=13,328,128; two-head parameters=19,761,024
both losses should generally fall on this tiny repeated training set. Evaluate on held-out text before making claims about which horizon is harder.
```

### the training curve

Both losses decrease over the updates:

| Step | `t+1` loss | `t+2` loss | Sum |
|---:|---:|---:|---:|
| 0 | 10.7201 | 10.8964 | 21.6164 |
| 10 | 9.5277 | 9.5873 | 19.1150 |
| 20 | 8.6713 | 8.6819 | 17.3533 |
| 30 | 7.9632 | 7.9850 | 15.9481 |
| 40 | 7.3085 | 7.3347 | 14.6431 |
| 50 | 6.6275 | 6.6604 | 13.2879 |
| 60 | 5.9083 | 5.9321 | 11.8404 |
| 70 | 5.1470 | 5.1674 | 10.3144 |
| 80 | 4.3652 | 4.3827 | 8.7479 |

The `t+2` loss starts higher than the `t+1` loss and remains slightly higher in this particular run. Both losses decrease as the model fits the repeated training example. That is consistent with the model learning the small training sequence.

However, **it does not prove that predicting two tokens ahead is always harder**. This is a tiny repeated training set, and there is no held-out evaluation here. The relative loss can change with data, model initialization, learning rate, and training duration. The meaningful observation is that both objectives are learnable on this example and both improve during training.

### Why does the combined loss start around 21.6?

The code defines the total loss as the **sum** of the two losses. At step 0:

\[
10.7201 + 10.8964 = 21.6165
\]

The tiny difference from the displayed `21.6164` comes from rounding of the printed components. The total is not the average of the two heads. If you instead used `(loss1 + loss2) / 2`, the numeric scale would be about half, and the gradient scale would also differ unless other settings compensated.

### Parameter count

```text
One-head parameters=13,328,128
Two-head parameters=19,761,024
```

The second head adds one more output projection of size `D × V` (plus any model-specific head parameters). In this configuration the increase is 6,432,896 parameters, which matches `128 × 50,257`. The input/output embedding tying used in the single-head comparison is a separate configuration; do not assume every model variant has the same tying setup unless the code explicitly shares those weights.

### Important limitation

This run repeats the same concatenated sequence eight times. It is useful for checking code and observing whether the losses decrease, but it is not evidence that the model generalizes to unseen text. For a stronger experiment, split documents into training and held-out sets, avoid leaking adjacent parts of the same text across the split, and report held-out losses for both horizons.

---

## 11. Overall results at a glance

| Requirement | result
|---|---|---|
| Print tensor shapes | `tokens (2,12)`, `hidden (2,12,128)`, `logits (2,12,50257)` | Shape flow is consistent with batch, sequence, model width, and vocabulary dimensions. |
| Verify target shift | Strings show `The → cat`, `cat → sat` | The displayed target is one token ahead of the input. |
| Mask padding | Count changes `22 → 21` | One padding target is excluded. |
| Mask packed-document seam | `'.' → 'A'`; count `22 → 21` | The false cross-document prediction is excluded from the loss. |
| Perplexity sanity check | `53,751.7`, with `V=50,257` | Near the vocabulary-size reference for an untrained model. |
| Tied vs untied head | Saves `6,432,896 = V×D` parameters | Weight tying removes one independent vocabulary projection matrix. |
| Ordinary vs chunked loss | Same loss; analytic logits estimate `7.94×` smaller | Chunking preserves the loss and reduces the estimated temporary logits size; this is not a measured CPU peak-RAM ratio. |
| Second prediction head | Both losses fall from about 10.7–10.9 to about 4.4 | Both objectives learn the tiny repeated sample; generalization is not tested. |

---


