# Session 10 Notebook — Output Interpretation

This README explains what each code cell in `Session 10(1).ipynb` does and how to interpret its saved output. It is intended to be read alongside the notebook; the code does not need to be copied into the README.

## Prepare the Shakespeare dataset

**Saved output:**

- Total characters: 1,115,394  
- Vocabulary size: 65 unique characters  
- Training tokens: 1,003,854  
- Validation tokens: 111,540

## Cell 4 — Trace tensor shapes through a Transformer training step

**Key saved outputs:**

- Input IDs: shape `(3, 10)`  
- Token embeddings: `(3, 10, 128)`  
- Positional embeddings: `(10, 128)`  
- Combined query/key/value projection: `(3, 10, 384)`  
- Attention scores: `(3, 4, 10, 10)`  
- Joined attention heads: `(3, 10, 128)`  
- MLP hidden representation: `(3, 10, 512)`  
- Final logits: `(3, 10, 65)`  
- Difference between manually traced logits and nanoGPT logits: about `3.58e-07`  
- Flattened logits: `(30, 65)`  
- Cross-entropy loss: scalar

**Interpretation:** There are three sequences of ten characters. Each character is represented by 128 features. The combined QKV projection contains three groups of 128 features. Attention is calculated independently across four heads; each head has 32 features and compares all ten query positions with ten key positions. The MLP expands the representation from 128 to 512 features and then returns it to 128\.

The final logits provide 65 character scores for each of the 30 token positions. They are flattened into 30 classification rows to match the 30 target character IDs. The very small difference from nanoGPT's own forward pass shows that the manual trace closely matches the reference implementation.

The backward output shows that gradients have the same shapes as their parameters. The token-embedding gradient receives contributions from both the input embedding and the tied language-model output head. The AdamW state tensors `exp_avg` and `exp_avg_sq` match the shape of the parameter they track.

## Cell 5 — Check one gradient using finite differences

**Saved output:**

- FP64 backpropagation gradient: approximately `-0.000073460783`  
- FP64 at epsilon `1e-3`: relative error approximately `4.10e-10`  
- FP32 at epsilon `1e-1`: relative error approximately 2.63%  
- FP32 at epsilon `1e-2`: relative error approximately 29.8%  
- FP32 at smaller epsilons: numerical gradient becomes zero and relative error reaches 100%

**Interpretation:** The FP64 numerical gradient agrees extremely closely with backpropagation at a suitable perturbation size. Making the perturbation smaller eventually makes the estimate less reliable because the loss difference is affected by floating-point rounding.

In FP32, the small changes in loss become too small to distinguish reliably, so the numerical gradient is reported as zero even though backpropagation returns a nonzero value.

## Cell 6 — Compare correct and incorrect gradient accumulation

**Saved output:**

- Short microbatches contain 384 tokens in total.  
- The long microbatch contains 1,024 tokens.  
- Total: 1,408 tokens.  
- Short tokens are about 27% of all tokens.  
- Under equal averaging of the four microbatch means, the three short microbatches collectively receive 75% of the microbatch-level weighting.  
- Correct gradient relative difference from the full-batch reference: `1.15e-07`  
- Incorrect gradient relative difference: `4.83e-01` (about 48.3%)  
- Incorrect gradient cosine similarity with reference: `0.927438`  
- Final training loss: correct `2.1169`; incorrect `2.2664`  
- Final validation loss: correct `2.1712`; incorrect `2.3220`  
- Final validation loss gap, incorrect minus correct: `+0.1508`

**Interpretation:** The token-weighted gradient is almost identical to the padded full-batch reference. Equal averaging of microbatch means gives short microbatches too much influence because each microbatch gets an equal share regardless of its number of valid targets.

In this run, the correctly weighted training method also achieves lower final training and validation losses. The validation gap of about 0.1508 supports the correct method for this particular setup; it is not a guaranteed improvement for every run.

**Important detail:** Equal averaging of microbatch mean losses is equivalent to a global token mean when every microbatch has the same number of valid targets. It is not generally equivalent when target counts differ.

**Numerical display issue:** The correct gradient's saved cosine similarity is `1.000045`, slightly greater than 1\. An exact cosine similarity cannot exceed 1; this is a floating-point artefact. Display it using FP64 and clamp the result to the interval from \-1 to 1\.

## Cell 7 — First attempt to find a gradient-norm lead

**Saved output:** The detector reports that it found no clear case and identifies step 190 as the largest relative gradient-norm jump.

**Interpretation:** The experiment did not find a step that satisfies all of the specified criteria. The training loss is calculated on changing random microbatches, so different batches can change both the loss and gradient norm. This makes it difficult to decide whether a norm spike precedes a loss change.

The norm is recorded before gradient clipping, while the optimizer receives clipped gradients. Therefore, a large pre-clipping norm does not necessarily cause a proportionally large update.

**Conclusion:** This attempt does not establish that a gradient-norm spike leads to a later loss increase. It motivates using a fixed evaluation batch and more carefully aligned measurements.

## Cell 8 — Second attempt using a fixed probe batch

**Saved output:**

- The run reports a candidate at learning rate `0.01`, step `339`.  
- Step 339 gradient norm: `1.053`  
- Probe loss at step 338: `2.7219`  
- Probe loss at step 339: `2.7128`  
- Probe loss at step 340: `2.7237`  
- Lag correlations: lag 0 `+0.919`, lag 1 `+0.581`, lag 2 `+0.133`, lag 3 `+0.152`, lag 4 `+0.286`, lag 5 `+0.384`

**Interpretation:** Using the same probe batch removes variation due to changing evaluation examples. The gradient norm at step 339 is larger than nearby norms, and the probe loss rises slightly in some subsequent steps.

However, the detector condition has a flaw: it checks whether the signed loss change is below a positive threshold rather than checking its absolute magnitude. A large loss decrease can therefore be accepted as “the loss has not moved much.” There is also a timing mismatch: the loss stored at step `t` is measured before update `t`, so the change between entries `t-1` and `t` is associated with the preceding update, not update `t`.

The lag-0 correlation is stronger than the lag-1 correlation. These correlations do not prove that gradient norm reliably predicts a future increase in loss.

## Cell 9 — CPU MFU estimate, throughput, and profiling

**Saved output:**

- CPU reported as Intel Xeon CPU at 2.20 GHz  
- Estimated CPU peak: `70.4 GFLOP/s`  
- Large matrix multiplication: `73.5 GFLOP/s`, or 104% of the estimated peak  
- Original training-loop estimate: `32.63 GFLOP/s`, labelled `46.3% MFU`  
- Fixed batch 16x64: `42.44 GFLOP/s`, labelled `60.3% MFU`  
- Bigger batch 64x64: `47.64 GFLOP/s`, labelled `67.7% MFU`  
- Wider model, embedding dimension 384: `53.12 GFLOP/s`, labelled `75.5% MFU`  
- Profiler reports `aten::mm` as about `58.98%` of self CPU time.  
- Attention kernels, GELU, AdamW, layer normalization, and other tensor operations also appear in the profiling table.

**Interpretation:** The measured throughput increases for the larger batch and wider model in this run, suggesting these configurations use the available CPU more efficiently in terms of estimated arithmetic throughput.

But the measured matrix multiplication exceeds the estimated theoretical peak. This indicates that the peak estimate is not a reliable upper bound under the assumptions used. The estimate assumes a number of physical cores based on half the logical CPU count, uses one reported CPU frequency, and assumes a specific number of FLOPs per cycle. Those assumptions may not match the available CPU or runtime.

Therefore, the calculated MFU percentages should not be treated as validated hardware-utilization measurements. The matrix multiplication result is an empirical reference for that operation, not theoretical CPU peak throughput.

The profiler indicates that matrix multiplication is the largest contributor to self CPU time in this sampled step. This is useful for identifying where optimization effort might matter, but the table represents a single profiled training step.

**Conclusion:** Use this cell to report measured throughput, approximate model GFLOP/s, relative workload performance, and profiler hotspots. Do not present the calculated MFU percentages as confirmed utilization values.

## Cell 10 — Compare FP32, BF16, and FP8 E4M3 representations of 0.1

**Purpose:** Manually encodes the decimal value 0.1 into FP32, BF16, and FP8 E4M3 formats, then compares the resulting bit patterns against PyTorch. It also tests whether a small addition changes the BF16 representation of 0.1.

**Saved output:**

| Format | Stored value | Absolute error | Relative error |
| :---- | ----: | ----: | ----: |
| FP32 | `0.10000000149` | `1.49e-09` | about `1.49e-06%` |
| BF16 | `0.10009765625` | `9.77e-05` | about `0.0977%` |
| FP8 E4M3FN | `0.1015625` | `1.56e-03` | about `1.5625%` |

The manual bit patterns match PyTorch's stored bits for all three tested formats.

The output also reports:

`bf16(0.1) + 1e-5 == bf16(0.1)? True`

**Interpretation:** FP32 represents 0.1 most accurately among these formats. BF16 uses the same exponent width as FP32 but has fewer mantissa bits, so its rounding error is larger. FP8 E4M3FN has even fewer mantissa bits and therefore represents this value less precisely.

The BF16 addition result means the small update is lost when rounded to the representable BF16 value at this magnitude. This illustrates why low-precision arithmetic needs to be considered carefully in numerical computation.

The displayed FP32 relative error rounds to `+0.0000%`, but the actual error is nonzero. More output precision would make that clearer.

**Limitation:** The manual encoder is a teaching example for a positive, nonzero normal value. It is not a complete encoder for zero, negative values, subnormal values, overflow, infinity, or NaN. One number also cannot determine which precision is best for a complete model's training.

---

