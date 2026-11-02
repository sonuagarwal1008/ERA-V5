# Reversible Transformer Experiment: Results and Analysis

## 1. Purpose

This experiment compares a conventional GPT-style Transformer with four reversible architectures. The central question is whether reversible backpropagation can reduce activation memory while retaining language-modelling quality, useful training speed, and numerically reliable reconstruction of activations during backward propagation.

The notebook evaluates:

- Initialization-time inversion accuracy in FP32 and mixed precision.
- Gradient fidelity against ordinary autograd.
- Peak GPU memory and the largest batch that fits a training step.
- Training quality and throughput at a common batch size and token budget.
- Inversion error after training, when numerical drift may have accumulated.

## 2. Experimental setup

| Setting | Value |
|---|---:|
| GPU | NVIDIA Tesla T4 |
| GPU memory reported by notebook | 15.6 GB |
| Precision | FP16 autocast + GradScaler |
| Dataset | WikiText-103 raw |
| Tokenizer | GPT-2 BPE (`tiktoken`) |
| Training tokens per run | 50 million |
| Validation tokens | 247,289 |
| Vocabulary size | 50,257 |
| Hidden dimension | 256 |
| Transformer layers | 9 |
| Attention heads | 8 |
| FFN expansion | 4x |
| Sequence length | 256 |
| Parameters per architecture | 20.02M |
| Fixed comparison batch | 32 sequences |
| Tokens per optimizer step | 8,192 |
| Optimizer | AdamW |
| Peak learning rate | 1e-3 |

The dataset loader successfully used WikiText-103 raw; the Tiny Shakespeare fallback was not used. Each fixed-batch run processed approximately 50M tokens over 6,104 optimizer steps.

## 3. Architectures compared

- **Standard:** conventional residual block, `x' = x + f(x)`, using ordinary autograd and stored activations.
- **Euler coupling:** additive coupling, `a' = a + F(b)` followed by `b' = b + G(a')`.
- **Midpoint:** two-state recurrence, `x_(k+1) = x_(k-1) + f(x_k)`.
- **Verlet:** reversible split update with half-step F updates around a G update.
- **Momentum:** `v' = γv + (1−γ)f(x)` and `x' = x + v'`, with γ = 0.9.

The four reversible models use custom reversible backward logic that reconstructs prior activations and recomputes layer operations rather than retaining all intermediate activations.

## 4. Initialization-time inversion and gradient fidelity

The notebook's initialization diagnostics show that all four reversible implementations reconstruct inputs very accurately in FP32. Mixed precision introduces a larger, but still initially small, inversion error.

| Architecture | Inversion relative error FP32 | Inversion relative error AMP | Gradient cosine FP32 | Gradient relative error FP32 | Gradient cosine AMP | Gradient relative error AMP |
|---|---:|---:|---:|---:|---:|---:|
| Euler | 9.37e-7 | 7.33e-4 | 1.00000000 | 4.46e-7 | 0.99999988 | 3.10e-4 |
| Midpoint | 1.49e-6 | 1.52e-3 | 1.00000000 | 5.57e-7 | 1.00000000 | 4.91e-4 |
| Verlet | 8.32e-7 | 6.74e-4 | 0.99999988 | 3.51e-7 | 1.00000000 | 3.05e-4 |
| Momentum | 1.06e-6 | 9.98e-4 | 1.00000000 | 4.31e-7 | 0.99999994 | 4.55e-4 |

**Interpretation:** At initialization, FP32 inversion is close to floating-point round-off, and the gradient cosine similarities are effectively 1.0. AMP inversion errors are higher (roughly 6.7e-4 to 1.5e-3), but the initial gradient comparisons remain very close. These initialization results alone do not establish that inversion will remain reliable after many training updates.

## 5. Memory profiling and maximum-batch experiment

The notebook profiled one complete training step (forward pass, backward pass, and AdamW update) while increasing batch size.

| Architecture | Largest batch tested successfully | Peak memory at largest successful batch | Activation estimate (MB/sequence) | Relative batch capacity vs standard |
|---|---:|---:|---:|---:|
| Standard | 608 | 15.02 GB | 22.15 | 1.00x |
| Euler | 2,048 (search cap) | 11.37 GB | 4.83 | 3.37x |
| Midpoint | 2,048 (search cap) | 11.38 GB | 4.84 | 3.37x |
| Verlet | 2,048 (search cap) | 13.28 GB | 5.79 | 3.37x |
| Momentum | 2,048 (search cap) | 12.72 GB | 5.52 | 3.37x |

All architectures have an estimated static parameter-plus-optimizer footprint of about 0.257 GB in the memory probe. The standard model ran out of memory at batch 640 and 768/1024 probes; the reversible variants completed at batch 2,048, which is the configured search cap. Therefore, 2,048 is a lower bound on their maximum capacity, not a measured true maximum.

**Interpretation:** The activation estimate per sequence is substantially lower for reversible models: approximately 4.83–5.79 MB versus 22.15 MB for standard. The search demonstrated at least 3.37x the batch capacity within the tested range. Verlet used the most memory among the reversible variants at the largest tested batch; Euler and midpoint used the least.

## 6. Fixed-batch training results

All five models were compared at batch size 32, sequence length 256, 8,192 tokens per step, and a 50M-token budget. This is the fairest direct comparison in the notebook because batch size and token budget are held constant.

| Model | Final train loss | Final validation loss | Validation perplexity | Best validation loss | Δ validation loss vs standard | Throughput (tokens/s) | Training time | Peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 4.2513 | 4.1930 | 66.22 | 4.1530 | 0.0000 | 32,264 | 25.83 min | 2.250 GB |
| Euler | 4.317 | 4.259 | 70.75 | 4.218 | +0.06609 | 28,500 approx. | 29.25 min | 1.570 GB |
| Midpoint | 4.344 | 4.284 | 72.52 | 4.244 | +0.09077 | 28,330 approx. | 29.42 min | 1.562 GB |
| Verlet | 4.269 | 4.212 | 67.48 | 4.170 | +0.01880 | 24,840 approx. | 33.55 min | 1.570 GB |
| Momentum | 4.255 | 4.197 | 66.49 | 4.155 | +0.004036 | 27,770 approx. | 30.01 min | 1.562 GB |

*Throughput and final-loss values above are taken from the notebook's final report/summary. The standard throughput is the reported 32,264 tokens/s. Reversible throughput is lower for every variant; Verlet is slowest.*

### What the training numbers show

1. **Momentum is closest to standard validation quality.** Its final validation loss is 4.1971 in the detailed findings (4.197 in the rounded final-results table), only about 0.004 above the standard result of 4.1930. Its best validation loss is 4.155, close to the standard best of 4.153.
2. **Verlet is the next closest on validation loss.** Its final validation loss is approximately 4.212, around 0.019 worse than standard, but it has the lowest throughput and longest training time among the reversible models.
3. **Euler and midpoint show larger quality gaps.** Their final validation losses are approximately 4.259 and 4.284, respectively. Midpoint has the largest validation-loss gap (+0.09077).
4. **All reversible models are slower at the same batch size.** Their throughput is below the standard model's 32.3k tokens/s. The notebook's results show a compute cost for reversible reconstruction/recomputation.
5. **Peak memory is lower for every reversible model.** At batch 32, standard uses 2.25 GB, while reversible models use about 1.56–1.57 GB—roughly 30% lower peak memory.

## 7. Inversion error after training

This is the key numerical-stability result. At initialization, AMP inversion errors were around 1e-3 or lower. After training, the final report shows substantially larger errors:

| Architecture | AMP inversion error at initialization | Trained inversion error | Approximate increase factor | Trained gradient cosine vs autograd | Non-finite gradient steps |
|---|---:|---:|---:|---:|---:|
| Euler | 7.329e-4 | 0.1562 | 213x | 0.997032 | 0 |
| Midpoint | 1.522e-3 | 0.1062 | 70x | 0.999980 | 1 |
| Verlet | 6.736e-4 | 0.1248 | 185x | 0.999285 | 0 |
| Momentum | 9.979e-4 | 0.04576 | 46x | 0.999931 | 0 |

The increase factors are approximate ratios of trained error to initialization AMP error.

**Interpretation:** The models start with small inversion error but do not preserve that accuracy through training. The final errors range from about 4.6e-2 to 1.56e-1. Momentum has the smallest trained inversion error, but it is still materially larger than its initialization error. Euler has the largest trained inversion error. Midpoint also records one non-finite-gradient step.

The gradient cosine similarities remain high (0.997–1.000), but this should not be interpreted as proof that reconstruction is numerically safe. Cosine similarity measures directional agreement and can remain high even when the reconstructed activations have a sizeable relative error. The inversion error is a separate warning signal.

The notebook's verdict marks every reversible model **UNRELIABLE (inversion drift)**. Under the notebook's reliability criterion, none of the reversible variants passes the trained inversion check.

## 8. Overall comparison

| Objective | Best result in this run | What it means |
|---|---|---|
| Validation quality among reversible models | Momentum | Closest to standard validation loss |
| Lowest trained inversion error | Momentum (0.04576) | Best of the four, but still shows substantial drift |
| Lowest activation estimate per sequence | Euler (4.83 MB) | Most memory-efficient in the probe |
| Highest tested batch capacity | All reversible variants (2,048 cap) | At least 3.37x standard's observed maximum batch |
| Highest reversible throughput | Euler (~28.5k tokens/s) | Fastest reversible variant in fixed-batch training |
| Lowest reversible throughput | Verlet (~24.8k tokens/s) | Highest compute/time cost in this run |
| Numerical reliability after training | None | All reversible variants flagged for inversion drift |

## 9. Conclusions

The experiment demonstrates a clear memory benefit from reversible backpropagation on this model and hardware. At the fixed batch size, reversible variants reduce peak memory from 2.25 GB to approximately 1.56–1.57 GB. In the batch-size search, they reach the 2,048-sequence search cap while the standard model's largest successful batch is 608.

That benefit comes with trade-offs. Every reversible variant trains more slowly than the standard model at the fixed batch size. Validation quality varies: momentum is very close to standard, Verlet is moderately close, and Euler/midpoint have larger gaps. Most importantly, all four reversible variants show substantial post-training inversion drift, despite very small initialization errors. Thus, the current run does not support treating any of these reversible implementations as numerically reliable under the tested FP16 training setup.

**Best overall reversible candidate in this run: Momentum**, because it has the smallest validation-loss gap and the lowest trained inversion error. However, it is not yet reliable enough to be considered a successful reversible implementation because its trained inversion error is still 0.04576 and the notebook flags it as unreliable.

## 10. Important qualifications

- The results are from one Tesla T4 run using FP16 autocast and GradScaler. They do not establish what would happen with FP32 or BF16 training.
- The maximum-batch results for reversible variants are capped at 2,048. The experiment did not find their actual maximum batch size.
- The activation-memory column is an estimate derived from profiling; it is not a direct measurement of activation tensors alone.
- The reversible architectures use different state-update rules and readout choices. Therefore, differences in validation quality are not caused solely by the custom reversible backward pass; architecture and readout differences also matter.
- The notebook's final report contains rounded values in its summary table and more precise values in its findings. This README preserves the reported precision where available and uses approximate labels where the notebook rounds the value.
- The current results show that inversion error grows during/after training, but they do not by themselves isolate the cause to FP16 rounding alone or identify which individual layer contributes most. Layer-wise, precision-controlled diagnostics are needed for that attribution.
