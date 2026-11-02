# Adam by Hand, Warmup, Schedules and LR-vs-Width

Assignment notebook for **ERA V5, Session 11: Optimizers and Learning-Rate Schedules**. Everything is a small char-level GPT trained on TinyShakespeare.

Notebook: `adam_by_hand.ipynb`. All outputs below come from a single run on a **Tesla T4** (the whole notebook is dominated by the Part 5 sweep, about 46 minutes).

---

## 1\. What is in the notebook

| Cell | Assignment item | Question answered |
| :---- | :---- | :---- |
| 1 | Setup | Data, minimal GPT, eval, cosine and WSD schedules |
| 2 | Part 1 | Does Adam computed by hand match `torch.optim.Adam`? |
| 3 | Part 2 | What does disabling bias correction do over the first 20 steps, and when does the difference stop mattering? |
| 3b | Part 2 (extended) | Theory vs measurement, dependence on beta2, a high-LR stress test |
| 4 | Part 3 | What is the update-to-weight ratio per layer, and what does warmup do to it? |
| 5 | Part 4 | Cosine vs WSD, both stopped at step 200 of 300: which model to keep? |
| 6 | Part 5 | LR sweep at widths 256/512/1024, and a prediction for width 4096 |

### Setup (Cell 1\)

- **Data:** TinyShakespeare, 1,115,394 characters, 90/10 train/val split, character-level vocabulary.  
- **Model:** pre-LayerNorm GPT with causal attention via `scaled_dot_product_attention`. Heads scale as `width // 64`, 4 layers, block size 128, batch size 32, init `N(0, 0.02)` everywhere.  
- **Evaluation:** a *fixed* validation set (generator seeded to 1234), so every reported loss is comparable.  
- **Schedules:** linear warmup then cosine to zero, and WSD (warmup, stable, linear decay over the last 20%).  
- **Fairness conventions:** ablations share identical initial weights and an identical data stream, so the only thing that changes is the factor under test.

### Running it

1. Open in Colab or Jupyter with a GPU runtime. A CPU fallback config is built in (smaller widths, depth and steps) but will not reproduce the numbers below.  
2. Run cells top to bottom. Cell 1 must run first. Cell 6 overwrites the globals `STEPS` and `WARMUP`, so if you re-run Cell 5 afterwards, run its first lines again.  
3. Requirements: `torch`, `numpy`, `matplotlib`, `requests` (the dataset is downloaded from GitHub).  
4. The Cell 6 sweep is 24 runs per width and takes about 46 minutes on a T4. Set `RUN_VERIFY_4096` or `RUN_MUP` to `False` to skip those blocks.

---

## 2\. Results and insights

### Part 1: Adam by hand matches PyTorch exactly

Five gradients `[1.0, 0.5, -0.2, 0.3, 0.1]`, one weight starting at 1.0, `lr=0.1`, `beta=(0.9, 0.999)`.

| t | g | m | v | m̂ | v̂ | step | w |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | 1.00 | 0.100000 | 0.00100000 | 1.000000 | 1.000000 | 0.100000 | 0.90000000 |
| 2 | 0.50 | 0.140000 | 0.00124900 | 0.736842 | 0.624812 | 0.093218 | 0.80678204 |
| 3 | \-0.20 | 0.106000 | 0.00128775 | 0.391144 | 0.429680 | 0.059671 | 0.74711094 |
| 4 | 0.30 | 0.125400 | 0.00137646 | 0.364641 | 0.344632 | 0.062114 | 0.68499725 |
| 5 | 0.10 | 0.122860 | 0.00138509 | 0.300017 | 0.277572 | 0.056945 | 0.62805191 |

The final weights agree with `torch.optim.Adam` (float64) to 9 decimal places at all five steps.

**Insights**

- **Step 1 moves exactly `lr`** (0.1) although the gradient is 1.0. With bias correction, `m̂/√v̂ = g/|g| = ±1` at the first step, so the gradient fixes the direction and the learning rate fixes the distance.  
- **A negative gradient (step 3\) does not reverse the weight.** Momentum still carries `m` positive (0.106), so the weight keeps moving the same way, but the step shrinks from 0.093 to 0.060. This is the "cancel what alternates" behaviour of the moving average.  
- **Steps shrink over time** (0.100, 0.093, 0.060, 0.062, 0.057) as the gradients stop agreeing in sign.

### Part 2: Bias correction on vs off

**Theory.** Both averages start at zero and are under-estimated. With a constant gradient, the uncorrected step is the corrected step multiplied by

`r(t) = (1 − β1^t) / √(1 − β2^t)`

| Quantity | Value |
| :---- | :---- |
| Inflation at step 1 | 3.16× |
| Peak inflation | **6.57× at step 12** |
| Inflation at step 20 | about 6.2× |
| Inflation at step 1000 (= 1/(1−β2)) | about 1.26× |

**When does the difference stop mattering?** For large t, `r(t) ≈ 1/√(1 − β2^t)`, so the step at which it falls within a tolerance `d` of 1 is `t* = ln(1 − (1+d)^−2) / ln β2`:

| Tolerance | Steps |
| :---- | :---- |
| within 10% | **about 1,750** |
| within 5% | about 2,374 |
| within 1% | about 3,925 |

**Answer to the assignment: roughly 1,750 steps (10%) to 3,900 steps (1%).** The scale is set by `β2` (about 1,000 steps of memory), not by the model.

**Measured over the first 20 steps** (`lr=1e-3`, same init and same batches): the loss gap between ON and OFF is 1.4% at step 20\. That window is far too short to show the difference "stopping"; it only shows the effect starting.

**Extended experiments (Cell 3b)**

- **(A) Formula vs real model.** Measured OFF/ON ratio overlaps r(t) (max deviation 0.029). Note that this is an *algebraic identity* rather than an independent test: the shadow uncorrected update is computed from the same `m` and `v`, so it equals `r(t)` times the corrected update exactly (apart from ε). What it does establish is that `r(t)` depends only on `t`, `β1`, `β2`, not on the model or the gradients.  
- **(B) Dependence on β2.** Peak height scales as `1/√(1−β2)`: 6.57× at 0.999 and 20.7× at 0.9999, both peaking near step 12\. (For `β2 = 0.9`, r(t) never exceeds 1, so the "peak" row is an artifact of the 300-step window.) Raising β2 makes the missing correction, and therefore warmup, matter more.  
- **(C)** Fade curve with the 1,750, 2,374 and 3,925 markers.  
- **(D) Per-coordinate spread.** The largest single-coordinate step relative to the RMS step starts at 1× (every coordinate moves exactly `lr` at step 1\) and settles near 10×. This is a property of the corrected update on real, heavy-tailed gradients, not of the bias term.  
- **(E) High-LR stress test** (`lr=1.5e-2`, no gradient clipping, 80 steps). The OFF run **spikes to a loss of about 28 around step 6** and recovers by about step 40, while the ON run stays smooth (peak about 4.4). Final losses are close (OFF 3.32 vs ON 3.16), so the failure is a **transient blow-up that heals**, not a NaN divergence. The final-loss comparison alone hides this, so look at the plot or at the peak loss.

### Part 3: Update-to-weight ratio and warmup

Tracked per tensor: `RMS(Δw) / RMS(w)`, for the token embedding, the first attention QKV weight, the last MLP weight, the final LayerNorm scale and the output head. Peak LR 3e-3, 30 warmup steps, 300 steps total, AdamW with weight decay 0.1.

**Insights**

- **The first update is enormous in relative terms.** At step 0 the lr-normalised ratio is about 50, which is `1 / 0.02`: Adam moves every weight by about `lr` while the weights have RMS 0.02.   
- **The raw ratio peaks at roughly 3–4e-2 around step 20–25** for the matrix weights, Peak LR 3e-3 is hot for this model. It is also visible as a loss spike near step 8 in Part 4\.  
- **The raw ratio mostly follows the LR schedule** (ramp up, then cosine down), which is why the notebook also plots ratio ÷ lr.  
- **The lr-normalised ratio** drops from about 50 and levels off at about 4–6 after roughly **step 50–60** (the last MLP weight is slower and sits lower at about 2–5). It then drifts slowly upward. Warmup is nominally over at step 30, but the ratio keeps adjusting for about another 20–30 steps.  
- **LayerNorm's scale sits about 30× *lower*, not higher.** Its weights are about 1 while matmul weights are about 0.02, so for a similar absolute step the *relative* update is far smaller. This is why norm scales are excluded from weight decay.  
- The knee statistic printed by the cell (steps 276–299) is not a good measure of "settling": it compares against a drifting end-of-run value while the learning rate tends to zero. The plot is the more reliable evidence here.

### Part 4: Cosine vs WSD, both stopped at step 200 of 300

Peak LR was tuned for both over `{1e-3, 3e-3, 1e-2}` for the 200-step objective; both picked **3e-3**.

| Model | LR at step 200 | Val loss at step 200 |
| :---- | :---- | :---- |
| Cosine (300-step schedule) | 9.06e-4 | **2.0742** |
| WSD (300-step schedule) | 3.00e-3 | 2.0847 |
| WSD checkpoint \+ 30-step linear decay | decayed to 0 | **2.0643** |

**Insights**

- **As-is at step 200, cosine is the better model** (2.0742 vs 2.0847). It has already annealed to about 30% of peak LR, while WSD is still at full LR (about 3.3× higher) and carries more noise.  
- **Decaying the WSD checkpoint reverses the order** (2.0643), a gain of about 0.02 over the raw WSD checkpoint. **Run to completion, WSD reaches a lower training loss than cosine** (about 1.75 vs 1.83 over the last steps, read off the plot, noisy).  
- **Decision: keep the WSD checkpoint.** The loss differences here are small (about 0.01) and come from one seed. WSD can stop, branch and continue, while a cosine run stopped early leaves a half-annealed model.

### Part 5: LR sweep vs width, and the width-4096 prediction

200 steps per run, 20 warmup steps, cosine, gradient clipping at 1.0, 3 seeds per point, 8 learning rates from 1e-4 to 1.28e-2 (factor 2 apart).

| Width | Grid best LR | Interpolated best LR | Best val loss (mean) |
| :---- | :---- | :---- | :---- |
| 256 | 1.6e-3 | **1.84e-3** | 2.2508 |
| 512 | 8e-4 | **6.28e-4** | 2.2748 |
| 1024 | 8e-4 | **5.85e-4** | 2.2171 |

**Insights**

- **The optimal LR falls as width grows**, The drop is mostly 256 → 512 (about 2.9×); 512 → 1024 barely moves (about 7%).  
    
- **At small LRs wider models are better** (at 1e-4: 2.556, 2.425, 2.375 for widths 256/512/1024), but they hit the instability cliff earlier. For width 1024, going from 8e-4 to 1.6e-3 costs 0.26 nats. This is the mechanism behind a shrinking optimal LR.  
    
- **Run-to-run noise is real.** Width 256 at 3.2e-3 has a std of 0.107 (one seed went bad), and several points near the optimum have stds of 0.02–0.05, comparable to the differences between widths at their best LR. Seeds vary initialisation only; the data stream is fixed.  
    
- **Prediction for width 4096: LR ≈ 1.5e-4** (log-log fit gives 1.57e-4, slope −0.83, R² \= 0.80).  
    
  - A second estimate, taking the 1024 optimum and applying the lecture's 1/width rule, gives 5.85e-4 / 4 ≈ 1.46e-4. The two agree, and value for 4096 is 1.9e-4.  
  - **Confidence is low to moderate.** The fit has three points, the leave-one-out error predicting 1024 from 256 and 512 is 63%, and the grid is only resolved to a factor of 2\. A realistic range is roughly a factor of 2–3 either side (about 0.6e-4 to 4e-4). The cell's own "within 2×: HIGH" line describes the flatness of the loss basin, which is not the same as certainty about where the optimum lies.


- **Sanity check at width 4096** (reduced to 2 layers, block 64, batch up to 16, 150 steps, one seed):


| LR | Val loss |
| :---- | :---- |
| 7.9e-5 (×0.5) | 2.4055 |
| 1.6e-4 (×1) | **2.3479** |
| 3.1e-4 (×2) | 2.4065 |


  The predicted LR wins and the two neighbours are nearly symmetric, which supports it as near-optimal *for that reduced configuration*. These losses are not comparable with the sweep (different depth, context, batch and length), so this is a consistency check, not a proof.


- **muP-style test (exploratory).** Scaling hidden-layer LR by `256/width` gives a best base LR of 1.6e-3 at both 512 and 1024, so the base value is steady. This is not a full muP implementation: the output head is not scaled, LayerNorm and bias parameters are scaled with the hidden group, and initialisation does not scale with width. It is also a single seed on a factor-2 grid. Note too that the standard-parameterisation optimum itself barely moved from 512 to 1024 here, so the comparison is suggestive rather than decisive.

---

## 3\. Summary of findings

| \# | Finding | Evidence |
| :---- | :---- | :---- |
| 1 | Hand-computed Adam equals PyTorch Adam | Agreement to 9 dp over 5 steps |
| 2 | Step 1 of Adam is exactly `lr` in magnitude | 0.100000 with g \= 1.0 |
| 3 | Without bias correction early steps are up to 6.57× too large | r(t), peak at step 12 |
| 4 | The correction stops mattering after about 1,750 steps (10%) to 3,900 (1%) | `ln(1−(1+d)^−2)/ln β2` |
| 5 | Without correction at high LR the loss spikes (about 28 vs about 4.4) and then recovers | Cell 3b (E) plot |
| 6 | Early updates are huge relative to weight size (ratio/lr ≈ 50\) and normalise after about 50–60 steps | Cell 4 plots |
| 7 | At step 200 as-is cosine beats WSD (2.0742 vs 2.0847); after a short decay WSD wins (2.0643) | Cell 5 |
| 8 | Best LR falls with width: 1.84e-3 → 6.3e-4 → 5.9e-4 | Cell 6 sweep |
| 9 | Width-4096 LR ≈ 1.5e-4, low to moderate confidence | Fit, 1/width cross-check, reduced-scale check |

---

## 4\. Limitations and suggested follow-ups

These are known gaps, listed so the results above are read at the right strength.

1. **Part 3 scope.** Only five tensors are logged, not every layer, and there is no no-warmup control run. A direct answer to "when does warmup stop changing the ratio" needs the same run with warmup disabled and the step at which the two per-layer curves agree.  
2. **Part 4 fairness.** The decayed WSD model got 30 extra steps (230 total) against cosine's 200, and the decay branch restarts AdamW with fresh optimizer state (the checkpoint stores weights only). Add a cosine schedule built deliberately for 200 steps, save and restore the optimizer state, tune on a split separate from the one you report on, and use 3–5 seeds.  
3. **Weight decay.** AdamW decay of 0.1 is applied to all parameters, including LayerNorm scales, biases and embeddings.  
4. **Part 5 resolution.** The grid is 2× spaced and the minima come from a symmetric log-parabola, which is biased by the sharp cliff above the optimum. Add 3–4 grid points between 4e-4 and 1.6e-3.  
5. **Part 5 verification.** Run the same reduced configuration at widths 256, 512 and 1024 so the width-4096 check is like for like.  
6. **muP.** Implement it properly (hidden and output LR proportional to 1/width, vector parameters constant, fan-in-scaled init) before drawing a conclusion about transfer.  
7. **Printed commentary.** Several analysis lines in the notebook are static strings that do not depend on the computed values. When reusing the notebook, derive each conclusion from the numbers.  
8. **Scale.** 300-step runs of a small model are short. The qualitative effects here (warmup transient, schedule behaviour, LR drift with width) are real, but absolute loss values and the exact optimal LRs will not transfer to long runs.

---

## 5\. Key takeaways

- The **gradient gives direction; the optimizer rule chooses distance**. Adam's first step is `lr·sign(g)`.  
- **Bias correction removes a large, β2-controlled early inflation** (up to 6.57×) that otherwise lasts thousands of steps. Warmup is the practical companion to it.  
- **Monitor update-to-weight ratio per layer**; it exposes both the early spike and layers (like LayerNorm) that live on a different scale.  
- **WSD gives you a model you can stop, branch and anneal**; cosine needs the run length in advance.  
- **Tune both sides of any comparison, use several seeds, and distrust single-run differences of about 0.01.**  
- **Learning rate depends on width** under standard parameterisation, so sweeping small and transferring naively overshoots; anchor on the widest measured point and apply the 1/width rule, with wide error bars.