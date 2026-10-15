# Structured Mathematical Encoding

**Embeddings that store mathematical structure, not just statistical meaning.**

`Session_7_Run.ipynb` is a self-contained experiment that asks a simple question: instead of hoping a neural network *learns* arithmetic, can we design number embeddings so that the mathematical meaning lives **inside the vector itself**?

If `9` is encoded such that the vector contains the genuine meaning of 9, then `9 + 9` should produce the vector of `18`, and `9 * 9` should produce the vector of `81` — as exact arithmetic, not as a learned approximation. The plan is to keep the existing 32 dimensions for language (alphabets/words) and append new dimensions for this numeric structure.

The notebook builds that encoding, compares it against a normal "guessing" transformer, and measures where it wins and where it breaks.

---

## Problem statement

Standard token embeddings treat numbers as arbitrary labels — `9` and `10` are unrelated IDs, and arithmetic has to be memorized. The notebook's premise is that embeddings can carry mathematical structure:

- `9 + 9` → the mathematical part of the embedding *is* `18`
- `9 * 9` → the mathematical part of the embedding *is* `81`

The experiment explores how far this can be pushed and whether whole operations can be described this way, reserving the existing language dimensions for words.

---

## File structure

```
Session 7/
├── Session_7_Run.ipynb   # the complete experiment (Colab-formatted, nbformat 4, Python 3)
└── README.md             # this file
```

The notebook is a linear pipeline of 12 cells (4 markdown, 8 code) that runs top to bottom. There are no external data files or downloads — everything is generated in-memory with fixed seeds (`random.seed(0)`, `torch.manual_seed(0)`).

---

## 1. The encoding scheme

To make vector arithmetic perform real math, numbers can't be randomly initialized. The notebook hand-designs each number's vector in three parts:

| Dimensions | Contents | Purpose |
|---|---|---|
| 0–3 | raw value, `log\|n\|`, sign, is-zero flag | Addition becomes ordinary vector addition; multiplication can read the log and sign |
| (first of each clock pair) | `cos(2π·(n mod T)/T)` | Periodic position of `n` on clock `T` |
| (second of each clock pair) | `sin(2π·(n mod T)/T)` | Companion phase for the same clock |

**1. Direct value & sign tracking.** The first few dimensions hold the raw scalar, the log magnitude, a sign indicator, and a zero flag, so linear addition happens naturally through vector addition.

**2. Log-space multiplication.** Multiplying in linear space is hard for neural vectors, but multiplication in log space is just addition: `log(a × b) = log(a) + log(b)`.

**3. Modular clock ratios.** Six coprime moduli `[7, 11, 13, 17, 19, 23]` are encoded as sine/cosine pairs, mirroring the Chinese Remainder Theorem. The product of the clocks is `7,438,428`, so the clocks uniquely identify integers up to roughly ±3.7 million. Crucially, adding two numbers just "turns the clock hands" by multiplying their complex phases.

The API:

| Function | Role |
|---|---|
| `encode(n)` | Number → vector |
| `add(x, y)` | Value adds; clock phases rotate |
| `negate(x)` | Flips value, sign, and clock phase direction |
| `sub(x, y)` | `add(x, negate(y))` |
| `mul(x, y)` | Adds logs and multiplies signs in log space |
| `decode_value(x)` | Read the number back from dimension 0 |
| `decode_clocks(x)` | Recover the number from the clocks alone (CRT) |

The opening demo confirms the core claim using only these hand-built vectors:

```text
9 + 9 = 18
9 * 9 = 81
9 + 9 (clocks only) = 18
```

---

## 2. Correctness sweep and precision limits

The encoding is verified exhaustively over every pair in `[-100, 100]` (40,401 pairs per operation):

```text
Every pair from -100 to 100 (40,401 per operation). Failures: {'add': 0, 'sub': 0, 'mul': 0, 'clocks': 0}
```

All four operations are **exact** — no failures. The limitation shows up only as numbers grow, because multiplication relies on floating-point `exp`/`log`:

```text
Multiplication exactness as numbers grow (computer rounding limit):
 2-digit x 2-digit: 100.0% exact
 3-digit x 3-digit: 100.0% exact
 4-digit x 4-digit: 100.0% exact
 5-digit x 5-digit: 100.0% exact
 6-digit x 6-digit: 100.0% exact
 7-digit x 7-digit: 100.0% exact
 8-digit x 8-digit: 9.7% exact
 9-digit x 9-digit: 0.2% exact
10-digit x 10-digit: 0.0% exact
```

The clock encoding behaves exactly as theory predicts:

```text
Clocks work inside ±3718214:
  3,000,000 -> 3000000
  4,000,000 -> -3436429 (wraps around, as theory predicts)
```

---

## 3. Dataset generation and tokenization

A regex tokenizer handles negative numbers, symbols, and operators, paired with a dynamic question generator.

- **Templates** (`TEMPLATES`) cover `+`, `-`, and `*` in multiple phrasings — both words and symbols, e.g. `"the sum of {x} and {y}"`, `"{x} * {y}"`, `"{x}×{y}"`, `"subtract {y} from {x}"`, `"multiply {x} by {y}"`.
- **`TOKEN_PATTERN`** = `(?<!\d)-\d+|\d+|[a-z]+|[+\-*/x×]` — a `-` is treated as a negative sign only when it does not directly follow a digit.
- **`tokenize(text)`** returns a word list and a number list, replacing each number with the `NUM` token.
- **`make_question(low, high)`** samples two signed numbers in range, picks a random template, and computes the ground-truth answer.

Data splits used for training and evaluation:

| Split | Size | Number range | Purpose |
|---|---|---|---|
| `train_qs` | 20,000 | 1–999 | Practice: small numbers only |
| `seen_qs` | 2,000 | 1–999 | Exam: familiar sizes |
| `unseen_qs` | 2,000 | 1,000–99,999 | Exam: never-seen sizes |

Tokenizer demo:

```text
9+9                  -> (['NUM', '+', 'NUM'], [9, 9])
9 * 9                -> (['NUM', '*', 'NUM'], [9, 9])
-9-5                 -> (['NUM', '-', 'NUM'], [-9, 5])
-9x3                 -> (['NUM', 'x', 'NUM'], [-9, 3])
subtract 5 from 20   -> (['subtract', 'NUM', 'from', 'NUM'], [5, 20])
```

---

## 4. Model architecture: guessing vs. rule transformers

Two models are compared to isolate what the structured embedding buys.

**`GuessingTransformer` (baseline).** Numbers are unmeaningful classification labels fed through a 2001-slot embedding; a linear head predicts the answer directly. This is the "normal AI" approach: memorize and interpolate. Targets are compressed with `squash` (signed `log1p` scaled to roughly `[-1, 1]`) and inverted with `unsquash`.

**`RuleTransformer` (the idea).** A hybrid. A shared `Reader` transformer reads only the *language* tokens (word embeddings + positions, 2 encoder layers, 4 heads, feedforward 64), and outputs two things:

- `coefficient` — a per-number routing weight (learned `+1` / `-1` distinctions for add/subtract), and
- `use_log` — a sigmoid `p_log` deciding between plain form (`+`, `-`) and log form (`×`).

The exact arithmetic is then executed by the structured embedding layer, not by a black-box head. During `predict`, the coefficients are snapped to whole numbers; plain form computes `c0·value(a) + c1·value(b)`, and log form computes `sign · exp(|c0|·log|a| + |c1|·log|b|)`. Two deliberate safeguards: `abs(c)` prevents a learned negative coefficient from silently turning log-addition into division, and a zero guard returns `0` if either operand is zero.

---

## 5. Training and evaluation

Both models train with Adam (learning rate `1e-3`), gradient clipping at `1.0`, on batches of 128 sampled questions:

- Baseline: 3,000 steps
- Rule transformer: 4,000 steps

```text
Guessing transformer:
 step    0 loss 0.19121
 step 1000 loss 0.10058
 step 2000 loss 0.07571
Rule transformer:
 step    0 loss 3.41475
 step 1000 loss 1.45202
 step 2000 loss 1.70959
 step 3000 loss 1.67701
```

Scoring reports both exact match and a ±1% tolerance:

```text
Model                  | Seen exact  Seen ±1% | Unseen exact Unseen ±1%
Guessing (normal AI)   |       0.0%      0.0% |         0.0%       0.1%
Rule (your idea)       |      90.0%     90.0% |        90.0%      90.0%
```

**Key finding:** the baseline generalizes to unseen numbers at essentially 0%, while the rule transformer holds 90% on both seen and unseen ranges — because the arithmetic is carried out by the embedding rather than guessed. Sizing up the unseen numbers to full 5-digit values, broken down by operation:

```text
Rule transformer, unseen numbers, by operation:
 '+': exact 100.0%
 '-': exact 100.0%
 '*': exact 73.7%
```

Addition and subtraction are effectively perfect (subtraction is implemented as vector negation, so it inherits addition's exactness). Multiplication is the weakest link, limited by log-space precision.

The notebook also inspects what the transformer actually learned — it recovers operation intent from phrasing alone:

```text
 {x} plus {y}                 -> plain form coefficients [1.0, 1.0]
 {x} minus {y}                -> plain form coefficients [1.0, -1.0]
 subtract {y} from {x}        -> plain form coefficients [-1.0, 1.0]
 take {y} away from {x}       -> plain form coefficients [-1.0, 1.0]
 {x} times {y}                -> log form   coefficients [-1.0, -1.0]
 multiply {x} by {y}          -> log form   coefficients [-1.0, -0.0]
 the product of {x} and {y}   -> log form   coefficients [1.0, 0.0]
```

Notice it correctly routes all `×` phrasings to **log form** and all `+`/`-` phrasings to **plain form**, with coefficients close to the mathematically correct `±1`.

---

## 6. Interactive examples

An `ask()` helper runs a single string through the tokenizer and the rule transformer:

```text
9+9                    = 18
9*9                    = 81
9 - 9                  = 0
9 x 9                  = 81
-7x6                   = -42
12--8                  = 20
multiply 9 by 9        = 9          <-- misclassified phrasing
subtract 9 from 100    = 91
48213 + 91077          = 139290
4821 * 9107            = 43904847
```

Clean symbolic layouts (`9*9`, `9 x 9`) and the trained subtraction phrasing parse correctly, while some word-order variants (e.g. `multiply 9 by 9`) can still trip the routing.

---

## 7. Limitations and tradeoffs

1. **Floating-point precision.** Multiplication uses `exp`/`log`, so beyond ~8 digits accuracy collapses (9.7% at 8 digits, ~0% at 10 digits). This is a property of the arithmetic layer, not the transformer.
2. **Decision-threshold sensitivity (`p_log`).** The plain-form vs. log-form switch is a hard `p > 0.5` boundary. If training fluctuates, this boundary can cause operation misclassifications (as seen with some multiply phrasings).
3. **Template-phrasing dependency.** Symbolic layouts parse reliably; unusual word orderings can hit edge cases in coefficient mapping.
4. **Clock range is finite.** The clock dimensions identify numbers only up to ±3,718,214; beyond that they wrap around, as CRT predicts.

---

## How to run

**Requirements:** Python 3 with `numpy` and `torch`. No dataset downloads — all questions are generated in code.

1. Open `Session_7_Run.ipynb` in Jupyter or Google Colab.
2. Run all cells top to bottom. The notebook is written as a single linear script: encoding → verification → tokenizer/data → models → training → evaluation → interactive demo.
3. Seeds are fixed (`0`), so results are stable in structure; expect minor run-to-run variation in the training losses and the multiplication exact-match percentage.

The notebook was authored with Colab metadata (`nbformat` 4, Python 3 kernel) and retains saved outputs from the original run.
