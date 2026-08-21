# InkCalc

**A standalone embedded handwriting-recognition calculator.** Write a math
expression by hand on a touchscreen and InkCalc recognizes it and solves it
entirely on-device, no network required.

Link to website: https://alanna-jc.github.io/InkCalc/ 

## How it works

A three-stage pipeline, designed to run on a Raspberry Pi 4 (with 4GB RAM) with a resistive touchscreen:

1. **Recognition** — a CTC-based Transformer encoder converts the online-handwriting
   pen strokes (`[dx, dy, dt, pen_state]` point sequences) into a LaTeX string.
   Trained on the [MathWriting-2024](https://github.com/google-research/google-research/tree/master/mathwriting)
   dataset and deployed as a self-contained ONNX model (runs under `onnxruntime`,
   no PyTorch on-device).
2. **Solving** — [SymPy](https://www.sympy.org) parses the LaTeX and computes a
   result: arithmetic, symbolic simplification, equation solving, and matrix operations.
3. **UI** — a `pygame` touchscreen interface for drawing, showing the inferred
   expression, and displaying the result.

## What it can solve

Once an expression is recognized, the solver
([`ui_5inch_resistive/solver.py`](ui_5inch_resistive/solver.py)) handles:

- **Arithmetic & numeric evaluation** — e.g. `2 + 4`, `\frac{10}{2}`, `2^{10}`, `\sqrt{16}`
- **Symbolic simplification** — expressions with variables, e.g. `x^2 + 2x + x^2` → `2x(x+1)`
- **Equation solving** — single- and multi-variable, e.g. `x^2 + 5x + 6 = 0` → `x = -2, -3`
- **Matrices** — display, multiplication, and equality checks
- **Linear systems** — `A x = b` (reports a unique solution, infinitely many, or inconsistent)
- **Big-operator notation** — summations `\sum`, products `\prod`, and integrals `\int`
  (definite and indefinite), e.g. `\sum_{i=1}^{n} i` → `n(n+1)/2`, `\int_{0}^{2} x^2\,dx` → `8/3`
- **Structural repair** — recovers from common recognizer mistakes before solving:
  dropped/added braces, doubled sub/superscript markers, and ragged matrix rows

**Not supported yet:**

- **Transcendental functions** (`sin`, `cos`, `tan`, `log`, `ln`, `exp`) — the model's
  vocabulary has no tokens for these, so they can't be recognized or output. Supporting
  them needs both a solver-side normalization layer and those tokens in the model.
- **Limits** (`\lim`) — also absent from the model's vocabulary.

## Repo layout

```
InkCalc/
├── model/                          # training pipeline (run scripts from here)
│   ├── model/                      # network definition
│   │   ├── encoder.py              # CTC-Transformer encoder
│   │   └── positional_encoding.py  # sinusoidal positional encoding
│   ├── preprocessing/              # ink → model input
│   │   ├── inkml_parser.py         # InkML stroke parsing
│   │   ├── feature_extraction.py   # [dx, dy, dt, pen_state] features from ink
│   │   └── dataset.py              # Dataset, batching, padding, collate
│   ├── postprocessing/             # logits → tokens
│   │   ├── ctc_decode.py           # greedy CTC decode + edit distance
│   │   └── beam_search.py          # beam search decoder (not implemented)
│   ├── training_loop.py            # single-GPU training loop
│   ├── vocab.py                    # LaTeX tokenizer + vocab build/load
│   ├── smoke_test.py               # fast end-to-end pipeline sanity check
│   ├── evaluate_test.py            # test-set CER / exact-match evaluation
│   ├── parity_check.py             # verify .pt and exported .onnx agree
│   ├── analyze_errors.py           # dump every test misprediction to error_analysis/
│   ├── make_poster_examples.py     # curate representative error examples (with ink SVG)
│   └── error_analysis/             # generated error report + curated examples (see below)
├── export_onnx.py                  # checkpoint (.pt) → ONNX model + vocab.json (for the Pi)
├── analyze_lengths.py              # sequence-length statistics over the dataset
└── ui_5inch_resistive/             # on-device app (Raspberry Pi 4)
    ├── recognition.py              # ONNX inference + matching preprocessing
    ├── solver.py                   # SymPy LaTeX solver
    └── ui.py                       # pygame touchscreen UI
```

### Error analysis (`model/error_analysis/`)

A generated breakdown of the model's mistakes on the test set, used for the report and
poster. Produced by `analyze_errors.py` and `make_poster_examples.py`:

- **`test_errors.csv`** — every non-exact-match prediction (reference vs. prediction,
  token/character edit distance, per-operation counts), sorted by how far off it is.
- **`test_error_summary.txt`** — an edit-distance histogram plus the most common
  substitutions, deletions, and insertions across all errors.
- **`poster_examples.csv`** / **`poster_examples.html`** — a curated set of representative
  errors grouped by type (case confusion, symbol look-alikes, structural/grouping, and
  catastrophic), with the original handwriting rendered inline as SVG.

## Trained model

The model is trained and exported on a workstation, then only the `.onnx` model
and `vocab.json` are copied to the Pi (in export folder of below).

The trained model is too large to commit. Download it here (vocab included):
https://drive.google.com/drive/folders/1VBTDO9-XxMkovqbcf12qfQrAg3RtgbuM?usp=sharing

