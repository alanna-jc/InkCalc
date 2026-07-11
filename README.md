# InkCalc

**A standalone embedded handwriting-recognition calculator.** Write a math
expression by hand on a touchscreen and InkCalc recognizes it and solves it
entirely on-device, no network required.

## How it works

A three-stage pipeline, designed to run on a Raspberry Pi with a resistive touchscreen:

1. **Recognition** — a CTC-based Transformer encoder converts the online-handwriting
   pen strokes (`[dx, dy, dt, pen_state]` point sequences) into a LaTeX string.
   Trained on the [MathWriting-2024](https://github.com/google-research/google-research/tree/master/mathwriting)
   dataset and deployed as a self-contained ONNX model (runs under `onnxruntime`,
   no PyTorch on-device).
2. **Solving** — [SymPy](https://www.sympy.org) parses the LaTeX and computes a
   result: arithmetic, symbolic simplification, equation solving, and matrix operations.
3. **UI** — a `pygame` touchscreen interface for drawing, showing the inferred
   expression, and displaying the result.

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
│   └── parity_check.py             # verify .pt and exported .onnx agree
├── export_onnx.py                  # checkpoint (.pt) → ONNX model + vocab.json (for the Pi)
├── analyze_lengths.py              # sequence-length statistics over the dataset
└── ui_5inch_resistive/             # on-device app (Raspberry Pi 4)
    ├── recognition.py              # ONNX inference + matching preprocessing
    ├── solver.py                   # SymPy LaTeX solver
    └── ui.py                  # pygame touchscreen UI
```

## Trained model

The model is trained and exported on a workstation, then only the `.onnx` model
and `vocab.json` are copied to the device (in export folder of below).

The trained model is too large to commit. Download it here (vocab included):
https://drive.google.com/drive/folders/1VBTDO9-XxMkovqbcf12qfQrAg3RtgbuM?usp=sharing

