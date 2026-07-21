# InkCalc — UI Overview (diagram-ready)

A self-contained, on-device handwriting math calculator running fullscreen on a
Raspberry Pi with a 5-inch resistive touchscreen. Everything runs locally — no
network. The screen is split into an **input side** and an **output side**.

---

## 1. UI Layout

The window is divided by a vertical divider into two panels (roughly 62% / 38% width).

**Left panel — Input (~62% width)**
- **Canvas** — the large drawing area where the user writes an expression by hand
  (finger, stylus, or mouse). Strokes render live as they're drawn.
- **Button row** (bottom, three equal buttons):
  - **Undo** — removes the last completed stroke.
  - **Clear** — erases the whole canvas.
  - **Submit** — sends the drawing for recognition.

**Right panel — Output (~38% width)**
- **Inferred area** (pinned at top, fixed height) — displays the recognized LaTeX
  expression. Long expressions wrap; the user reads this to verify recognition
  before solving.
- **Results area** (scrollable, middle) — a history of solved entries. Each entry
  shows the inferred expression and its result, or an error message. Scrolls by dragging.
- **Button row** (bottom, two equal buttons):
  - **Solve** — computes a result from the current inferred expression.
  - **Clear** — clears the inferred expression and results history.

---

## 2. Component Breakdown (boxes + arrows)

**Boxes (nodes):**

| Box | Role |
|-----|------|
| **Touchscreen input** | Captures finger/stylus/mouse events |
| **Canvas (stroke state)** | Holds the drawn strokes as sequences of (x, y, time) points |
| **Undo / Clear / Submit** | Input-side controls |
| **Recognition** (background) | Preprocess → Model → Decode |
| **Inferred area** | Shows recognized LaTeX; user verification point |
| **Solve / Clear** | Output-side controls |
| **Solver** | Computes the result from LaTeX |
| **Results area** | History of results / errors |

**Arrows (flow):**
- Touchscreen input → Canvas (add stroke)
- Undo → Canvas (remove last stroke); Clear → Canvas (reset)
- Submit → Recognition → Inferred area
- Solve → Solver → Results area
- Clear (right) → Inferred area + Results area (reset)

---

## 3. Functional Overview (input flow)

Consistent stage names: **Capture → Preprocess → Recognize → Decode → Verify → Solve → Display.**

1. **Capture** — The user draws on the Canvas. Each stroke is stored as a list of
   (x, y, time) points. Undo/Clear edit this stroke list directly.
2. **Submit** — Recognition runs on a background thread so the UI stays responsive;
   a "recognizing…" indicator shows meanwhile. (On the first submit, the model and
   vocabulary are loaded from disk.)
3. **Preprocess** — The strokes are normalized to a unit box, resampled at even
   spacing, and converted into per-point features `[dx, dy, dt, pen_state]`, then
   padded to a fixed length with a padding mask.
4. **Recognize** — The features go through the on-device model (ONNX CTC-Transformer),
   producing per-position character probabilities.
5. **Decode** — Greedy CTC decoding turns those probabilities into a **LaTeX string**.
6. **Verify** — The LaTeX appears in the **Inferred area**. The user reads it and can
   re-draw if recognition slipped. (This verify step is important — recognition is the
   system's accuracy ceiling.)
7. **Solve** — Pressing Solve sends the inferred LaTeX to the **Solver** (symbolic
   math: evaluate, simplify, solve equations, matrix operations / linear systems).
8. **Display** — The result (or an error message) is appended to the **Results area**
   as a new history entry.

---

## 4. Suggested Diagram Layout

### (a) Screen layout mock

```
┌──────────────────────────────────────────────┬───────────────────────────┐
│  LEFT PANEL — INPUT (~62%)                     │  RIGHT PANEL — OUTPUT(~38%)│
│                                                │                            │
│  ┌──────────────────────────────────────────┐ │  ┌──────────────────────┐ │
│  │                                          │ │  │ INFERRED             │ │
│  │              CANVAS                       │ │  │  <recognized LaTeX>  │ │ ← wraps
│  │   draw expression                        │ │  ├──────────────────────┤ │
│  │   (finger / stylus / mouse)              │ │  │ RESULTS  (scrolls)   │ │
│  │                                          │ │  │  INFERRED: …         │ │
│  │                                          │ │  │  RESULT:   …         │ │
│  │                                          │ │  │  … history entries … │ │
│  └──────────────────────────────────────────┘ │  └──────────────────────┘ │
│  ┌────────┐  ┌────────┐  ┌────────┐            │  ┌────────┐  ┌────────┐   │
│  │  UNDO  │  │ CLEAR  │  │ SUBMIT │            │  │ SOLVE  │  │ CLEAR  │   │
│  └────────┘  └────────┘  └────────┘            │  └────────┘  └────────┘   │
└──────────────────────────────────────────────┴───────────────────────────┘
                              ↑ vertical divider
```

### (b) Data-flow diagram

```
        ┌────────────────────┐
        │  Touchscreen input │
        └─────────┬──────────┘
                  │ strokes (x, y, time)
                  ▼
        ┌────────────────────┐   undo  → remove last stroke
        │  Canvas / strokes  │   clear → reset
        └─────────┬──────────┘
                  │ SUBMIT
                  ▼
   ┌──────────────────────────────────────┐
   │  RECOGNITION  (background thread)      │
   │   Preprocess → Model (ONNX) → Decode  │
   │   strokes → features → probs → LaTeX  │
   └─────────┬────────────────────────────┘
             │ inferred LaTeX
             ▼
   ┌────────────────────┐
   │   INFERRED area     │ ◀── user verifies (re-draw if wrong)
   └─────────┬──────────┘
             │ SOLVE
             ▼
   ┌────────────────────┐
   │   Solver (SymPy)    │ → result  OR  error
   └─────────┬──────────┘
             ▼
   ┌────────────────────┐
   │   Results area      │  (scrollable history)
   └────────────────────┘
```

Color/grouping hint for the visual version: shade the **Recognition** box
(Preprocess / Model / Decode) as one group to show it's the ML pipeline, and mark the
**Inferred area** as the human-in-the-loop verification checkpoint.

---

## 5. Future Work

*(Proposed; not yet implemented. Shown as dashed/future nodes in a diagram.)*

**A. User-calibration routine**
A short, one-time flow where the user is prompted to write a small set of reference
glyphs before regular use. The intent is to capture per-user handwriting
characteristics the model cannot infer from a single expression — most usefully the
**relative size of upper- vs lower-case letters**, since preprocessing normalizes
absolute size away and case confusion is the most common recognition error. The
captured "calibration profile" would feed into the recognition path (as a
preprocessing reference or decode-time hint).

Diagram placement:
```
[Calibration prompts] → [Calibration profile] ┄┄▶ (into) RECOGNITION
```

**B. Editable inferred expression (keyboard)** — *TODO, may implement later*
After recognition, present the inferred LaTeX in an **editable field** so the user can
correct a misrecognition (e.g., a constant read as the wrong digit) before solving,
using an on-screen or attached keyboard. The edited text — not the raw recognition —
is what feeds the Solver. This makes the "verify" step actionable rather than read-only.

Diagram placement:
```
INFERRED area ┄┄▶ [Editable text field (keyboard)] ┄┄▶ Solver
```

---

Terminology is kept consistent throughout: **Canvas, Inferred area, Results area,
Recognition (Preprocess/Model/Decode), Solver**, and the buttons
**Undo / Clear / Submit / Solve / Clear**. Nothing above assumes features beyond what
the current UI does, and both Future Work items are clearly marked as not-yet-built.
