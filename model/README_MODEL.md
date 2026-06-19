# InkCalc Paper-Compatible InkML Preprocessing

This folder contains the first two implementation files for InkCalc's **online handwriting preprocessing pipeline**:

```text
inkml_parser.py
feature_extraction.py
```

It deliberately does **not** implement the paper's Bézier-curve representation. The current goal is to build and verify the simpler raw-point baseline first.

The implementation follows the raw-input preprocessing described in:

> Victor Carbune et al.  
> **Fast Multi-language LSTM-based Online Handwriting Recognition.**  
> arXiv:1902.10525, especially Sections 2.1.1 and 2.1.2.

Paper: https://arxiv.org/abs/1902.10525  
InkML specification: https://www.w3.org/TR/InkML/

---

## 1. What problem are these files solving?

An InkML file stores handwriting as **ordered pen trajectories** rather than only as a picture.

A simplified stroke is:

```text
(x0, y0, t0)
(x1, y1, t1)
(x2, y2, t2)
...
```

The coordinates show where the pen was. The timestamp shows when each point was captured. Separate `<trace>` elements normally represent separate strokes.

The model should not receive raw device values such as:

```text
x = 738
y = 412
t = 16854 milliseconds
```

Those values depend on screen resolution, writing-area size, device sampling rate, starting position, and timestamp units.

The preprocessing pipeline converts them into:

```text
[delta_x, delta_y, delta_t, pen_state, new_stroke]
```

or:

```text
[dx, dy, dt, p, n]
```

That sequence is suitable for an LSTM, GRU, Transformer, or another sequence model.

---

## 2. Why `dt` matters

The cited paper explicitly retains time. The raw points are:

```text
(x_i, y_i, t_i)
```

After preprocessing, the model receives:

```text
(x_i - x_(i-1),
 y_i - y_(i-1),
 t_i - t_(i-1),
 p_i,
 n_i)
```

Therefore:

```text
dt_i = t_i - t_(i-1)
```

`dt` gives the model information about handwriting dynamics:

- slow versus fast movement,
- pauses before a new symbol,
- pauses before a superscript or denominator,
- hesitation around difficult symbols,
- inter-stroke timing.

The parser converts timestamps to **seconds** when the InkML channel declares a recognized unit. The feature extractor then shifts the first timestamp to zero and calculates `dt`.

---

## 3. End-to-end processing pipeline

```text
InkML file
    ↓
InkMLParser
    ↓
absolute x, absolute y, absolute t in seconds
    ↓
shift first timestamp to zero
    ↓
paper-style coordinate normalization
    ↓
equidistant spatial resampling at delta = 0.05
    ↓
linear interpolation of x, y, and t
    ↓
flatten strokes in original writing order
    ↓
create dx, dy, dt, p, n
    ↓
variable-length NumPy array with shape [T, 5]
```

`T` is the resulting number of time steps. The code does not force every sample to a fixed length. Padding belongs in the model's batch collator.

---

# 4. `inkml_parser.py`

## Responsibility

`inkml_parser.py` converts XML into clean Python objects. It does **not** normalize coordinates or create model features.

Its central output is:

```python
InkSample(
    strokes=(...),
    label="x+1",
    sample_id="sample_0001",
    source_path="...",
    writing_area=None,
    metadata={...},
)
```

Each stroke contains absolute points:

```python
InkStroke(
    points=(
        InkPoint(x=..., y=..., t=...),
        ...
    )
)
```

## Main classes

### `InkPoint`

```python
InkPoint(x: float, y: float, t: float)
```

`t` is in seconds after parsing when time conversion is enabled.

### `InkStroke`

Preserves one original InkML stroke boundary:

```python
InkStroke(points: tuple[InkPoint, ...])
```

### `WritingArea`

Stores known writing-surface bounds when metadata is available:

```python
WritingArea(
    min_x=0,
    min_y=0,
    width=800,
    height=480,
)
```

### `InkSample`

Contains the complete parsed sample.

### `InkMLParser`

```python
from inkml_parser import InkMLParser

parser = InkMLParser(
    require_time=True,
    default_time_unit="s",
)

sample = parser.parse("example.inkml")
```

## Trace-format support

InkML can declare its channel order:

```xml
<traceFormat>
    <channel name="X"/>
    <channel name="Y"/>
    <channel name="T" units="ms"/>
</traceFormat>
```

The parser reads these declarations rather than blindly assuming channel order.

## Absolute and delta-encoded InkML values

InkML can compactly encode values using prefixes:

```text
!  absolute value
'  first difference
"  second difference
```

The parser reconstructs absolute channel values before passing them to feature extraction.

## Time-unit conversion

Recognized examples include seconds, milliseconds, microseconds, and nanoseconds.

If the file declares no unit, pass the correct dataset assumption:

```python
InkMLParser(default_time_unit="ms")
```

Do not guess silently if dataset documentation gives an explicit unit.

## Labels

The parser checks annotations such as:

```xml
<annotation type="truth">x^2+1</annotation>
```

and stores the result in `sample.label`.

---

# 5. `feature_extraction.py`

## Responsibility

`feature_extraction.py` converts a parsed `InkSample` into a model-ready variable-length sequence.

```python
from feature_extraction import FeatureExtractionConfig, PaperFeatureExtractor

extractor = PaperFeatureExtractor(
    FeatureExtractionConfig(
        spatial_step=0.05,
        surrogate_area_scale=1.20,
    )
)

sequence = extractor.transform(sample)
features = sequence.features
```

Output:

```text
features.shape == [T, 5]
```

Columns:

```text
0: delta_x
1: delta_y
2: delta_t
3: pen_state
4: new_stroke
```

---

# 6. Coordinate normalization from the paper

The implementation does not center every expression into `[-0.5, 0.5]`. It follows the paper:

1. Shift `x` so the first point has `x = 0`.
2. Scale `x` and `y` using the **same scale**.
3. Use writing-area height as the scale.
4. Shift `y` so the writing area's vertical range maps to `[0, 1]`.

Let:

```text
x_0        = first recorded x coordinate
y_area_min = minimum y of the writing area
H          = writing-area height
```

Then:

```text
x_normalized = (x - x_0) / H
y_normalized = (y - y_area_min) / H
```

## Why use one shared scale?

This is **isometric normalization**. It preserves aspect ratio, symbol width-to-height relationships, exponent placement, fraction structure, and horizontal spacing.

Do not independently force x and y into `[0, 1]`, because that deforms the handwriting.

## Why x is not restricted to `[0, 1]`

Only the writing-area y range is normalized to `[0, 1]`. A long expression might have:

```text
x range: 0 to 7.2
y range: 0 to 1
```

That is valid.

## No y-axis flip

The baseline preserves the source orientation. Training data and live touchscreen input must use the same convention.

---

# 7. Unknown writing area and the 20% surrogate area

When the real writing-area bounding box is unknown, the paper uses a surrogate area 20% larger than the observed vertical range.

```text
observed_height = y_max - y_min
surrogate_height = 1.20 * observed_height
```

The implementation divides the extra 20% evenly around the ink:

```text
total_padding = surrogate_height - observed_height
y_area_min = y_min - total_padding / 2
```

The observed ink then occupies approximately `0.0833` to `0.9167` vertically.

## Dot and zero-height fallback

A single dot has zero observed height and cannot be used as a divisor. The code preserves such strokes and uses a conservative fallback scale instead of deleting the sample or dividing by zero.

---

# 8. Equidistant spatial resampling

Devices produce inconsistent point densities. The paper uses spatial resampling with:

```text
delta = 0.05
```

A normalized line of spatial length 1 therefore has approximately 20 equal intervals.

For consecutive normalized points:

```text
distance_i = sqrt((x_i-x_(i-1))^2 + (y_i-y_(i-1))^2)
```

Cumulative distance is:

```text
s_i = sum(distance_1 ... distance_i)
```

New sample positions are:

```text
0.00, 0.05, 0.10, 0.15, ...
```

until the stroke endpoint.

## Interpolating time

At every new spatial location, the code linearly interpolates `x`, `y`, and `t`. Spatial regularity is achieved without discarding timing information.

## Single-point strokes

Single-point and nearly zero-length strokes are preserved because they may represent decimal points, multiplication dots, punctuation, or tiny marks.

---

# 9. Delta feature creation

After resampling, strokes are flattened in original writing order.

```text
dx_i = x_i - x_(i-1)
dy_i = y_i - y_(i-1)
dt_i = t_i - t_(i-1)
```

For the first point:

```text
dx_0 = 0
dy_0 = 0
dt_0 = 0
```

## Do not reset deltas at stroke boundaries

The jump from one stroke to the next can encode useful layout information, such as movement from a base symbol to a superscript or from numerator to denominator. The stroke flags tell the model that the jump was not a continuous drawn line.

---

# 10. Pen-state conventions for InkML

The paper uses:

```text
p_i = 1 for pen down
p_i = 0 for pen up
n_i = 1 at the beginning of a new stroke
n_i = 0 otherwise
```

InkML normally stores pen-down trajectories as separate traces but may not include explicit hover samples. The code supports two conventions.

## Default: `last_point_up`

```text
all non-final points: p = 1
final point:          p = 0
first point:          n = 1
other points:         n = 0
```

## Alternative: `explicit_up_event`

All real points receive `p = 1`, then a duplicate endpoint with `p = 0` is added.

Use one convention consistently for training, validation, test, and live input.

---

# 11. Variable-length output

The extractor returns:

```python
FeatureSequence(
    features=np.ndarray[T, 5],
    sequence_length=T,
    ...
)
```

A later batch collator should pad only to the longest sample in each batch and preserve original sequence lengths.

---

# 12. Installation

```bash
pip install numpy
```

Optional safer XML parser:

```bash
pip install defusedxml
```

---

# 13. Command-line usage

Inspect one InkML file:

```bash
python inkml_parser.py example.inkml --default-time-unit ms
```

Write parsed JSON:

```bash
python inkml_parser.py example.inkml \
    --default-time-unit ms \
    --json-output parsed/example.json
```

Extract features:

```bash
python feature_extraction.py example.inkml \
    --default-time-unit ms \
    --output processed/example_features.npz
```

The `.npz` file contains:

```text
features
sequence_length
label
sample_id
feature_names
```

---

# 14. Python usage

```python
from inkml_parser import InkMLParser
from feature_extraction import FeatureExtractionConfig, PaperFeatureExtractor

parser = InkMLParser(
    require_time=True,
    default_time_unit="ms",
)

sample = parser.parse("data/raw/example.inkml")

extractor = PaperFeatureExtractor(
    FeatureExtractionConfig(
        spatial_step=0.05,
        surrogate_area_scale=1.20,
        pen_state_mode="last_point_up",
    )
)

sequence = extractor.transform(sample)

print(sequence.features.shape)
print(sequence.features[:10])
print(sequence.label)
```

---

# 15. Validation checks

The implementation checks for missing files, invalid XML, missing traces, missing channels, unsupported time units, mismatched channel counts, non-finite coordinates, empty strokes, non-monotonic timestamps, invalid scales, negative `dt`, and invalid output shape.

The first feature row always has:

```text
dx = 0
dy = 0
dt = 0
```

Recommended tests include absolute traces, reordered channels, millisecond timestamps, first- and second-difference encodings, multiple strokes, one-point dots, and truth annotations.

For visual debugging, plot raw points, normalized points, resampled points, stroke boundaries, and point colour versus time.

---

# 16. Important assumptions

1. Timestamps are globally ordered across strokes.
2. One InkML `<trace>` corresponds to one stroke.
3. Live touchscreen input will reuse the same preprocessing pipeline.
4. Bézier curves are intentionally excluded for now.

If a dataset restarts time at zero for every trace, that must be handled explicitly from dataset documentation. The code does not silently invent cross-stroke timing.

---

# 17. Theory summary of the paper

The paper uses an end-to-end sequence recognizer:

```text
input sequence
    ↓
multiple bidirectional LSTM layers
    ↓
softmax classification at each time step
    ↓
CTC decoding
```

Its main preprocessing lesson is that a sufficiently deep network can learn useful representations from compact raw features rather than requiring a large handcrafted feature set.

For raw points, the paper uses:

```text
dx, dy, dt, p, n
```

The key preprocessing steps are device-independent normalization, isotropic scaling, a 20%-expanded surrogate area when needed, equidistant spatial resampling, temporal differences, and stroke-state indicators.

The paper also studies Bézier curves mainly to shorten sequences and improve recognition speed. InkCalc is intentionally implementing the raw representation first because it is easier to inspect, unit test, and benchmark.

---

# 18. Recommended next modules

```text
label_processing.py
dataset.py
collate.py
model.py
ctc_decoder.py
evaluation.py
```

Recommended order:

```text
1. verify InkML parsing
2. verify feature extraction visually and numerically
3. analyze post-resampling sequence-length percentiles
4. implement label normalization and vocabulary
5. build a variable-length dataset and collator
6. train a raw-point CTC baseline
7. measure CER and exact-match accuracy
8. only then evaluate more advanced representations
```

---

# 19. References

## Primary paper

Carbune et al., **Fast Multi-language LSTM-based Online Handwriting Recognition**  
https://arxiv.org/abs/1902.10525

Relevant sections:

```text
2.1.1 Raw Touch Points
2.1.2 Bézier Curves
3.1 Connectionist Temporal Classification Loss
```

## InkML specification

https://www.w3.org/TR/InkML/

Use it to validate trace formats, channel names, units, derivative encodings, contexts, and writing-area metadata.

## MathWriting dataset

Before processing the full dataset, inspect several real MathWriting files and confirm channel order, time unit, timestamp scope, trace encoding, label annotation type, and writing-area availability. The parser exposes these details instead of hard-coding undocumented assumptions.
