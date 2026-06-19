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

`feature_extraction.py` converts a parsed `InkSample` into a model-ready variable-length sequence. It takes the clean absolute-coordinate data that the parser produced and runs it through four sequential stages: validation, time-shifting, coordinate normalization, spatial resampling, and delta encoding.

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

## 5.1 `FeatureExtractionConfig` — Tunable Parameters

All behavior is controlled through a single frozen config object that is validated before any processing begins.

### `spatial_step` (default: `0.05`)

The equidistant arc-length interval used during spatial resampling, in normalized coordinate units. After normalization, the writing area height equals `1.0`, so a step of `0.05` means approximately 20 resampled intervals per unit of height. Smaller values produce longer sequences with finer detail; larger values produce shorter sequences.

### `surrogate_area_scale` (default: `1.20`)

When no writing-area bounding box is available in the InkML metadata, the code estimates one from the observed ink. This scale factor expands that observed vertical extent. A value of `1.20` means the surrogate area is 20% taller than the tallest observed ink, matching the paper's stated approach. Must be at least `1.0`.

### `min_coordinate_scale` (default: `1e-6`)

A floor value used in two places. First, it guards against dividing by a near-zero scale when the observed ink has almost no vertical extent (such as a single dot). Second, during resampling, it defines the minimum distance two consecutive points must be apart before they are treated as distinct — points closer than this are deduplicated before interpolation, preventing numerical instability.

### `monotonic_time_tolerance` (default: `1e-9`)

Timestamps must be globally non-decreasing. Floating-point arithmetic can occasionally produce a tiny negative delta-t even in correctly recorded data. This tolerance defines how large a negative delta-t is considered a rounding artifact (clamped to zero) versus a genuine recording error (raises an exception). The tolerance is in seconds.

### `require_monotonic_time` (default: `True`)

When `True`, the validator raises an error if any timestamp decreases beyond the tolerance. Set to `False` only if the dataset is known to use per-stroke-local clocks that reset to zero at each new trace — this is unusual and must be confirmed from dataset documentation rather than assumed.

### `preserve_single_point_strokes` (default: `True`)

Single-point strokes (a tap, a decimal point, a dot) have zero arc length and cannot be spatially resampled in the normal way. When `True`, they are preserved as-is. When `False`, encountering one raises an error. Keep this `True` for mathematical handwriting, which regularly contains dots, decimal points, and diacritical marks.

### `pen_state_mode` (default: `"last_point_up"`)

Controls how pen-up events are encoded. See section 10 for full detail.

---

## 5.2 Input Validation (`_validate_sample`)

Before any transformation, the extractor checks two things:

**Finite values.** Every x, y, and t coordinate across all strokes must be a normal floating-point number — not `NaN`, not `+Inf`, not `-Inf`. Non-finite values would silently propagate through normalization and resampling and corrupt every downstream delta.

**Globally monotonic timestamps.** Timestamps must never decrease as you move forward through all strokes in written order. The check is global, not per-stroke — it treats the entire sample as one continuous timeline. A timestamp that goes backwards at a stroke boundary is a sign either that the device restarted its clock, or that the InkML was assembled from mis-ordered traces. The code raises a detailed error pointing to the exact flattened point index where the reversal occurs, rather than silently repairing it. Silent repair can introduce impossible negative `dt` values into the final features.

---

## 5.3 Time Shifting (`_shift_time_to_zero`)

The first timestamp of the first stroke is subtracted from every point in the sample. After this step, the first point always has `t = 0`, and all other timestamps represent elapsed time from the pen-down start.

This is necessary because the delta encoding only produces time differences — the absolute starting time from the device clock carries no useful information for the model and would otherwise appear as a large `dt` on the very first point.

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

until the stroke endpoint. The last target is always snapped to the exact total stroke length so the endpoint is always included.

## Deduplication before interpolation

Before interpolation, any two consecutive source points that are within `min_coordinate_scale` of each other spatially are collapsed to one. The collapsing rule is: replace the earlier point with the later one. This means the surviving representative of a run of near-coincident points is always the last one (which has the latest timestamp), ensuring `t` stays non-decreasing.

This step is necessary because linear interpolation (`np.interp`) requires strictly increasing x-values. Near-zero-length segments produce near-duplicate cumulative distances, which would make `np.interp` produce undefined or wildly incorrect results.

## Interpolating time

At every new spatial location, the code linearly interpolates `x`, `y`, and `t` simultaneously. This means time is spread proportionally according to the spatial distance covered, not the original sample-point index. Spatial regularity is achieved without discarding timing information.

## Single-point strokes

Single-point and nearly zero-length strokes are preserved because they may represent decimal points, multiplication dots, punctuation, or tiny marks. A zero-length stroke is reduced to a single representative point whose position comes from the first recorded point and whose timestamp comes from the last, preserving the correct elapsed time.

---

# 9. Delta feature creation

After resampling, strokes are flattened in original writing order into a single sequence of absolute `(x, y, t, pen_state, new_stroke)` rows. The spatial differences are then computed across the entire flattened sequence at once.

```text
dx_i = x_i - x_(i-1)
dy_i = y_i - y_(i-1)
dt_i = t_i - t_(i-1)
```

For the first point, all three deltas are set to zero:

```text
dx_0 = 0
dy_0 = 0
dt_0 = 0
```

`pen_state` and `new_stroke` are copied directly from the absolute rows rather than differenced — they are already binary indicators, not quantities.

## Do not reset deltas at stroke boundaries

The jump from one stroke to the next can encode useful layout information, such as movement from a base symbol to a superscript or from numerator to denominator. The stroke flags (`new_stroke`) tell the model that the jump was not a continuous drawn line, but the spatial and temporal displacement itself is part of the signal.

## Tiny-negative `dt` correction

After computing deltas, it is possible for a `dt` value to be a very small negative number — for example `-2e-11` — due to floating-point rounding during the time interpolation step that occurred inside resampling. These are not genuine time reversals. Any `dt` that is negative but within `monotonic_time_tolerance` is clamped to exactly zero. Any `dt` more negative than that threshold raises a `FeatureExtractionError`, because it indicates a real problem in the source data that should be fixed upstream rather than papered over.

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

# 11. Output: `FeatureSequence` and `NormalizationParameters`

The extractor returns a `FeatureSequence` object containing everything needed by a downstream model and everything needed to reconstruct the original ink.

```python
FeatureSequence(
    features          = np.ndarray,  # shape [T, 5], dtype float32
    sequence_length   = T,           # == features.shape[0]
    label             = "x^2+1",     # from the InkML annotation, or None
    sample_id         = "sample_0001",
    normalized_strokes  = (...),     # after normalization, before resampling
    resampled_strokes   = (...),     # after resampling, before delta encoding
    normalization       = NormalizationParameters(...),
    metadata            = {...},
)
```

A later batch collator should pad only to the longest sample in each batch and pass original sequence lengths to the model so it ignores padding.

## `NormalizationParameters`

```python
NormalizationParameters(
    x_origin              = 738.0,   # first recorded x in device pixels
    y_area_min            = 80.0,    # bottom of writing area (or surrogate)
    scale                 = 480.0,   # height used for both x and y scaling
    used_known_writing_area = True,
)
```

These four numbers record exactly what transform was applied. They are stored alongside the features because:

1. **Invertibility.** The normalized coordinates can be converted back to device pixels at any time: `x_device = x_normalized * scale + x_origin` and `y_device = y_normalized * scale + y_area_min`. This is required for the planned ink-editing functionality, where a user modifies a stroke and the pipeline must re-extract features from the corrected absolute coordinates.

2. **Consistency checking.** If you process training data and live input through the same code, comparing `used_known_writing_area` across splits lets you detect data-collection differences before they affect training.

## Reconstructing absolute coordinates from features

`PaperFeatureExtractor` exposes a static method that inverts the delta encoding:

```python
absolute = PaperFeatureExtractor.reconstruct_absolute_deltas(features)
```

`np.cumsum` on the `dx`, `dy`, `dt` columns recovers normalized absolute x, y, t. The result starts at `(0, initial_y, initial_t)` — `initial_y` defaults to `0.0` because x was shifted so the first point is at `x = 0`, but y's starting position in normalized space depends on where in the writing area the first point fell.

To get back to device pixels:

```python
absolute_normalized = PaperFeatureExtractor.reconstruct_absolute_deltas(features)
x_device = absolute_normalized[:, 0] * normalization.scale + normalization.x_origin
y_device = absolute_normalized[:, 1] * normalization.scale + normalization.y_area_min
```

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
