"""
recognition.py  —  Handwriting → LaTeX inference module
=========================================================================
Runtime dependencies (Pi):
    pip install numpy onnxruntime

Required files alongside this script:
    model.onnx    —  exported by export_onnx.py on the training machine
    vocab.json    —  exported by export_onnx.py on the training machine

In ui.py, the submit handler calls:
    inferred, error = run_recognition(draw_state.strokes)
"""

import json
import time
import numpy as np

# ---------------------------------------------------------------------------
# ── Paths (everything needs to be in the same directory)
# ---------------------------------------------------------------------------

MODEL_PATH = 'model.onnx'
VOCAB_PATH  = 'vocab.json'

# ---------------------------------------------------------------------------
# ── Lazy singletons
# ---------------------------------------------------------------------------

_session = None   # onnxruntime.InferenceSession
_vocab   = None   # dict[int, str]  index → character
_meta    = None   # dict  (max_points, blank_idx, vocab_size)


def _load_assets():
    global _session, _vocab, _meta

    import onnxruntime as ort

    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        bundle = json.load(f)

    raw_vocab = bundle['vocab']
    _meta     = bundle['meta']
    _vocab    = {int(k): v for k, v in raw_vocab.items()}
    print(f'[recognition] Vocab: {len(_vocab)} tokens  '
          f'max_points={_meta["max_points"]}')

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2

    _session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=opts,
        providers=['CPUExecutionProvider'],
    )
    inp = _session.get_inputs()[0]
    print(f'[recognition] Model ready  '
          f'input="{inp.name}" shape={inp.shape}')


def _get_session():
    global _session
    if _session is None:
        t0 = time.time()
        _load_assets()
        print(f'[recognition] Assets loaded in {time.time()-t0:.1f}s')
    return _session


# ---------------------------------------------------------------------------
# ── Preprocessing  (must match training exactly)
# ---------------------------------------------------------------------------

_SPATIAL_STEP   = 0.05   # matches feature_extraction.py FeatureExtractionConfig
_MIN_COORD_SCALE = 1e-6  # floor for bounding-box ranges and deduplication


def _resample_stroke(points: list[tuple]) -> list[tuple]:
    """
    Resample one normalized stroke to equidistant arc-length intervals.
    Mirrors _resample_stroke in feature_extraction.py exactly.

    points: list of (x, y, t) already in normalized [0,1] coordinates.
    Returns a new list of (x, y, t) tuples at spatial_step spacing.
    """
    if len(points) == 1:
        return points

    values = np.array(points, dtype=np.float64)          # (M, 3)

    segment_lengths = np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])

    if total_length < _MIN_COORD_SCALE:
        # Zero-length stroke: single representative point (first xy, last t)
        return [(float(values[0, 0]), float(values[0, 1]), float(values[-1, 2]))]

    # Deduplicate near-coincident points so np.interp gets strictly
    # increasing x values.
    keep = [0]
    for i in range(1, len(cumulative)):
        if cumulative[i] > cumulative[keep[-1]] + _MIN_COORD_SCALE:
            keep.append(i)
        else:
            keep[-1] = i

    cum_u = cumulative[keep]
    val_u = values[keep]

    targets = np.arange(0.0, total_length, _SPATIAL_STEP)
    if targets.size == 0 or not np.isclose(targets[-1], total_length):
        targets = np.append(targets, total_length)
    else:
        targets[-1] = total_length

    x_r = np.interp(targets, cum_u, val_u[:, 0])
    y_r = np.interp(targets, cum_u, val_u[:, 1])
    t_r = np.interp(targets, cum_u, val_u[:, 2])

    return list(zip(x_r.tolist(), y_r.tolist(), t_r.tolist()))


def _preprocess(strokes: list[list[tuple]], max_points: int):
    """
    Strokes → (points_tensor, lengths_tensor) ready for ONNX inference.

    strokes: draw_state.strokes — [ [(x, y, t), …], … ]
    x, y are screen pixel coords; t is time.time() epoch seconds.

    Pipeline (matches feature_extraction.py exactly):
        1. Normalize x and y independently to [0, 1] using bounding box.
        2. Resample each stroke to equidistant arc-length intervals of 0.05.
        3. Flatten strokes; tag last point of each stroke with pen_up = 1.
        4. Compute dx, dy, dt deltas (first row = 0).
        5. Pad or truncate to max_points.

    Output features per point: [dx, dy, dt, pen_up]
    """
    strokes = [s for s in strokes if s]

    # ── Spatial normalisation ─────────────────────────────────────────────
    # Compute bounding box across all strokes before resampling so the scale
    # is consistent across the whole drawing.
    all_x = [x for stroke in strokes for x, _, _ in stroke]
    all_y = [y for stroke in strokes for _, y, _ in stroke]
    x_min = min(all_x);  x_max = max(all_x)
    y_min = min(all_y);  y_max = max(all_y)
    x_range = max(x_max - x_min, _MIN_COORD_SCALE)
    y_range = max(y_max - y_min, _MIN_COORD_SCALE)

    norm_strokes = [
        [((x - x_min) / x_range, (y - y_min) / y_range, t) for x, y, t in stroke]
        for stroke in strokes
    ]

    # ── Equidistant spatial resampling ────────────────────────────────────
    resampled = [_resample_stroke(stroke) for stroke in norm_strokes]

    # ── Flatten with pen_up tag ───────────────────────────────────────────
    rows = []
    for stroke in resampled:
        for i, (x, y, t) in enumerate(stroke):
            pen_up = 1.0 if i == len(stroke) - 1 else 0.0
            rows.append((x, y, t, pen_up))

    abs_arr = np.array(rows, dtype=np.float64)   # (N, 4)
    N = len(abs_arr)

    # ── Delta encoding ────────────────────────────────────────────────────
    features = np.zeros((N, 4), dtype=np.float32)
    features[1:, 0] = np.diff(abs_arr[:, 0])   # dx
    features[1:, 1] = np.diff(abs_arr[:, 1])   # dy
    features[1:, 2] = np.diff(abs_arr[:, 2])   # dt  (absolute seconds)
    features[:,  3] = abs_arr[:, 3]             # pen_up

    # ── Pad / truncate to max_points ──────────────────────────────────────
    if N > max_points:
        print(f'[recognition] Warning: {N} points > max_points '
              f'{max_points}, truncating.')
        features = features[:max_points]
    elif N < max_points:
        pad = np.zeros((max_points - N, 4), dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)

    actual_len = min(N, max_points)
    return features[np.newaxis], np.array([actual_len], dtype=np.int64)


# ---------------------------------------------------------------------------
# ── CTC greedy decoder
# ---------------------------------------------------------------------------

def _ctc_greedy_decode(logits: np.ndarray,
                       vocab: dict[int, str],
                       blank_idx: int,
                       seq_len: int | None = None) -> str:
    """
    Greedy CTC decode on a (T, V) or (1, T, V) float array.
    argmax at each timestep → collapse consecutive repeats → strip blank.

    seq_len: true (pre-padding) number of timesteps. Outputs at padded
             timesteps are never supervised during training, so decoding
             past seq_len appends spurious tokens. Always pass it.
    """
    if logits.ndim == 3:
        logits = logits[0]

    if seq_len is not None:
        logits = logits[:seq_len]

    indices = logits.argmax(axis=-1).tolist()

    tokens, prev = [], None
    for idx in indices:
        if idx != prev:
            if idx != blank_idx:
                ch = vocab.get(idx, '')
                if ch and ch != '<blank>':
                    tokens.append(ch)
            prev = idx

    return ''.join(tokens)


# ---------------------------------------------------------------------------
# ── Public API
# ---------------------------------------------------------------------------

def run_recognition(strokes: list[list[tuple]]) -> tuple[str | None, str | None]:
    """
    Full recognition pipeline.
    Returns (inferred_latex, error_str) — one is always None.

    strokes: draw_state.strokes — list of strokes, each a list of (x, y, t).
    The first call is slower (asset load); subsequent calls are fast.
    """
    try:
        # Drop empty strokes first, THEN check for content — a bare tap can
        # produce strokes=[[]] which is truthy but has no points, and would
        # crash _preprocess on min([]) of an empty coordinate list.
        strokes = [s for s in strokes if s]
        if not strokes:
            return None, 'Nothing drawn.'

        session = _get_session()
        points, lengths = _preprocess(strokes, _meta['max_points'])
        actual_len = int(lengths[0])

        # The CTCTransformer export takes TWO inputs:
        #     points           float32 (batch, max_points, 4)
        #     key_padding_mask bool    (batch, max_points)  True = padding
        # and outputs log-probabilities (log_softmax applied in-graph). The
        # legacy model.onnx instead took an int64 lengths tensor as its second
        # input, so pick the feed by inspecting the model's declared type.
        input_names = [inp.name for inp in session.get_inputs()]
        output_name = session.get_outputs()[0].name
        feed = {input_names[0]: points}
        if len(input_names) > 1:
            second = session.get_inputs()[1]
            if 'bool' in second.type:
                mask = (np.arange(_meta['max_points']) >= actual_len)[None, :]  # (1, max_points) bool
                feed[second.name] = mask
            else:  # legacy model.onnx — int64 lengths
                feed[second.name] = lengths

        logits = session.run([output_name], feed)[0]
        # Decode only the real timesteps; padded outputs are unsupervised.
        latex  = _ctc_greedy_decode(
            logits, _vocab, _meta['blank_idx'], seq_len=actual_len
        )

        if not latex.strip():
            return None, 'Model returned an empty expression.'

        return latex, None

    except FileNotFoundError as e:
        return None, f'Missing file — copy model.onnx and vocab.json to the project folder. ({e})'
    except Exception as e:
        # Log the full traceback so shape/type mismatches and other real bugs
        # are diagnosable instead of collapsing into one opaque string.
        import traceback
        traceback.print_exc()
        return None, f'Recognition error: {e}'


# ---------------------------------------------------------------------------
# ── Smoke test  (python recognition.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    # Synthetic strokes: two short pen strokes with timestamps
    test_strokes = [
        [(100.0, 200.0, 0.000),
         (110.0, 205.0, 0.050),
         (120.0, 210.0, 0.100)],
        [(150.0, 200.0, 0.200),
         (160.0, 195.0, 0.250)],
    ]

    print('Running smoke test…')
    latex, err = run_recognition(test_strokes)
    if err:
        print(f'ERROR: {err}')
        sys.exit(1)
    print(f'OK — inferred: {latex}')