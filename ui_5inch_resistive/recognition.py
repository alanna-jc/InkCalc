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

def _preprocess(strokes: list[list[tuple]], max_points: int):
    """
    Strokes → (points_tensor, lengths_tensor) ready for ONNX inference.

    strokes is draw_state.strokes: [ [(x, y, t), …], … ]
    where x, y are screen pixel coords and t is time.time() epoch seconds.

    Feature layout per point — matches training repo inkml.py exactly:
        [dx, dy, dt, pen_up]
        dx / dy  : normalised spatial delta from previous point
        dt       : normalised time delta from previous point
        pen_up   : 1.0 on the last point of each stroke, else 0.0

    Normalisation makes the model invariant to screen resolution, drawing
    size, and drawing speed — matching what the training parser applied to
    the raw MathWriting InkML files before any sample was fed to the model.
    """
    # ── Gap interpolation ─────────────────────────────────────────────
    # Insert linearly interpolated points wherever consecutive samples
    # are further apart than INTERP_DIST_PX. This compensates for events
    # dropped by the touchscreen digitizer during fast strokes, making
    # the input distribution closer to the evenly-sampled MathWriting
    # training data. x, y, and t are all interpolated linearly.
    # Tune independently of the renderer's INTERP_DIST in ui.py.
    INTERP_DIST_PX = 4

    filled = []
    for stroke in strokes:
        if not stroke:
            continue
        filled_stroke = [stroke[0]]
        for j in range(1, len(stroke)):
            x0, y0, t0 = stroke[j - 1]
            x1, y1, t1 = stroke[j]
            dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            if dist > INTERP_DIST_PX:
                steps = int(dist / INTERP_DIST_PX)
                for s in range(1, steps):
                    alpha = s / steps
                    filled_stroke.append((
                        x0 + (x1 - x0) * alpha,
                        y0 + (y1 - y0) * alpha,
                        t0 + (t1 - t0) * alpha,
                    ))
            filled_stroke.append(stroke[j])
        filled.append(filled_stroke)
    strokes = filled

    # Flatten all strokes into rows, tagging the last point of each stroke
    rows = []
    for stroke in strokes:
        for i, (x, y, t) in enumerate(stroke):
            pen_up = 1.0 if i == len(stroke) - 1 else 0.0
            rows.append([x, y, t, pen_up])

    arr = np.array(rows, dtype=np.float32)   # (N, 4)
    N   = len(arr)

    # ── Spatial normalisation ─────────────────────────────────────────────
    # Scale x and y independently to [0, 1] based on the bounding box of
    # this drawing. Makes the model invariant to position on screen and to
    # how large or small the user writes.
    x_min, x_max = arr[:, 0].min(), arr[:, 0].max()
    y_min, y_max = arr[:, 1].min(), arr[:, 1].max()
    x_range = max(float(x_max - x_min), 1.0)
    y_range = max(float(y_max - y_min), 1.0)
    arr[:, 0] = (arr[:, 0] - x_min) / x_range
    arr[:, 1] = (arr[:, 1] - y_min) / y_range

    # ── Spatial deltas ────────────────────────────────────────────────────
    # Replace absolute normalised coords with the delta from the previous
    # point. The model learns pen movement patterns, not positions.
    # The first point has no predecessor so its delta is 0.
    arr[1:, 0] = np.diff(arr[:, 0])
    arr[1:, 1] = np.diff(arr[:, 1])
    arr[0,  0] = 0.0
    arr[0,  1] = 0.0

    # ── Temporal normalisation ────────────────────────────────────────────
    # Convert raw timestamps to deltas, then scale by total session duration
    # so the model is invariant to whether the user drew quickly or slowly.
    # Unit (epoch seconds vs ms) cancels out because we divide by t_total.
    raw_t   = arr[:, 2].copy()
    t_total = max(float(raw_t[-1] - raw_t[0]), 1.0)
    dt      = np.empty(N, dtype=np.float32)
    dt[0]   = 0.0
    dt[1:]  = np.diff(raw_t) / t_total
    arr[:, 2] = dt

    # ── Pad / truncate to max_points ──────────────────────────────────────
    # The model expects a fixed-length tensor. Real data length is passed
    # separately as input_lengths so the BiGRU ignores padding.
    if N > max_points:
        print(f'[recognition] Warning: {N} points > max_points '
              f'{max_points}, truncating.')
        arr = arr[:max_points]
    elif N < max_points:
        pad = np.zeros((max_points - N, 4), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)

    actual_len = min(N, max_points)
    return arr[np.newaxis], np.array([actual_len], dtype=np.int64)


# ---------------------------------------------------------------------------
# ── CTC greedy decoder
# ---------------------------------------------------------------------------

def _ctc_greedy_decode(logits: np.ndarray,
                       vocab: dict[int, str],
                       blank_idx: int) -> str:
    """
    Greedy CTC decode on a (T, V) or (1, T, V) float array.
    argmax at each timestep → collapse consecutive repeats → strip blank.
    """
    if logits.ndim == 3:
        logits = logits[0]

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
        if not strokes:
            return None, 'Nothing drawn.'

        session = _get_session()
        points, lengths = _preprocess(strokes, _meta['max_points'])

        input_names = [inp.name for inp in session.get_inputs()]
        output_name = session.get_outputs()[0].name
        feed = {input_names[0]: points}
        if len(input_names) > 1:
            feed[input_names[1]] = lengths

        logits = session.run([output_name], feed)[0]
        latex  = _ctc_greedy_decode(logits, _vocab, _meta['blank_idx'])

        if not latex.strip():
            return None, 'Model returned an empty expression.'

        return latex, None

    except FileNotFoundError as e:
        return None, f'Missing file — copy model.onnx and vocab.json to the project folder. ({e})'
    except Exception as e:
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