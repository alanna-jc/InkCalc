"""
parity_check.py - verify the exported ONNX model matches the PyTorch model.

Runs the SAME real test samples (with real padding) through both the PyTorch
CTCTransformer and the exported ONNX graph, then checks that:
  - the log-probabilities agree to within a small float32 tolerance, and
  - the per-timestep argmax is identical (this is what CTC greedy decoding uses,
    so identical argmax => identical predictions).

Use this after every re-export to confirm export_onnx.py didn't change behavior
(e.g. the MultiheadAttention masked_fill graph rewrite is benign only if this
still passes). Exits non-zero if any sample fails, so it can gate a deploy.

Run from model/ (venv active, needs onnxruntime installed):
  python parity_check.py
  python parity_check.py --num-samples 20 --tol 1e-3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from model.encoder import CTCTransformer
from vocab import load_vocab
from preprocessing.dataset import MathWritingDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='best_ctc_transformer.pt')
    parser.add_argument('--vocab',      default='vocab.json')
    parser.add_argument('--onnx',       default='../export/model.onnx')
    parser.add_argument('--test-dir',   default='mathwriting-2024/test')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='how many real test samples to compare')
    parser.add_argument('--tol', type=float, default=1e-3,
                        help='max allowed |log_prob difference| over real timesteps')
    args = parser.parse_args()

    import onnxruntime as ort   # imported here so the file still parses without it

    # -- PyTorch model (output wrapped in log_softmax, exactly like ExportWrapper) --
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = CTCTransformer(
        vocab_size     = ckpt['vocab_size'],
        max_points     = ckpt['max_points'],
        num_layers     = ckpt['num_layers'],
        num_heads      = ckpt['num_heads'],
        ffn_num_hidden = ckpt['ffn_num_hidden'],
        embed_dim      = ckpt['embed_dim'],
        dropout        = ckpt['dropout'],
    ).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    max_points = ckpt['max_points']

    # -- ONNX session --------------------------------------------------------
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])

    # -- collect real test samples (each padded to max_points, with a mask) ---
    tok2idx, idx2tok, meta = load_vocab(args.vocab)
    paths = sorted(Path(args.test_dir).glob('*.inkml'))
    ds = MathWritingDataset(paths[:args.num_samples * 3], tok2idx)  # x3 for dropped ones

    samples = []
    for i in range(len(ds)):
        item = ds[i]
        if item is not None:
            samples.append(item[0])            # features (T, 4)
        if len(samples) >= args.num_samples:
            break
    assert samples, 'no usable test samples found'

    print(f'Comparing {len(samples)} real samples on CPU '
          f'(tolerance {args.tol:.0e})\n')

    worst_diff = 0.0
    all_argmax_ok = True
    failures = 0

    for n, feat in enumerate(samples):
        T = min(len(feat), max_points)
        points = np.zeros((1, max_points, 4), np.float32)
        points[0, :T] = feat[:T]
        mask = (np.arange(max_points) >= T)[None, :]       # True = padding

        with torch.no_grad():
            logits = model(torch.from_numpy(points),
                           key_padding_mask=torch.from_numpy(mask))
            pt = torch.log_softmax(logits, dim=-1).numpy()

        onx = sess.run(['log_probs'],
                       {'points': points, 'key_padding_mask': mask})[0]

        # Compare over the REAL region only — padded outputs are unused by decoding.
        diff = float(np.abs(pt[:, :T] - onx[:, :T]).max())
        argmax_ok = bool((pt[0, :T].argmax(-1) == onx[0, :T].argmax(-1)).all())

        worst_diff = max(worst_diff, diff)
        all_argmax_ok &= argmax_ok
        ok = (diff <= args.tol) and argmax_ok
        failures += (not ok)
        status = 'PASS' if ok else 'FAIL'
        print(f'  [{status}] sample {n:>2}  T={T:>3}  max|diff|={diff:.2e}  '
              f'argmax_match={argmax_ok}')

    print()
    print(f'worst max|diff| over real region : {worst_diff:.2e}')
    print(f'argmax identical on every sample : {all_argmax_ok}')

    if failures:
        print(f'\nRESULT: FAIL ({failures}/{len(samples)} samples exceeded tolerance)')
        sys.exit(1)
    print('\nRESULT: PASS - ONNX model matches PyTorch; safe to deploy.')


if __name__ == '__main__':
    main()
