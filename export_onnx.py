"""
export_onnx.py  —  Run this ONCE on the training machine (not the Pi).
=======================================================================
Produces two files to copy to the Pi:
  model.onnx          — self-contained model, no PyTorch needed at runtime
  vocab.json          — index→character mapping for CTC decoding

Usage (from the root of the handwriting-ctc repo):
  python export_onnx.py \
      --checkpoint checkpoints/best_real_ctc_gpu.pt \
      --vocab     data/mathwriting-vocab.json \
      --out-dir   export/

Then i'll scp export/ to the Pi.
"""

import argparse
import json
import sys
import pathlib

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_real_ctc_gpu.pt')
    parser.add_argument('--vocab',      default='data/mathwriting-vocab.json')
    parser.add_argument('--out-dir',    default='export')
    args = parser.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    print('Checkpoint keys:', list(ckpt.keys()))

    hidden_dim  = ckpt['hidden_dim']
    num_layers  = ckpt['layers']
    max_points  = ckpt['max_points']
    vocab_size  = ckpt.get('vocab_size', 90)
    print(f'  hidden_dim={hidden_dim}  layers={num_layers}  '
          f'max_points={max_points}  vocab_size={vocab_size}')

    # ── 2. Build model and load weights ─────────────────────────────────────
    # Imports from the training repo's src package.
    sys.path.insert(0, str(pathlib.Path(__file__).parent / 'src'))
    from handwriting_ctc.model import CTCRecognizer

    model = CTCRecognizer(
        input_dim=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print('Model loaded.')

    # ── 3. Export to ONNX ────────────────────────────────────────────────────
    # Input is fixed shape (1, max_points, 4) — same as training.
    # The Pi will pad/truncate sequences to max_points before calling the model.
    dummy = torch.zeros(1, max_points, 4)
    onnx_path = out / 'model.onnx'

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=['points'],
        output_names=['logits'],
        # Keep seq_len dynamic in case inference sequences vary in length;
        # if export fails due to control-flow issues, remove dynamic_axes and
        # always pad to exactly max_points (recognition.py already does this).
        dynamic_axes={'points': {1: 'seq_len'}},
        opset_version=14,
        do_constant_folding=True,
    )
    print(f'Exported → {onnx_path}')

    # ── 4. Convert vocab to index→character format ───────────────────────────
    # mathwriting-vocab.json may be {"char": index} or a list.
    # We normalise to {"0": "<blank>", "1": "a", ...} for the Pi.
    with open(args.vocab, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    if isinstance(raw, list):
        # Already ordered list — index is position
        idx_to_char = {str(i): ch for i, ch in enumerate(raw)}
    elif isinstance(raw, dict):
        first_val = next(iter(raw.values()))
        if isinstance(first_val, int):
            # {"char": index} — flip it
            idx_to_char = {str(v): k for k, v in raw.items()}
        else:
            # {"index_str": char} — use as-is
            idx_to_char = {str(k): v for k, v in raw.items()}
    else:
        sys.exit(f'Unexpected vocab format: {type(raw)}')

    # Ensure blank slot exists (CTC blank = index 0)
    if '0' not in idx_to_char:
        print('Warning: no entry for index 0 in vocab; inserting <blank>.')
        idx_to_char['0'] = '<blank>'

    # Save max_points alongside so recognition.py can read it
    export_meta = {
        'max_points': max_points,
        'vocab_size': vocab_size,
        'blank_idx':  0,
    }

    vocab_path = out / 'vocab.json'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({'vocab': idx_to_char, 'meta': export_meta}, f,
                  ensure_ascii=False, indent=2)
    print(f'Exported → {vocab_path}')

    # ── 5. Quick sanity check ────────────────────────────────────────────────
    print('\nRunning forward pass sanity check…')
    with torch.no_grad():
        out_tensor = model(dummy)
    print(f'  Input shape:  {tuple(dummy.shape)}')
    print(f'  Output shape: {tuple(out_tensor.shape)}')
    print('\nDone. Copy the export/ folder to the Pi.')
    print(f'  Pi install:  pip install numpy onnxruntime')


if __name__ == '__main__':
    main()