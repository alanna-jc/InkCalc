"""
export_onnx.py  —  Run this ONCE on the training machine (not the Pi).
=======================================================================
Produces two files to copy to the Pi:
  model.onnx          — self-contained model, no PyTorch needed at runtime
  vocab.json          — index->token mapping for CTC decoding + meta

Usage (from the repo root):
  python export_onnx.py \
      --checkpoint model/best_ctc_transformer.pt \
      --vocab      model/vocab.json \
      --out-dir    export/

The exported model has TWO inputs:
  points           float32 (batch, max_points, 4)   [dx, dy, dt, pen_state], zero-padded
  key_padding_mask bool    (batch, max_points)      True where padding

recognition.py on the Pi must build the mask after padding:
  mask = np.arange(max_points) >= actual_len        # shape (max_points,), then add batch dim

Output:
  log_probs        float32 (batch, max_points, vocab_size)  — log-softmax already applied

Then scp export/ to the Pi.
"""

import argparse
import json
import pathlib
import sys

import torch
from torch import nn

# training_loop.py runs with model/ as its root, so the package layout is
# model.encoder / model.positional_encoding. Add model/ to the path so the
# same imports resolve from the repo root here.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'model'))

from model.encoder import CTCTransformer


class ExportWrapper(nn.Module):
    """
    Wraps CTCTransformer for ONNX export:
      - takes the padding mask as an explicit input (ONNX graphs can't
        conveniently build it from lengths across all runtimes)
      - applies log_softmax so the Pi gets ready-to-decode log-probabilities
    """
    def __init__(self, model: CTCTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, points: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        logits = self.model(points, key_padding_mask=key_padding_mask)
        return logits.log_softmax(dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='model/best_ctc_transformer.pt')
    parser.add_argument('--vocab',      default='model/vocab.json')
    parser.add_argument('--out-dir',    default='export')
    parser.add_argument('--opset', type=int, default=14)
    args = parser.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load checkpoint (dict written by training_loop.py) ────────────────
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    print('Checkpoint keys:', list(ckpt.keys()))

    required = ('model_state_dict', 'embed_dim', 'num_layers', 'num_heads',
                'ffn_num_hidden', 'dropout', 'max_points', 'vocab_size')
    missing = [k for k in required if k not in ckpt]
    if missing:
        sys.exit(f'Checkpoint is missing keys {missing}. '
                 f'Was it saved by the current training_loop.py?')

    max_points = int(ckpt['max_points'])
    vocab_size = int(ckpt['vocab_size'])
    input_dim  = int(ckpt.get('input_dim', 4))
    print(f"  embed_dim={ckpt['embed_dim']}  layers={ckpt['num_layers']}  "
          f"heads={ckpt['num_heads']}  max_points={max_points}  vocab_size={vocab_size}  "
          f"epoch={ckpt.get('epoch')}  valid_loss={ckpt.get('valid_loss')}")

    # ── 2. Rebuild the exact architecture and load weights ───────────────────
    model = CTCTransformer(
        vocab_size     = vocab_size,
        max_points     = max_points,
        num_layers     = int(ckpt['num_layers']),
        num_heads      = int(ckpt['num_heads']),
        ffn_num_hidden = int(ckpt['ffn_num_hidden']),
        embed_dim      = int(ckpt['embed_dim']),
        dropout        = float(ckpt['dropout']),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print('Model loaded.')

    export_model = ExportWrapper(model)
    export_model.eval()

    # ── 3. Export to ONNX ─────────────────────────────────────────────────────
    # Pi pads/truncates to exactly max_points, so fixed seq_len keeps it simple.
    dummy_points = torch.zeros(1, max_points, input_dim, dtype=torch.float32)
    dummy_mask   = torch.zeros(1, max_points, dtype=torch.bool)
    onnx_path = out / 'model.onnx'

    torch.onnx.export(
        export_model,
        (dummy_points, dummy_mask),
        str(onnx_path),
        input_names=['points', 'key_padding_mask'],
        output_names=['log_probs'],
        dynamic_axes={
            'points':           {0: 'batch'},
            'key_padding_mask': {0: 'batch'},
            'log_probs':        {0: 'batch'},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,   # use the legacy TorchScript exporter: no onnxscript dep,
                        # honors opset_version, and yields a graph the Pi's
                        # (often older) onnxruntime handles reliably.
    )
    print(f'Exported -> {onnx_path}')

    # ── 4. Copy vocab.json through, refreshing meta from the checkpoint ──────
    # vocab.json was written by vocab.save_vocab() as {'vocab': {...}, 'meta': {...}}.
    with open(args.vocab, 'r', encoding='utf-8') as f:
        bundle = json.load(f)

    idx_to_tok = bundle['vocab']
    if '0' not in idx_to_tok:
        print('Warning: no entry for index 0 in vocab; inserting <blank>.')
        idx_to_tok['0'] = '<blank>'

    bundle['meta'] = {
        'max_points': max_points,
        'vocab_size': vocab_size,
        'blank_idx':  int(ckpt.get('blank_idx', 0)),
        'input_dim':  input_dim,
        'features':   ['dx', 'dy', 'dt', 'pen_state'],
        'input_names':  ['points', 'key_padding_mask'],
        'output_name':  'log_probs',
        'checkpoint_epoch':      ckpt.get('epoch'),
        'checkpoint_valid_loss': ckpt.get('valid_loss'),
    }

    vocab_path = out / 'vocab.json'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    print(f'Exported -> {vocab_path}')

    # ── 5. Quick sanity check ─────────────────────────────────────────────────
    print('\nRunning forward pass sanity check…')
    with torch.no_grad():
        out_tensor = export_model(dummy_points, dummy_mask)
    print(f'  points shape:    {tuple(dummy_points.shape)}')
    print(f'  mask shape:      {tuple(dummy_mask.shape)}')
    print(f'  log_probs shape: {tuple(out_tensor.shape)}')
    assert out_tensor.shape == (1, max_points, vocab_size), 'unexpected output shape'
    print('\nDone. Copy the export/ folder to the Pi.')
    print('  Pi install:  pip install numpy onnxruntime')


if __name__ == '__main__':
    main()
