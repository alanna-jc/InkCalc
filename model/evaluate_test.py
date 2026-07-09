"""
evaluate_test.py - evaluate a trained CTCTransformer on the MathWriting test split.

Reports:
  - Test CER (token) : token-level error rate == training's CER metric
                       (sum edit distance / sum reference tokens)
  - Test CER (char)  : true character-level error rate over the LaTeX strings
  - Test EM          : exact-match accuracy (fraction decoded exactly right)
  - Coverage         : evaluated vs dropped test files (parse / feature / OOV)

Run from model/ (venv active):
  python evaluate_test.py
  python evaluate_test.py --examples 15     # also print some pred-vs-ref samples
"""
import argparse
from pathlib import Path

import torch

from model.encoder import CTCTransformer
from vocab import load_vocab, BLANK_IDX
from preprocessing.dataset import build_dataloader, MAX_POINTS
from postprocessing.ctc_decode import greedy_ctc_decode, edit_distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='best_ctc_transformer.pt')
    parser.add_argument('--vocab',      default='vocab.json')
    parser.add_argument('--test-dir',   default='mathwriting-2024/test')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--examples',   type=int, default=0,
                        help='print this many decoded pred-vs-ref samples')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[eval] device: {device}')

    # -- vocab + model -------------------------------------------------------
    tok2idx, idx2tok, meta = load_vocab(args.vocab)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CTCTransformer(
        vocab_size     = ckpt['vocab_size'],
        max_points     = ckpt['max_points'],
        num_layers     = ckpt['num_layers'],
        num_heads      = ckpt['num_heads'],
        ffn_num_hidden = ckpt['ffn_num_hidden'],
        embed_dim      = ckpt['embed_dim'],
        dropout        = ckpt['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[eval] loaded checkpoint (epoch {ckpt.get('epoch')}, "
          f"val CER {ckpt.get('valid_cer', float('nan')):.4f})")

    # -- data ----------------------------------------------------------------
    test_paths = sorted(Path(args.test_dir).glob('*.inkml'))
    assert test_paths, f'no .inkml files found in {args.test_dir}'
    total_files = len(test_paths)
    test_loader = build_dataloader(
        test_paths, tok2idx, batch_size=args.batch_size,
        shuffle=False, num_workers=8,
    )

    def to_str(seq):
        return ''.join(idx2tok.get(i, '?') for i in seq)

    tok_edits = tok_ref = char_edits = char_ref = exact = evaluated = 0
    shown = 0

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            inputs           = batch['inputs'].to(device)
            input_lengths    = batch['input_lengths']       # CPU (CTC length arg)
            target_lengths   = batch['target_lengths']      # CPU
            key_padding_mask = batch['key_padding_mask'].to(device)

            logits = model(inputs, key_padding_mask=key_padding_mask)
            preds  = greedy_ctc_decode(logits, input_lengths, BLANK_IDX)

            targets = batch['targets']   # concatenated CPU int64
            offset = 0
            for b, tlen in enumerate(target_lengths.tolist()):
                ref  = targets[offset:offset + tlen].tolist()
                offset += tlen
                pred = preds[b]

                # token-level (matches the training CER metric)
                tok_edits += edit_distance(pred, ref)
                tok_ref   += tlen

                # char-level (true CER over the reconstructed LaTeX strings)
                ref_s, pred_s = to_str(ref), to_str(pred)
                char_edits += edit_distance(list(pred_s), list(ref_s))
                char_ref   += len(ref_s)

                if pred == ref:
                    exact += 1
                evaluated += 1

                if shown < args.examples:
                    mark = 'OK  ' if pred == ref else 'DIFF'
                    print(f'  [{mark}] ref : {ref_s}')
                    print(f'         pred: {pred_s}')
                    shown += 1

    tok_cer  = tok_edits / max(tok_ref, 1)
    char_cer = char_edits / max(char_ref, 1)
    em       = exact / max(evaluated, 1)
    dropped  = total_files - evaluated

    print('\n==================== TEST RESULTS ====================')
    print(f'  test files             : {total_files}')
    print(f'  evaluated              : {evaluated}')
    print(f'  dropped (parse/feat/OOV): {dropped} ({100 * dropped / total_files:.2f}%)')
    print(f'  Test CER (token-level) : {tok_cer:.4f}   <- comparable to training val CER')
    print(f'  Test EM  (exact match) : {em:.4f}   ({exact}/{evaluated})')
    print('======================================================')


if __name__ == '__main__':
    main()
