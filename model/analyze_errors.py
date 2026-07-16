"""
analyze_errors.py - collect every non-exact-match test prediction and summarize
the common mistake patterns, for final-report / poster examples.

Outputs (default --out-dir error_analysis/):
  test_errors.csv        - one row per mismatch, sorted by token edit distance:
                             sample_id, ref, pred, tok_dist, ref_len,
                             n_sub, n_ins, n_del, char_dist
                           (ascending edit distance -> cleanest single-token
                            errors first, the best illustrative examples)
  test_error_summary.txt - edit-distance histogram + the top substitutions,
                           deletions, and insertions across all mismatches

Run from model/ (venv active):
  python analyze_errors.py
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from model.encoder import CTCTransformer
from vocab import load_vocab, BLANK_IDX
from preprocessing.dataset import MathWritingDataset, MAX_POINTS
from postprocessing.ctc_decode import greedy_ctc_decode, edit_distance


def align_ops(ref, pred):
    """Levenshtein with backtrace over token-index lists.
    Returns a list of ('match'|'sub'|'del'|'ins', ref_tok|None, pred_tok|None)."""
    m, n = len(ref), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == pred[j - 1] else 1):
            ops.append(('match' if ref[i - 1] == pred[j - 1] else 'sub', ref[i - 1], pred[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(('del', ref[i - 1], None)); i -= 1
        else:
            ops.append(('ins', None, pred[j - 1])); j -= 1
    ops.reverse()
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='best_ctc_transformer.pt')
    ap.add_argument('--vocab',      default='vocab.json')
    ap.add_argument('--test-dir',   default='mathwriting-2024/test')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--out-dir',    default='error_analysis')
    ap.add_argument('--top',        type=int, default=30, help='top-N patterns to report')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok2idx, idx2tok, meta = load_vocab(args.vocab)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CTCTransformer(
        vocab_size=ckpt['vocab_size'], max_points=ckpt['max_points'],
        num_layers=ckpt['num_layers'], num_heads=ckpt['num_heads'],
        ffn_num_hidden=ckpt['ffn_num_hidden'], embed_dim=ckpt['embed_dim'],
        dropout=ckpt['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    def to_str(seq):
        return ''.join(idx2tok.get(i, '?') for i in seq)

    test_paths = sorted(Path(args.test_dir).glob('*.inkml'))
    assert test_paths, f'no .inkml files found in {args.test_dir}'
    ds = MathWritingDataset(test_paths, tok2idx, max_points=MAX_POINTS)
    print(f'[errors] scoring {len(test_paths)} test files on {device} …')

    rows = []
    subs, dels, inss = Counter(), Counter(), Counter()
    dist_hist = Counter()
    evaluated = mismatches = 0

    def flush(buf):
        """buf: list of (sample_id, features, ref_encoded). Run model + record errors."""
        nonlocal evaluated, mismatches
        if not buf:
            return
        lengths = [len(f) for _, f, _ in buf]
        T = max(lengths)
        padded = np.zeros((len(buf), T, 4), dtype=np.float32)
        for i, (_, f, _) in enumerate(buf):
            padded[i, :len(f)] = f
        inputs = torch.from_numpy(padded).to(device)
        input_lengths = torch.tensor(lengths)
        mask = (torch.arange(T)[None, :] >= input_lengths[:, None]).to(device)
        with torch.no_grad():
            logits = model(inputs, key_padding_mask=mask)
        preds = greedy_ctc_decode(logits, input_lengths, BLANK_IDX)

        for (sid, _, ref), pred in zip(buf, preds):
            evaluated += 1
            if pred == ref:
                continue
            mismatches += 1
            ops = align_ops(ref, pred)
            n_sub = n_ins = n_del = 0
            for kind, a, b in ops:
                if kind == 'sub':
                    subs[(idx2tok.get(a, '?'), idx2tok.get(b, '?'))] += 1; n_sub += 1
                elif kind == 'del':
                    dels[idx2tok.get(a, '?')] += 1; n_del += 1
                elif kind == 'ins':
                    inss[idx2tok.get(b, '?')] += 1; n_ins += 1
            tok_dist = n_sub + n_ins + n_del
            dist_hist[tok_dist] += 1
            ref_s, pred_s = to_str(ref), to_str(pred)
            rows.append({
                'sample_id': sid, 'ref': ref_s, 'pred': pred_s,
                'tok_dist': tok_dist, 'ref_len': len(ref),
                'n_sub': n_sub, 'n_ins': n_ins, 'n_del': n_del,
                'char_dist': edit_distance(list(pred_s), list(ref_s)),
            })

    buf = []
    for idx in range(len(ds)):
        item = ds[idx]
        if item is None:
            continue
        buf.append((test_paths[idx].stem, item[0], item[1]))
        if len(buf) >= args.batch_size:
            flush(buf); buf = []
        if (idx + 1) % 1000 == 0:
            print(f'  {idx + 1}/{len(test_paths)} …')
    flush(buf)

    # -- write CSV (sorted by token edit distance, cleanest errors first) ------
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r['tok_dist'], r['ref_len']))
    csv_path = out / 'test_errors.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # -- write summary --------------------------------------------------------
    lines = []
    lines.append(f'Test error analysis — {mismatches} mismatches / {evaluated} evaluated '
                 f'({100 * mismatches / max(evaluated, 1):.1f}%)\n')
    lines.append('Token edit-distance histogram (how far off each mismatch is):')
    for d in sorted(dist_hist):
        bar = '#' * min(60, dist_hist[d] * 60 // max(dist_hist.values()))
        lines.append(f'  dist {d:>3}: {dist_hist[d]:>5}  {bar}')
    lines.append(f'\nTop {args.top} SUBSTITUTIONS  (ref -> pred):')
    for (a, b), c in subs.most_common(args.top):
        lines.append(f'  {c:>5}   {a!r:>12} -> {b!r}')
    lines.append(f'\nTop {args.top} DELETIONS  (ref token the model dropped):')
    for a, c in dels.most_common(args.top):
        lines.append(f'  {c:>5}   {a!r}')
    lines.append(f'\nTop {args.top} INSERTIONS  (token the model added):')
    for b, c in inss.most_common(args.top):
        lines.append(f'  {c:>5}   {b!r}')
    summary = '\n'.join(lines)

    with open(out / 'test_error_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary + '\n')

    print('\n' + summary)
    print(f'\n[errors] wrote {len(rows)} rows -> {csv_path}')
    print(f'[errors] summary       -> {out / "test_error_summary.txt"}')


if __name__ == '__main__':
    main()
