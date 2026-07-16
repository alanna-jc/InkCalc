"""
make_poster_examples.py - pick representative error examples per category,
render each sample's handwriting to inline SVG, and emit a poster/report HTML
table + a CSV.

Categories:
  case        - single-token upper/lower-case confusion (x<->X, P<->p, ...)
  lookalike   - single-token visual look-alikes (Greek/Latin, l/1, o/0, ./\\cdot)
  structural  - dropped/added braces { } and sub/superscript markers _ ^

Run from model/ (after analyze_errors.py has produced test_errors.csv):
  python make_poster_examples.py
"""
import argparse
import csv
import html
from pathlib import Path

from vocab import load_vocab, tokenize_label
from preprocessing.inkml_parser import InkMLParser, InkMLParseError

STRUCT = {'{', '}', '_', '^'}


def align_ops(ref, pred):
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
    ops, i, j = [], m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == pred[j - 1] else 1):
            ops.append(('match' if ref[i - 1] == pred[j - 1] else 'sub', ref[i - 1], pred[j - 1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(('del', ref[i - 1], None)); i -= 1
        else:
            ops.append(('ins', None, pred[j - 1])); j -= 1
    ops.reverse()
    return ops


def categorize(ref_toks, pred_toks):
    ops = align_ops(ref_toks, pred_toks)
    diff = [o for o in ops if o[0] != 'match']
    subs = [(a, b) for k, a, b in diff if k == 'sub']
    dels = [a for k, a, b in diff if k == 'del']
    inss = [b for k, a, b in diff if k == 'ins']

    if len(diff) == 1 and len(subs) == 1:
        a, b = subs[0]
        if len(a) == 1 and len(b) == 1 and a.isalpha() and b.isalpha() and a.lower() == b.lower():
            return 'case', ops, a.lower()                      # dedupe key = letter
        return 'lookalike', ops, frozenset((a, b))             # dedupe key = pair
    involved = set(dels) | set(inss) | {t for p in subs for t in p}
    if involved and involved <= STRUCT and not any(len(d) > 1 for d in diff if isinstance(d, str)):
        sig = (tuple(sorted(dels)), tuple(sorted(inss)))
        return 'structural', ops, sig
    return None, ops, None


def note_for(ops):
    diff = [o for o in ops if o[0] != 'match']
    parts = []
    for k, a, b in diff[:4]:
        if k == 'sub':
            parts.append(f"‘{a}’ read as ‘{b}’")
        elif k == 'del':
            parts.append(f"dropped ‘{a}’")
        elif k == 'ins':
            parts.append(f"added ‘{b}’")
    return '; '.join(parts)


def highlight(ops, side):
    """Rebuild the LaTeX string as HTML, wrapping differing tokens in a diff span."""
    out = []
    for k, a, b in ops:
        if k == 'match':
            out.append(html.escape(a))
        elif k == 'sub':
            tok = a if side == 'ref' else b
            out.append(f'<span class="d">{html.escape(tok)}</span>')
        elif k == 'del' and side == 'ref':
            out.append(f'<span class="d">{html.escape(a)}</span>')
        elif k == 'ins' and side == 'pred':
            out.append(f'<span class="d">{html.escape(b)}</span>')
    return ''.join(out) or '<span class="empty">(empty)</span>'


def render_svg(strokes, target_h=64, pad=7):
    pts = [(p.x, p.y) for s in strokes for p in s.points]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    w, h = max(maxx - minx, 1e-6), max(maxy - miny, 1e-6)
    scale = target_h / h
    def tx(x): return (x - minx) * scale + pad
    def ty(y): return (y - miny) * scale + pad          # InkML is y-down => upright in SVG
    lines = []
    for s in strokes:
        p = ' '.join(f'{tx(pt.x):.1f},{ty(pt.y):.1f}' for pt in s.points)
        lines.append(f'<polyline points="{p}"/>')
    W, H = w * scale + 2 * pad, target_h + 2 * pad
    return (f'<svg viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" '
            f'class="ink" role="img">{"".join(lines)}</svg>')


CATS = {
    'case': ('Case confusion',
             'Upper- vs lower-case letters. Coordinate normalization to [0,1] removes '
             'absolute size — the main visual cue for case — so the model often cannot tell them apart.'),
    'lookalike': ('Symbol look-alikes',
                  'Visually similar glyphs: Greek vs Latin (ω/w, κ/k, ρ/p), letters vs digits '
                  '(l/1, o/0), and the decimal dot vs multiplication (· / .).'),
    'structural': ('Structure &amp; grouping',
                   'Dropped or added braces { } and sub/superscript markers _ ^ — the model mis-handles '
                   'nesting, producing mismatched grouping rather than wrong symbols.'),
    'catastrophic': ('Catastrophic failures',
                     'The hardest cases — long or dense expressions (nested scripts, stacked fractions, '
                     'matrices) where recognition breaks down and the output diverges substantially from '
                     'the target. Rare, but they set the tail of the error distribution.'),
}

CSS = """
:root{
  --bg:#f5f6f8; --panel:#ffffff; --text:#1b2029; --muted:#5c6673; --line:#e3e6eb;
  --accent:#0f766e; --accent-soft:#dcefec; --ref:#0b7a4b; --err:#c2410c;
  --paper:#ffffff; --stroke:#161a20;
}
@media (prefers-color-scheme:dark){
  :root{ --bg:#0e1218; --panel:#161b22; --text:#e7ebf0; --muted:#98a2b0; --line:#28303a;
         --accent:#2dd4bf; --accent-soft:#12312e; --ref:#4ade80; --err:#fb923c;
         --paper:#ffffff; --stroke:#161a20; }
}
:root[data-theme="light"]{ --bg:#f5f6f8; --panel:#ffffff; --text:#1b2029; --muted:#5c6673;
  --line:#e3e6eb; --accent:#0f766e; --accent-soft:#dcefec; --ref:#0b7a4b; --err:#c2410c;
  --paper:#ffffff; --stroke:#161a20; }
:root[data-theme="dark"]{ --bg:#0e1218; --panel:#161b22; --text:#e7ebf0; --muted:#98a2b0;
  --line:#28303a; --accent:#2dd4bf; --accent-soft:#12312e; --ref:#4ade80; --err:#fb923c;
  --paper:#ffffff; --stroke:#161a20; }

*{ box-sizing:border-box; }
body{ margin:0; background:var(--bg); color:var(--text);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; line-height:1.5; }
.wrap{ max-width:1000px; margin:0 auto; padding:40px 24px 64px; }
.eyebrow{ text-transform:uppercase; letter-spacing:.12em; font-size:12px; font-weight:600;
  color:var(--accent); margin:0 0 8px; }
h1{ font-size:30px; line-height:1.15; margin:0 0 8px; text-wrap:balance; letter-spacing:-.01em; }
.sub{ color:var(--muted); margin:0 0 28px; max-width:65ch; }
.stat{ background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:10px; padding:16px 20px; margin:0 0 40px; }
.stat b{ color:var(--accent); }
h2{ font-size:20px; margin:36px 0 4px; letter-spacing:-.01em; }
.rootcause{ color:var(--muted); font-size:14px; margin:0 0 16px; max-width:70ch; }
.tablewrap{ overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:var(--panel); }
table{ border-collapse:collapse; width:100%; font-size:14px; }
th{ text-align:left; text-transform:uppercase; letter-spacing:.06em; font-size:11px; color:var(--muted);
  font-weight:600; padding:11px 14px; border-bottom:1px solid var(--line); white-space:nowrap; }
td{ padding:12px 14px; border-bottom:1px solid var(--line); vertical-align:middle; }
tr:last-child td{ border-bottom:none; }
.paper{ background:var(--paper); border:1px solid var(--line); border-radius:7px; padding:5px 8px;
  display:inline-block; }
.ink{ display:block; }
.ink polyline{ fill:none; stroke:var(--stroke); stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }
.latex{ font-family:ui-monospace,"Cascadia Code","Consolas",monospace; font-size:13px;
  white-space:nowrap; }
.ref{ color:var(--ref); }
.pred{ color:var(--text); }
.latex.wrap{ white-space:normal; word-break:break-word; max-width:280px; display:inline-block; }
.latex .d{ background:var(--accent-soft); border-radius:3px; padding:0 2px; }
.pred .d{ color:var(--err); font-weight:700; background:transparent; text-decoration:underline;
  text-decoration-color:var(--err); text-underline-offset:2px; }
.ref .d{ font-weight:700; }
.note{ color:var(--muted); font-size:13px; }
.empty{ color:var(--muted); font-style:italic; }
.id{ font-family:ui-monospace,monospace; font-size:11px; color:var(--muted); }
footer{ margin-top:40px; color:var(--muted); font-size:12px; border-top:1px solid var(--line); padding-top:16px; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--errors-csv', default='error_analysis/test_errors.csv')
    ap.add_argument('--test-dir', default='mathwriting-2024/test')
    ap.add_argument('--vocab', default='vocab.json')
    ap.add_argument('--per-category', type=int, default=5)
    ap.add_argument('--catastrophic', type=int, default=4)
    ap.add_argument('--out', default='error_analysis/poster_examples.html')
    ap.add_argument('--csv-out', default='error_analysis/poster_examples.csv')
    args = ap.parse_args()

    load_vocab(args.vocab)  # not strictly needed, but validates the vocab is present
    parser = InkMLParser(require_time=False)

    rows = list(csv.DictReader(open(args.errors_csv, encoding='utf-8')))
    total = len(rows)
    d1 = sum(1 for r in rows if int(r['tok_dist']) == 1)
    d3 = sum(1 for r in rows if int(r['tok_dist']) <= 3)

    picked = {'case': [], 'lookalike': [], 'structural': [], 'catastrophic': []}
    seen_keys = {'case': set(), 'lookalike': set(), 'structural': set()}

    for r in rows:  # already sorted by tok_dist ascending -> cleanest first
        cat, ops, key = categorize(tokenize_label(r['ref']), tokenize_label(r['pred']))
        if cat is None or len(picked[cat]) >= args.per_category or key in seen_keys[cat]:
            continue
        ink = args.test_dir + '/' + r['sample_id'] + '.inkml'
        try:
            sample = parser.parse(Path(ink))
            svg = render_svg(sample.strokes)
        except (InkMLParseError, FileNotFoundError, ValueError):
            continue
        seen_keys[cat].add(key)
        picked[cat].append({
            'sample_id': r['sample_id'], 'ref': r['ref'], 'pred': r['pred'],
            'tok_dist': r['tok_dist'], 'ops': ops, 'svg': svg, 'note': note_for(ops),
        })

    # Catastrophic: the worst mismatches (highest token edit distance). Rendered
    # plain (ops=None) since nearly every token differs.
    seen_ref = set()
    for r in sorted(rows, key=lambda x: -int(x['tok_dist'])):
        if len(picked['catastrophic']) >= args.catastrophic:
            break
        if r['ref'] in seen_ref:
            continue
        ink = args.test_dir + '/' + r['sample_id'] + '.inkml'
        try:
            sample = parser.parse(Path(ink))
            svg = render_svg(sample.strokes)
        except (InkMLParseError, FileNotFoundError, ValueError):
            continue
        seen_ref.add(r['ref'])
        picked['catastrophic'].append({
            'sample_id': r['sample_id'], 'ref': r['ref'], 'pred': r['pred'],
            'tok_dist': r['tok_dist'], 'ops': None, 'svg': svg,
            'note': f"{r['tok_dist']} tokens off (reference is {r['ref_len']} tokens)",
        })

    # -- HTML ---------------------------------------------------------------
    parts = ['<title>InkCalc — Recognition Error Modes</title>',
             f'<style>{CSS}</style>', '<main class="wrap">']
    parts.append('<p class="eyebrow">InkCalc · model evaluation</p>')
    parts.append('<h1>Handwriting recognition: representative error modes</h1>')
    parts.append('<p class="sub">Where the CTC-Transformer\'s greedy decode disagrees with the '
                 'ground-truth LaTeX on the MathWriting test set, shown with the original ink. '
                 'The mispredicted token is highlighted.</p>')
    parts.append(f'<div class="stat"><b>{100*d1/total:.0f}% of misses are a single token</b>, and '
                 f'{100*d3/total:.0f}% are within three — errors are usually small, not catastrophic '
                 f'({total:,} mismatches over 7,644 test files).</div>')

    for cat in ('case', 'lookalike', 'structural', 'catastrophic'):
        title, root = CATS[cat]
        parts.append(f'<h2>{title}</h2><p class="rootcause">{root}</p>')
        parts.append('<div class="tablewrap"><table><thead><tr>'
                     '<th>Handwriting</th><th>Reference (truth)</th>'
                     '<th>Model prediction</th><th>What went wrong</th></tr></thead><tbody>')
        for e in picked[cat]:
            if e['ops'] is None:                       # catastrophic: plain, wrapping
                ref_in, pred_in, wrap = html.escape(e['ref']), html.escape(e['pred']), ' wrap'
            else:
                ref_in, pred_in, wrap = highlight(e['ops'], 'ref'), highlight(e['ops'], 'pred'), ''
            parts.append(
                '<tr>'
                f'<td><span class="paper">{e["svg"]}</span><div class="id">{html.escape(e["sample_id"])}</div></td>'
                f'<td class="latex ref{wrap}">{ref_in}</td>'
                f'<td class="latex pred{wrap}">{pred_in}</td>'
                f'<td class="note">{html.escape(e["note"])}</td>'
                '</tr>'
            )
        parts.append('</tbody></table></div>')

    parts.append('<footer>Generated by make_poster_examples.py from test_errors.csv. '
                 'Ink rendered directly from the source InkML strokes.</footer>')
    parts.append('</main>')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(parts), encoding='utf-8')

    # -- CSV ----------------------------------------------------------------
    with open(args.csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['category', 'sample_id', 'ref', 'pred', 'tok_dist', 'note'])
        for cat in ('case', 'lookalike', 'structural', 'catastrophic'):
            for e in picked[cat]:
                w.writerow([cat, e['sample_id'], e['ref'], e['pred'], e['tok_dist'], e['note']])

    n = sum(len(v) for v in picked.values())
    print(f'[poster] selected {n} examples '
          f'(case={len(picked["case"])}, lookalike={len(picked["lookalike"])}, '
          f'structural={len(picked["structural"])})')
    print(f'[poster] HTML -> {out}')
    print(f'[poster] CSV  -> {args.csv_out}')


if __name__ == '__main__':
    main()
