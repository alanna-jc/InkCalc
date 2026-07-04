"""
analyze_lengths.py — How long are our sequences, really?
Answers: is MAX_POINTS = 512 enough, or are we truncating strokes?

Run from model/:   python analyze_lengths.py ../data/mathwriting-2024/train
"""
import sys
import random
from pathlib import Path

import numpy as np

from model.preprocessing.inkml_parser import InkMLParser, InkMLParseError
from model.preprocessing.feature_extraction import (
    PaperFeatureExtractor, FeatureExtractionConfig, FeatureExtractionError,
)

SAMPLE_SIZE = 5000   # files to check; None = all (slower but exact)

def main():
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/mathwriting-2024/train')
    paths = sorted(data_dir.glob('*.inkml'))
    if not paths:
        sys.exit(f'No .inkml files found in {data_dir}')
    if SAMPLE_SIZE and len(paths) > SAMPLE_SIZE:
        paths = random.sample(paths, SAMPLE_SIZE)   # random subset is fine for stats

    # Same config as dataset.py — spatial_step must match or lengths are wrong
    parser    = InkMLParser(require_time=True, convert_time_to_seconds=True)
    extractor = PaperFeatureExtractor(FeatureExtractionConfig(spatial_step=0.05))

    lengths, failures = [], 0
    for i, path in enumerate(paths):
        try:
            sample = parser.parse(path)
            seq = extractor.transform(sample)
            lengths.append(len(seq.features))
        except (InkMLParseError, FeatureExtractionError, FileNotFoundError):
            failures += 1
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(paths)} …')

    lengths = np.array(lengths)
    print(f'\nAnalyzed {len(lengths)} files ({failures} failed to parse)\n')
    print(f'  min / mean / max : {lengths.min()} / {lengths.mean():.0f} / {lengths.max()}')
    for p in (50, 90, 95, 99, 99.9):
        print(f'  p{p:<5} : {np.percentile(lengths, p):.0f}')
    for cap in (256, 512, 768, 1024):
        pct = (lengths <= cap).mean() * 100
        print(f'  fits in {cap:>4} points : {pct:.2f}%')

    # Crude text histogram
    print()
    counts, edges = np.histogram(lengths, bins=20)
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = '#' * int(60 * c / counts.max())
        print(f'  {lo:5.0f}–{hi:5.0f} | {bar}')

if __name__ == '__main__':
    main()