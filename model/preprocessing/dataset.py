"""
dataset.py  —  PyTorch Dataset and DataLoader for MathWriting InkML files.

MathWritingDataset(inkml_paths, tok2idx, ...)   map-style Dataset
build_dataloader(inkml_paths, tok2idx, ...)     convenience wrapper
collect_labels(inkml_dir, use_normalized)       gather labels for vocab building

Batch dict produced by the collate function (matches training_loop.py):
    batch["inputs"]         float32  (B, T_max, 4)         padded feature sequences
    batch["targets"]        int64    (sum_T_target,)       concatenated label indices
    batch["input_lengths"]  int64    (B,)                  true T per sample (pre-pad)
    batch["target_lengths"] int64    (B,)                  true label length per sample

Why targets is 1-D (concatenated rather than 2-D padded):
    nn.CTCLoss accepts either shape. The 1-D form avoids any ambiguity about
    what the padding value should be, since CTC loss only reads up to
    target_lengths[i] indices for each sample in the batch.

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .inkml_parser import InkMLParser, InkMLParseError
    from .feature_extraction import (
        PaperFeatureExtractor,
        FeatureExtractionConfig,
        FeatureExtractionError,
    )
    from vocab import encode_label
except ImportError:
    from inkml_parser import InkMLParser, InkMLParseError
    from feature_extraction import (
        PaperFeatureExtractor,
        FeatureExtractionConfig,
        FeatureExtractionError,
    )
    from vocab import encode_label

# Where did this number come from originally? I used it on the Pi 
# but that was using the orginal Robin mystery model as reference, so now 
# im unsure if we still need this exact number
# Must match recognition.py on the Pi and the meta["max_points"] in vocab.json.
MAX_POINTS = 512


class MathWritingDataset(Dataset):
    """
    Map-style PyTorch Dataset over a list of MathWriting InkML files.

    Each item returned by __getitem__ is either:
        (features: np.ndarray shape (T, 4), encoded_label: list[int])
    or:
        None  — if the file failed to parse, produced empty features, or
                had a label with no tokens in the vocabulary.

    None items are dropped by collate_fn, so the effective batch size may
    be smaller than requested when bad samples appear.

    Parameters
    ----------
    inkml_paths         : paths to .inkml files from one split (train / valid / test)
    tok2idx             : token-to-index mapping from vocab.build_vocab()
    max_points          : truncate feature sequences longer than this
    spatial_step        : resampling interval for PaperFeatureExtractor (0.05 = paper default)
    use_normalized_label: if True, prefer the normalizedLabel annotation over the raw label
                          (Section 2.4 of MathWriting paper — strongly recommended for training)
    """

    def __init__(
        self,
        inkml_paths: list[str | Path],
        tok2idx: dict[str, int],
        max_points: int = MAX_POINTS,
        spatial_step: float = 0.05,
        use_normalized_label: bool = True,
    ) -> None:
        self.paths      = [Path(p) for p in inkml_paths]
        self.tok2idx    = tok2idx
        self.max_points = max_points

        # Label preference: normalizedLabel first for training.
        # The inkml_parser stores all annotation elements in sample.metadata
        # under their original type key, so we can inspect them here.
        self._label_preference = (
            ('normalizedlabel', 'truth', 'label', 'transcription', 'groundtruth')
            if use_normalized_label
            else ('truth', 'label', 'transcription', 'groundtruth')
        )

        self._parser    = InkMLParser(require_time=True, convert_time_to_seconds=True)
        self._extractor = PaperFeatureExtractor(
            FeatureExtractionConfig(spatial_step=spatial_step)
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]

        # -- 1. Parse InkML ----------------------------------------------------
        try:
            sample = self._parser.parse(path)
        except (InkMLParseError, FileNotFoundError):
            return None

        # -- 2. Pick the best available label ----------------------------------
        label = self._pick_label(sample)
        if not label:
            return None

        # -- 3. Feature extraction ---------------------------------------------
        try:
            seq = self._extractor.transform(sample)
        except FeatureExtractionError:
            return None

        features = seq.features   # np.float32, shape (T, 4)

        # Truncate long sequences (same rule as recognition.py)
        if len(features) > self.max_points:
            features = features[:self.max_points]

        # -- 4. Encode label → integer indices ---------------------------------
        encoded = encode_label(label, self.tok2idx)
        if not encoded:
            # All tokens in this label were OOV — skip the sample
            return None

        return features, encoded

    def _pick_label(self, sample) -> Optional[str]:
        """
        Inspect sample.metadata (all annotations) and return the label from
        the highest-priority annotation type found.

        inkml_parser stores annotations in sample.metadata with their original
        case, e.g. 'normalizedLabel'. We compare case-insensitively.
        """
        meta_lower = {k.lower(): v for k, v in sample.metadata.items()}
        for preferred in self._label_preference:
            if preferred in meta_lower:
                return meta_lower[preferred]
        # Fall back to whatever the parser extracted as sample.label
        return sample.label


# -- Collate function -----------------------------------------------------------

def _collate_fn(batch: list) -> Optional[dict]:
    """
    Collate a list of (features, encoded_label) pairs into a training batch.

    None entries (failed samples) are filtered out silently.
    Returns None if the entire batch is empty after filtering — the training
    loop should skip None batches.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    features_list, labels_list = zip(*batch)

    # -- Pad feature sequences to the longest in this batch --------------------
    input_lengths = torch.tensor(
        [len(f) for f in features_list], dtype=torch.long
    )
    T_max    = int(input_lengths.max().item())
    feat_dim = features_list[0].shape[-1]   # always 4: [dx, dy, dt, pen_state]

    padded = np.zeros((len(batch), T_max, feat_dim), dtype=np.float32)
    for i, f in enumerate(features_list):
        T = len(f)
        padded[i, :T] = f

    inputs = torch.from_numpy(padded)   # (B, T_max, 4)

    # -- Concatenate labels into a 1-D tensor for nn.CTCLoss -------------------
    target_lengths = torch.tensor(
        [len(label) for label in labels_list], dtype=torch.long
    )
    targets = torch.tensor(
        [idx for label in labels_list for idx in label],
        dtype=torch.long,
    )   # shape: (sum of all target_lengths,)

    return {
        'inputs':         inputs,           # float32 (B, T_max, 4)
        'targets':        targets,          # int64   (sum_T_target,)
        'input_lengths':  input_lengths,    # int64   (B,)
        'target_lengths': target_lengths,   # int64   (B,)
    }

def build_dataloader(
    inkml_paths: list[str | Path],
    tok2idx: dict[str, int],
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    max_points: int = MAX_POINTS,
    spatial_step: float = 0.05,
) -> DataLoader:
    """
    Build a DataLoader for one MathWriting split.

    batch_size=256 matches the paper's training configuration.
    Set shuffle=False for val and test splits.
    """
    dataset = MathWritingDataset(
        inkml_paths,
        tok2idx,
        max_points=max_points,
        spatial_step=spatial_step,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


def collect_labels(
    inkml_dir: str | Path,
    use_normalized: bool = True,
) -> list[str]:
    """
    Scan all .inkml files in a directory and return their labels.

    Call this on the train split (and optionally the synthetic split) before
    training to gather all labels for vocab.build_vocab().

    Parameters
    ----------
    inkml_dir      : directory containing .inkml files for one split
    use_normalized : if True, prefer normalizedLabel annotations
    """
    parser = InkMLParser(require_time=False)
    labels: list[str] = []
    paths  = sorted(Path(inkml_dir).glob('*.inkml'))
    print(f'[dataset] Scanning {len(paths)} files in {inkml_dir} …')

    for path in paths:
        try:
            sample = parser.parse(path)
        except (InkMLParseError, FileNotFoundError):
            continue

        meta_lower = {k.lower(): v for k, v in sample.metadata.items()}
        if use_normalized and 'normalizedlabel' in meta_lower:
            label = meta_lower['normalizedlabel']
        elif sample.label:
            label = sample.label
        else:
            continue

        labels.append(label)

    print(f'[dataset] Collected {len(labels)} labels.')
    return labels
