"""
vocab.py  —  Vocabulary building and label tokenisation.
=========================================================================
tokenize_label(label)              → list[str]
build_vocab(labels)                → (tok2idx, idx2tok)
encode_label(label, tok2idx)       → list[int]
save_vocab(idx2tok, meta, path)    → None   (JSON recognised by recognition.py)
load_vocab(path)                   → (tok2idx, idx2tok, meta)

Index 0 is always the CTC blank token (BLANK_IDX = 0)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# MathWriting tokeniser (directly from Appendix M of the paper) 
# Handles multi-character LaTeX commands like \frac, \begin{matrix}, \mathbb{R}
# as single tokens rather than splitting them character-by-character.
_COMMAND_RE = re.compile(
    r'\\(?:mathbb\{[a-zA-Z]\}|begin\{[a-z]+\}|end\{[a-z]+\}|operatorname\*|[a-zA-Z]+|.)'
)

BLANK_TOKEN = '<blank>'
BLANK_IDX   = 0


def tokenize_label(label: str) -> list[str]:
    """
    Split a normalised LaTeX string into tokens.

    Multi-character commands like \\frac, \\begin{matrix}, and \\mathbb{R}
    are kept as single tokens. Everything else is split character-by-character.

    This is taken from Mathwriting paper Appendix I.
    """
    tokens: list[str] = []
    s = label
    while s:
        if s[0] == '\\':
            m = _COMMAND_RE.match(s)
            if m is None:           # bare backslash with no valid continuation
                tokens.append(s[0])
                s = s[1:]
            else:
                tokens.append(m.group(0))
                s = s[len(tokens[-1]):]
        else:
            tokens.append(s[0])
            s = s[1:]
    return tokens


def build_vocab(
    labels: list[str],
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build a complete vocabulary from a list of normalised label strings.
    Blank token is placed at index 0; all other tokens are sorted alphabetically

    Parameters
    labels : list of normalised LaTeX strings (use collect_labels() to gather them)

    Returns
    tok2idx : dict[str, int]   token → integer index
    idx2tok : dict[int, str]   integer index → token
    """
    all_tokens: set[str] = set()
    for label in labels:
        all_tokens.update(tokenize_label(label))

    tok2idx: dict[str, int] = {BLANK_TOKEN: BLANK_IDX}
    idx2tok: dict[int, str] = {BLANK_IDX: BLANK_TOKEN}

    for i, token in enumerate(sorted(all_tokens), start=1):
        tok2idx[token] = i
        idx2tok[i]     = token

    return tok2idx, idx2tok


def encode_label(
    label: str,
    tok2idx: dict[str, int],
    *,
    drop_if_oov: bool = True,
) -> list[int]:
    """
    Convert a normalised LaTeX string to a sequence of vocabulary indices.

    An out-of-vocabulary (OOV) token can appear if the vocab was built on the
    train split but a val/test label contains a token not seen in training.

    By default (drop_if_oov=True) a label containing ANY OOV token is rejected
    outright — the function returns [] so the caller drops the whole sample.
    Silently skipping only the OOV tokens (the old behaviour) produces a
    shorter, semantically-wrong target (e.g. '\\frac{a}{b}' -> 'ab') that would
    then be trained against as if it were correct, which is worse than dropping.

    Pass drop_if_oov=False to restore the lossy skip-only-OOV-tokens behaviour.
    """
    ids: list[int] = []
    for t in tokenize_label(label):
        if t in tok2idx:
            ids.append(tok2idx[t])
        elif drop_if_oov:
            return []          # reject the whole label; caller drops the sample
        # else: skip only this token (lossy)
    return ids


def save_vocab(
    idx2tok: dict[int, str],
    meta: dict,
    path: str | Path,
) -> None:
    """
    Save vocabulary to JSON in the format recognition.py already expects:

        {
            "vocab": { "0": "<blank>", "1": "!", "2": "#", ... },
            "meta":  { "max_points": 512, "blank_idx": 0, "vocab_size": 255 }
        }

    The meta dict should contain at minimum:
        max_points  (int)  — must match the padding in dataset.py and recognition.py
        blank_idx   (int)  — always 0
        vocab_size  (int)  — len(idx2tok)
    """
    bundle = {
        'vocab': {str(k): v for k, v in idx2tok.items()},
        'meta':  meta,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    print(f'[vocab] Saved {len(idx2tok)} tokens to {path}')


def load_vocab(
    path: str | Path,
) -> tuple[dict[str, int], dict[int, str], dict]:
    """
    Load a saved vocab.json.

    Returns
    tok2idx : dict[str, int]
    idx2tok : dict[int, str]
    meta    : dict  (max_points, blank_idx, vocab_size, ...)
    """
    with open(path, 'r', encoding='utf-8') as f:
        bundle = json.load(f)
    idx2tok = {int(k): v for k, v in bundle['vocab'].items()}
    tok2idx = {v: k for k, v in idx2tok.items()}
    return tok2idx, idx2tok, bundle['meta']
