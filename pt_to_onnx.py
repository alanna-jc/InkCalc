"""
Export the trained MathWriting CTC model to ONNX for Raspberry Pi deployment.

Run from the repo root:

    python scripts/export_onnx.py ^
      --checkpoint checkpoints/best_real_ctc_gpu.pt ^
      --out-dir export

The output folder contains:

    model.onnx
    vocab.json

The main ONNX model accepts:

    points:        float32 tensor with shape (batch, seq_len, 4)
    input_lengths: int64 tensor with shape (batch,)

Each point feature is [dx, dy, dt, pen_up]. For Pi deployment, pad or
truncate each sequence to max_points from vocab.json, and pass the true
unpadded length as input_lengths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    print(
        """PyTorch is not installed in this virtual environment.
Activate your venv and run:

python -m pip install -r requirements.txt

If installation fails, make sure you are using Python 3.11, not Python 3.13.
""",
        file=sys.stderr,
    )
    raise SystemExit(1)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handwriting_ctc.model import CTCRecognizer


class FixedInputCTCRecognizer(nn.Module):
    """Compatibility wrapper for loaders that only support a points input."""

    def __init__(self, model: CTCRecognizer) -> None:
        super().__init__()
        self.model = model

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.model.encoder(points)
        logits = self.model.classifier(encoded)
        return logits.log_softmax(dim=-1)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def vocab_from_checkpoint(checkpoint: dict) -> tuple[dict[str, str], int, int]:
    vocab = checkpoint.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(f"Unexpected checkpoint vocab format: {type(vocab)}")

    blank = vocab.get("blank", "<blank>")
    tokens = vocab.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError("Checkpoint vocab does not contain a token list at vocab['tokens']")

    idx_to_char = {"0": blank}
    for offset, token in enumerate(tokens, start=1):
        idx_to_char[str(offset)] = token
    return idx_to_char, 0, len(tokens) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a MathWriting CTC checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best_real_ctc_gpu.pt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "export")
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument(
        "--skip-points-only",
        action="store_true",
        help="Only export the faithful two-input model.onnx, not the compatibility model_points_only.onnx.",
    )
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    out_dir = resolve_path(args.out_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("Checkpoint keys:", list(checkpoint.keys()))
    print(
        "metadata:",
        {
            key: checkpoint.get(key)
            for key in ("epoch", "valid_cer", "valid_loss", "input_dim", "hidden_dim", "layers", "dropout", "max_points")
        },
    )
    print("checkpoint vocab format:", type(checkpoint.get("vocab")).__name__)

    idx_to_char, blank_idx, vocab_size = vocab_from_checkpoint(checkpoint)
    input_dim = int(checkpoint.get("input_dim", 4))
    hidden_dim = int(checkpoint["hidden_dim"])
    layers = int(checkpoint["layers"])
    dropout = float(checkpoint.get("dropout", 0.0))
    max_points = int(checkpoint.get("max_points", 512))

    model = CTCRecognizer(
        input_dim=input_dim,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=layers,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dummy = torch.zeros(1, max_points, input_dim, dtype=torch.float32)
    dummy_lengths = torch.tensor([max_points], dtype=torch.long)
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        model,
        (dummy, dummy_lengths),
        str(onnx_path),
        input_names=["points", "input_lengths"],
        output_names=["log_probs"],
        dynamic_axes={
            "points": {0: "batch", 1: "seq_len"},
            "input_lengths": {0: "batch"},
            "log_probs": {0: "batch", 1: "seq_len"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    points_only_path = out_dir / "model_points_only.onnx"
    if not args.skip_points_only:
        export_model = FixedInputCTCRecognizer(model)
        export_model.eval()
        torch.onnx.export(
            export_model,
            dummy,
            str(points_only_path),
            input_names=["points"],
            output_names=["log_probs"],
            dynamic_axes={
                "points": {0: "batch", 1: "seq_len"},
                "log_probs": {0: "batch", 1: "seq_len"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )

    vocab_path = out_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "vocab": idx_to_char,
                "meta": {
                    "blank_idx": blank_idx,
                    "vocab_size": vocab_size,
                    "max_points": max_points,
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "layers": layers,
                    "dropout": dropout,
                    "features": ["dx", "dy", "dt", "pen_up"],
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_epoch": checkpoint.get("epoch"),
                    "checkpoint_valid_cer": checkpoint.get("valid_cer"),
                    "input_names": ["points", "input_lengths"],
                    "output_name": "log_probs",
                    "points_only_compat_model": "model_points_only.onnx" if not args.skip_points_only else None,
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Exported ONNX model: {onnx_path}")
    if not args.skip_points_only:
        print(f"Exported compatibility model: {points_only_path}")
    print(f"Exported vocab:      {vocab_path}")
    print()
    print("Running forward pass sanity check...")
    with torch.no_grad():
        output = model(dummy, dummy_lengths)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Length shape: {tuple(dummy_lengths.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print("Done. Copy the export folder to the Pi.")
    print("Pi install: pip install numpy onnxruntime")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Export failed: {error}", file=sys.stderr)
        raise SystemExit(1)
