r"""
train_ddp.py — Multi-GPU (DistributedDataParallel) training for InkCalc.
=========================================================================
Runs the SAME model / loss / decode as training_loop.py, but splits each
batch across all visible GPUs so two RTX A4500s give ~2x throughput AND an
effective batch of (per_gpu_batch * num_gpus) — e.g. 128 * 2 = 256, matching
the MathWriting paper without OOMing a single 20 GB card.

training_loop.py is intentionally left untouched. This file imports the
reusable pieces from it (train_one_batch, validate_one_epoch, the model
hyperparameters, and the data paths) and only adds the distributed plumbing.
Because DDP averages gradients across ranks, the effective batch is genuinely
128*2 = 256, so the paper's LEARNING_RATE = 1e-3 applies directly — no
linear-scaling tweak needed.


=========================================================================
1. ENVIRONMENT SETUP  (do this once per machine)
=========================================================================
The ML stack has no wheels for Python 3.14 yet, so use Python 3.12 (or 3.11)
in a dedicated virtual environment. Pick the torch CUDA wheel that matches
the GPU architecture:
    - Ampere (RTX A4500 / A100 / 30-series, compute cap 8.x):  cu124
    - Blackwell (RTX 50-series, compute cap 12.x):             cu128

Full copy-paste sequence for the LINUX LAB MACHINE (Ampere A4500 x2).
Run these in order from a shell:

    # (a) Confirm Python 3.12 is available. If "command not found":
    python3.12 --version
    #   - HPC cluster? try:   module avail python   then   module load python/3.12
    #   - otherwise install via pyenv:  pyenv install 3.12.10 && pyenv local 3.12.10
    #     (or ask the cluster admin; do NOT use the system python if it's 3.14)

    # (b) Go to the repo, then into model/ (all commands run from model/).
    cd /path/to/InkCalc

    # (c) Create + activate the venv (at the repo root).
    python3.12 -m venv .venv
    source .venv/bin/activate

    # (d) Upgrade pip, then install PyTorch (cu124 for Ampere) + numpy.
    python -m pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install numpy

    # (e) OPTIONAL — only needed for the UI / solver / ONNX export, not training:
    pip install onnxruntime sympy "antlr4-python3-runtime==4.11.1" pygame defusedxml

    # (f) Verify torch sees BOTH GPUs before doing anything else:
    python -c "import torch; print(torch.cuda.device_count(), \
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
    #   expect: 2 ['NVIDIA RTX A4500', 'NVIDIA RTX A4500']

    # (g) Check usable CPU cores (drives num_workers; respects SLURM/affinity):
    python -c "import os; print('usable cores:', len(os.sched_getaffinity(0)))"
    #   NUM_WORKERS_PER_PROC below is 6 (=12 of 16 cores across 2 procs)

    # (h) Finally cd into model/ before launching (see section 3):
    cd model

Every later session (after the venv already exists) only needs:
    cd /path/to/InkCalc && source .venv/bin/activate && cd model


=========================================================================
2. DATASET LOCATION
=========================================================================
Paths come from training_loop.py (TRAIN_DIR / VAL_DIR), which are RELATIVE
and resolve against the directory you launch from — so run from model/.
Put the data at  model/mathwriting-2024/{train,valid}  OR point a junction/
symlink there so the bytes can live elsewhere:

    Windows (cmd):   mklink /J model\mathwriting-2024 D:\datasets\mathwriting-2024
    Linux:           ln -s /data/mathwriting-2024 model/mathwriting-2024


=========================================================================
3. LAUNCH  (always via torchrun — NOT `python train_ddp.py`)
=========================================================================
All commands run from the model/ directory with the venv active.

Full training (defaults: 50 epochs, per-GPU batch 128 -> effective 256):
    torchrun --standalone --nproc_per_node=2 train_ddp.py

The first launch does a one-time full vocab scan of all 229k train files
(tens of minutes) and writes vocab.json; later runs load it instantly.

Useful flags:
    --epochs N        override epoch count
    --batch-size N    PER-GPU batch (effective = N * num_gpus)
    --limit N         use only the first N train/val files (fast test; also
                      builds a throwaway vocab_test.json and writes
                      checkpoint_test.pt so real artifacts are never touched)
    --checkpoint PATH resume-checkpoint filename (default checkpoint.pt)


=========================================================================
4. HOW TO TEST IT WORKS  (staged, cheapest first)
=========================================================================
Step 1 — DDP plumbing on ONE GPU (isolates logic from multi-GPU comms):
    torchrun --standalone --nproc_per_node=1 train_ddp.py --epochs 1 --limit 400
    -> should finish in <1 min and print "Training complete".

Step 2 — BOTH GPUs on a small subset:
    torchrun --standalone --nproc_per_node=2 train_ddp.py --epochs 1 --limit 400
    -> header should say  world_size=2 ... effective_batch=256
    -> in another terminal:  watch -n 1 nvidia-smi
       expect TWO python processes, utilization > 0 on BOTH cards.

Step 3 — does it actually learn? (correctness under DDP):
    torchrun --standalone --nproc_per_node=2 train_ddp.py --epochs 15 --limit 400
    -> val CER should trend DOWN across epochs (won't hit 0 on 400 files).

Step 4 — resume works:
    Run a 2-epoch limited job, let it finish (writes checkpoint_test.pt), then
    run the SAME command again -> should print "Resumed from epoch ...".

Step 5 — the real run:
    torchrun --standalone --nproc_per_node=2 train_ddp.py
    Watch nvidia-smi for the first few minutes; if GPU util keeps sagging to 0
    you're data-starved -> raise NUM_WORKERS_PER_PROC.

If NCCL hangs at init or errors, try:
    NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 train_ddp.py --epochs 1 --limit 400
    (last resort: set backend="gloo" in setup_ddp() — slower, but proves logic.)


=========================================================================
CHECKPOINTS  (written by rank 0 only; formats match training_loop.py)
=========================================================================
    checkpoint.pt              — resume state (weights + optim + sched + epoch)
    best_ctc_transformer.pt    — best-CER model + architecture metadata
                                 (export_onnx.py reads this one)
    (--limit runs write *_test.pt / vocab_test.json instead.)
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.encoder import CTCTransformer
from vocab import build_vocab, save_vocab, load_vocab, BLANK_IDX
from preprocessing.dataset import (
    MathWritingDataset, _collate_fn, collect_labels, MAX_POINTS,
)
from preprocessing.inkml_parser import InkMLParser, InkMLParseError
# Reuse the exact training / validation logic from the single-GPU script so
# the two stay in lock-step. main() there is __main__-guarded, so importing
# runs no training.
from training_loop import (
    train_one_batch, validate_one_epoch,
    TRAIN_DIR, VAL_DIR, VOCAB_PATH,
    LEARNING_RATE, WARMUP_STEPS, NUM_EPOCHS,
    INPUT_DIM, EMBED_DIM, NUM_LAYERS, NUM_HEADS, FFN_NUM_HIDDEN, DROPOUT,
)

# Per-GPU batch size. Effective batch = PER_GPU_BATCH_SIZE * num_gpus.
# 128 * 2 A4500s = 256 (paper). 128 fits comfortably in 20 GB.
PER_GPU_BATCH_SIZE = 128
NUM_WORKERS_PER_PROC = 6   # 6 * 2 procs = 12 of your 16 cores; leaves headroom


# --------------------------------------------------------------------------
# DDP setup / teardown
# --------------------------------------------------------------------------
def setup_ddp() -> int:
    """Initialise the process group from torchrun's env vars. Returns local rank."""
    dist.init_process_group(backend="nccl")   # nccl = fast GPU-GPU on one node
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def is_main_process() -> bool:
    return dist.get_rank() == 0


def _labels_for_paths(paths):
    """Labels for an explicit file list (prefer normalizedLabel).

    Used only for the fast --limit test vocab, so we don't scan all 229k train
    files just to sanity-check the DDP plumbing.
    """
    parser = InkMLParser(require_time=False)
    labels = []
    for p in paths:
        try:
            sample = parser.parse(p)
        except (InkMLParseError, FileNotFoundError):
            continue
        meta_lower = {k.lower(): v for k, v in sample.metadata.items()}
        label = meta_lower.get("normalizedlabel") or sample.label
        if label:
            labels.append(label)
    return labels


def log(msg: str) -> None:
    """Print only from rank 0 to avoid N duplicate lines."""
    if is_main_process():
        print(msg, flush=True)


# --------------------------------------------------------------------------
# Checkpoint helpers (DDP-aware: always save/load the UNWRAPPED module so the
# files are interchangeable with training_loop.py and export_onnx.py).
# --------------------------------------------------------------------------
def save_resume_checkpoint(path, epoch, ddp_model, optimizer, scheduler, best_val_cer):
    torch.save({
        "epoch": epoch,
        "model_state_dict": ddp_model.module.state_dict(),   # unwrap DDP
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_cer": best_val_cer,
    }, path)


def save_best_checkpoint(path, epoch, ddp_model, optimizer, scheduler,
                         val_loss, val_cer, vocab_size):
    # Mirrors training_loop.py's best-model dict so export_onnx.py can rebuild
    # the architecture from the checkpoint alone.
    torch.save({
        "model_state_dict": ddp_model.module.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":          epoch + 1,
        "valid_loss":     val_loss,
        "valid_cer":      val_cer,
        "input_dim":      INPUT_DIM,
        "embed_dim":      EMBED_DIM,
        "num_layers":     NUM_LAYERS,
        "num_heads":      NUM_HEADS,
        "ffn_num_hidden": FFN_NUM_HIDDEN,
        "dropout":        DROPOUT,
        "max_points":     MAX_POINTS,
        "vocab_size":     vocab_size,
        "blank_idx":      BLANK_IDX,
    }, path)


def load_resume_checkpoint(path, ddp_model, optimizer, scheduler, device):
    if not os.path.exists(path):
        return 0, float("inf")
    ckpt = torch.load(path, map_location=device)
    ddp_model.module.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"] + 1, ckpt["best_val_cer"]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="override number of epochs (use a small value to test)")
    parser.add_argument("--batch-size", type=int, default=PER_GPU_BATCH_SIZE,
                        help="PER-GPU batch size; effective batch = this * num_gpus")
    parser.add_argument("--limit", type=int, default=0,
                        help="if >0, use only the first N train and val files (for a quick test run)")
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    args = parser.parse_args()

    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    log(f"[ddp] world_size={world_size}  per_gpu_batch={args.batch_size}  "
        f"effective_batch={args.batch_size * world_size}  epochs={args.epochs}")

    # -- Data paths (needed before vocab so --limit can build a fast vocab) --
    train_paths = sorted(TRAIN_DIR.glob("*.inkml"))
    val_paths   = sorted(VAL_DIR.glob("*.inkml"))
    if args.limit > 0:
        train_paths = train_paths[:args.limit]
        val_paths   = val_paths[:args.limit]
        log(f"[ddp] --limit active: {len(train_paths)} train / {len(val_paths)} val files")

    # -- Vocab: rank 0 builds/saves once; everyone else waits, then all load --
    # (Prevents 2 processes both scanning files and racing on the write.)
    # With --limit we build a small throwaway vocab from just the limited files
    # (fast) and write it to a separate path so the real vocab.json is untouched.
    vocab_path = Path("vocab_test.json") if args.limit > 0 else VOCAB_PATH
    if is_main_process():
        if not vocab_path.exists():
            if args.limit > 0:
                log(f"[ddp] Building test vocab from {len(train_paths)} limited files…")
                labels = _labels_for_paths(train_paths)
            else:
                log("[ddp] Building vocab from full train split (rank 0 only)…")
                labels = collect_labels(TRAIN_DIR, use_normalized=True)
            tok2idx, idx2tok = build_vocab(labels)
            meta = {"max_points": MAX_POINTS, "blank_idx": BLANK_IDX,
                    "vocab_size": len(idx2tok)}
            save_vocab(idx2tok, meta, vocab_path)
    dist.barrier()
    tok2idx, idx2tok, meta = load_vocab(vocab_path)
    vocab_size = meta["vocab_size"]
    log(f"[ddp] Vocab size: {vocab_size}  (from {vocab_path})")

    # -- Data ----------------------------------------------------------------
    train_ds = MathWritingDataset(train_paths, tok2idx, max_points=MAX_POINTS)
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=NUM_WORKERS_PER_PROC,
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=True,          # equal #batches per rank → no collective desync
    )

    # Validation runs on rank 0 only over the full val split (simple + correct).
    val_loader = None
    if is_main_process():
        val_ds = MathWritingDataset(val_paths, tok2idx, max_points=MAX_POINTS)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=NUM_WORKERS_PER_PROC, collate_fn=_collate_fn, pin_memory=True,
        )

    # -- Model / loss / optim / schedule ------------------------------------
    model = CTCTransformer(
        vocab_size=vocab_size, max_points=MAX_POINTS,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        ffn_num_hidden=FFN_NUM_HIDDEN, embed_dim=EMBED_DIM, dropout=DROPOUT,
    ).to(device)
    model = DDP(model, device_ids=[local_rank])   # broadcasts rank-0 weights on init

    ctc_loss  = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    # LR stays at the paper's 1e-3: DDP averages grads across ranks, so the
    # effective batch is 256 and the paper's LR applies directly.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_loader) * args.epochs   # per-rank steps (same on all ranks)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Route --limit test runs to separate checkpoint files so they never
    # clobber real training artifacts (and a real run can't accidentally
    # resume from a test checkpoint built on the small test vocab).
    if args.limit > 0:
        resume_path = "checkpoint_test.pt" if args.checkpoint == "checkpoint.pt" else args.checkpoint
        best_path   = "best_ctc_transformer_test.pt"
    else:
        resume_path = args.checkpoint
        best_path   = "best_ctc_transformer.pt"

    # -- Resume (all ranks load the same file, mapped to their own device) ---
    start_epoch, best_val_cer = load_resume_checkpoint(
        resume_path, model, optimizer, scheduler, device
    )
    if start_epoch > 0:
        log(f"[ddp] Resumed from epoch {start_epoch - 1} (best CER {best_val_cer:.4f})")

    # -- Training loop -------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)   # REQUIRED: reshuffles differently each epoch

        for batch in train_loader:
            if batch is None:
                # NOTE: a fully-None batch on one rank but not another would
                # desync DDP's gradient all-reduce and hang. With batch_size
                # 128 on real data this ~never happens (would need all 128
                # samples in a batch to fail parsing). If you ever see a hang,
                # pre-filter unparseable files so __getitem__ never returns None.
                continue
            train_one_batch(model, batch, optimizer, scheduler, ctc_loss, device)

        # -- Validate + checkpoint on rank 0 only ---------------------------
        if is_main_process():
            val_loss, val_cer = validate_one_epoch(
                model.module, val_loader, ctc_loss, device
            )
            log(f"Epoch {epoch+1}/{args.epochs} | val loss: {val_loss:.4f} | "
                f"val CER: {val_cer:.4f}")

            if val_cer < best_val_cer:
                best_val_cer = val_cer
                save_best_checkpoint(
                    best_path, epoch, model, optimizer, scheduler,
                    val_loss, val_cer, vocab_size,
                )
                log("Saved new best model.")

            save_resume_checkpoint(
                resume_path, epoch, model, optimizer, scheduler, best_val_cer
            )

        # Keep all ranks in step each epoch (rank 0 is busy validating/saving).
        dist.barrier()

    log("Training complete")
    cleanup_ddp()


if __name__ == "__main__":
    main()
