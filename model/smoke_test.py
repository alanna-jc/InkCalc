"""
smoke_test.py — cheap proof the pipeline works before burning GPU-days.
Run from model/:   python smoke_test.py            (stages 0+1, no data needed)
                   python smoke_test.py --overfit  (adds stage 2, needs dataset)
"""
import argparse
import torch

from model.encoder import CTCTransformer

def stage0_shapes_and_gradients():
    """Forward + backward on random tensors. Catches wiring/shape bugs."""
    VOCAB_SIZE = 100
    model = CTCTransformer(vocab_size=VOCAB_SIZE, max_points=512)

    B, T = 4, 300                                  # NOT 512 — proves the PE slice works
    x = torch.randn(B, T, 4)
    lengths = torch.tensor([300, 250, 100, 7])
    mask = torch.arange(T)[None, :] >= lengths[:, None]

    logits = model(x, key_padding_mask=mask)
    assert logits.shape == (B, T, VOCAB_SIZE), f"bad output shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "non-finite logits"

    log_probs = logits.log_softmax(-1).transpose(0, 1)
    targets = torch.randint(1, VOCAB_SIZE, (20,))
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(
        log_probs, targets, lengths, torch.tensor([8, 6, 4, 2]))
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached: {name}"
        assert torch.isfinite(p.grad).all(), f"NaN/inf gradient in: {name}"

    print(f"Stage 0 OK — output shape right, loss {loss.item():.3f}, "
          f"gradients flow to all {sum(1 for _ in model.parameters())} parameter tensors")
    return model, x


def stage1_mask_and_pe(model, x):
    """The two bugs specific to this architecture: dead mask, missing PE."""
    model.eval()
    no_mask = torch.zeros(1, 100, dtype=torch.bool)

    with torch.no_grad():
        # Padding invariance: extra padding must not change real positions' outputs
        a = model(x[:1, :100], no_mask)
        padded = torch.cat([x[:1, :100], torch.zeros(1, 200, 4)], dim=1)
        pmask = torch.arange(300)[None, :] >= 100
        b = model(padded, pmask)
        assert torch.allclose(a, b[:, :100], atol=1e-4), \
            "MASK BUG: padding changed outputs on real positions"

        # Order sensitivity: shuffling points in time MUST change outputs
        perm = torch.randperm(100)
        c = model(x[:1, :100][:, perm], no_mask)
        assert not torch.allclose(a, c, atol=1e-3), \
            "PE BUG: model gives identical outputs for shuffled input — position info missing"

    print("Stage 1 OK — mask isolates padding, positional encoding is active")


def _labels_for(paths):
    """Extract labels for a specific set of files (prefer normalizedLabel).

    Mirrors dataset.collect_labels but for an explicit path list, so the
    overfit vocab can be built from exactly the files we train on (no OOV).
    """
    from preprocessing.inkml_parser import InkMLParser, InkMLParseError

    parser = InkMLParser(require_time=False)
    labels = []
    for p in paths:
        try:
            sample = parser.parse(p)
        except (InkMLParseError, FileNotFoundError):
            continue
        meta_lower = {k.lower(): v for k, v in sample.metadata.items()}
        label = meta_lower.get('normalizedlabel') or sample.label
        if label:
            labels.append(label)
    return labels


def stage2_overfit():
    """Train on 50 real files, validate on the same 50. Must approach CER 0.

    Self-contained: reuses train_one_batch/validate_one_epoch and the model
    hyperparameters from training_loop.py, but drives the loop here so no
    changes to training_loop.py are needed. main() there is __main__-guarded,
    so importing it runs no training.
    """
    import torch.nn as nn
    import torch.optim as optim

    from vocab import build_vocab, BLANK_IDX
    from preprocessing.dataset import build_dataloader, MAX_POINTS
    from training_loop import (
        train_one_batch, validate_one_epoch, TRAIN_DIR,
        EMBED_DIM, NUM_LAYERS, NUM_HEADS, FFN_NUM_HIDDEN, DROPOUT,
    )

    paths = sorted(TRAIN_DIR.glob('*.inkml'))[:50]
    assert paths, f"no .inkml files found in {TRAIN_DIR}"

    NUM_EPOCHS = 300
    LR         = 3e-4
    BATCH      = 16

    # Vocab from exactly these files → guarantees no OOV drops for this test.
    tok2idx, idx2tok = build_vocab(_labels_for(paths))

    # num_workers=0: solo, deterministic, no Windows spawn overhead for 50 files.
    train_loader = build_dataloader(
        paths, tok2idx, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader = build_dataloader(
        paths, tok2idx, batch_size=BATCH, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CTCTransformer(
        vocab_size     = len(idx2tok),
        max_points     = MAX_POINTS,
        num_layers     = NUM_LAYERS,
        num_heads      = NUM_HEADS,
        ffn_num_hidden = FFN_NUM_HIDDEN,
        embed_dim      = EMBED_DIM,
        dropout        = DROPOUT,
    ).to(device)

    ctc_loss  = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Constant LR: the real warmup schedule (WARMUP_STEPS=4000) would never
    # leave the ramp in a ~300-step overfit and the test would fail spuriously.
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    # train_one_batch now requires a GradScaler (AMP). Enable it only on CUDA;
    # on CPU it's disabled and the calls become plain fp32 passthroughs. Running
    # on the GPU here means this smoke test also exercises the real AMP path.
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    print(f"Stage 2 — overfitting {len(paths)} samples on {device}, "
          f"expect CER -> ~0 …")

    best_cer = float("inf")
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            if batch is None:
                continue
            train_one_batch(model, batch, optimizer, scheduler, ctc_loss, device, scaler)

        _, val_cer = validate_one_epoch(model, val_loader, ctc_loss, device)
        best_cer = min(best_cer, val_cer)

        if epoch % 20 == 0 or best_cer < 0.10:
            print(f"  epoch {epoch+1:>3}/{NUM_EPOCHS} | CER {val_cer:.4f} | best {best_cer:.4f}")
        if best_cer < 0.01:                       # memorized — no need to keep going
            break

    print(f"Stage 2 result: best CER on training data = {best_cer:.4f}")
    assert best_cer < 0.10, (
        "OVERFIT FAILED: model can't memorize 50 samples — suspect decode, "
        "target alignment, blank index, or loss wiring")
    print("Stage 2 OK — model can learn; pipeline validated end to end")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--overfit', action='store_true', help='run stage 2 (needs dataset)')
    args = parser.parse_args()

    model, x = stage0_shapes_and_gradients()
    stage1_mask_and_pe(model, x)
    if args.overfit:
        stage2_overfit()
    else:
        print("\nStages 0+1 passed. Run with --overfit for stage 2 (needs dataset).")
