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


def stage2_overfit():
    """Train on 50 real files, validate on the same 50. Must approach CER 0."""
    from training_loop import train_model, TRAIN_DIR   # safe: main() is __main__-guarded

    paths = sorted(TRAIN_DIR.glob('*.inkml'))[:50]
    assert paths, f"no .inkml files found in {TRAIN_DIR}"

    print(f"Stage 2 — overfitting {len(paths)} samples, expect CER -> ~0 …")
    best_cer = train_model(
        train_paths=paths,
        val_paths=paths,          # same files on purpose — testing memorization
        lr=3e-4,
        num_epochs=300,
        checkpoint_name="smoke.pt",
    )
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
