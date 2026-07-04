import math
#from sched import scheduler
import optuna
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from model.encoder import CTCTransformer
from pathlib import Path
from vocab import build_vocab, save_vocab, load_vocab, BLANK_IDX
from preprocessing.dataset import build_dataloader, collect_labels, MAX_POINTS
from postprocessing.ctc_decode import greedy_ctc_decode, edit_distance


# Paths: these are dummy paths right now!!!!
TRAIN_DIR     = Path('data/mathwriting-2024/train')
VAL_DIR       = Path('data/mathwriting-2024/valid')
VOCAB_PATH    = Path('vocab.json')
CHECKPOINT_PATH = "checkpoint.pt"

BATCH_SIZE    = 256
LEARNING_RATE = 1e-3 # MathWriting used 1e-3 i think?
WARMUP_STEPS  = 4000   # batches spent ramping 0 -> peak ("Attention is all you need" used 4000)
NUM_EPOCHS = 50 # dummy number

# Model hyperparameters — single source of truth.
# Used for (1) constructing the model, (2) the checkpoint dict, so that
# export_onnx.py can rebuild the exact same architecture without guessing.
INPUT_DIM      = 4      # [dx, dy, dt, pen_state]
EMBED_DIM      = 512    # MathWriting section 4.2
NUM_LAYERS     = 11     # MathWriting section 4.2
NUM_HEADS      = 8
FFN_NUM_HIDDEN = 2048   # "Attention is all you need"
DROPOUT        = 0.15   # MathWriting section 4.2


def train_one_batch(model, batch, optimizer, scheduler, ctc_loss, device):
    """
    batch should contain: AC TODO 
        inputs:          
        targets:         
        input_lengths:   
        target_lengths:  
    """
    optimizer.zero_grad()

    inputs = batch["inputs"].to(device)
    targets = batch["targets"].to(device)
    input_lengths = batch["input_lengths"].to(device) # the true ones after padding !
    target_lengths = batch["target_lengths"].to(device) # the true ones after padding !
    key_padding_mask = batch["key_padding_mask"].to(device)
    
    # need to pass in key_padding_mask 
    logits = model(inputs, key_padding_mask=key_padding_mask) # (B, T, C)

    # Convert logits to probabilities
    log_probs = F.log_softmax(logits, dim=-1) # (B, T, C)

    # CTC loss expects (T, B, C) (PyTorch)
    log_probs = log_probs.transpose(0, 1) 
    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

    # Backprop
    loss.backward()

    # Gradient clipping: if the combined size (norm) of all gradients exceeds
    # 1.0, scale them down to 1.0. Direction is kept, magnitude is capped —
    # prevents one bad batch from catapulting the weights. Must run after
    # backward() (gradients exist) and before step() (they get applied).
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()          # advance the LR schedule one batch
    return loss.item()

def validate_one_epoch(model, val_loader, ctc_loss, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_edits = 0          # sum of edit distances
    total_target_tokens = 0  # sum of reference lengths

    with torch.no_grad():
    
        for batch in val_loader:
            if batch is None: # in case batch is bad
                continue
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            input_lengths = batch["input_lengths"].to(device) # the true ones after padding !
            target_lengths = batch["target_lengths"].to(device) # the true ones after padding !
            key_padding_mask = batch["key_padding_mask"].to(device)

            # this is detailed in the 'train_one_batch()' function
            logits = model(inputs, key_padding_mask=key_padding_mask)                   
            log_probs = F.log_softmax(logits, dim=-1) 
            log_probs = log_probs.transpose(0, 1)     

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            num_batches += 1

            # ── CER ──
            preds = greedy_ctc_decode(logits, input_lengths, BLANK_IDX)
            targets_cpu = targets.cpu()
            offset = 0
            for b, tlen in enumerate(target_lengths.tolist()):
                ref = targets_cpu[offset : offset + tlen].tolist()
                offset += tlen
                total_edits += edit_distance(preds[b], ref)
                total_target_tokens += tlen

    avg_loss = total_loss / max(num_batches, 1)
    cer = total_edits / max(total_target_tokens, 1)
    return avg_loss, cer


def save_checkpoint(epoch, model, optimizer, scheduler, best_val_cer, checkpoint_path="checkpoint.pt"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_cer": best_val_cer,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path="checkpoint.pt"):
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found.")
        return 0, float("inf")
    
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_cer = checkpoint["best_val_cer"]

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    return start_epoch, best_val_cer

def objective(trial):
    # hyperparams for optuna to to train defined here 
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    # TODO dropout and warmup?

    # Build vocabulary 
    if VOCAB_PATH.exists():
        print('[main] Loading existing vocab …')
        tok2idx, idx2tok, meta = load_vocab(VOCAB_PATH)
    else:
        print('[main] Building vocab from train split …')
        labels = collect_labels(TRAIN_DIR, use_normalized=True)
        tok2idx, idx2tok = build_vocab(labels)
        meta = {
            'max_points':  MAX_POINTS,
            'blank_idx':   BLANK_IDX,
            'vocab_size':  len(idx2tok),
        }
        save_vocab(idx2tok, meta, VOCAB_PATH)

    VOCAB_SIZE = meta['vocab_size']
    print(f'[main] Vocab size: {VOCAB_SIZE}')

   
    # -------------------------------------------------------------------
    # load dataset 
    # create input to model
    # this involves parsing and positional embeddings
    train_paths = sorted(TRAIN_DIR.glob('*.inkml'))
    val_paths   = sorted(VAL_DIR.glob('*.inkml'))

    train_loader = build_dataloader(
        train_paths, tok2idx, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = build_dataloader(
        val_paths, tok2idx, batch_size=BATCH_SIZE, shuffle=False
    )
    # -------------------------------------------------------------------

    # init of model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CTCTransformer(
        vocab_size     = VOCAB_SIZE,
        max_points     = MAX_POINTS,
        num_layers     = NUM_LAYERS,
        num_heads      = NUM_HEADS,
        ffn_num_hidden = FFN_NUM_HIDDEN,
        embed_dim      = EMBED_DIM,
        dropout        = DROPOUT,
    )
    model = model.to(device)

    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # LR schedule: linear warmup for WARMUP_STEPS batches, then cosine decay
    # to ~0 over the rest of training. Warmup matters because at step 0 the
    # attention weights are random — big updates then are pure noise. Decay
    # matters at the end — small steps let the model settle into a minimum.
    total_steps = len(train_loader) * NUM_EPOCHS

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)              # 0 -> 1 linearly
        progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0 cosine

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    #best_val_loss = float("inf")
    best_val_cer = float("inf")

    start_epoch = 0
    # not recommended with optuna as it needs a clean state every time to test hyperparams
    '''
    if os.path.exists(CHECKPOINT_PATH):
            start_epoch, best_val_cer = load_checkpoint(
                model,
                optimizer,
                scheduler,
                CHECKPOINT_PATH
            )
    '''

    # Training Loop
    # Epoch is one clean sweep through all training examples
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        for batch in train_loader:
            if batch is None:   # whole batch was bad samples
                continue # skip

            batch_loss = train_one_batch(model, batch, optimizer, scheduler, ctc_loss, device)
            total_train_loss += batch_loss
            num_train_batches += 1

        avg_train_loss = total_train_loss / max(num_train_batches, 1)

        val_loss, val_cer = validate_one_epoch(model, val_loader, ctc_loss, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train loss: {avg_train_loss:.4f} | val loss: {val_loss:.4f} | val CER: {val_cer:.4f}")

        # save best
        if val_cer < best_val_cer:

            best_val_cer = val_cer
            
            # Save a full checkpoint dict, not a bare state_dict.
            # 'model_state_dict' holds the learned weights (keyed by nn.Module
            # attribute names — which is why every layer needs its own instance).
            # The rest is the metadata export_onnx.py needs to rebuild the
            # exact same architecture before loading the weights into it.
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch':          epoch + 1,
                'valid_loss':     val_loss,
                'valid_cer':      val_cer,
                'input_dim':      INPUT_DIM,
                'embed_dim':      EMBED_DIM,
                'num_layers':     NUM_LAYERS,
                'num_heads':      NUM_HEADS,
                'ffn_num_hidden': FFN_NUM_HIDDEN,
                'dropout':        DROPOUT,
                'max_points':     MAX_POINTS,
                'vocab_size':     VOCAB_SIZE,
                'blank_idx':      BLANK_IDX,
            }
            torch.save(checkpoint, f"best_ctc_transformer{trial.number}.pt")
            print("Saved new best model.")

        # not recommended with optuna as it needs a clean state every time to test hyperparams
        '''
        save_checkpoint(
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_cer,
            CHECKPOINT_PATH
        )
        '''

        trial.report(val_cer, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    #done
    print("Training complete")
    return best_val_cer

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()  # Drops hopeless trials early
    )
    
    # Run the optimization search
    # TODO num trials? maybe 30 later
    study.optimize(objective, n_trials=15)

    print("\n--- Optimization Complete ---")
    print(f"Best Trial Value (CER): {study.best_trial.value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")