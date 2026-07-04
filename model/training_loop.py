import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from model.encoder import CTCTransformer
from pathlib import Path
from vocab import build_vocab, save_vocab, load_vocab, BLANK_IDX
from preprocessing.dataset import build_dataloader, collect_labels, MAX_POINTS


# i dont think these get used ever? 
#SEQ_LENGTH = 100
#DEPTH = 4

# how do we get vocab?
#NUM_LABELS = 3 we get this in main now
#BLANK_IDX = 0 defined in vocab.py now
#VOCAB_SIZE = NUM_LABELS + 1   # we get this in main now

# Paths: these are dummy paths right now!!!!
TRAIN_DIR     = Path('data/mathwriting-2024/train')
VAL_DIR       = Path('data/mathwriting-2024/valid')
VOCAB_PATH    = Path('vocab.json')
BATCH_SIZE    = 256
LEARNING_RATE = 1e-4 # MathWriting used 1e-3 i think?
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


def train_one_batch(model, batch, optimizer, ctc_loss, device):
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
    optimizer.step()

    return loss.item()

def validate_one_epoch(model, val_loader, ctc_loss, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

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

    return total_loss / max(num_batches, 1)

def main():

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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # AC TODO should be initialized at what?
    best_val_loss = float("inf")

    # AC TODO how do we want to deal with batches and epochs?

    # Training Loop
    # Epoch is one clean sweep through all training examples
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        for batch in train_loader:
            
            if batch is None:   # whole batch was bad samples
                continue # skip

            # AC TODO add model.train() into v this function??
            batch_loss = train_one_batch(model, batch, optimizer, ctc_loss, device)
            total_train_loss += batch_loss
            num_train_batches += 1

        avg_train_loss = total_train_loss / max(num_train_batches, 1)

        val_loss = validate_one_epoch(model, val_loader, ctc_loss, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train loss: {avg_train_loss:.4f} | val loss: {val_loss:.4f}")
        
        # save best
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            # Save a full checkpoint dict, not a bare state_dict.
            # 'model_state_dict' holds the learned weights (keyed by nn.Module
            # attribute names — which is why every layer needs its own instance).
            # The rest is the metadata export_onnx.py needs to rebuild the
            # exact same architecture before loading the weights into it.
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch':          epoch + 1,
                'valid_loss':     val_loss,
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
            torch.save(checkpoint, "best_ctc_transformer.pt")
            print("Saved new best model.")

        # report CER and ES? not part of training loop AC TODO
    #done
    print("Training complete")

if __name__ == "__main__":
    main()