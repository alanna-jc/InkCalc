import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from blocks import CTCTransformer

SEQ_LENGTH = 100
DEPTH = 4

# how do we get vocab?
NUM_LABELS = 3 
BLANK_IDX = 0
VOCAB_SIZE = NUM_LABELS + 1   # +1 for CTC blank

LEARNING_RATE = 1e-4 # dummy number
NUM_EPOCHS = 50 # dummy number


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

    logits = model(inputs) # (B, T, C)

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

    # AC TODO how do we want to do this
    for batch in val_loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        input_lengths = batch["input_lengths"].to(device) # the true ones after padding !
        target_lengths = batch["target_lengths"].to(device) # the true ones after padding !

        # this is detailed in the 'train_one_batch()' function
        logits = model(inputs)                     
        log_probs = F.log_softmax(logits, dim=-1) 
        log_probs = log_probs.transpose(0, 1)     

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -------------------------------------------------------------------
    # load dataset 
    # create input to model
    # this involves parsing and positional embeddings
    train_loader = None
    val_loader = None
    # -------------------------------------------------------------------

    # init of model and optimizer
    model = CTCTransformer(vocab_size = VOCAB_SIZE)
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
            # AC TODO change to not a .pt
            # AC TODO note how this works. This saves weights that are learnt at every stage of model in a tensor
            # It recognizes each section of the model using nn.Module
            # This is also why you need a different instance of each layer in the model
            torch.save(model.state_dict(), "best_ctc_transformer.pt")
            print("Saved new best model.")

        # report CER and ES? not part of training loop AC TODO
    #done
    print("Training complete")

if __name__ == "__main__":
    main()





