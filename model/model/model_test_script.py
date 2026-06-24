# Made with Chat GPT
import torch
from encoder import CTCTransformer


def main():
    # -------------------------
    # Hyperparameters for test
    # -------------------------
    batch_size = 2
    seq_len = 10
    embed_dim = 512
    vocab_size = 20

    num_layers = 2
    num_heads = 8
    ffn_num_hidden = 2048
    dropout = 0.15

    # -------------------------
    # Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Create model
    # -------------------------
    model = CTCTransformer(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_num_hidden=ffn_num_hidden,
        embed_dim=embed_dim,
        dropout=dropout
    ).to(device)

    model.train()

    # -------------------------
    # Fake input
    # x shape: (batch_size, seq_len, embed_dim)
    # -------------------------
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    # -------------------------
    # Fake padding mask
    # shape: (batch_size, seq_len)
    # True = padding, False = real token
    #
    # Example:
    # sample 0 has full length 10
    # sample 1 has true length 7, so last 3 positions are padding
    # -------------------------
    key_padding_mask = torch.tensor([
        [False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, True,  True,  True ]
    ], dtype=torch.bool, device=device)

    # -------------------------
    # Forward pass
    # -------------------------
    logits = model(x, key_padding_mask)

    print("Input shape: ", x.shape)
    print("Mask shape:  ", key_padding_mask.shape)
    print("Logits shape:", logits.shape)

    # Expected shape: (batch_size, seq_len, vocab_size)
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, (
        f"Expected logits shape {expected_shape}, got {logits.shape}"
    )

    # -------------------------
    # Backward pass sanity check
    # Use a fake scalar loss
    # -------------------------
    loss = logits.mean()
    loss.backward()

    print("Loss:", loss.item())
    print("Forward + backward pass successful.")

    # Optional: check one parameter got a gradient
    grad_found = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient found for parameter: {name}, grad shape = {param.grad.shape}")
            grad_found = True
            break

    assert grad_found, "No gradients found after backward pass."
    print("Gradient check passed.")


if __name__ == "__main__":
    main()