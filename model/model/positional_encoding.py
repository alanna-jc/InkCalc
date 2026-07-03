import math
import torch

# Adapted from Alanna's version which is commented out at the bottom of this file.
# This new version returns a torch tensor instead of a numpy array.


def positional_encoding(
    seq_length: int,
    depth: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
   
    # Create sinusoidal positional encodings.

    # Args:
    #     seq_length:
    #         Number of points in the padded sequence.

    #     depth:
    #         Transformer embedding dimension, such as 512.
    #         This is NOT the raw feature count of 4.

    #     device:
    #         CPU or GPU device on which to create the tensor.

    #     dtype:
    #         Tensor dtype matching the model input.

    # Returns:
    #     Tensor with shape:
    #         [seq_length, depth]
   

    if seq_length <= 0:
        raise ValueError("seq_length must be greater than zero.")

    if depth <= 0:
        raise ValueError("depth must be greater than zero.")

    # Perform the calculations in float32 for numerical stability.
    position = torch.arange(
        seq_length,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(
            0,
            depth,
            2,
            device=device,
            dtype=torch.float32,
        )
        * (-math.log(10000.0) / depth)
    )

    positional_encodings = torch.zeros(
        seq_length,
        depth,
        device=device,
        dtype=torch.float32,
    )

    angles = position * div_term

    positional_encodings[:, 0::2] = torch.sin(angles)

    # supports an odd embedding dimension (which we don't have but just for completeness)
    cosine_columns = positional_encodings[:, 1::2].shape[1]
    positional_encodings[:, 1::2] = torch.cos(
        angles[:, :cosine_columns]
    )

    return positional_encodings.to(dtype=dtype)

# OLD VERSION!
# def positional_encoding(seq_length, depth):
#     """
#     This class implements the Positional Encoding formulas from "Attention is all you need" section 3.5.

#     Args:
#         seq_length: Length of sequence
#         depth: number of features for each point in sequence

#     Returns:
#         pe: A positional encoding matrix of the size (seq_length, depth)
#     """
    
#     pe = np.zeros((seq_length, depth))

#     # every row is a position
#     # every column is a depth of dim
#     position = np.arange(seq_length)[:, np.newaxis]

#     div_term = np.exp(np.arange(0, depth, 2) * -(np.log(10000.0) / depth)) 

#     # python broadcasts the (5,1) * (1, depth/2) into pe
#     pe[:,0::2] = np.sin(position * div_term)
#     pe[:,1::2] = np.cos(position * div_term)

#     return pe