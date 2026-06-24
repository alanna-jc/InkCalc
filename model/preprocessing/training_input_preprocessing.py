import os
import numpy as np


# TODO copilot suggest something about supporting device / dtype
def positional_encoding(seq_length, depth):
    """
    This class implements the Positional Encoding formulas from "Attention is all you need" section 3.5.

    Args:
        seq_length: Length of sequence
        depth: number of features for each point in sequence

    Returns:
        pe: A positional encoding matrix of the size (seq_length, depth)
    """
    
    pe = np.zeros((seq_length, depth))

    # every row is a position
    # every column is a depth of dim
    position = np.arange(seq_length)[:, np.newaxis]

    div_term = np.exp(np.arange(0, depth, 2) * -(np.log(10000.0) / depth)) 

    # python broadcasts the (5,1) * (1, depth/2) into pe
    pe[:,0::2] = np.sin(position * div_term)
    pe[:,1::2] = np.cos(position * div_term)

    return pe