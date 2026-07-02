#Collate.py apparently Pads features and Creates input_lengths and key padding mask? 

# Apparently - each feature neads to be padded into the same length becasue they are variable. 
#Each sample must contain features with shape:
#    [T, 4]
#where the four features are:
#   [dx, dy, dt, pen_state]


from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


NUM_INPUT_FEATURES = 4


def _get_features(sample: Any) -> torch.Tensor:
    """Extract and validate the feature tensor from one dataset sample."""

    if isinstance(sample, dict):
        features = sample["features"]
    elif hasattr(sample, "features"):
        features = sample.features
    else:
        features = sample

    if isinstance(features, torch.Tensor):
        features = features.to(dtype=torch.float32)
    else:
        features = torch.tensor(
            np.asarray(features),
            dtype=torch.float32,
        )

    if features.ndim != 2:
        raise ValueError(
            f"Features must have shape [T, 4], got {tuple(features.shape)}."
        )

    if features.shape[1] != NUM_INPUT_FEATURES:
        raise ValueError(
            f"Expected {NUM_INPUT_FEATURES} features per point, "
            f"got {features.shape[1]}."
        )

    if features.shape[0] == 0:
        raise ValueError("Feature sequence cannot be empty.")

    return features

# This function pads a batch of variable length handwriting sequences and returns the long tensor with shape B
# Returns:    
#       inputs:
    #      Float tensor with shape [B, T_max, 4].

    #   input_lengths:
    #       Long tensor with shape [B].
    #    key_padding_mask:
    #       Boolean tensor with shape [B, T_max].
    #       False = real handwriting point
    #       True = padded point

def collate_ink_batch(batch: Sequence[Any]) -> Dict[str, torch.Tensor]:


    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch.")

    sequences = [_get_features(sample) for sample in batch]

    input_lengths = torch.tensor(
        [sequence.shape[0] for sequence in sequences],
        dtype=torch.long,
    )

    inputs = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0,
    )

    max_length = inputs.shape[1]

    positions = torch.arange(max_length).unsqueeze(0)

    key_padding_mask = positions >= input_lengths.unsqueeze(1)

    return {
        "inputs": inputs,
        "input_lengths": input_lengths,
        "key_padding_mask": key_padding_mask,
    }