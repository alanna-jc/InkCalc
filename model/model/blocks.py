import os
import numpy as np
import torch
from torch import nn
import tensorflow as tf


# AC TODO copilot suggest something about supporting device / dtype
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


class PositionWiseFFN(nn.Module):
    """
        A positionwise feedforward network that expands and then contracts the amount of params
        
        Activation function used is Swish, as mentioned in "MathWriting: A Dataset For Handwritten
        Mathematical Expression Recognition" section 4.2.

    Args:
        ffn_num_hidden: dim of expanded layer, default of 2048 from "Attention is all you need"
        embed_dim: hidden dim of model, default of 512 from "MathWriting: ..."
        dropout: probability of random zeroing of some elements, default of 0.15 from "MathWriting: ..."

    Source: 
        Linear, SiLU, Dropout (all from pyTorch)
    """
    def __init__(self, ffn_num_hidden = 2048, embed_dim = 512, dropout = 0.15):
        super().__init__()

        # fc layer: expansion
        self.expansion = nn.Linear(embed_dim, ffn_num_hidden)
        
        # activation 
        self.activation = nn.SiLU()

        # dropout
        self.dropout = nn.Dropout(dropout)

        # fc layer: contraction
        self.contraction = nn.Linear(ffn_num_hidden, embed_dim)
        
    def forward(self, x):
        x = self.expansion(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.contraction(x)  
        return x     
        

class TransformerEncoderBlock(nn.Module):
    """
    Encoder block of transformer with Pre-LN (motivated by: "On Layer Normalization in the Transformer Architecture").
    Flow is: Layer Norm, Multi Head Attention, Residual Add, PW FFN, Residual Add
 
    Args:
        num_heads: Number of heads for multi head attention. Default is 8, which divides embed_dim (AC TODO where this # from)
        ffn_num_hidden: dim of expanded layer, default of 2048 from "Attention is all you need"
        embed_dim: hidden dim of model, default of 512 from "MathWriting: ..."
        dropout: probability of random zeroing of some elements, default of 0.15 from "MathWriting: ..."

    Source:
        MultiHeadAttnetion: https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html
        LayerNorm, Dropout from PyTorch
    """
    def __init__(self, num_heads = 8, ffn_num_hidden = 2048, embed_dim = 512, dropout = 0.15):
        super().__init__()

        # AC TODO 
        if embed_dim % num_heads != 0:
            print("no!")

        # AC TODO masking for padding
        self.self_attention = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            dropout = dropout,
            batch_first = True, # input and output is (batch, seq, feature)
        )
    
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
        self.position_wise_ffn = PositionWiseFFN(ffn_num_hidden, embed_dim)
        
    def forward(self, x):

        norm = self.norm1(x)
        attn_output, _ = self.self_attention(norm, norm, norm, need_weights = False) # attn_output is (N: batch size, L: seq_length , E: embed_dim) 
        
        x = x + self.dropout1(attn_output)

        norm = self.norm2(x)
        ffn_output = self.position_wise_ffn(norm)

        x = x + self.dropout2(ffn_output)

        return x   


class TransformerEncoder(nn.Module):    
    """
    Transformer Encoder. Repeats the Transformer Encoder Block.
 
    Args:
        num_layers: Number of transformer blocks stacked. Default of 8 is motivated by "MathWriting: ..."
        num_heads: Number of heads for multi head attention. Default is 8, which divides embed_dim (AC TODO where this # from)
        ffn_num_hidden: dim of expanded layer, default of 2048 from "Attention is all you need"
        embed_dim: hidden dim of model, default of 512 from "MathWriting: ..."
        dropout: probability of random zeroing of some elements, default of 0.15 from "MathWriting: ..."

    Source:
        ModuleList by PyTorch
    """
    def __init__(self, num_layers = 11, num_heads = 8, ffn_num_hidden = 2048, embed_dim = 512, dropout = 0.15):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                num_heads=num_heads,
                ffn_num_hidden=ffn_num_hidden,
                embed_dim=embed_dim,
                dropout=dropout
            ) for _ in range(num_layers) # this repeats the input 11 times
        ])
    
    def forward(self, x):

        # ModuleList output is array like and can be treated as such, with each position being an instance of the transformer encoder block
        for block in self.blocks:
            x = block(x)

        return x


class CTCTransformer(nn.Module):
    """
    Decoder: We can view the decoder of a CTC model as a simple linear transformation followed by a softmax normalization. This layer should project all  
        steps of the encoder output into the dimensionality of the output alphabet. Source: https://distill.pub/2017/ctc/
        
    Args:
        encoder: the transformer encoder 
        vocab_size: total number of output classes including one blank for CTC 
        embed_dim: hidden dim of model, default of 512 from "MathWriting: ..."
    """
    def __init__(self, encoder, vocab_size, embed_dim = 512):
        super().__init__()

        # AC TODO : any inputs needed here
        self.encoder = encoder

        # AC this I BELIEVE is where you add in your tree aware stuff (next steps though, NOT NOW!)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        Returns: 
            logits: 
        """
        x = self.encoder(x)
        # AC TODO draw it out: logits output shape is (batch, time, vocab size) 
        logits = self.linear(x)

        return logits






