# reorg these classes into diff files

import os
import numpy as np
import torch
from torch import nn
import tensorflow as tf

# TODO make this have correct input and output (draw it out)
# TODO copilot suggest something about supporting device / dtype
def positional_encoding(seq_length, depth):
    """
    This class implements the Positional Encoding formulas from "Attention is all you need" section 3.5

    Args:
        nn (_type_): _description_
    """
        
    # each dimension of corresponds to a sinusoid
    # making zeros
    pe = np.zeros((seq_length, depth))

    # python 'np.arange' creates ab array from 0 to length of input
    # the second term reshapes it so that the array is vertical not horizontal	

    # every row is a position
    # every column is a depth of dim
    position = np.arange(seq_length)[:, np.newaxis]

    div_term = np.exp(np.arange(0, depth, 2) * -(np.log(10000.0) / depth)) 

    # all rows, every other column
    # python broadcatss the (5,1) * (1, depth/2) into pe
    pe[:,0::2] = np.sin(position * div_term)
    pe[:,1::2] = np.cos(position * div_term)

    # ok now we output pe (idk how i want to do this yet, but this is the input into the encoders block)
    return pe


# position wise feed forward network
class PositionWiseFFN(nn.Module):
    """
        A feedforward network that expands and then contracts the amount of params
        Activation function used is LeakyReLU (will change TODO)
        
        (aka: "two convolutions with kernel size 1")
        attenttion is all you need: 512 input output, 2048 hidden
        TODO following attnetion is all you need you use ReLU
        TODO following MathWriting uses swish i believe, so we use swish

    Args:
        num_hidden: dim of hidden layer to input
        num_output: dim of output
    """
    def __init__(self, ffn_num_hidden = 2048, embed_dim = 512, dropout = 0.15):
        super().__init__()
        # dense layer: expansion
        self.expansion = nn.Linear(embed_dim, ffn_num_hidden)
        
        # activation 
        self.activation = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

        # dense layer: contraction
        self.contraction = nn.Linear(ffn_num_hidden, embed_dim)
        
    def forward(self, x):

        x = self.expansion(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.contraction(x)  

        return x     
        

# nn.Module is pytorch's neural network class
class TransformerEncoderBlock(nn.Module):
    """
    Encoder block of transformer
 
    Args:
        num_hidden:
        num_heads:
        ffn_num_hidden:
        dropout:

    Returns:
        output of encoder block

    Source:
        MultiHeadAttnetion: https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html
    """
    
    # init
    def __init__(self, num_heads, ffn_num_hidden, embed_dim = 512, dropout = 0.15):
        super().__init__()

        # AC TODO 
        if embed_dim % num_heads != 0:
            print("no!")

        # multi head attention
        # AC TODO masking for padding
        self.self_attention = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            dropout = dropout,
            batch_first = True, # input and output is (batch, seq, feature)
            # AC ad bias?
        )
    
        # add and norm input
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
        # position wise feed forward network
        self.position_wise_ffn = PositionWiseFFN(ffn_num_hidden, embed_dim)
        
    def forward(self, x):

        # TODO AC is this true? not used in example codes
        # attn_output is (N, L, E) if batched(how do i do or not do this AC)
        # L is target sequence length, N is batch size, E is embedding dim
        "Pre-LN addition brought to you by: 'On Layer Normalization in the Transformer Architecture'"
        # self attention there query key and value are all the same
        norm = self.norm1(x)
        attn_output, _ = self.self_attention(norm, norm, norm, need_weights = False)
        
        x = x + self.dropout1(attn_output)

        norm = self.norm2(x)
        ffn_output = self.position_wise_ffn(norm)

        x = x + self.dropout2(ffn_output)

        return x   


class TransformerEncoder(nn.Module):    
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

        # ModuleList output is array like and can be treated as such
        # each block is a different instance of the transformer block
        for block in self.blocks:
            x = block(x)

        return x


class CTCTransformer(nn.Module):
    
    def __init__(self, encoder,  vocab_size, embed_dim=512):
        super().__init__()

        """Decoder: We can view the decoder of a CTC model as a simple linear transformation followed by a softmax normalization. This layer should project all  
        steps of the encoder output into the dimensionality of the output alphabet. Source: https://distill.pub/2017/ctc/"""
        # run the encoder
        self.encoder = encoder
        # a linear layer here??
        # this I BELIEVE is where you add in your tree aware stuff (next steps though)
        self.linear = nn.Linear(embed_dim, vocab_size)

        # 1 softmax layer that also outputs a blank label required by the CTC loss decoder
        # this is the layer that provides the probability distribution aka the logits
        
    
        #  and then we run it through the beam search ctc bad boy guy. not sure if thats here tho
    def forward(self, x):
        x = self.encoder(x)
        logits = self.linear(x)
        # now put through soft max
        # return out put of softmax

def CTC_loss():
    # this is where we do the actual loss function and beam search


