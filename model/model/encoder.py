import torch
from torch import nn

class PositionWiseFFN(nn.Module):
    """
        A positionwise feedforward network that expands and then contracts the amount of dims
        
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

        self.expansion = nn.Linear(embed_dim, ffn_num_hidden)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.contraction = nn.Linear(ffn_num_hidden, embed_dim)
        
    def forward(self, x):
        x = self.expansion(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.contraction(x)  
        return x     
        

class TransformerEncoderBlock(nn.Module):
    """
    Encoder block of transformer with Peri-LN motivated by "Peri-LN: Revisiting Layer Normalization in the Transformer Architecture"
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

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

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
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
    
        self.position_wise_ffn = PositionWiseFFN(ffn_num_hidden, embed_dim, dropout)
        
    def forward(self, x, key_padding_mask):
        """
        Args:
            x: input of shape 
            key_padding_mask: shape (batch_size, seq_length), (contains bools: False if not padding, True if padding)

        Returns:
            x: insert shape 
        """
        norm = self.norm1(x)
        # AC TODO add key_padding_mask
        attn_output, _ = self.self_attention(norm, norm, norm, 
                                             key_padding_mask = key_padding_mask,
                                             need_weights = False) 
        # attn_output is (N: batch size, L: seq_length , E: embed_dim) 
        
        x = x + self.norm2(self.dropout1(attn_output))

        norm = self.norm3(x)
        ffn_output = self.position_wise_ffn(norm)

        x = x + self.norm4(self.dropout2(ffn_output))

        return x   


class TransformerEncoder(nn.Module):    
    """
    Transformer Encoder. Repeats the Transformer Encoder Block num_layers times.
 
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
            ) for _ in range(num_layers) # this repeats the input 'num_layers' times
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, key_padding_mask):

        # ModuleList output is array like and can be treated as such, with each position being an instance of the transformer encoder block
        for block in self.blocks:
            x = block(x, key_padding_mask)

        x = self.final_norm(x)

        return x


class CTCTransformer(nn.Module):
    """
    Wraps entire transformer and adds a linear layer
        
    Args:
        encoder: the transformer encoder 
        vocab_size: total number of output classes including one blank for CTC 
        embed_dim: hidden dim of model, default of 512 from "MathWriting: ..."
    """
    def __init__(self, vocab_size, num_layers = 11, num_heads = 8, ffn_num_hidden = 2048, embed_dim = 512, dropout = 0.15):
        super().__init__()

        # AC TODO : any inputs needed here
        self.encoder = TransformerEncoder(num_layers = num_layers, 
                                          num_heads = num_heads, 
                                          ffn_num_hidden = ffn_num_hidden, 
                                          embed_dim = embed_dim, 
                                          dropout = dropout)

        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
            key_padding_mask: Bool tensor of shape (batch_size, seq_len)

        Returns: 
            logits: of shape (batch, seq_len, vocab size) 
        """
        ## Explicitly route the mask down to the encoder
        x = self.encoder(x,key_padding_mask=key_padding_mask)
        logits = self.linear(x)

        return logits