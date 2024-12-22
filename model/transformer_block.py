import torch
from torch import nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim = 256,
        num_heads = 4,
        dropout_p = 0.1
        ):
        super().__init__()
        
        self.multihead_attn_layer = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            batch_first = True
        )
        self.norm_attn_layer = nn.LayerNorm(embed_dim)
        
        self.feed_forward_layer = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim),
            nn.Dropout(dropout_p),
        )
        self.norm_feed_layer = nn.LayerNorm(embed_dim)
        
    def forward(self, tokens, attn_mask):
        attn, attn_weights = self.multihead_attn_layer(
            query = tokens,
            key = tokens,
            value = tokens,
            key_padding_mask = attn_mask
        )
        tokens = self.norm_attn_layer(tokens + attn)
        
        feed_forward_result = self.feed_forward_layer(tokens)
        return self.norm_feed_layer(tokens + feed_forward_result)
        
        