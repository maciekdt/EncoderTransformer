import torch
from torch import nn
import torch.nn.functional as F
from model.transformer_block import TransformerBlock

class EncoderModel(nn.Module):
    def __init__(
        self,
        vocab_size = 50_000,
        seq_dim = 512,
        embed_dim = 256,
        num_heads = 1,
        num_transformer_blocks = 2,
        dropout_p = .1,
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_dim = seq_dim
        
        self.token_embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_layer = nn.Embedding(seq_dim, embed_dim)
        self.norm_embed_layer = nn.LayerNorm(embed_dim)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim = embed_dim,
                num_heads = num_heads,
                dropout_p = dropout_p,
            )
            for _ in range(num_transformer_blocks)
        ])
        
        self.flatten_layer = nn.Linear(embed_dim, 1)
        self.bin_layer = nn.Sequential(
            nn.Linear(seq_dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, input):
        positions = torch.arange(0, self.seq_dim)
        positions = self.position_embedding_layer(positions)
        
        tokens = input["input_ids"]
        tokens = self.token_embedding_layer(tokens)
        tokens = tokens + positions
        tokens = self.norm_embed_layer(tokens)
        
        attn_mask = input["attention_mask"]
        attn_mask = torch.where(
            condition = (attn_mask == 1),
            input = torch.tensor(0.0),
            other = torch.tensor(float('-inf'))
        )
        
        for block in self.transformer_blocks:
            tokens = block(tokens = tokens, attn_mask = attn_mask)
        
        tokens = self.flatten_layer(tokens)
        tokens = tokens.squeeze(2)
        
        pred = self.bin_layer(tokens)
        pred = pred.squeeze(1)
        return pred