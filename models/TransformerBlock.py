from models.MultiHeadAttention import MultiHeadAttention
from models.FeedForward import FeedForward
from models.LayerNorm import LayerNorm
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(d_in=cfg['emb_dim'], d_out=cfg['emb_dim'], context_length=cfg['context_length'], 
                                            dropout=cfg['dropout'], num_heads=cfg['num_heads'], bias=cfg['bias'])
        self.ffn = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.dropout = nn.Dropout(cfg['dropout'])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut
        return x
