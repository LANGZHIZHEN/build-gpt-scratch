import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.input_dim = d_in
        self.output_dim = d_out
        self.nums_heads = num_heads
        self.head_dim = d_out // self.nums_heads

        self.w_q = nn.Linear(self.input_dim, self.output_dim, bias=bias)
        self.w_k = nn.Linear(self.input_dim, self.output_dim, bias=bias)
        self.w_v = nn.Linear(self.input_dim, self.output_dim, bias=bias)

        self.out_proj = nn.Linear(self.output_dim, self.output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.w_k(x)
        queries = self.w_q(x)
        values = self.w_v(x)
        keys = keys.view(b, num_tokens, self.nums_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.nums_heads, self.head_dim)
        values = values.view(b, num_tokens, self.nums_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_score = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[0:num_tokens, 0:num_tokens]
        attn_score = attn_score.masked_fill_(mask_bool, -torch.inf)
        attn_weight = torch.softmax(attn_score / keys.shape[-1] ** 0.5, dim=-1)
        attn_weight = self.dropout(attn_weight)

        context_vec = (attn_weight @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.output_dim)
        context_vec = self.out_proj(context_vec)
        return context_vec

