import torch
import torch.nn as nn
class LayerNorm(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.eps = 1e-5
        self.gamma = nn.Parameter(torch.ones(d_in))
        self.beta = nn.Parameter(torch.zeros(d_in))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / (torch.sqrt(var) + self.eps)
        return self.gamma * norm_x + self.beta