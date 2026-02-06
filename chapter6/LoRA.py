import math
import torch

class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_features, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_features))
        self.alpha = alpha
        

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank=4, alpha=1.0):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank=rank, alpha=alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)
