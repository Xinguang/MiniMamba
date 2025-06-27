import torch
import torch.nn as nn
from torch import Tensor

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    A simplified form of LayerNorm that does not subtract the mean.

    Reference:
    https://arxiv.org/abs/1910.07467
    """
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        # Scale input by RMS and learned weight
        return x * rms * self.weight