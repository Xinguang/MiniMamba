import torch
import torch.nn as nn
from torch import Tensor

from .s6 import S6
from .norm import RMSNorm

class MambaBlock(nn.Module):
    """
    A single Mamba block with pre-normalization and residual connection.
    Combines normalization, the S6 mixer, and residual addition.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model

        # Use RMSNorm or LayerNorm for pre-normalization
        if rms_norm:
            self.norm = RMSNorm(d_model, eps=norm_epsilon)
        else:
            self.norm = nn.LayerNorm(d_model, eps=norm_epsilon)

        # S6 (Selective State Space) mixer layer
        self.mixer = S6(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )

    def forward(self, hidden_states: Tensor, inference_params=None) -> Tensor:
        """
        Forward pass of the MambaBlock.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, d_model)
            inference_params: Optional parameters for generation mode

        Returns:
            Output tensor of same shape (batch, seq_len, d_model)
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states, inference_params)
        return hidden_states + residual
