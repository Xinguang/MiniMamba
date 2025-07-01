import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Optional

from .s6 import S6
from .norm import RMSNorm
from .config import BaseMambaConfig, MambaConfig

class MambaBlock(nn.Module):
    """
    A single Mamba block with pre-normalization and residual connection.
    Combines normalization, the S6 mixer, and residual addition.
    """
    def __init__(self, config: BaseMambaConfig, mixer_cls: Optional[Type] = None):
        super().__init__()

        self.config = config

        # Use RMSNorm or LayerNorm for pre-normalization
        if config.rms_norm:
            self.norm = RMSNorm(config.d_model, eps=config.norm_epsilon)
        else:
            self.norm = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

        # S6 (Selective State Space) mixer layer - now pluggable
        mixer_class = mixer_cls or S6
        self.mixer = mixer_class(config=config)

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