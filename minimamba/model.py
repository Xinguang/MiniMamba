import torch
import torch.nn as nn
from torch import Tensor

from .block import MambaBlock
from .norm import RMSNorm
from .config import MambaConfig

class Mamba(nn.Module):
    """
    Complete Mamba model with multiple layers.
    Includes embedding, stacked MambaBlocks, normalization, and output head.
    """
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        # --- FIX: Use the pre-calculated padded_vocab_size from the config ---
        # Token embedding
        self.embedding = nn.Embedding(config.padded_vocab_size, config.d_model)

        # Stacked MambaBlocks
        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(config.n_layer)
        ])

        # Final normalization
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_epsilon)

        # --- FIX: Use the pre-calculated padded_vocab_size from the config ---
        # Output linear layer
        self.lm_head = nn.Linear(config.d_model, config.padded_vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

        # For inference: assign layer index to each block
        for i, layer in enumerate(self.layers):
            layer.mixer.layer_idx = i

    def forward(self, input_ids: Tensor, inference_params=None) -> Tensor:
        """
        Forward pass of the full Mamba model.

        Args:
            input_ids: Input tensor of shape (batch, seq_len)
            inference_params: Optional inference context for autoregressive generation

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size)
        """
        hidden_states = self.embedding(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, inference_params)

        hidden_states = self.norm_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits