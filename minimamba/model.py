import torch
import torch.nn as nn
from torch import Tensor

from .block import MambaBlock
from .norm import RMSNorm

class Mamba(nn.Module):
    """
    Complete Mamba model with multiple layers.
    Includes embedding, stacked MambaBlocks, normalization, and output head.
    """
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        pad_vocab_size_multiple: int = 1,
        norm_epsilon: float = 1e-5,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size

        # Pad vocab size to be divisible by a multiple (for efficient matmul)
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Stacked MambaBlocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, norm_epsilon, **kwargs)
            for _ in range(n_layer)
        ])

        # Final normalization
        self.norm_f = RMSNorm(d_model, eps=norm_epsilon)

        # Output linear layer
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

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
