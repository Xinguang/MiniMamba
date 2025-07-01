import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Type, Union

from .block import MambaBlock
from .norm import RMSNorm
from .config import BaseMambaConfig, InferenceParams


class MambaEncoder(nn.Module):
    """
    Core Mamba encoder that can be used for various tasks.
    
    This is the pure encoder component without task-specific heads,
    making it reusable across different applications.
    """
    
    def __init__(self, config: BaseMambaConfig, mixer_cls: Optional[Type] = None):
        super().__init__()
        self.config = config
        
        # Stacked MambaBlocks with optional custom mixer
        self.layers = nn.ModuleList([
            MambaBlock(config, mixer_cls=mixer_cls) for _ in range(config.n_layer)
        ])
        
        # Final normalization
        if config.rms_norm:
            self.norm = RMSNorm(config.d_model, eps=config.norm_epsilon)
        else:
            self.norm = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        
        # For inference: assign layer index to each block
        for i, layer in enumerate(self.layers):
            layer.mixer.layer_idx = i
    
    def forward(self, hidden_states: Tensor, inference_params: Optional[InferenceParams] = None) -> Tensor:
        """
        Forward pass through all Mamba layers.
        
        Args:
            hidden_states: Input embeddings (batch, seq_len, d_model)
            inference_params: Optional inference parameters for generation
            
        Returns:
            Encoded hidden states (batch, seq_len, d_model)
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, inference_params)
        
        return self.norm(hidden_states)
    
    def reset_cache(self, inference_params: Optional[InferenceParams] = None) -> None:
        """Reset all cached states for inference."""
        if inference_params is not None:
            inference_params.reset_cache()
    
    def get_cache_info(self, inference_params: Optional[InferenceParams] = None) -> dict:
        """Get information about cached states."""
        if inference_params is None:
            return {'cached_layers': 0, 'memory_mb': 0, 'total_tensors': 0}
        return inference_params.get_cache_info()


class MambaEmbedding(nn.Module):
    """
    Embedding layer for Mamba models.
    
    Supports both token embeddings and positional embeddings if needed.
    """
    
    def __init__(self, vocab_size: int, d_model: int, pad_token_id: Optional[int] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if pad_token_id is not None:
            nn.init.constant_(self.token_embedding.weight[pad_token_id], 0.0)
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Convert token IDs to embeddings.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            
        Returns:
            Token embeddings (batch, seq_len, d_model)
        """
        return self.token_embedding(input_ids)


class MambaLMHead(nn.Module):
    """
    Language modeling head for Mamba models.
    
    Supports weight tying with embeddings and optional bias.
    """
    
    def __init__(self, d_model: int, vocab_size: int, bias: bool = False, 
                 tie_weights: bool = True, embedding_layer: Optional[nn.Embedding] = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=bias)
        
        # Weight tying with embeddings
        if tie_weights and embedding_layer is not None:
            self.lm_head.weight = embedding_layer.weight
        else:
            nn.init.normal_(self.lm_head.weight, std=0.02)
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Compute logits from hidden states.
        
        Args:
            hidden_states: Encoded states (batch, seq_len, d_model)
            
        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        return self.lm_head(hidden_states)


class MambaClassificationHead(nn.Module):
    """
    Classification head for Mamba models.
    
    Supports different pooling strategies and dropout.
    """
    
    def __init__(self, d_model: int, num_labels: int, pooling_strategy: str = "last", 
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels
        self.pooling_strategy = pooling_strategy
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(d_model, num_labels)
        
        # Initialize classifier
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Compute classification logits.
        
        Args:
            hidden_states: Encoded states (batch, seq_len, d_model)
            attention_mask: Optional attention mask (batch, seq_len)
            
        Returns:
            Classification logits (batch, num_labels)
        """
        if self.pooling_strategy == "last":
            # Use the last non-padding token
            if attention_mask is not None:
                sequence_lengths = (attention_mask.sum(dim=1) - 1).long()
                pooled = hidden_states[torch.arange(hidden_states.size(0)), sequence_lengths]
            else:
                pooled = hidden_states[:, -1]
        elif self.pooling_strategy == "mean":
            # Mean pooling over sequence length
            if attention_mask is not None:
                masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled = hidden_states.mean(dim=1)
        elif self.pooling_strategy == "max":
            # Max pooling over sequence length
            if attention_mask is not None:
                masked_hidden = hidden_states.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, float('-inf')
                )
                pooled = masked_hidden.max(dim=1)[0]
            else:
                pooled = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        pooled = self.dropout(pooled)
        return self.classifier(pooled)