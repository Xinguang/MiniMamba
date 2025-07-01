import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union


@dataclass
class BaseMambaConfig:
    """
    Base Mamba configuration containing core SSM parameters.
    
    Args:
        d_model: Model dimension, controls representation capacity
        n_layer: Number of Mamba layers
        d_state: SSM state dimension, affects long-range dependency modeling capability
        d_conv: Convolution kernel size, controls local interaction range
        expand: Internal dimension expansion ratio (d_inner = expand * d_model)
        dt_rank: Rank of time step parameter, "auto" means automatic calculation
        dt_min: Minimum value for time step initialization
        dt_max: Maximum value for time step initialization
        dt_init: Time step initialization method ("random" or "constant")
        dt_scale: Time step scaling factor
        dt_init_floor: Floor value for time step initialization
        conv_bias: Whether to use bias in convolution layers
        bias: Whether to use bias in linear layers
        norm_epsilon: Epsilon for RMS normalization
        rms_norm: Whether to use RMS normalization
        use_fast_path: Whether to use optimized computation paths
        device: Device to place the model on
        dtype: Data type for model parameters
    """
    d_model: int = 768
    n_layer: int = 12
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    norm_epsilon: float = 1e-5
    rms_norm: bool = True
    use_fast_path: bool = True
    device: Optional[str] = None
    dtype: Optional[str] = None

    def __post_init__(self):
        if self.dt_rank == "auto":
            self.dt_rank = max(1, self.d_model // 16)
        
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def d_inner(self) -> int:
        """Internal dimension after expansion"""
        return int(self.expand * self.d_model)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseMambaConfig':
        """Create configuration from dictionary"""
        # Filter out computed fields that shouldn't be passed to constructor
        constructor_dict = {k: v for k, v in config_dict.items()
                          if k not in ['d_inner', 'padded_vocab_size']}
        return cls(**constructor_dict)

    def save_to_json(self, file_path: str):
        """Save configuration to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, file_path: str) -> 'BaseMambaConfig':
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json_file(cls, json_file: str):
        """Creates an instance from a JSON file."""
        return cls.from_json(json_file)


@dataclass
class MambaLMConfig(BaseMambaConfig):
    """
    Mamba configuration specialized for language modeling tasks.
    
    Additional Args:
        vocab_size: Vocabulary size for language modeling
        pad_vocab_size_multiple: Pad vocabulary size to multiple of this value
        tie_embeddings: Whether to tie embedding and output projection weights
    """
    vocab_size: int = 32000
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True

    def __post_init__(self):
        super().__post_init__()
        
        # Pad vocabulary size for efficiency
        remainder = self.vocab_size % self.pad_vocab_size_multiple
        if remainder != 0:
            self.padded_vocab_size = self.vocab_size + (self.pad_vocab_size_multiple - remainder)
        else:
            self.padded_vocab_size = self.vocab_size


@dataclass
class MambaClassificationConfig(BaseMambaConfig):
    """
    Mamba configuration for sequence classification tasks.
    
    Additional Args:
        num_labels: Number of classification labels
        pooling_strategy: How to pool sequence representations ("last", "mean", "max")
        dropout: Dropout rate for classification head
    """
    num_labels: int = 2
    pooling_strategy: str = "last"
    dropout: float = 0.1


@dataclass
class InferenceParams:
    """
    Parameters for inference mode.
    
    Args:
        cache: Dictionary to store cached states
        seqlen_offset: Sequence length offset for incremental generation
        max_sequence_len: Maximum sequence length supported
        max_batch_size: Maximum batch size supported
    """
    cache: Dict[str, Any] = field(default_factory=dict)
    seqlen_offset: int = 0
    max_sequence_len: int = 2048
    max_batch_size: int = 1

    def reset_cache(self):
        """Reset all cached states"""
        self.cache.clear()
        self.seqlen_offset = 0

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached states"""
        import torch
        total_memory = 0
        layer_count = 0
        
        for key, value in self.cache.items():
            if isinstance(value, torch.Tensor):
                total_memory += value.numel() * value.element_size()
            if 'conv_state_' in key or 'ssm_state_' in key:
                layer_count += 1
        
        return {
            'cached_layers': layer_count // 2,  # Each layer has conv_state and ssm_state
            'total_tensors': len(self.cache),
            'memory_mb': total_memory / 1024 / 1024,
            'seqlen_offset': self.seqlen_offset
        }


# Backward compatibility
MambaConfig = MambaLMConfig