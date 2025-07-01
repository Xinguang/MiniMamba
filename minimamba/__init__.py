# minimamba/__init__.py

# Version information
from ._version import __version__, __version_info__

# Configuration classes
from .config import (
    BaseMambaConfig,
    MambaConfig,
    MambaLMConfig,
    MambaClassificationConfig,
    InferenceParams
)

# Core components
from .core import (
    MambaEncoder,
    MambaEmbedding,
    MambaLMHead,
    MambaClassificationHead
)

# Specialized models
from .models import (
    MambaForCausalLM,
    MambaForSequenceClassification,
    MambaForFeatureExtraction,
    Mamba  # Backward compatibility
)

# Low-level components (for advanced users)
from .norm import RMSNorm
from .s6 import S6
from .block import MambaBlock
from .model import Mamba as MambaLegacy  # Legacy model

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    
    # Configuration
    "BaseMambaConfig",
    "MambaConfig",
    "MambaLMConfig",
    "MambaClassificationConfig",
    "InferenceParams",
    
    # Core components
    "MambaEncoder",
    "MambaEmbedding",
    "MambaLMHead",
    "MambaClassificationHead",
    
    # Specialized models
    "MambaForCausalLM",
    "MambaForSequenceClassification",
    "MambaForFeatureExtraction",
    "Mamba",
    
    # Low-level components
    "RMSNorm",
    "S6",
    "MambaBlock",
    "MambaLegacy"
]