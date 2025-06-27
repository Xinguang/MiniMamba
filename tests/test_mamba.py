import torch
import pytest
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from minimamba.model import Mamba
from minimamba.config import MambaConfig

# Fixture to initialize a small Mamba model and its config for testing
@pytest.fixture
def mamba_setup():
    config = MambaConfig(
        d_model=128,
        n_layer=2,
        vocab_size=500,
        d_state=8,
        d_conv=3,
        expand=2,
    )
    model = Mamba(config=config)
    return model, config

def test_model_construction(mamba_setup):
    model, config = mamba_setup
    # Ensure the model has parameters and can be built
    assert sum(p.numel() for p in model.parameters()) > 0

def test_forward_output_shape(mamba_setup):
    model, config = mamba_setup
    # Create dummy input. Note: We still use original vocab_size here, which is correct.
    input_ids = torch.randint(0, config.vocab_size, (4, 64))  # (batch_size=4, seq_len=64)
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    # --- FIX: Check output shape against the PADDED vocab size ---
    assert logits.shape == (4, 64, config.padded_vocab_size)

def test_empty_input(mamba_setup):
    model, config = mamba_setup
    # Handle edge case: zero-length sequence
    input_ids = torch.empty((2, 0), dtype=torch.long)
    
    with torch.no_grad():
        logits = model(input_ids)
    
    # --- FIX: Should return empty tensor with PADDED vocab size ---
    assert logits.shape == (2, 0, config.padded_vocab_size)