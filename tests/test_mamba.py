import torch
import pytest
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from minimamba.model import Mamba

# Fixture to initialize a small Mamba model for testing
@pytest.fixture
def model():
    config = {
        'd_model': 128,
        'n_layer': 2,
        'vocab_size': 500,
        'd_state': 8,
        'd_conv': 3,
        'expand': 2,
    }
    return Mamba(**config)

def test_model_construction(model):
    # Ensure the model has parameters and can be built
    assert sum(p.numel() for p in model.parameters()) > 0

def test_forward_output_shape(model):
    # Create dummy input
    input_ids = torch.randint(0, 500, (4, 64))  # (batch_size=4, seq_len=64)
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    # Check output shape
    assert logits.shape == (4, 64, 500)

def test_empty_input(model):
    # Handle edge case: zero-length sequence
    input_ids = torch.empty((2, 0), dtype=torch.long)
    
    with torch.no_grad():
        logits = model(input_ids)
    
    # Should return empty tensor with correct shape
    assert logits.shape == (2, 0, 500)
