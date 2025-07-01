"""
Example demonstrating the improved Mamba implementation.

This example shows how to use the new modular architecture, improved parallel scan,
cache management, and specialized model classes.
"""

import torch
import torch.nn.functional as F
from minimamba import (
    # Configuration classes
    BaseMambaConfig,
    MambaLMConfig,
    MambaClassificationConfig,
    InferenceParams,
    
    # Core components
    MambaEncoder,
    
    # Specialized models
    MambaForCausalLM,
    MambaForSequenceClassification,
    MambaForFeatureExtraction,
    
    # Backward compatibility
    Mamba,
    MambaConfig
)


def example_1_configuration_system():
    """Example 1: Using the new configuration system."""
    print("=== Example 1: Configuration System ===")
    
    # Base configuration for core SSM parameters
    base_config = BaseMambaConfig(
        d_model=512,
        n_layer=12,
        d_state=16,
        d_conv=4,
        expand=2
    )
    print(f"Base config d_inner: {base_config.d_inner}")
    
    # Language modeling configuration
    lm_config = MambaLMConfig(
        d_model=512,
        n_layer=12,
        vocab_size=32000,
        tie_embeddings=True
    )
    print(f"LM config padded vocab size: {lm_config.padded_vocab_size}")
    
    # Classification configuration
    class_config = MambaClassificationConfig(
        d_model=512,
        n_layer=12,
        num_labels=3,
        pooling_strategy="mean",
        dropout=0.1
    )
    print(f"Classification config: {class_config.num_labels} labels, {class_config.pooling_strategy} pooling")
    
    # Save and load configurations
    lm_config.save_to_json("temp_config.json")
    loaded_config = MambaLMConfig.from_json("temp_config.json")
    print(f"Config loaded successfully: {loaded_config.d_model}")
    
    print()


def example_2_causal_language_modeling():
    """Example 2: Causal language modeling with generation."""
    print("=== Example 2: Causal Language Modeling ===")
    
    # Create configuration and model
    config = MambaLMConfig(
        d_model=256,
        n_layer=6,
        vocab_size=1000,
        d_state=16
    )
    
    model = MambaForCausalLM(config)
    model.eval()
    
    # Sample input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    print(f"Logits shape: {logits.shape}")
    
    # Training with labels
    model.train()
    labels = input_ids.clone()
    result = model(input_ids, labels=labels)
    print(f"Training loss: {result['loss'].item():.4f}")
    
    # Generation
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids[:1, :5],  # Single sequence, first 5 tokens
            max_new_tokens=5,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            use_cache=True
        )
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    print()


def example_3_sequence_classification():
    """Example 3: Sequence classification."""
    print("=== Example 3: Sequence Classification ===")
    
    # Create configuration and model
    config = MambaClassificationConfig(
        d_model=256,
        n_layer=6,
        num_labels=3,
        pooling_strategy="last",
        dropout=0.1
    )
    
    model = MambaForSequenceClassification(config)
    
    # Sample input (assuming we have a vocab_size in config)
    batch_size, seq_len = 2, 20
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 15:] = 0  # Mask last 5 tokens for first sequence
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
    print(f"Classification logits shape: {logits.shape}")
    print(f"Predicted classes: {torch.argmax(logits, dim=-1).tolist()}")
    
    # Training with labels
    model.train()
    labels = torch.randint(0, config.num_labels, (batch_size,))
    result = model(input_ids, attention_mask=attention_mask, labels=labels)
    print(f"Classification loss: {result['loss'].item():.4f}")
    
    print()


def example_4_feature_extraction():
    """Example 4: Feature extraction."""
    print("=== Example 4: Feature Extraction ===")
    
    # Create configuration and model
    config = BaseMambaConfig(
        d_model=256,
        n_layer=6,
        d_state=16
    )
    
    model = MambaForFeatureExtraction(config)
    
    # Sample input (pre-computed embeddings)
    batch_size, seq_len = 2, 15
    embeddings = torch.randn(batch_size, seq_len, config.d_model)
    
    # Extract features
    with torch.no_grad():
        features = model(embeddings)
    print(f"Features shape: {features.shape}")
    print(f"Feature vector norm (first token): {torch.norm(features[0, 0]).item():.4f}")
    
    print()


def example_5_cache_management():
    """Example 5: Cache management and streaming generation."""
    print("=== Example 5: Cache Management ===")
    
    config = MambaLMConfig(
        d_model=256,
        n_layer=4,
        vocab_size=1000
    )
    
    model = MambaForCausalLM(config)
    model.eval()
    
    # Initialize inference parameters
    inference_params = InferenceParams()
    
    # Process initial sequence
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    with torch.no_grad():
        logits = model(input_ids, inference_params)
    
    # Check cache info
    cache_info = model.get_cache_info(inference_params)
    print(f"Cache info: {cache_info}")
    
    # Streaming generation example
    print("Streaming generation:")
    generated_tokens = []
    
    for i, token in enumerate(model.generate_streaming(
        input_ids[:, :5],  # Start with first 5 tokens
        max_new_tokens=5,
        temperature=1.0,
        top_p=0.9
    )):
        generated_tokens.append(token)
        print(f"  Step {i+1}: Generated token {token}")
    
    print(f"All generated tokens: {generated_tokens}")
    
    # Reset cache
    model.reset_cache(inference_params)
    cache_info_after = model.get_cache_info(inference_params)
    print(f"Cache after reset: {cache_info_after}")
    
    print()


def example_6_parallel_scan_comparison():
    """Example 6: Parallel scan performance comparison."""
    print("=== Example 6: Parallel Scan Comparison ===")
    
    from minimamba.s6 import S6
    
    config = BaseMambaConfig(
        d_model=256,
        n_layer=1,
        d_state=16
    )
    
    s6_layer = S6(config)
    
    # Test with different sequence lengths
    for seq_len in [16, 64, 256]:
        print(f"\nSequence length: {seq_len}")
        
        batch_size, d_inner, d_state = 1, config.d_inner, config.d_state
        A = torch.rand(batch_size, seq_len, d_inner, d_state) * 0.9 + 0.1
        Bu = torch.randn(batch_size, seq_len, d_inner, d_state) * 0.1
        
        # Time sequential scan
        import time
        start_time = time.time()
        with torch.no_grad():
            states_seq = s6_layer._sequential_scan(A, Bu)
        seq_time = time.time() - start_time
        
        # Time parallel scan
        start_time = time.time()
        with torch.no_grad():
            states_par = s6_layer._parallel_scan_log_space(A, Bu)
        par_time = time.time() - start_time
        
        # Compare accuracy
        max_diff = torch.max(torch.abs(states_seq - states_par)).item()
        
        print(f"  Sequential time: {seq_time*1000:.2f}ms")
        print(f"  Parallel time: {par_time*1000:.2f}ms")
        print(f"  Speedup: {seq_time/par_time:.2f}x")
        print(f"  Max difference: {max_diff:.2e}")
    
    print()


def example_7_backward_compatibility():
    """Example 7: Backward compatibility."""
    print("=== Example 7: Backward Compatibility ===")
    
    # Original API still works
    config = MambaConfig(
        d_model=256,
        n_layer=4,
        vocab_size=1000
    )
    
    model = Mamba(config)
    
    # Original usage pattern
    input_ids = torch.randint(0, 1000, (2, 10))
    
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Legacy model output shape: {logits.shape}")
    print("✓ Backward compatibility maintained")
    
    print()


def example_8_custom_mixer():
    """Example 8: Using custom mixer components."""
    print("=== Example 8: Custom Mixer Components ===")
    
    from minimamba.block import MambaBlock
    from minimamba.s6 import S6
    
    # Create a custom mixer (example: S6 with different initialization)
    class CustomS6(S6):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            # Custom initialization or modifications
            print("Using custom S6 mixer")
    
    # Use custom mixer in block
    config = BaseMambaConfig(d_model=128, n_layer=2)
    block = MambaBlock(config, mixer_cls=CustomS6)
    
    # Test the custom block
    x = torch.randn(1, 10, config.d_model)
    with torch.no_grad():
        output = block(x)
    
    print(f"Custom mixer output shape: {output.shape}")
    print()


if __name__ == "__main__":
    print("Improved Mamba Implementation Examples")
    print("=" * 50)
    
    example_1_configuration_system()
    example_2_causal_language_modeling()
    example_3_sequence_classification()
    example_4_feature_extraction()
    example_5_cache_management()
    example_6_parallel_scan_comparison()
    example_7_backward_compatibility()
    example_8_custom_mixer()
    
    print("All examples completed successfully! 🎉")
    
    # Cleanup
    import os
    if os.path.exists("temp_config.json"):
        os.remove("temp_config.json")