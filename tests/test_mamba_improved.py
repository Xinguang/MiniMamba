import unittest
import torch
import torch.nn as nn
import math
from minimamba import (
    BaseMambaConfig,
    MambaLMConfig,
    MambaClassificationConfig,
    InferenceParams,
    MambaEncoder,
    MambaForCausalLM,
    MambaForSequenceClassification,
    MambaForFeatureExtraction,
    S6
)


class TestImprovedMamba(unittest.TestCase):
    """Test suite for the improved Mamba implementation."""
    
    def setUp(self):
        """Set up test configurations."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Small config for fast testing
        self.base_config = BaseMambaConfig(
            d_model=64,
            n_layer=2,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        self.lm_config = MambaLMConfig(
            d_model=64,
            n_layer=2,
            vocab_size=1000,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        self.classification_config = MambaClassificationConfig(
            d_model=64,
            n_layer=2,
            num_labels=3,
            d_state=16,
            d_conv=4,
            expand=2
        )
    
    def test_config_decoupling(self):
        """Test that configuration classes are properly decoupled."""
        # Base config should not require vocab_size
        base_config = BaseMambaConfig(d_model=128, n_layer=4)
        self.assertEqual(base_config.d_model, 128)
        self.assertEqual(base_config.n_layer, 4)
        self.assertFalse(hasattr(base_config, 'vocab_size'))
        
        # LM config should have vocab_size and padded_vocab_size
        lm_config = MambaLMConfig(d_model=128, n_layer=4, vocab_size=1000)
        self.assertEqual(lm_config.vocab_size, 1000)
        self.assertTrue(hasattr(lm_config, 'padded_vocab_size'))
        
        # Classification config should have num_labels
        class_config = MambaClassificationConfig(d_model=128, n_layer=4, num_labels=5)
        self.assertEqual(class_config.num_labels, 5)
        self.assertEqual(class_config.pooling_strategy, "last")
    
    def test_inference_params(self):
        """Test inference parameters functionality."""
        params = InferenceParams()
        
        # Test cache management
        params.cache['test_key'] = torch.randn(2, 64, 16)
        cache_info = params.get_cache_info()
        
        self.assertEqual(cache_info['total_tensors'], 1)
        self.assertGreater(cache_info['memory_mb'], 0)
        
        # Test cache reset
        params.reset_cache()
        self.assertEqual(len(params.cache), 0)
        self.assertEqual(params.seqlen_offset, 0)
    
    def test_mamba_encoder(self):
        """Test the core MambaEncoder functionality."""
        encoder = MambaEncoder(self.base_config)
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, self.base_config.d_model)
        
        output = encoder(x)
        self.assertEqual(output.shape, (batch_size, seq_len, self.base_config.d_model))
        
        # Test with inference params
        inference_params = InferenceParams()
        output_cached = encoder(x, inference_params)
        self.assertEqual(output_cached.shape, (batch_size, seq_len, self.base_config.d_model))
        
        # Check that cache was populated
        cache_info = encoder.get_cache_info(inference_params)
        self.assertEqual(cache_info['cached_layers'], self.base_config.n_layer)
    
    def test_causal_lm_model(self):
        """Test MambaForCausalLM functionality."""
        model = MambaForCausalLM(self.lm_config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, self.lm_config.vocab_size, (batch_size, seq_len))
        
        # Test forward pass without labels
        logits = model(input_ids)
        expected_shape = (batch_size, seq_len, self.lm_config.padded_vocab_size)
        self.assertEqual(logits.shape, expected_shape)
        
        # Test forward pass with labels
        labels = torch.randint(0, self.lm_config.vocab_size, (batch_size, seq_len))
        result = model(input_ids, labels=labels)
        
        self.assertIsInstance(result, dict)
        self.assertIn('loss', result)
        self.assertIn('logits', result)
        self.assertEqual(result['logits'].shape, expected_shape)
        
        # Test generation
        generated = model.generate(
            input_ids[:1, :5],  # Single batch, shorter sequence
            max_new_tokens=5,
            do_sample=False,  # Greedy for deterministic test
            use_cache=True
        )
        
        self.assertEqual(generated.shape[0], 1)
        self.assertEqual(generated.shape[1], 10)  # 5 input + 5 generated
    
    def test_classification_model(self):
        """Test MambaForSequenceClassification functionality."""
        model = MambaForSequenceClassification(self.classification_config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Test forward pass without labels
        logits = model(input_ids)
        expected_shape = (batch_size, self.classification_config.num_labels)
        self.assertEqual(logits.shape, expected_shape)
        
        # Test forward pass with labels
        labels = torch.randint(0, self.classification_config.num_labels, (batch_size,))
        result = model(input_ids, labels=labels)
        
        self.assertIsInstance(result, dict)
        self.assertIn('loss', result)
        self.assertIn('logits', result)
        
        # Test with attention mask
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 7:] = 0  # Mask last 3 tokens for first sequence
        
        logits_masked = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(logits_masked.shape, expected_shape)
    
    def test_feature_extraction_model(self):
        """Test MambaForFeatureExtraction functionality."""
        model = MambaForFeatureExtraction(self.base_config)
        
        batch_size, seq_len = 2, 10
        
        # Test with embedding inputs (token IDs)
        if hasattr(model, 'has_embeddings') and model.has_embeddings:
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            features = model(input_ids)
        else:
            # Test with pre-computed embeddings
            embeddings = torch.randn(batch_size, seq_len, self.base_config.d_model)
            features = model(embeddings)
        
        expected_shape = (batch_size, seq_len, self.base_config.d_model)
        self.assertEqual(features.shape, expected_shape)
    
    def test_parallel_scan_correctness(self):
        """Test that the parallel scan produces correct results."""
        config = BaseMambaConfig(d_model=32, n_layer=1, d_state=8)
        s6_layer = S6(config)
        
        batch_size, seq_len, d_inner, d_state = 1, 16, config.d_inner, config.d_state
        
        # Create test inputs
        A = torch.rand(batch_size, seq_len, d_inner, d_state) * 0.9 + 0.1
        Bu = torch.randn(batch_size, seq_len, d_inner, d_state) * 0.1
        
        # Test different scan methods
        states_sequential = s6_layer._sequential_scan(A, Bu)
        states_parallel = s6_layer._true_parallel_scan(A, Bu)
        
        # They should produce very similar results (within numerical precision)
        max_diff = torch.max(torch.abs(states_sequential - states_parallel))
        self.assertLess(max_diff.item(), 1e-4, 
                       f"Parallel scan differs from sequential by {max_diff.item()}")
    
    def test_parallel_scan_performance(self):
        """Test that parallel scan works with different sequence lengths."""
        config = BaseMambaConfig(d_model=64, n_layer=1, d_state=16)
        s6_layer = S6(config)
        
        # Test with different sequence lengths
        for seq_len in [8, 32, 128]:
            batch_size, d_inner, d_state = 1, config.d_inner, config.d_state
            
            A = torch.rand(batch_size, seq_len, d_inner, d_state) * 0.9 + 0.1
            Bu = torch.randn(batch_size, seq_len, d_inner, d_state) * 0.1
            
            # Should not raise any errors
            states = s6_layer._parallel_scan_log_space(A, Bu)
            self.assertEqual(states.shape, (batch_size, seq_len, d_inner, d_state))
    
    def test_training_vs_inference_consistency(self):
        """Test that training and inference modes produce consistent results."""
        model = MambaForCausalLM(self.lm_config)
        model.eval()
        
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, self.lm_config.vocab_size, (batch_size, seq_len))
        
        # Training mode (no cache)
        with torch.no_grad():
            logits_training = model(input_ids)
        
        # Inference mode (with cache, but process full sequence at once)
        inference_params = InferenceParams()
        with torch.no_grad():
            logits_inference = model(input_ids, inference_params)
        
        # Should produce identical results
        max_diff = torch.max(torch.abs(logits_training - logits_inference))
        self.assertLess(max_diff.item(), 1e-5,
                       f"Training vs inference mode differs by {max_diff.item()}")
    
    def test_incremental_generation_consistency(self):
        """Test that incremental generation produces consistent results."""
        model = MambaForCausalLM(self.lm_config)
        model.eval()
        
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, self.lm_config.vocab_size, (batch_size, seq_len))
        
        # Generate one token at a time
        inference_params = InferenceParams()
        generated_tokens = []
        
        current_ids = input_ids.clone()
        
        with torch.no_grad():
            for step in range(3):  # Generate 3 tokens
                if step == 0:
                    logits = model(current_ids, inference_params)
                else:
                    logits = model(current_ids[:, -1:], inference_params)
                
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated_tokens.append(next_token.item())
                current_ids = torch.cat([current_ids, next_token], dim=1)
                inference_params.seqlen_offset += 1
        
        # Compare with batch generation
        with torch.no_grad():
            batch_generated = model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=False,
                use_cache=True
            )
        
        # The generated tokens should match
        for i, token in enumerate(generated_tokens):
            expected_token = batch_generated[0, seq_len + i].item()
            self.assertEqual(token, expected_token,
                           f"Token {i}: incremental={token}, batch={expected_token}")
    
    def test_memory_efficiency(self):
        """Test that caching provides memory efficiency benefits."""
        model = MambaForCausalLM(self.lm_config)
        model.eval()
        
        batch_size, seq_len = 1, 20
        input_ids = torch.randint(0, self.lm_config.vocab_size, (batch_size, seq_len))
        
        inference_params = InferenceParams()
        
        with torch.no_grad():
            # Process initial sequence
            _ = model(input_ids, inference_params)
            
            # Check cache info
            cache_info = model.get_cache_info(inference_params)
            self.assertEqual(cache_info['cached_layers'], self.lm_config.n_layer)
            self.assertGreater(cache_info['memory_mb'], 0)
            
            # Process additional token (should be much faster)
            next_token = torch.randint(0, self.lm_config.vocab_size, (1, 1))
            _ = model(next_token, inference_params)
            
            # Cache should still be there
            cache_info_after = model.get_cache_info(inference_params)
            self.assertEqual(cache_info_after['cached_layers'], self.lm_config.n_layer)
    
    def test_backward_compatibility(self):
        """Test that the original API still works."""
        from minimamba import Mamba, MambaConfig
        
        # Original API should still work
        config = MambaConfig(
            d_model=64,
            n_layer=2,
            vocab_size=1000
        )
        
        model = Mamba(config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        logits = model(input_ids)
        expected_shape = (batch_size, seq_len, config.padded_vocab_size)
        self.assertEqual(logits.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()