import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, List
from types import SimpleNamespace

from .block import MambaBlock
from .norm import RMSNorm
from .config import MambaConfig, MambaLMConfig, InferenceParams

class Mamba(nn.Module):
    """
    Complete Mamba model with multiple layers.
    Includes embedding, stacked MambaBlocks, normalization, and output head.
    """
    def __init__(self, config: Union[MambaConfig, MambaLMConfig]):
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
    
    def reset_cache(self, inference_params: Optional[InferenceParams] = None) -> None:
        """
        Reset all cached states for inference.
        
        Args:
            inference_params: Inference parameters containing cache
        """
        if inference_params is None:
            return
        inference_params.reset_cache()
    
    def get_cache_info(self, inference_params: Optional[InferenceParams] = None) -> dict:
        """
        Get information about cached states.
        
        Args:
            inference_params: Inference parameters containing cache
            
        Returns:
            Dictionary with cache information
        """
        if inference_params is None:
            return {'cached_layers': 0, 'memory_mb': 0, 'total_tensors': 0}
        return inference_params.get_cache_info()
    
    def prepare_inputs_for_generation(self, input_ids: Tensor,
                                    inference_params: Optional[InferenceParams] = None) -> dict:
        """
        Prepare inputs for autoregressive generation.
        
        Args:
            input_ids: Input token IDs
            inference_params: Inference parameters
            
        Returns:
            Dictionary with prepared inputs
        """
        if inference_params is None:
            inference_params = InferenceParams()
        
        return {
            'input_ids': input_ids,
            'inference_params': inference_params
        }
    
    def _top_p_filter(self, logits: Tensor, top_p: float) -> Tensor:
        """
        Apply top-p (nucleus) filtering to logits.
        
        Args:
            logits: Logits tensor (batch_size, vocab_size)
            top_p: Cumulative probability threshold
            
        Returns:
            Filtered logits
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits
    
    @torch.no_grad()
    def generate(self,
                input_ids: Tensor,
                max_length: int = 50,
                max_new_tokens: Optional[int] = None,
                temperature: float = 1.0,
                top_p: float = 0.9,
                top_k: Optional[int] = None,
                do_sample: bool = True,
                pad_token_id: Optional[int] = None,
                eos_token_id: Optional[int] = None,
                use_cache: bool = True) -> Tensor:
        """
        Generate sequences using autoregressive sampling.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            max_length: Maximum total sequence length
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Top-p (nucleus) sampling threshold
            top_k: Top-k sampling threshold
            do_sample: Whether to use sampling (vs greedy)
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            use_cache: Whether to use inference cache for efficiency
            
        Returns:
            Generated token sequences (batch_size, total_length)
        """
        batch_size, input_length = input_ids.shape
        device = input_ids.device
        
        # Determine actual max length
        if max_new_tokens is not None:
            actual_max_length = input_length + max_new_tokens
        else:
            actual_max_length = max_length
        
        # Initialize inference parameters if using cache
        inference_params = None
        if use_cache:
            inference_params = InferenceParams()
        
        # Track generated sequences
        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generation loop
        for step in range(actual_max_length - input_length):
            # Prepare inputs
            if use_cache and step > 0:
                # Only use the last token for cached generation
                model_input = generated[:, -1:]
            else:
                model_input = generated
            
            # Forward pass
            logits = self(model_input, inference_params)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply sampling strategies
            if top_k is not None and top_k > 0:
                # Top-k filtering
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
            
            if top_p < 1.0:
                # Top-p filtering
                next_token_logits = self._top_p_filter(next_token_logits, top_p)
            
            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Update generated sequences
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update inference cache offset
            if use_cache:
                inference_params.seqlen_offset += 1
            
            # Check for EOS tokens
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break
        
        return generated
    
    @torch.no_grad()
    def generate_streaming(self,
                          input_ids: Tensor,
                          max_new_tokens: int = 50,
                          temperature: float = 1.0,
                          top_p: float = 0.9,
                          eos_token_id: Optional[int] = None):
        """
        Generate tokens one by one (streaming generation).
        
        Args:
            input_ids: Input token IDs (1, seq_len) - only batch_size=1 supported
            max_new_tokens: Maximum number of new tokens
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            eos_token_id: End-of-sequence token ID
            
        Yields:
            Generated tokens one by one
        """
        if input_ids.size(0) != 1:
            raise ValueError("Streaming generation only supports batch_size=1")
        
        inference_params = InferenceParams()
        current_ids = input_ids.clone()
        
        for step in range(max_new_tokens):
            # Forward pass
            if step == 0:
                logits = self(current_ids, inference_params)
            else:
                logits = self(current_ids[:, -1:], inference_params)
            
            # Sample next token
            next_token_logits = logits[:, -1, :] / temperature
            if top_p < 1.0:
                next_token_logits = self._top_p_filter(next_token_logits, top_p)
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Update sequence and cache
            current_ids = torch.cat([current_ids, next_token], dim=1)
            inference_params.seqlen_offset += 1
            
            # Yield the new token
            yield next_token.item()
            
            # Check for EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break