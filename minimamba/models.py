import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, Dict, Any

from .core import MambaEncoder, MambaEmbedding, MambaLMHead, MambaClassificationHead
from .config import (
    BaseMambaConfig, 
    MambaLMConfig, 
    MambaClassificationConfig, 
    InferenceParams
)


class MambaForCausalLM(nn.Module):
    """
    Mamba model specialized for causal language modeling.
    
    This model includes embeddings, the Mamba encoder, and a language modeling head
    with support for weight tying and efficient generation.
    """
    
    def __init__(self, config: MambaLMConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.embedding = MambaEmbedding(
            vocab_size=config.padded_vocab_size,
            d_model=config.d_model
        )
        
        self.backbone = MambaEncoder(config)
        
        self.lm_head = MambaLMHead(
            d_model=config.d_model,
            vocab_size=config.padded_vocab_size,
            bias=config.bias,
            tie_weights=config.tie_embeddings,
            embedding_layer=self.embedding.token_embedding if config.tie_embeddings else None
        )
        
        # For inference: assign layer index to each block
        for i, layer in enumerate(self.backbone.layers):
            layer.mixer.layer_idx = i
    
    def forward(self, 
                input_ids: Tensor, 
                inference_params: Optional[InferenceParams] = None,
                labels: Optional[Tensor] = None) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Forward pass for causal language modeling.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            inference_params: Optional inference parameters for generation
            labels: Optional labels for computing loss (batch, seq_len)
            
        Returns:
            If labels provided: dict with 'loss' and 'logits'
            Otherwise: logits tensor (batch, seq_len, vocab_size)
        """
        # Convert tokens to embeddings
        hidden_states = self.embedding(input_ids)
        
        # Pass through encoder
        hidden_states = self.backbone(hidden_states, inference_params)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        if labels is not None:
            # Compute cross-entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            return {
                'loss': loss,
                'logits': logits
            }
        
        return logits
    
    def reset_cache(self, inference_params: Optional[InferenceParams] = None) -> None:
        """Reset all cached states for inference."""
        self.backbone.reset_cache(inference_params)
    
    def get_cache_info(self, inference_params: Optional[InferenceParams] = None) -> dict:
        """Get information about cached states."""
        return self.backbone.get_cache_info(inference_params)
    
    def prepare_inputs_for_generation(self, input_ids: Tensor, 
                                    inference_params: Optional[InferenceParams] = None) -> dict:
        """Prepare inputs for autoregressive generation."""
        if inference_params is None:
            inference_params = InferenceParams()
        
        return {
            'input_ids': input_ids,
            'inference_params': inference_params
        }
    
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
        """Generate sequences using autoregressive sampling."""
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
    
    def _top_p_filter(self, logits: Tensor, top_p: float) -> Tensor:
        """Apply top-p (nucleus) filtering to logits."""
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


class MambaForSequenceClassification(nn.Module):
    """
    Mamba model for sequence classification tasks.
    
    This model includes embeddings, the Mamba encoder, and a classification head
    with configurable pooling strategies.
    """
    
    def __init__(self, config: MambaClassificationConfig):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        
        # We need embedding for token inputs
        # For classification, we don't need padded vocab size
        self.embedding = MambaEmbedding(
            vocab_size=getattr(config, 'vocab_size', 32000),
            d_model=config.d_model
        )
        
        self.backbone = MambaEncoder(config)
        
        self.classifier = MambaClassificationHead(
            d_model=config.d_model,
            num_labels=config.num_labels,
            pooling_strategy=config.pooling_strategy,
            dropout=config.dropout
        )
        
        # For inference: assign layer index to each block
        for i, layer in enumerate(self.backbone.layers):
            layer.mixer.layer_idx = i
    
    def forward(self, 
                input_ids: Tensor, 
                attention_mask: Optional[Tensor] = None,
                labels: Optional[Tensor] = None) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Forward pass for sequence classification.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Optional attention mask (batch, seq_len)
            labels: Optional labels for computing loss (batch,)
            
        Returns:
            If labels provided: dict with 'loss' and 'logits'
            Otherwise: logits tensor (batch, num_labels)
        """
        # Convert tokens to embeddings
        hidden_states = self.embedding(input_ids)
        
        # Pass through encoder
        hidden_states = self.backbone(hidden_states)
        
        # Compute classification logits
        logits = self.classifier(hidden_states, attention_mask)
        
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss = F.mse_loss(logits.squeeze(), labels.squeeze())
            else:
                # Classification
                loss = F.cross_entropy(logits, labels)
            
            return {
                'loss': loss,
                'logits': logits
            }
        
        return logits


class MambaForFeatureExtraction(nn.Module):
    """
    Mamba model for feature extraction (embeddings).
    
    This model returns the hidden states from the encoder without any task-specific head.
    Useful for getting contextualized embeddings.
    """
    
    def __init__(self, config: BaseMambaConfig):
        super().__init__()
        self.config = config
        
        # We might not need embeddings if input is already embeddings
        self.has_embeddings = hasattr(config, 'vocab_size') and config.vocab_size > 0
        
        if self.has_embeddings:
            vocab_size = getattr(config, 'padded_vocab_size', config.vocab_size)
            self.embedding = MambaEmbedding(
                vocab_size=vocab_size,
                d_model=config.d_model
            )
        
        self.backbone = MambaEncoder(config)
        
        # For inference: assign layer index to each block
        for i, layer in enumerate(self.backbone.layers):
            layer.mixer.layer_idx = i
    
    def forward(self, 
                inputs: Tensor,
                inference_params: Optional[InferenceParams] = None) -> Tensor:
        """
        Forward pass for feature extraction.
        
        Args:
            inputs: Input token IDs (batch, seq_len) or embeddings (batch, seq_len, d_model)
            inference_params: Optional inference parameters
            
        Returns:
            Hidden states (batch, seq_len, d_model)
        """
        if self.has_embeddings and inputs.dim() == 2:
            # Assume token IDs, convert to embeddings
            hidden_states = self.embedding(inputs)
        else:
            # Assume already embeddings
            hidden_states = inputs
        
        # Pass through encoder
        return self.backbone(hidden_states, inference_params)


# For backward compatibility, keep the original Mamba class
class Mamba(MambaForCausalLM):
    """
    Backward compatibility alias for MambaForCausalLM.
    """
    pass