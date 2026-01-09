"""
Transformer architecture with Hyper-Connections.

This module implements a complete Transformer model using hyper-connections
as described in "HYPER-CONNECTIONS" (ICLR 2025).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from hyper_connections import HyperConnection


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with support for hyper-connections."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        expansion_rate: int = 1,  # For output projection scaling
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.expansion_rate = expansion_rate
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Scale output projection by √n for proper variance
        # This is critical when using expansion_rate > 1
        if expansion_rate > 1:
            with torch.no_grad():
                self.out_proj.weight.data *= 1.0 / math.sqrt(expansion_rate)
                if self.out_proj.bias is not None:
                    self.out_proj.bias.data *= 1.0 / math.sqrt(expansion_rate)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            attention_mask: Optional mask [batch, 1, seq_len, seq_len] or [batch, seq_len]
        
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.qkv(x)  # [batch, seq_len, 3 * hidden_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [batch, num_heads, seq_len, seq_len]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Handle different mask shapes
            if attention_mask.dim() == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len] (padding mask)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
                attention_mask = attention_mask.unsqueeze(1)
            # else dim == 4, already correct shape [batch, 1, seq_len, seq_len]
            
            # Apply mask (True/1 = attend, False/0 = don't attend)
            # Handle both bool and int/float masks
            if attention_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~attention_mask, float('-inf'))
            else:
                attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # [batch, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network with support for hyper-connections."""
    
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        bias: bool = True,
        expansion_rate: int = 1,  # For output projection scaling
    ):
        super().__init__()
        
        if ffn_dim is None:
            ffn_dim = 4 * hidden_dim
        
        self.expansion_rate = expansion_rate
        
        self.fc1 = nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim, bias=bias)
        
        # Scale output projection by √n for proper variance
        # This is critical when using expansion_rate > 1
        if expansion_rate > 1:
            with torch.no_grad():
                self.fc2.weight.data *= 1.0 / math.sqrt(expansion_rate)
                if self.fc2.bias is not None:
                    self.fc2.bias.data *= 1.0 / math.sqrt(expansion_rate)
        
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swiglu':
            # SwiGLU: requires different handling
            self.fc1 = nn.Linear(hidden_dim, ffn_dim * 2, bias=bias)
            self.activation = nn.SiLU()
            self.use_swiglu = True
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.use_swiglu = activation == 'swiglu'
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
        
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        if self.use_swiglu:
            # SwiGLU: split into gate and value
            gate_and_value = self.fc1(x)
            gate, value = gate_and_value.chunk(2, dim=-1)
            x = self.activation(gate) * value
        else:
            x = self.fc1(x)
            x = self.activation(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class TransformerBlockWithHC(nn.Module):
    """
    Transformer block with hyper-connections.
    
    Applies hyper-connections to both attention and FFN sub-layers.
    Each block maintains n hidden vectors throughout the computation.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        expansion_rate: int = 4,
        layer_idx: int = 0,
        dynamic: bool = True,
        use_tanh: bool = True,
        activation: str = 'gelu',
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.layer_idx = layer_idx
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Attention module (with output scaled by √n)
        self.attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            expansion_rate=expansion_rate,
        )
        
        # Feed-forward module (with output scaled by √n)
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            activation=activation,
            expansion_rate=expansion_rate,
        )
        
        # Hyper-connections for attention and FFN
        self.hc_attn = HyperConnection(
            hidden_dim=hidden_dim,
            expansion_rate=expansion_rate,
            layer_idx=layer_idx * 2,  # Each block has 2 sub-layers
            dynamic=dynamic,
            use_tanh=use_tanh,
        )
        
        self.hc_ffn = HyperConnection(
            hidden_dim=hidden_dim,
            expansion_rate=expansion_rate,
            layer_idx=layer_idx * 2 + 1,
            dynamic=dynamic,
            use_tanh=use_tanh,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, n, hidden_dim] or [batch, seq_len, hidden_dim]
                          If 3D, will be expanded to 4D
            attention_mask: Optional attention mask
        
        Returns:
            Updated hidden states [batch, seq_len, n, hidden_dim]
        """
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Initialize hidden states if this is the first block
        if hidden_states.dim() == 3:
            # [batch, seq_len, hidden_dim] -> [batch, seq_len, n, hidden_dim]
            hidden_states = hidden_states.unsqueeze(2).expand(
                -1, -1, self.expansion_rate, -1
            )
        
        # Attention sub-layer with hyper-connections
        # We need to process each of the n hidden vectors
        # For simplicity, we average them for the attention input (could also use other strategies)
        h_for_attn = hidden_states.mean(dim=2)  # [batch, seq_len, hidden_dim]
        h_for_attn = self.norm1(h_for_attn)
        attn_output = self.attention(h_for_attn, attention_mask)
        
        # Apply hyper-connection
        hidden_states = self.hc_attn(attn_output, hidden_states)
        
        # FFN sub-layer with hyper-connections
        h_for_ffn = hidden_states.mean(dim=2)  # [batch, seq_len, hidden_dim]
        h_for_ffn = self.norm2(h_for_ffn)
        ffn_output = self.ffn(h_for_ffn)
        
        # Apply hyper-connection
        hidden_states = self.hc_ffn(ffn_output, hidden_states)
        
        return hidden_states


class TransformerWithHC(nn.Module):
    """
    Complete Transformer model with hyper-connections.
    
    This model replaces standard residual connections with hyper-connections,
    which learn optimal connection strengths and can rearrange layer computations.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        expansion_rate: int = 4,
        dynamic: bool = True,
        use_tanh: bool = True,
        activation: str = 'gelu',
        tie_weights: bool = True,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            ffn_dim: Feed-forward dimension (default: 4 * hidden_dim)
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            expansion_rate: Number of parallel hidden vectors (n)
            dynamic: Use Dynamic HC (DHC) vs Static HC (SHC)
            use_tanh: Use tanh activation in DHC
            activation: Activation function ('gelu', 'relu', 'swiglu')
            tie_weights: Tie input/output embeddings
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.expansion_rate = expansion_rate
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks with hyper-connections
        self.blocks = nn.ModuleList([
            TransformerBlockWithHC(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                expansion_rate=expansion_rate,
                layer_idx=i,
                dynamic=dynamic,
                use_tanh=use_tanh,
                activation=activation,
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection (unembedding)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie weights if specified
        if tie_weights:
            self.output_proj.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following standard practices."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = False,
        return_hidden_states: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len] for padding
            use_causal_mask: Whether to use causal masking (for autoregressive LM)
            return_hidden_states: Whether to return all hidden states
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            or (logits, hidden_states) if return_hidden_states=True
        """
        batch_size, seq_len = input_ids.shape
        
        # Check sequence length
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch, seq_len, hidden_dim]
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)  # [1, seq_len, hidden_dim]
        
        # Combine embeddings
        hidden_states = token_embeds + pos_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Process attention mask
        if attention_mask is None and use_causal_mask:
            # Create causal mask for autoregressive modeling
            # Shape: [1, 1, seq_len, seq_len]
            attention_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool)
            ).unsqueeze(0).unsqueeze(0)
        
        # If attention_mask provided, ensure it's in correct format
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [batch, seq_len] padding mask
                # Expand to [batch, 1, 1, seq_len] and broadcast
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                if use_causal_mask:
                    # Combine with causal mask
                    causal = torch.tril(
                        torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool)
                    ).unsqueeze(0).unsqueeze(0)
                    attention_mask = attention_mask & causal
        
        all_hidden_states = [] if return_hidden_states else None
        
        # Apply transformer blocks with hyper-connections
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Final processing: sum over n hidden vectors
        # hidden_states: [batch, seq_len, n, hidden_dim]
        hidden_states = hidden_states.sum(dim=2)  # [batch, seq_len, hidden_dim]
        
        # Final layer norm
        hidden_states = self.final_norm(hidden_states)
        
        # Output projection
        logits = self.output_proj(hidden_states)  # [batch, seq_len, vocab_size]
        
        if return_hidden_states:
            return logits, all_hidden_states
        return logits
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.pos_embedding.weight.numel()
        return n_params


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal attention mask.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
    
    Returns:
        Causal mask [1, 1, seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)


def count_parameters(model: nn.Module) -> dict:
    """
    Count parameters in the model, separating hyper-connection parameters.
    
    Returns dict with counts for different parameter types.
    """
    total = 0
    hc_params = 0
    hc_static = 0
    hc_dynamic = 0
    other = 0
    
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        
        if 'hc_' in name:
            hc_params += count
            if any(x in name for x in ['W_beta', 'W_m', 'W_r']):
                hc_dynamic += count
            else:
                hc_static += count
        else:
            other += count
    
    return {
        'total': total,
        'hyper_connection': hc_params,
        'hc_static': hc_static,
        'hc_dynamic': hc_dynamic,
        'other': other,
        'hc_percentage': 100.0 * hc_params / total if total > 0 else 0,
    }
