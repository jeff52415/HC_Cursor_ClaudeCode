"""
Hyper-Connections Implementation

This module implements hyper-connections as described in the ICLR 2025 paper:
"HYPER-CONNECTIONS" by Zhu et al., ByteDance.

Hyper-connections serve as an alternative to residual connections, addressing
the seesaw effect between gradient vanishing and representation collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class HyperConnection(nn.Module):
    """
    Hyper-Connection Block implementing both depth-connections and width-connections.
    
    Args:
        hidden_size: Dimension of hidden states
        expansion_rate: Number of intermediate hidden states (n in paper)
        use_tanh: Whether to apply tanh to learned weights (default True for DHC)
        static_weights: If True, uses static (non-learnable) weights (SHC variant)
    """
    def __init__(
        self, 
        hidden_size: int, 
        expansion_rate: int = 4,
        use_tanh: bool = True,
        static_weights: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expansion_rate = expansion_rate
        self.use_tanh = use_tanh
        self.static_weights = static_weights
        
        if static_weights:
            # Static Hyper-Connections (SHC)
            # Initialize with fixed values
            self.register_buffer('alpha', torch.ones(expansion_rate, expansion_rate))
            self.register_buffer('beta', torch.ones(expansion_rate + 1))
        else:
            # Dynamic Hyper-Connections (DHC)
            # Alpha: width-connection weights (n x n matrix for connections between h_i states)
            # Each h_i can receive information from all h_j states
            self.alpha = nn.Parameter(torch.zeros(expansion_rate, expansion_rate))
            
            # Beta: depth-connection weights (n+1 weights for combining layer output with h_i states)
            # beta[0] for layer output, beta[1:] for h_1, h_2, ..., h_n
            self.beta = nn.Parameter(torch.ones(expansion_rate + 1))
            
        # Initialize hidden state projections
        # These project the input into multiple hidden states
        self.hidden_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False) 
            for _ in range(expansion_rate)
        ])
        
        # Output projection to combine everything
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        if not self.static_weights:
            # Initialize alpha with small random values
            nn.init.normal_(self.alpha, mean=0.0, std=0.02)
            # Initialize beta with ones (similar to residual connections)
            nn.init.constant_(self.beta, 1.0)
        
        # Initialize projection layers
        for proj in self.hidden_projections:
            nn.init.xavier_uniform_(proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
    
    def forward(
        self, 
        layer_output: torch.Tensor, 
        input_hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of hyper-connection block.
        
        Args:
            layer_output: Output from the layer (e.g., attention or FFN output)
                         Shape: (batch_size, seq_len, hidden_size)
            input_hidden: Input hidden state to the layer
                         Shape: (batch_size, seq_len, hidden_size)
        
        Returns:
            Combined output with hyper-connections applied
            Shape: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = input_hidden.shape
        
        # Step 1: Create multiple hidden states h_1, h_2, ..., h_n from input
        hidden_states = []
        for i, proj in enumerate(self.hidden_projections):
            h_i = proj(input_hidden)  # (batch, seq, hidden)
            hidden_states.append(h_i)
        
        # Step 2: Apply width-connections (lateral information exchange)
        # Each h_i receives weighted information from all h_j states
        width_connected_states = []
        
        # Get alpha weights and apply tanh if needed
        alpha_weights = torch.tanh(self.alpha) if self.use_tanh else self.alpha
        
        for i in range(self.expansion_rate):
            # Compute weighted sum of all hidden states for h_i
            h_i_new = hidden_states[i].clone()  # Start with original h_i
            
            for j in range(self.expansion_rate):
                if i != j:  # Skip self-connection in width-connections
                    # Add weighted contribution from h_j
                    h_i_new = h_i_new + alpha_weights[i, j] * hidden_states[j]
            
            width_connected_states.append(h_i_new)
        
        # Step 3: Apply depth-connections (combine layer output with hidden states)
        # Get beta weights and apply tanh if needed
        beta_weights = torch.tanh(self.beta) if self.use_tanh else self.beta
        
        # Start with weighted layer output
        output = beta_weights[0] * layer_output
        
        # Add weighted contributions from all width-connected hidden states
        for i in range(self.expansion_rate):
            output = output + beta_weights[i + 1] * width_connected_states[i]
        
        # Step 4: Final output projection
        output = self.output_proj(output)
        
        return output


class TransformerLayerWithHC(nn.Module):
    """
    Transformer layer using Hyper-Connections instead of residual connections.
    
    This implements a standard Transformer layer (with self-attention and FFN)
    but replaces residual connections with hyper-connections.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        expansion_rate: int = 4,
        dropout: float = 0.1,
        use_tanh: bool = True,
        static_weights: bool = False,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Layer normalization (Pre-Norm style)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            hidden_size, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Hyper-connections for attention and FFN
        self.attention_hc = HyperConnection(
            hidden_size, 
            expansion_rate, 
            use_tanh, 
            static_weights
        )
        self.ffn_hc = HyperConnection(
            hidden_size, 
            expansion_rate, 
            use_tanh, 
            static_weights
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Transformer layer with hyper-connections.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask for causal/masked attention
            key_padding_mask: Padding mask
        
        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        # Self-attention block with hyper-connection
        residual = x
        x_norm = self.attention_norm(x)
        
        attn_output, _ = self.attention(
            x_norm, x_norm, x_norm,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Apply hyper-connection instead of residual
        x = self.attention_hc(attn_output, residual)
        
        # FFN block with hyper-connection
        residual = x
        x_norm = self.ffn_norm(x)
        ffn_output = self.ffn(x_norm)
        
        # Apply hyper-connection instead of residual
        x = self.ffn_hc(ffn_output, residual)
        
        return x


class TransformerWithHC(nn.Module):
    """
    Complete Transformer model with Hyper-Connections.
    
    This implements a full Transformer architecture suitable for language modeling,
    using hyper-connections instead of traditional residual connections.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_hidden_size: Optional[int] = None,
        max_seq_length: int = 2048,
        expansion_rate: int = 4,
        dropout: float = 0.1,
        use_tanh: bool = True,
        static_weights: bool = False,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Positional embeddings
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers with hyper-connections
        self.layers = nn.ModuleList([
            TransformerLayerWithHC(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_hidden_size=ffn_hidden_size,
                expansion_rate=expansion_rate,
                dropout=dropout,
                use_tanh=use_tanh,
                static_weights=static_weights,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights between token embeddings and output projection
        self.lm_head.weight = self.token_embedding.weight
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Transformer with hyper-connections.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            position_ids: Position IDs (batch_size, seq_len)
        
        Returns:
            Logits over vocabulary (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens and positions
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Create causal attention mask for autoregressive modeling
        if attention_mask is None:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device),
                diagonal=1
            ).bool()
            attention_mask = causal_mask
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask
            )
        
        # Final layer norm
        hidden_states = self.final_norm(hidden_states)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
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
            n_params -= self.position_embedding.weight.numel()
        return n_params
