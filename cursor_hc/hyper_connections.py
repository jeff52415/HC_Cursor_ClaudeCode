"""
Hyper-Connections: A novel alternative to residual connections.

This module implements hyper-connections as described in "HYPER-CONNECTIONS" (ICLR 2025).
Hyper-connections allow networks to:
1. Adjust connection strength between features at different depths
2. Rearrange layers dynamically
3. Eliminate representation collapse while maintaining gradient flow

Key components:
- Expansion rate (n): Number of parallel hidden vectors (default n=4)
- Depth-connections (DC): Weighted connections between layer inputs/outputs
- Width-connections (WC): Information exchange between n hidden vectors
- Static (SHC) vs Dynamic (DHC) variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class HyperConnection(nn.Module):
    """
    Base Hyper-Connection module.
    
    The hyper-connection matrix HC ∈ ℝ^(n+1)×(n+1) has structure:
    HC = [0      B     ]
         [Am     Ar    ]
    
    Where:
    - B ∈ ℝ^(1×n): Weights for layer output
    - Am ∈ ℝ^(n×1): Weights for mixing inputs
    - Ar ∈ ℝ^(n×n): Weights for residual connections
    """
    
    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int = 4,
        layer_idx: int = 0,
        dynamic: bool = True,
        use_tanh: bool = True,
        scaling_factor_alpha: float = 0.01,
        scaling_factor_beta: float = 0.01,
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states
            expansion_rate: Number of parallel hidden vectors (n)
            layer_idx: Index of the layer (for initialization)
            dynamic: Whether to use Dynamic HC (DHC) or Static HC (SHC)
            use_tanh: Whether to use tanh activation in DHC
            scaling_factor_alpha: Scaling factor s_α for Am and Ar in DHC
            scaling_factor_beta: Scaling factor s_β for B in DHC
        """
        super().__init__()
        
        assert expansion_rate >= 1, "Expansion rate must be at least 1"
        if expansion_rate == 1:
            print("Warning: expansion_rate=1 not recommended, may revert to seesaw effect")
        
        self.hidden_dim = hidden_dim
        self.n = expansion_rate
        self.layer_idx = layer_idx
        self.dynamic = dynamic
        self.use_tanh = use_tanh
        
        # Static components: B, Am, Ar (no weight decay should be applied)
        # B: weights for layer output (1 x n)
        self.B = nn.Parameter(torch.ones(1, self.n))
        
        # Am: weights for mixing inputs (n x 1)
        self.Am = nn.Parameter(torch.zeros(self.n, 1))
        
        # Ar: weights for residual connections (n x n)
        self.Ar = nn.Parameter(torch.eye(self.n))
        
        # Initialize to match Pre-Norm residual connections
        # HC^k = [0_{1×1}    1_{1×n}     ]
        #        [e_{k mod n} e_{n×n}     ]
        with torch.no_grad():
            # B initialized to all ones
            self.B.fill_(1.0)
            
            # Am initialized as k-th column of identity (k mod n)
            k = layer_idx % self.n
            self.Am.zero_()
            self.Am[k, 0] = 1.0
            
            # Ar initialized as identity
            self.Ar.copy_(torch.eye(self.n))
        
        # Dynamic components (only if dynamic=True)
        if self.dynamic:
            # Scaling factors (learnable)
            self.s_alpha = nn.Parameter(torch.ones(self.n, self.n) * scaling_factor_alpha)
            self.s_beta = nn.Parameter(torch.ones(1, self.n) * scaling_factor_beta)
            
            # Weight matrices for dynamic adjustments
            # These SHOULD have weight decay applied
            self.W_beta = nn.Parameter(torch.randn(hidden_dim, self.n) * 0.02)
            self.W_m = nn.Parameter(torch.randn(hidden_dim, self.n) * 0.02)
            self.W_r = nn.Parameter(torch.randn(hidden_dim, self.n, self.n) * 0.02)
    
    def forward(
        self,
        layer_output: torch.Tensor,
        hidden_states: torch.Tensor,
        norm: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Apply hyper-connections.
        
        Args:
            layer_output: Output from the layer [batch, seq_len, hidden_dim]
            hidden_states: n hidden vectors [batch, seq_len, n, hidden_dim]
            norm: Optional normalization layer (e.g., LayerNorm)
        
        Returns:
            Updated hidden states [batch, seq_len, n, hidden_dim]
        """
        batch_size, seq_len, _ = layer_output.shape
        
        # Get connection matrices
        if self.dynamic:
            B, Am, Ar = self._get_dynamic_matrices(hidden_states)
        else:
            B = self.B
            Am = self.Am
            Ar = self.Ar
        
        # Expand dimensions for broadcasting
        # layer_output: [batch, seq_len, hidden_dim] -> [batch, seq_len, 1, hidden_dim]
        layer_output_expanded = layer_output.unsqueeze(2)
        
        # Apply depth-connections: weighted sum with layer output
        # B: [1, n] or [batch, seq_len, 1, n]
        # layer_output_expanded: [batch, seq_len, 1, hidden_dim]
        # Result: [batch, seq_len, n, hidden_dim]
        if self.dynamic:
            # B is [batch, seq_len, 1, n], we need to transpose to [batch, seq_len, n, 1]
            # then broadcast with layer_output_expanded [batch, seq_len, 1, hidden_dim]
            B_transposed = B.transpose(2, 3)  # [batch, seq_len, n, 1]
            depth_contribution = B_transposed * layer_output_expanded
        else:
            # B is [1, n], reshape to [1, 1, n, 1] for broadcasting
            B_expanded = B.view(1, 1, self.n, 1)
            depth_contribution = B_expanded * layer_output_expanded
        
        # Apply width-connections: information exchange between hidden vectors
        # Am: [n, 1] or [batch, seq_len, n, 1]
        # Ar: [n, n] or [batch, seq_len, n, n]
        
        # Mix all hidden states (Am contribution)
        if self.dynamic:
            # hidden_states: [batch, seq_len, n, hidden_dim]
            # Am: [batch, seq_len, n, 1]
            # We need to sum over n dimension weighted by Am
            mixed_input = (hidden_states * Am).sum(dim=2, keepdim=True)
            # mixed_input: [batch, seq_len, 1, hidden_dim]
            mixed_input = mixed_input.expand(-1, -1, self.n, -1)
        else:
            # Am: [n, 1], reshape to [1, 1, n, 1]
            Am_expanded = Am.view(1, 1, self.n, 1)
            mixed_input = (hidden_states * Am_expanded).sum(dim=2, keepdim=True)
            mixed_input = mixed_input.expand(-1, -1, self.n, -1)
        
        # Residual connections (Ar contribution)
        if self.dynamic:
            # Ar: [batch, seq_len, n, n]
            # hidden_states: [batch, seq_len, n, hidden_dim]
            residual = torch.einsum('bsij,bsjd->bsid', Ar, hidden_states)
        else:
            # Ar: [n, n]
            residual = torch.einsum('ij,bsjd->bsid', Ar, hidden_states)
        
        # Combine all contributions
        new_hidden_states = depth_contribution + mixed_input + residual
        
        return new_hidden_states
    
    def _get_dynamic_matrices(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute dynamic connection matrices based on input.
        
        B(H) = s_β ⊙ tanh(H̄W_β)^T + B
        Am(H) = s_α ⊙ tanh(H̄W_m) + Am
        Ar(H) = s_α ⊙ tanh(H̄W_r) + Ar
        
        Args:
            hidden_states: [batch, seq_len, n, hidden_dim]
        
        Returns:
            B, Am, Ar with shapes adapted for dynamic case
        """
        batch_size, seq_len, n, hidden_dim = hidden_states.shape
        
        # Normalize and average over n hidden vectors
        # H̄ = norm(H)
        H_bar = F.layer_norm(
            hidden_states,
            (hidden_dim,),
        )
        # Average over the n dimension
        H_bar = H_bar.mean(dim=2)  # [batch, seq_len, hidden_dim]
        
        # Compute dynamic adjustments
        if self.use_tanh:
            # B(H) = s_β ⊙ tanh(H̄W_β)^T + B
            delta_B = torch.tanh(torch.matmul(H_bar, self.W_beta))  # [batch, seq_len, n]
            # self.s_beta is [1, n], delta_B is [batch, seq_len, n], self.B is [1, n]
            B = self.s_beta * delta_B + self.B  # [batch, seq_len, n]
            B = B.unsqueeze(2)  # [batch, seq_len, 1, n]
            
            # Am(H) = s_α ⊙ tanh(H̄W_m) + Am
            delta_Am = torch.tanh(torch.matmul(H_bar, self.W_m))  # [batch, seq_len, n]
            # Broadcast s_alpha and Am correctly
            # self.s_alpha[:, 0] gives us [n], we need to broadcast with [batch, seq_len, n]
            Am = self.s_alpha[:, 0].view(1, 1, self.n) * delta_Am + self.Am.squeeze().view(1, 1, self.n)
            Am = Am.unsqueeze(-1)  # [batch, seq_len, n, 1]
            
            # Ar(H) = s_α ⊙ tanh(H̄W_r) + Ar
            # H_bar: [batch, seq_len, hidden_dim]
            # W_r: [hidden_dim, n, n]
            delta_Ar = torch.einsum('bsd,dij->bsij', H_bar, self.W_r)  # [batch, seq_len, n, n]
            delta_Ar = torch.tanh(delta_Ar)
            Ar = self.s_alpha * delta_Ar + self.Ar  # [batch, seq_len, n, n]
        else:
            # Without tanh
            delta_B = torch.matmul(H_bar, self.W_beta)  # [batch, seq_len, n]
            B = self.s_beta * delta_B + self.B
            B = B.unsqueeze(2)  # [batch, seq_len, 1, n]
            
            delta_Am = torch.matmul(H_bar, self.W_m)  # [batch, seq_len, n]
            # Broadcast s_alpha and Am correctly
            Am = self.s_alpha[:, 0].view(1, 1, self.n) * delta_Am + self.Am.squeeze().view(1, 1, self.n)
            Am = Am.unsqueeze(-1)  # [batch, seq_len, n, 1]
            
            delta_Ar = torch.einsum('bsd,dij->bsij', H_bar, self.W_r)
            Ar = self.s_alpha * delta_Ar + self.Ar
        
        return B, Am, Ar
    
    def get_parameters_for_weight_decay(self):
        """
        Returns parameters that SHOULD have weight decay applied.
        Only dynamic components should have weight decay.
        """
        if self.dynamic:
            return [self.W_beta, self.W_m, self.W_r]
        else:
            return []
    
    def get_parameters_no_weight_decay(self):
        """
        Returns parameters that should NOT have weight decay applied.
        Static components (B, Am, Ar) and scaling factors should not have weight decay.
        """
        params = [self.B, self.Am, self.Ar]
        if self.dynamic:
            params.extend([self.s_alpha, self.s_beta])
        return params


def configure_optimizers_for_hyper_connections(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """
    Configure optimizer with proper weight decay for hyper-connections.
    
    Args:
        model: Model containing HyperConnection modules
        learning_rate: Learning rate
        weight_decay: Weight decay (applied only to appropriate parameters)
        betas: Adam betas
    
    Returns:
        Configured optimizer
    """
    # Separate parameters into those with and without weight decay
    decay_params = []
    no_decay_params = []
    
    for name, module in model.named_modules():
        if isinstance(module, HyperConnection):
            # Get HC-specific parameter groups
            decay_params.extend(module.get_parameters_for_weight_decay())
            no_decay_params.extend(module.get_parameters_no_weight_decay())
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            # Regular weight decay for standard layers
            decay_params.append(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                no_decay_params.append(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            # No weight decay for normalization layers
            if hasattr(module, 'weight') and module.weight is not None:
                no_decay_params.append(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                no_decay_params.append(module.bias)
    
    # Remove duplicates while preserving parameter identity
    decay_params = list({id(p): p for p in decay_params}.values())
    no_decay_params = list({id(p): p for p in no_decay_params}.values())
    
    # Create parameter groups
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas)
    return optimizer
