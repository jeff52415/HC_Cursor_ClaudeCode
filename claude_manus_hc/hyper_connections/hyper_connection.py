"""
Hyper-Connections Implementation
Paper: "HYPER-CONNECTIONS" (ICLR 2025)
Authors: Defa Zhu, Hongzhi Huang, et al., ByteDance

This module implements the hyper-connection block as described in the paper,
which serves as an alternative to residual connections in Transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HyperConnection(nn.Module):
    """
    Hyper-Connection module that replaces residual connections in Transformers.

    Implements both static (SHC) and dynamic (DHC) hyper-connections as described
    in Sections 2.1 and 2.2 of the paper.

    Args:
        dim (int): Hidden dimension size
        expansion_rate (int): Expansion rate n (number of hyper hidden vectors)
        layer_id (int): Layer index (used for initialization)
        dynamic (bool): Whether to use dynamic hyper-connections
        use_tanh (bool): Whether to use tanh activation in dynamic version
        device: Device to place parameters on
    """

    def __init__(
        self,
        dim: int,
        expansion_rate: int = 4,
        layer_id: int = 0,
        dynamic: bool = True,
        use_tanh: bool = True,
        device=None
    ):
        super().__init__()
        self.dim = dim
        self.rate = expansion_rate
        self.layer_id = layer_id
        self.dynamic = dynamic
        self.use_tanh = use_tanh

        # Static parameters - Equation 1 in the paper
        # B: weights for layer output (β1, β2, ..., βn)
        self.static_beta = nn.Parameter(torch.ones(expansion_rate, device=device))

        # Am and Ar: weights for width and depth connections
        # Initialize according to Equation 14 to match Pre-Norm residual
        init_alpha_m = torch.zeros(expansion_rate, 1, device=device)
        init_alpha_m[layer_id % expansion_rate, 0] = 1.0  # e_(k mod n)

        init_alpha_r = torch.eye(expansion_rate, device=device)  # I_(n×n)

        # Combine Am and Ar: [Am | Ar] shape (n, n+1)
        self.static_alpha = nn.Parameter(
            torch.cat([init_alpha_m, init_alpha_r], dim=1)
        )

        # Dynamic parameters - Equations 10-13 in the paper
        if self.dynamic:
            # Layer normalization before dynamic computation
            self.layer_norm = nn.LayerNorm(dim, device=device)

            # Weight matrices for computing dynamic parameters
            # beta_fn: (d,) - computes single weight per hidden vector
            # alpha_fn: (d, n+1) - computes weights for all connections
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim, device=device))
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, expansion_rate + 1, device=device))

            # Scaling factors (initialized to 0.01 as per paper)
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)

    def width_connection(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute width connections - mixes hidden vectors and computes weights.

        Implements Equation 3-4 and 10-13 from the paper.

        Args:
            h: Hyper hidden matrix of shape (batch, seq_len, n, dim)

        Returns:
            mix_h: Mixed hidden states of shape (batch, seq_len, n+1, dim)
            beta: Depth connection weights of shape (batch, seq_len, n)
        """
        # Compute alpha and beta weights
        if self.dynamic:
            # Normalize input - Equation 10
            norm_h = self.layer_norm(h)  # (B, L, n, d)

            # Compute dynamic alpha - Equation 12-13
            alpha_weight = norm_h @ self.dynamic_alpha_fn  # (B, L, n, n+1)
            if self.use_tanh:
                alpha_weight = torch.tanh(alpha_weight)
            dynamic_alpha = alpha_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha + self.static_alpha[None, None, :, :]  # (B, L, n, n+1)

            # Compute dynamic beta - Equation 11
            beta_weight = norm_h @ self.dynamic_beta_fn  # (B, L, n)
            if self.use_tanh:
                beta_weight = torch.tanh(beta_weight)
            dynamic_beta = beta_weight * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta[None, None, :]  # (B, L, n)
        else:
            # Static weights
            alpha = self.static_alpha[None, None, :, :]  # (B, L, n, n+1)
            beta = self.static_beta[None, None, :]  # (B, L, n)

        # Width connection - Equation 3-4
        # mix_h = alpha^T @ h, shape: (B, L, n+1, d)
        mix_h = torch.einsum('blnm,blnd->blmd', alpha, h)

        return mix_h, beta

    def depth_connection(
        self,
        mix_h: torch.Tensor,
        layer_output: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute depth connections - combines layer output with hidden states.

        Implements Equation 5 from the paper.

        Args:
            mix_h: Mixed hidden states from width_connection (B, L, n+1, d)
            layer_output: Output from transformer layer (B, L, d)
            beta: Depth connection weights (B, L, n)

        Returns:
            Updated hyper hidden matrix of shape (B, L, n, d)
        """
        # Equation 5: Ĥ = B^T * (T(h0))^T + H'
        # layer_output is T(h0), shape: (B, L, d)
        # beta weights the layer output for each hyper hidden
        # mix_h[..., 1:, :] is H' (remaining n vectors after h0)

        # Broadcast and weight: (B, L, d) * (B, L, n) -> (B, L, n, d)
        weighted_output = torch.einsum('bld,bln->blnd', layer_output, beta)

        # Add residual from width connections
        h_out = weighted_output + mix_h[..., 1:, :]  # (B, L, n, d)

        return h_out

    def forward(
        self,
        h: torch.Tensor,
        layer: nn.Module
    ) -> torch.Tensor:
        """
        Complete forward pass through hyper-connection.

        Args:
            h: Hyper hidden matrix (B, L, n, d)
            layer: Transformer layer (attention or FFN)

        Returns:
            Updated hyper hidden matrix (B, L, n, d)
        """
        # Width connections: mix inputs and get weights
        mix_h, beta = self.width_connection(h)

        # Layer input is first element (h0)
        layer_input = mix_h[..., 0, :]  # (B, L, d)

        # Apply transformer layer
        layer_output = layer(layer_input)  # (B, L, d)

        # Depth connections: combine output with hidden states
        h_out = self.depth_connection(mix_h, layer_output, beta)

        return h_out


def test_hyper_connection():
    """Test the HyperConnection module"""
    print("Testing HyperConnection module...")

    batch_size = 2
    seq_len = 10
    dim = 512
    expansion_rate = 4

    # Create a simple test layer
    test_layer = nn.Linear(dim, dim)

    # Test static hyper-connection
    print("\n1. Testing Static Hyper-Connection (SHC)")
    shc = HyperConnection(
        dim=dim,
        expansion_rate=expansion_rate,
        layer_id=0,
        dynamic=False
    )

    # Create input hyper hidden matrix
    h = torch.randn(batch_size, seq_len, expansion_rate, dim)
    h_out = shc(h, test_layer)

    print(f"   Input shape: {h.shape}")
    print(f"   Output shape: {h_out.shape}")
    assert h_out.shape == h.shape, "Output shape mismatch!"
    print("   ✓ Static hyper-connection test passed")

    # Test dynamic hyper-connection
    print("\n2. Testing Dynamic Hyper-Connection (DHC)")
    dhc = HyperConnection(
        dim=dim,
        expansion_rate=expansion_rate,
        layer_id=0,
        dynamic=True,
        use_tanh=True
    )

    h_out = dhc(h, test_layer)
    print(f"   Input shape: {h.shape}")
    print(f"   Output shape: {h_out.shape}")
    assert h_out.shape == h.shape, "Output shape mismatch!"
    print("   ✓ Dynamic hyper-connection test passed")

    # Test backward pass
    print("\n3. Testing backward pass")
    loss = h_out.sum()
    loss.backward()
    print("   ✓ Backward pass successful")

    # Check parameter counts
    print("\n4. Parameter counts:")
    shc_params = sum(p.numel() for p in shc.parameters())
    dhc_params = sum(p.numel() for p in dhc.parameters())
    print(f"   SHC parameters: {shc_params}")
    print(f"   DHC parameters: {dhc_params}")
    print(f"   DHC overhead: {dhc_params - shc_params} parameters")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_hyper_connection()
