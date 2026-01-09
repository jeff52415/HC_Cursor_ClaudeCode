"""
Transformer with Hyper-Connections
Paper: "HYPER-CONNECTIONS" (ICLR 2025)

Complete Transformer implementation using hyper-connections instead of
residual connections, following the architecture in Figure 8 (Appendix A).
"""

import torch
import torch.nn as nn
import math
from hyper_connections.hyper_connection import HyperConnection


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, L, 3*dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, H, L, d_head)
        out = out.transpose(1, 2).contiguous()  # (B, L, H, d_head)
        out = out.reshape(batch_size, seq_len, self.dim)  # (B, L, dim)

        # Output projection
        out = self.out_proj(out)
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlockHyper(nn.Module):
    """
    Transformer block with hyper-connections.

    Replaces standard residual connections with hyper-connections as described
    in the paper (see Algorithm 3 in Appendix J).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        expansion_rate: int = 4,
        ffn_hidden_dim: int = None,
        dropout: float = 0.1,
        layer_id: int = 0,
        dynamic: bool = True,
        use_tanh: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.expansion_rate = expansion_rate

        # Normalization layers
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

        # Transformer sublayers
        self.attention = MultiHeadAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, ffn_hidden_dim, dropout)

        # Hyper-connections (replacing residual connections)
        self.attn_hyper_connection = HyperConnection(
            dim=dim,
            expansion_rate=expansion_rate,
            layer_id=layer_id * 2,  # Each block has 2 sublayers
            dynamic=dynamic,
            use_tanh=use_tanh
        )

        self.ffn_hyper_connection = HyperConnection(
            dim=dim,
            expansion_rate=expansion_rate,
            layer_id=layer_id * 2 + 1,
            dynamic=dynamic,
            use_tanh=use_tanh
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Forward pass following Algorithm 3 from the paper.

        Args:
            h: Hyper hidden matrix of shape (batch, seq_len, n, dim)
            mask: Optional attention mask

        Returns:
            Updated hyper hidden matrix of shape (batch, seq_len, n, dim)
        """
        # Attention block with hyper-connection
        mix_h, beta = self.attn_hyper_connection.width_connection(h)
        x = self.attn_norm(mix_h[..., 0, :])  # Normalize first hyper hidden
        x = self.attention(x, mask)
        x = self.dropout(x)
        h = self.attn_hyper_connection.depth_connection(mix_h, x, beta)

        # FFN block with hyper-connection
        mix_h, beta = self.ffn_hyper_connection.width_connection(h)
        x = self.ffn_norm(mix_h[..., 0, :])  # Normalize first hyper hidden
        x = self.ffn(x)
        x = self.dropout(x)
        h = self.ffn_hyper_connection.depth_connection(mix_h, x, beta)

        return h


class TransformerEncoderHyper(nn.Module):
    """
    Transformer Encoder with Hyper-Connections.

    Implements the complete architecture shown in Figure 8 of the paper.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        expansion_rate: int = 4,
        ffn_hidden_dim: int = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        dynamic: bool = True,
        use_tanh: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.expansion_rate = expansion_rate

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlockHyper(
                dim=dim,
                num_heads=num_heads,
                expansion_rate=expansion_rate,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                layer_id=i,
                dynamic=dynamic,
                use_tanh=use_tanh
            )
            for i in range(num_layers)
        ])

        # Final normalization and output projection
        self.final_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling for hyper-connections"""
        # Standard initialization
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Scale output projections by sqrt(n) as mentioned in paper (Section 4)
        # This ensures proper std of output when summing n hyper hidden vectors
        scale_factor = math.sqrt(self.expansion_rate)
        for block in self.blocks:
            # Scale attention output projection
            if hasattr(block.attention, 'out_proj'):
                block.attention.out_proj.weight.data /= scale_factor

            # Scale FFN output projection
            if hasattr(block.ffn, 'fc2'):
                block.ffn.fc2.weight.data /= scale_factor

    def forward(self, input_ids: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            mask: Optional attention mask

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token and position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        # Replicate input n times to form initial hyper hidden matrix H^0
        # As described in Section 2.1: H^0 = [h^0, h^0, ..., h^0]^T
        h = x.unsqueeze(2).repeat(1, 1, self.expansion_rate, 1)  # (B, L, n, d)

        # Pass through transformer blocks
        for block in self.blocks:
            h = block(h, mask)

        # Sum hyper hidden vectors row-wise (Section 2.1)
        # "we sum the last hyper hidden matrix row-wise to obtain the required hidden vector"
        h_final = h.sum(dim=2)  # (B, L, d)

        # Final normalization and projection
        h_final = self.final_norm(h_final)
        logits = self.output_proj(h_final)

        return logits


def test_transformer_hyper():
    """Test the Transformer with hyper-connections"""
    print("Testing Transformer with Hyper-Connections...\n")

    # Model hyperparameters
    vocab_size = 1000
    dim = 256
    num_layers = 4
    num_heads = 8
    expansion_rate = 4
    max_seq_len = 128
    batch_size = 2
    seq_len = 32

    # Create model
    print("1. Creating model...")
    model = TransformerEncoderHyper(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        expansion_rate=expansion_rate,
        max_seq_len=max_seq_len,
        dropout=0.1,
        dynamic=True,
        use_tanh=True
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\n2. Testing forward pass...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)

    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size), "Output shape mismatch!"
    print("   ✓ Forward pass successful")

    # Test backward pass
    print("\n3. Testing backward pass...")
    target = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        target.view(-1)
    )
    loss.backward()
    print(f"   Loss: {loss.item():.4f}")
    print("   ✓ Backward pass successful")

    # Compare with static hyper-connections
    print("\n4. Testing static hyper-connections...")
    model_static = TransformerEncoderHyper(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        expansion_rate=expansion_rate,
        max_seq_len=max_seq_len,
        dropout=0.1,
        dynamic=False  # Use static HC
    )

    logits_static = model_static(input_ids)
    print(f"   Output shape (static): {logits_static.shape}")
    print("   ✓ Static hyper-connection test passed")

    # Parameter comparison
    static_params = sum(p.numel() for p in model_static.parameters())
    dynamic_params = sum(p.numel() for p in model.parameters())
    overhead = dynamic_params - static_params
    overhead_pct = (overhead / static_params) * 100

    print(f"\n5. Parameter comparison:")
    print(f"   Static HC parameters: {static_params:,}")
    print(f"   Dynamic HC parameters: {dynamic_params:,}")
    print(f"   Overhead: {overhead:,} ({overhead_pct:.2f}%)")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_transformer_hyper()
