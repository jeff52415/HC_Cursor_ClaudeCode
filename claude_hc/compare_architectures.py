"""
Compare Transformer architectures: Standard Residual vs Hyper-Connections

This script demonstrates the difference between standard residual connections
and hyper-connections, and validates the implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from hyper_connections import (
    TransformerWithHyperConnections,
    HyperConnection,
    count_parameters
)


class StandardTransformerBlock(nn.Module):
    """Standard Transformer block with Pre-Norm residual connections"""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        from hyper_connections import MultiHeadAttention, FeedForward, RMSNorm

        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads, dropout, causal=True)
        self.ffn = FeedForward(dim, 4 * dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention with residual
        x = x + self.dropout(self.attention(self.attn_norm(x), mask))

        # FFN with residual
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))

        return x


class StandardTransformer(nn.Module):
    """Standard Transformer with residual connections"""

    def __init__(self, vocab_size, dim=512, num_layers=6, num_heads=8, max_seq_len=2048, tie_weights=True):
        super().__init__()
        from hyper_connections import RMSNorm

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)

        self.blocks = nn.ModuleList([
            StandardTransformerBlock(dim, num_heads)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights if specified (standard practice for LLMs)
        if tie_weights:
            self.output_proj.weight = self.token_embedding.weight

    def forward(self, input_ids):
        B, L = input_ids.shape

        # Embeddings
        x = self.token_embedding(input_ids)
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.final_norm(x)
        logits = self.output_proj(x)

        return logits


def compare_model_sizes():
    """Compare model sizes and computational costs"""

    print("="*80)
    print("Model Size Comparison (Weight Tying Enabled - Standard Practice)")
    print("="*80)

    config = {
        'vocab_size': 10000,
        'dim': 512,
        'num_layers': 6,
        'num_heads': 8,
        'max_seq_len': 1024
    }

    # Standard Transformer with weight tying
    standard_model = StandardTransformer(**config, tie_weights=True)
    standard_params = count_parameters(standard_model)

    print(f"\nStandard Transformer (Pre-Norm Residual):")
    print(f"  Parameters: {standard_params:,}")

    # Hyper-Connections with different expansion rates (weight tying enabled)
    for expansion_rate in [2, 4, 8]:
        hc_model = TransformerWithHyperConnections(
            vocab_size=config['vocab_size'],
            dim=config['dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            max_seq_len=config['max_seq_len'],
            expansion_rate=expansion_rate,
            dynamic=True,
            tie_weights=True
        )

        hc_params = count_parameters(hc_model)
        overhead = (hc_params - standard_params) / standard_params * 100

        print(f"\nHyper-Connections (n={expansion_rate}):")
        print(f"  Parameters: {hc_params:,}")
        print(f"  Overhead: +{overhead:.2f}%")

    print(f"\n{'='*80}")
    print("Note: Weight tying shares parameters between input embeddings and output")
    print("projection. This is standard practice in modern LLMs (GPT, LLaMA, etc.)")
    print("and saves ~vocab_size * dim parameters with no performance loss.")
    print("="*80)


def test_initialization():
    """Test that initialization matches Pre-Norm residual connections"""

    print("\n" + "="*80)
    print("Initialization Test")
    print("="*80)

    dim = 64
    rate = 4
    batch_size = 2
    seq_len = 16

    # Create hyper-connection module
    hc = HyperConnection(dim=dim, rate=rate, layer_id=0, dynamic=False)

    # Create random input
    H = torch.randn(batch_size, seq_len, rate, dim)
    layer_output = torch.randn(batch_size, seq_len, dim)

    # Perform width and depth connections
    mix_h, beta = hc.width_connection(H)
    H_new = hc.depth_connection(mix_h, layer_output, beta)

    print(f"\nInput shape: {H.shape}")
    print(f"Layer output shape: {layer_output.shape}")
    print(f"Output shape: {H_new.shape}")

    # Check that initialization is correct
    # At initialization, static_beta should be all ones
    print(f"\nInitial beta values: {hc.static_beta.data}")

    # At initialization, static_alpha should be [e_0, I]
    print(f"\nInitial alpha shape: {hc.static_alpha.shape}")
    print(f"Initial alpha:\n{hc.static_alpha.data}")

    # Verify behavior matches residual connection
    # h_0 should be selected (first column of alpha is e_0)
    h_0 = H[:, :, 0, :]  # First hidden vector
    expected_output = layer_output.unsqueeze(2) + H  # Broadcast addition

    # Check if any output matches expected (approximately)
    print(f"\nVerifying Pre-Norm equivalence at initialization...")

    # For layer 0, Am should select h_0 (first vector)
    # So h_input = h_0
    # Then layer output is computed
    # Then depth connection: beta * layer_output + H'
    # where H' comes from width connection

    # At initialization with layer_id=0:
    # Am = e_0 (selects first vector)
    # Ar = I (identity)
    # B = 1 (all ones)

    # So: h_input = H @ e_0 = h_0
    # layer_output = T(h_0)
    # H' = Ar^T @ H = I^T @ H = H
    # H_new = B^T @ layer_output + H' = 1 * layer_output + H

    # This should give each position: h_new[i] = layer_output + h[i]
    manual_output = layer_output.unsqueeze(2) + H
    difference = torch.abs(H_new - manual_output).max().item()

    print(f"Maximum difference from expected: {difference:.6f}")

    if difference < 1e-5:
        print("✓ Initialization matches Pre-Norm residual connection!")
    else:
        print("✗ Initialization does not match (this may be expected for dynamic HC)")


def test_forward_backward():
    """Test forward and backward passes"""

    print("\n" + "="*80)
    print("Forward/Backward Pass Test")
    print("="*80)

    config = {
        'vocab_size': 1000,
        'dim': 128,
        'num_layers': 2,
        'num_heads': 4,
        'max_seq_len': 64,
        'expansion_rate': 4,
        'dynamic': True
    }

    model = TransformerWithHyperConnections(**config)

    # Test input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")

    # Forward pass
    logits = model(input_ids)
    print(f"Output shape: {logits.shape}")

    # Compute loss
    target_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1)
    )

    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check gradients
    has_gradients = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"All gradients valid: {has_gradients}")

    if has_gradients:
        print("✓ Forward/backward pass successful!")
    else:
        print("✗ Some gradients are None or NaN")


def analyze_connection_patterns():
    """Analyze learned connection patterns"""

    print("\n" + "="*80)
    print("Connection Pattern Analysis")
    print("="*80)

    dim = 64
    rate = 4
    batch_size = 2
    seq_len = 16

    # Static HC
    print("\nStatic Hyper-Connections:")
    hc_static = HyperConnection(dim=dim, rate=rate, layer_id=0, dynamic=False)

    H = torch.randn(batch_size, seq_len, rate, dim)
    alpha_static, beta_static = hc_static.get_alpha_beta(H)

    print(f"  Alpha shape: {alpha_static.shape}")
    print(f"  Beta shape: {beta_static.shape}")
    print(f"  Alpha (static):\n{alpha_static}")
    print(f"  Beta (static): {beta_static}")

    # Dynamic HC
    print("\nDynamic Hyper-Connections:")
    hc_dynamic = HyperConnection(dim=dim, rate=rate, layer_id=0, dynamic=True)

    alpha_dynamic, beta_dynamic = hc_dynamic.get_alpha_beta(H)

    print(f"  Alpha shape: {alpha_dynamic.shape}")
    print(f"  Beta shape: {beta_dynamic.shape}")
    print(f"  Alpha min: {alpha_dynamic.min().item():.4f}, max: {alpha_dynamic.max().item():.4f}")
    print(f"  Beta min: {beta_dynamic.min().item():.4f}, max: {beta_dynamic.max().item():.4f}")

    # Check that dynamic parameters modify the static ones
    alpha_diff = torch.abs(alpha_dynamic - alpha_static.unsqueeze(0).unsqueeze(0)).mean().item()
    beta_diff = torch.abs(beta_dynamic - beta_static.unsqueeze(0).unsqueeze(0)).mean().item()

    print(f"\n  Mean alpha difference: {alpha_diff:.6f}")
    print(f"  Mean beta difference: {beta_diff:.6f}")

    if alpha_diff > 0 and beta_diff > 0:
        print("✓ Dynamic HC modifies static parameters!")
    else:
        print("  Note: Differences are small at initialization (expected)")


def test_generation():
    """Test text generation"""

    print("\n" + "="*80)
    print("Generation Test")
    print("="*80)

    config = {
        'vocab_size': 1000,
        'dim': 256,
        'num_layers': 4,
        'num_heads': 4,
        'max_seq_len': 128,
        'expansion_rate': 4,
        'dynamic': True
    }

    model = TransformerWithHyperConnections(**config)
    model.eval()

    # Generate text
    prompt = torch.randint(0, config['vocab_size'], (1, 10))

    print(f"Prompt: {prompt[0].tolist()}")

    with torch.no_grad():
        # Greedy decoding
        generated = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=1.0,
            top_k=50
        )

    print(f"Generated: {generated[0].tolist()}")
    print(f"Length: {generated.size(1)} tokens")

    if generated.size(1) == 30:
        print("✓ Generation successful!")
    else:
        print(f"✗ Expected 30 tokens, got {generated.size(1)}")


def main():
    """Run all comparisons and tests"""

    print("\n" + "="*80)
    print("Transformer with Hyper-Connections - Validation Suite")
    print("="*80)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run tests
    compare_model_sizes()
    test_initialization()
    test_forward_backward()
    analyze_connection_patterns()
    test_generation()

    print("\n" + "="*80)
    print("All Tests Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
