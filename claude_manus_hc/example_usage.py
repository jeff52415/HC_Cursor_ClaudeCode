"""
Example Usage: Transformer with Hyper-Connections
Paper: "HYPER-CONNECTIONS" (ICLR 2025)

This script demonstrates how to use the Transformer with hyper-connections
for various tasks, comparing it with standard residual connections.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from hyper_connections import TransformerEncoderHyper


def example_language_modeling():
    """
    Example: Language modeling with hyper-connections
    Demonstrates training a small language model on toy data
    """
    print("=" * 70)
    print("Example 1: Language Modeling with Hyper-Connections")
    print("=" * 70)

    # Model configuration
    vocab_size = 5000
    dim = 512
    num_layers = 6
    num_heads = 8
    expansion_rate = 4
    max_seq_len = 256
    batch_size = 8
    seq_len = 128

    # Create model with dynamic hyper-connections
    print("\n1. Creating Transformer with Dynamic Hyper-Connections (DHC)...")
    model_dhc = TransformerEncoderHyper(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        expansion_rate=expansion_rate,
        max_seq_len=max_seq_len,
        dropout=0.1,
        dynamic=True,  # Use dynamic hyper-connections
        use_tanh=True
    )

    total_params = sum(p.numel() for p in model_dhc.parameters())
    print(f"   Model parameters: {total_params:,}")

    # Generate synthetic data
    print("\n2. Generating synthetic training data...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Target shape: {target_ids.shape}")

    # Training setup
    optimizer = optim.Adam(model_dhc.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop (few iterations for demonstration)
    print("\n3. Training for 5 iterations...")
    model_dhc.train()

    for step in range(5):
        optimizer.zero_grad()

        # Forward pass
        logits = model_dhc(input_ids)

        # Compute loss
        loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"   Step {step + 1}/5: Loss = {loss.item():.4f}")

    print("\n   ✓ Training completed successfully!")

    # Evaluation
    print("\n4. Evaluation mode...")
    model_dhc.eval()
    with torch.no_grad():
        logits = model_dhc(input_ids)
        eval_loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
        print(f"   Evaluation loss: {eval_loss.item():.4f}")

    return model_dhc


def example_comparison_dhc_vs_shc():
    """
    Example: Compare Dynamic vs Static Hyper-Connections
    Shows the difference in parameter count and behavior
    """
    print("\n" + "=" * 70)
    print("Example 2: Dynamic HC vs Static HC Comparison")
    print("=" * 70)

    # Shared configuration
    config = {
        'vocab_size': 1000,
        'dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'expansion_rate': 4,
        'max_seq_len': 128,
        'dropout': 0.1
    }

    # Create models
    print("\n1. Creating models...")
    model_dhc = TransformerEncoderHyper(**config, dynamic=True, use_tanh=True)
    model_shc = TransformerEncoderHyper(**config, dynamic=False)

    # Parameter counts
    dhc_params = sum(p.numel() for p in model_dhc.parameters())
    shc_params = sum(p.numel() for p in model_shc.parameters())
    overhead = dhc_params - shc_params
    overhead_pct = (overhead / shc_params) * 100

    print(f"\n2. Parameter comparison:")
    print(f"   Dynamic HC (DHC): {dhc_params:,} parameters")
    print(f"   Static HC (SHC):  {shc_params:,} parameters")
    print(f"   Overhead:         {overhead:,} parameters ({overhead_pct:.2f}%)")
    print(f"   → As noted in paper, overhead is negligible (<0.1% for typical models)")

    # Forward pass comparison
    print("\n3. Forward pass comparison...")
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))

    model_dhc.eval()
    model_shc.eval()

    with torch.no_grad():
        logits_dhc = model_dhc(input_ids)
        logits_shc = model_shc(input_ids)

    print(f"   DHC output shape: {logits_dhc.shape}")
    print(f"   SHC output shape: {logits_shc.shape}")
    print(f"   → Both produce same output shape")

    return model_dhc, model_shc


def example_expansion_rates():
    """
    Example: Effect of different expansion rates (n)
    As shown in Table 1 of the paper, n=4 typically performs best
    """
    print("\n" + "=" * 70)
    print("Example 3: Effect of Expansion Rate (n)")
    print("=" * 70)

    config = {
        'vocab_size': 1000,
        'dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'max_seq_len': 128,
        'dropout': 0.1,
        'dynamic': True
    }

    expansion_rates = [1, 2, 4, 8]

    print("\n1. Creating models with different expansion rates...")
    for n in expansion_rates:
        model = TransformerEncoderHyper(**config, expansion_rate=n)
        params = sum(p.numel() for p in model.parameters())
        print(f"   n={n}: {params:,} parameters")

    print("\n   → From paper experiments (Table 1):")
    print("     - n=1: Poor performance (worse than baseline)")
    print("     - n=2: Good performance (+1.4% improvement)")
    print("     - n=4: Best performance (+1.9% improvement) ✓ RECOMMENDED")
    print("     - n=8: Similar to n=4, no significant gain")

    return expansion_rates


def example_architecture_visualization():
    """
    Example: Visualize the hyper-connection architecture
    Shows how the hyper hidden matrix flows through the network
    """
    print("\n" + "=" * 70)
    print("Example 4: Architecture Visualization")
    print("=" * 70)

    print("\n1. Standard Residual Connection (Pre-Norm):")
    print("   ┌─────────┐")
    print("   │ Input h │")
    print("   └────┬────┘")
    print("        │")
    print("        ├──────────┐")
    print("        │          │")
    print("        ▼          │")
    print("   ┌─────────┐     │")
    print("   │LayerNorm│     │")
    print("   └────┬────┘     │")
    print("        │          │")
    print("        ▼          │")
    print("   ┌─────────┐     │")
    print("   │  Layer  │     │")
    print("   └────┬────┘     │")
    print("        │          │")
    print("        ▼          │")
    print("        +◄─────────┘")
    print("        │")
    print("        ▼")
    print("   ┌─────────┐")
    print("   │ Output  │")
    print("   └─────────┘")

    print("\n2. Hyper-Connection (n=4):")
    print("   ┌──────────────────────────┐")
    print("   │ Hyper Hidden Matrix H    │")
    print("   │ [h₁, h₂, h₃, h₄]^T      │")
    print("   └──────────┬───────────────┘")
    print("              │")
    print("              ▼")
    print("   ┌─────────────────────────┐")
    print("   │  Width Connections      │  ← Mix h₁,h₂,h₃,h₄")
    print("   │  h₀ = Σᵢ αᵢ·hᵢ          │     Get dynamic α,β")
    print("   └──────────┬──────────────┘")
    print("              │")
    print("              ▼")
    print("   ┌─────────────┐")
    print("   │  LayerNorm  │")
    print("   └──────┬──────┘")
    print("          │")
    print("          ▼")
    print("   ┌─────────────┐")
    print("   │    Layer    │")
    print("   └──────┬──────┘")
    print("          │")
    print("          ▼")
    print("   ┌─────────────────────────┐")
    print("   │  Depth Connections      │  ← Weighted combination")
    print("   │  ĥᵢ = βᵢ·output + h'ᵢ   │     using β weights")
    print("   └──────────┬──────────────┘")
    print("              │")
    print("              ▼")
    print("   ┌──────────────────────────┐")
    print("   │ Updated Hyper Hidden Ĥ   │")
    print("   │ [ĥ₁, ĥ₂, ĥ₃, ĥ₄]^T       │")
    print("   └──────────────────────────┘")

    print("\n3. Key Advantages:")
    print("   ✓ Adjustable connection strength (via α and β)")
    print("   ✓ Multiple information pathways (n parallel paths)")
    print("   ✓ Dynamic adaptation (weights depend on input)")
    print("   ✓ Reduces representation collapse")
    print("   ✓ Better gradient flow")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("TRANSFORMER WITH HYPER-CONNECTIONS - EXAMPLES")
    print("Paper: 'HYPER-CONNECTIONS' (ICLR 2025)")
    print("=" * 70)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run examples
    try:
        model_dhc = example_language_modeling()
        model_dhc, model_shc = example_comparison_dhc_vs_shc()
        expansion_rates = example_expansion_rates()
        example_architecture_visualization()

        print("\n" + "=" * 70)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        print("\nKey Takeaways:")
        print("1. Hyper-connections are a drop-in replacement for residual connections")
        print("2. Dynamic HC (DHC) adapts connection weights based on input")
        print("3. Expansion rate n=4 provides best performance (from paper)")
        print("4. Minimal computational overhead (<1% additional parameters)")
        print("5. Significant improvements: 1.8× faster convergence (OLMoE)")
        print("\nFor more details, see HYPER-CONNECTIONS.pdf paper")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
