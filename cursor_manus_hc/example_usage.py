"""
Example usage of Transformer with Hyper-Connections

This script demonstrates how to instantiate and use the Transformer
with hyper-connections for language modeling tasks.
"""

import torch
from hyper_connections import TransformerWithHC, HyperConnection


def example_basic_usage():
    """Basic example of using the Transformer with hyper-connections"""
    print("=" * 60)
    print("Example 1: Basic Transformer with Hyper-Connections (DHC×4)")
    print("=" * 60)
    
    # Model configuration similar to a small language model
    config = {
        'vocab_size': 50257,  # GPT-2 vocab size
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12,
        'ffn_hidden_size': 3072,  # 4 * hidden_size
        'max_seq_length': 1024,
        'expansion_rate': 4,  # DHC×4 as in paper
        'dropout': 0.1,
        'use_tanh': True,  # DHC (with tanh)
        'static_weights': False,  # Dynamic weights
    }
    
    # Create model
    model = TransformerWithHC(**config)
    
    # Print model info
    print(f"\nModel Configuration:")
    print(f"  Vocab Size: {config['vocab_size']}")
    print(f"  Hidden Size: {config['hidden_size']}")
    print(f"  Number of Layers: {config['num_layers']}")
    print(f"  Number of Heads: {config['num_heads']}")
    print(f"  Expansion Rate (n): {config['expansion_rate']}")
    print(f"  Total Parameters: {model.get_num_params():,}")
    print(f"  Non-embedding Parameters: {model.get_num_params(non_embedding=True):,}")
    
    # Create sample input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_length))
    
    # Forward pass
    print(f"\nInput shape: {input_ids.shape}")
    with torch.no_grad():
        logits = model(input_ids)
    print(f"Output shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_length}, {config['vocab_size']})")
    
    print("\n✓ Basic forward pass successful!\n")


def example_different_configurations():
    """Examples with different hyper-connection configurations"""
    print("=" * 60)
    print("Example 2: Different Hyper-Connection Configurations")
    print("=" * 60)
    
    configs = [
        {
            'name': 'DHC×2 (with tanh)',
            'expansion_rate': 2,
            'use_tanh': True,
            'static_weights': False,
        },
        {
            'name': 'DHC×4 (with tanh) - Paper\'s best',
            'expansion_rate': 4,
            'use_tanh': True,
            'static_weights': False,
        },
        {
            'name': 'DHC×8 (with tanh)',
            'expansion_rate': 8,
            'use_tanh': True,
            'static_weights': False,
        },
        {
            'name': 'DHC×4 (without tanh)',
            'expansion_rate': 4,
            'use_tanh': False,
            'static_weights': False,
        },
        {
            'name': 'SHC×4 (Static Hyper-Connections)',
            'expansion_rate': 4,
            'use_tanh': False,
            'static_weights': True,
        },
    ]
    
    base_config = {
        'vocab_size': 50257,
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12,
        'max_seq_length': 1024,
    }
    
    for cfg in configs:
        name = cfg.pop('name')
        model_config = {**base_config, **cfg}
        model = TransformerWithHC(**model_config)
        
        print(f"\n{name}:")
        print(f"  Expansion rate: {model_config['expansion_rate']}")
        print(f"  Use tanh: {model_config['use_tanh']}")
        print(f"  Static weights: {model_config['static_weights']}")
        print(f"  Total params: {model.get_num_params():,}")


def example_hyper_connection_block():
    """Example of using just the hyper-connection block"""
    print("\n" + "=" * 60)
    print("Example 3: Standalone Hyper-Connection Block")
    print("=" * 60)
    
    hidden_size = 768
    expansion_rate = 4
    batch_size = 2
    seq_len = 128
    
    # Create hyper-connection block
    hc_block = HyperConnection(
        hidden_size=hidden_size,
        expansion_rate=expansion_rate,
        use_tanh=True,
        static_weights=False
    )
    
    print(f"\nHyper-Connection Block:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Expansion rate: {expansion_rate}")
    print(f"  Alpha shape (width-connections): {hc_block.alpha.shape}")
    print(f"  Beta shape (depth-connections): {hc_block.beta.shape}")
    
    # Create sample inputs
    layer_output = torch.randn(batch_size, seq_len, hidden_size)
    input_hidden = torch.randn(batch_size, seq_len, hidden_size)
    
    print(f"\nInput shapes:")
    print(f"  Layer output: {layer_output.shape}")
    print(f"  Input hidden: {input_hidden.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = hc_block(layer_output, input_hidden)
    
    print(f"\nOutput shape: {output.shape}")
    print("\n✓ Hyper-connection block forward pass successful!\n")


def example_training_setup():
    """Example of setting up model for training"""
    print("=" * 60)
    print("Example 4: Training Setup")
    print("=" * 60)
    
    # Create model
    model = TransformerWithHC(
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        expansion_rate=4,
        use_tanh=True,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    print(f"\nModel created with {model.get_num_params():,} parameters")
    print(f"Optimizer: AdamW with lr=3e-4")
    
    # Example training step
    model.train()
    batch_size = 4
    seq_length = 256
    
    # Sample batch
    input_ids = torch.randint(0, 50257, (batch_size, seq_length))
    target_ids = torch.randint(0, 50257, (batch_size, seq_length))
    
    # Forward pass
    logits = model(input_ids)
    
    # Compute loss (cross-entropy)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1)
    )
    
    print(f"\nTraining step:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("\n✓ Training step successful!")
    print("\nNote: The hyper-connection parameters (α and β) are")
    print("automatically optimized during training.")
    
    # Show learned parameters
    print("\nExample learned hyper-connection parameters from first layer:")
    first_layer = model.layers[0]
    print(f"  Attention HC - Alpha (width-connections):")
    print(f"    Shape: {first_layer.attention_hc.alpha.shape}")
    print(f"    Sample values: {first_layer.attention_hc.alpha.data[:2, :2]}")
    print(f"  Attention HC - Beta (depth-connections):")
    print(f"    Shape: {first_layer.attention_hc.beta.shape}")
    print(f"    Sample values: {first_layer.attention_hc.beta.data[:3]}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Transformer with Hyper-Connections - Example Usage")
    print("Based on: HYPER-CONNECTIONS (ICLR 2025)")
    print("=" * 60 + "\n")
    
    # Run examples
    example_basic_usage()
    example_different_configurations()
    example_hyper_connection_block()
    example_training_setup()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
