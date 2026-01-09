"""
Debug Example for Hyper-Connections

This script demonstrates the debug mode that shows tensor shapes
at each step of the forward pass through the transformer.
"""

import torch
from hyper_connections import TransformerWithHyperConnections

def main():
    print("=" * 80)
    print("Hyper-Connections Debug Mode Example")
    print("=" * 80)

    # Small model for demonstration
    config = {
        'vocab_size': 1000,
        'dim': 128,           # Small hidden dimension
        'num_layers': 2,      # Only 2 layers for clarity
        'num_heads': 4,
        'ffn_hidden_dim': 512,
        'max_seq_len': 512,
        'dropout': 0.1,
        'expansion_rate': 4,  # 4 hidden vectors
        'dynamic': True,
        'use_tanh': True,
        'causal': True,
        'norm_type': 'rmsnorm',
        'tie_weights': True,
        'debug': True         # Enable debug mode!
    }

    # Create model with debug enabled
    model = TransformerWithHyperConnections(**config)
    model.eval()  # Set to eval mode to avoid randomness from dropout

    print(f"\nModel Configuration:")
    print(f"  Vocab Size: {config['vocab_size']}")
    print(f"  Hidden Dim: {config['dim']}")
    print(f"  Num Layers: {config['num_layers']}")
    print(f"  Expansion Rate: {config['expansion_rate']}")
    print(f"  Debug Mode: {config['debug']}")

    # Create small test input
    batch_size = 2
    seq_len = 8  # Short sequence for clarity
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))

    print(f"\nRunning forward pass with debug output...")
    print(f"Input: batch_size={batch_size}, seq_len={seq_len}")
    print("=" * 80)

    # Forward pass - this will print detailed shape information
    with torch.no_grad():
        logits = model(input_ids)

    print("\n" + "=" * 80)
    print("Debug mode output complete!")
    print("=" * 80)

    print(f"\nKey Observations:")
    print(f"1. Input shape: [{batch_size}, {seq_len}]")
    print(f"2. After embeddings: [{batch_size}, {seq_len}, {config['dim']}]")
    print(f"3. Hyper hidden matrix H: [{batch_size}, {seq_len}, {config['expansion_rate']}, {config['dim']}]")
    print(f"   - Note the extra dimension for {config['expansion_rate']} parallel hidden vectors")
    print(f"4. After width_connection: mix_h has {config['expansion_rate']+1} vectors (for input mixing)")
    print(f"5. Layer input h: Takes first vector from mix_h")
    print(f"6. After depth_connection: H is updated with new information")
    print(f"7. Final sum: Collapse {config['expansion_rate']} vectors to 1")
    print(f"8. Output logits: [{batch_size}, {seq_len}, {config['vocab_size']}]")

    print("\n" + "=" * 80)
    print("To use debug mode in your own code:")
    print("=" * 80)
    print("""
# Simply set debug=True when creating the model:
model = TransformerWithHyperConnections(
    vocab_size=10000,
    dim=512,
    num_layers=6,
    expansion_rate=4,
    debug=True  # <-- Enable debug mode
)

# Then run a forward pass:
logits = model(input_ids)

# The model will print detailed shape information for each operation
    """)

if __name__ == "__main__":
    main()
