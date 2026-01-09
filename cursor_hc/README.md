# Hyper-Connections for Transformers

Implementation of **Hyper-Connections**, a novel alternative to residual connections for deep neural networks, as published at **ICLR 2025**.

## Overview

This repository implements hyper-connections, which address common drawbacks of residual connection variants (Pre-Norm and Post-Norm) such as the seesaw effect between gradient vanishing and representation collapse.

### Key Features

- ✅ **Faithful implementation** of the ICLR 2025 paper
- ✅ **Dynamic and Static variants** (DHC and SHC)
- ✅ **Proper initialization** matching Pre-Norm residual connections
- ✅ **Correct weight decay** configuration for HC parameters
- ✅ **Output scaling** by √n for stable training
- ✅ **Minimal overhead**: <0.04% parameter increase

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv add torch

# Or using pip
pip install torch>=2.0.0
```

### Basic Usage

```python
import torch
from transformer_hc import TransformerWithHC

# Create a Transformer with hyper-connections
model = TransformerWithHC(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    expansion_rate=4,      # n=4 (recommended default)
    dynamic=True,          # Use Dynamic HC (DHC)
    use_tanh=True,         # Use tanh in DHC
)

# Forward pass
input_ids = torch.randint(0, 10000, (2, 128))  # [batch, seq_len]
logits = model(input_ids)  # [batch, seq_len, vocab_size]

# For language modeling with causal masking:
# logits = model(input_ids, use_causal_mask=True)

print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {logits.shape}")
```

### Training with Proper Weight Decay

Hyper-connections require careful weight decay configuration:

```python
from hyper_connections import configure_optimizers_for_hyper_connections

# Configure optimizer with correct weight decay
optimizer = configure_optimizers_for_hyper_connections(
    model,
    learning_rate=3e-4,
    weight_decay=0.01,  # Applied only to appropriate parameters
)
```

## Architecture

### Hyper-Connection Matrix

The hyper-connection matrix HC ∈ ℝ^(n+1)×(n+1):

```
HC = [0      B     ]
     [Am     Ar    ]
```

Where:
- **B** ∈ ℝ^(1×n): Weights for layer output (depth-connections)
- **Am** ∈ ℝ^(n×1): Weights for mixing inputs (width-connections)
- **Ar** ∈ ℝ^(n×n): Weights for residual connections (width-connections)

### Dynamic Hyper-Connections (DHC)

DHC adds input-dependent adjustments:

```python
B(H) = s_β ⊙ tanh(H̄W_β)^T + B
Am(H) = s_α ⊙ tanh(H̄W_m) + Am
Ar(H) = s_α ⊙ tanh(H̄W_r) + Ar
```

## Files

- `hyper_connections.py` - Core hyper-connection module
- `transformer_hc.py` - Transformer architecture with HC
- `example_usage.py` - Complete training example
- `test_implementation.py` - Comprehensive test suite
- `CLAUDE.md` - Implementation guide for AI assistants

## Testing

Run the comprehensive test suite to validate the implementation:

```bash
python test_implementation.py
```

Tests validate:
- ✓ Correct initialization matching Pre-Norm
- ✓ Output scaling by √n
- ✓ Weight decay configuration
- ✓ Forward/backward passes
- ✓ Multiple expansion rates (n=1,2,4,8)
- ✓ Static vs Dynamic variants

## Example Training

Run a complete training example:

```bash
python example_usage.py
```

This demonstrates:
- Model creation with hyper-connections
- Training loop with proper optimizer
- Connection pattern visualization
- Text generation

## Recommended Hyperparameters

### Language Models (OLMo-style)
- Expansion rate: **n=4** (default), try n=2 or n=8
- Use **Dynamic HC (DHC)** for best performance
- Include **tanh activation** in DHC
- Train with same learning rate as baseline

### Vision Models (ViT-style)
- Expansion rate: **n=2** typically sufficient
- Both SHC and DHC show improvements
- DHC shows larger gains at larger scales

## Expected Performance

### Language Models
- **1B models**: ~0.03-0.04 loss reduction on validation
- **7B models**: ~0.02 loss reduction
- **MoE models**: 1.8× faster convergence, 6+ point improvements on downstream tasks

### Vision Models
- **ImageNet**: +1-3% accuracy improvement
- **Image generation (DiT)**: Comparable to 50% larger baseline

## Key Implementation Details

### Critical Points

1. **Never use n=1**: Reverts to seesaw effect
2. **Always scale output by √n**: Essential for stability
3. **Proper weight decay**: Static components (B, Am, Ar) should NOT have weight decay
4. **Train both B and WC**: Both depth and width connections are critical

### Initialization

To match Pre-Norm residual connections:

```python
HC^k = [0_{1×1}    1_{1×n}     ]
       [e_{k mod n} e_{n×n}     ]
```

Where e_i is the i-th column of the identity matrix.

## Reference

If you use this implementation, please cite:

```bibtex
@inproceedings{zhu2025hyperconnections,
  title={Hyper-Connections},
  author={Zhu, Defa and Huang, Hongzhi and Huang, Zihao and Zeng, Yutao and Mao, Yunyao and Wu, Banggu and Min, Qiyang and Zhou, Xun},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## License

See the paper for licensing details.

## Contributing

This implementation faithfully follows the ICLR 2025 paper. For questions about the method, please refer to the paper or contact the authors.
