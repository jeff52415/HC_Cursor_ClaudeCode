# Hyper-Connections: Transformer Implementation

A faithful PyTorch implementation of **Hyper-Connections** as described in the ICLR 2025 paper ["HYPER-CONNECTIONS"](https://openreview.net/forum?id=HYPER-CONNECTIONS) by Zhu et al., ByteDance Seed-Foundation-Model Team.

## Overview

Hyper-connections serve as an alternative to residual connections in deep neural networks, specifically addressing the seesaw effect between gradient vanishing and representation collapse. This implementation provides:

- ✅ **Complete Hyper-Connection Block** with depth-connections and width-connections
- ✅ **Dynamic Hyper-Connections (DHC)** with learnable α and β parameters
- ✅ **Static Hyper-Connections (SHC)** variant
- ✅ **Full Transformer architecture** with hyper-connections
- ✅ **Configurable expansion rates** (n = 2, 4, 8, etc.)
- ✅ **Pre-Norm Transformer layers** with hyper-connections replacing residual connections

## Key Features

### Hyper-Connection Architecture

The implementation includes:

1. **Width-Connections**: Lateral information exchange between hidden states at the same depth
   - Controlled by learnable α parameters (n × n matrix)
   - Enables feature interaction across parallel pathways

2. **Depth-Connections**: Vertical integration combining layer output with hidden states
   - Controlled by learnable β parameters (n+1 vector)
   - Dynamically adjusts connection strengths between depths

3. **Expansion Rate (n)**: Number of intermediate hidden states
   - Paper shows best results with n=4 or n=8
   - Allows network to learn optimal connection patterns

### Performance Benefits (from paper)

- **1.8× faster convergence** compared to baseline residual connections
- **Improved perplexity** across multiple benchmarks
- **Better gradient flow** without representation collapse
- **Applicable to both dense and sparse models**

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

#### Step 1: Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

#### Step 2: Create a virtual environment and install dependencies

```bash
# Navigate to the project directory
cd cursor_manus_hc

# Create a virtual environment
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

#### Step 3: Install development dependencies (optional)

```bash
uv pip install -e ".[dev]"
```

### Using pip (Alternative)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
from hyper_connections import TransformerWithHC

# Create a Transformer with Hyper-Connections (DHC×4)
model = TransformerWithHC(
    vocab_size=50257,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    expansion_rate=4,  # DHC×4 as in paper
    use_tanh=True,     # Use tanh activation (DHC)
    static_weights=False,  # Dynamic (learnable) weights
)

# Forward pass
input_ids = torch.randint(0, 50257, (2, 128))  # (batch_size, seq_length)
logits = model(input_ids)  # (batch_size, seq_length, vocab_size)

print(f"Parameters: {model.get_num_params():,}")
```

### Using the Hyper-Connection Block

```python
from hyper_connections import HyperConnection

# Create a hyper-connection block
hc_block = HyperConnection(
    hidden_size=768,
    expansion_rate=4,
    use_tanh=True,
    static_weights=False
)

# Apply to layer output and input hidden state
layer_output = torch.randn(2, 128, 768)  # From attention or FFN
input_hidden = torch.randn(2, 128, 768)  # Layer input
output = hc_block(layer_output, input_hidden)
```

### Different Configurations

```python
# DHC×2 (smaller expansion)
model_dhc2 = TransformerWithHC(
    vocab_size=50257,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    expansion_rate=2,
)

# DHC×8 (larger expansion)
model_dhc8 = TransformerWithHC(
    vocab_size=50257,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    expansion_rate=8,
)

# SHC×4 (Static Hyper-Connections)
model_shc4 = TransformerWithHC(
    vocab_size=50257,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    expansion_rate=4,
    static_weights=True,  # Non-learnable weights
)

# DHC×4 without tanh
model_dhc4_no_tanh = TransformerWithHC(
    vocab_size=50257,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    expansion_rate=4,
    use_tanh=False,  # No tanh activation
)
```

## Running Examples

The repository includes comprehensive examples:

```bash
# Run all examples
python example_usage.py
```

This will demonstrate:
1. Basic Transformer with Hyper-Connections
2. Different hyper-connection configurations
3. Standalone hyper-connection block usage
4. Training setup with optimizer

## Architecture Details

### Hyper-Connection Block

The hyper-connection block processes layer outputs through:

1. **Hidden State Projection**: Projects input into n hidden states (h₁, h₂, ..., hₙ)
2. **Width-Connections**: Each hᵢ receives weighted information from all other hⱼ states
   ```
   h'ᵢ = hᵢ + Σⱼ₌₁ⁿ αᵢⱼ · hⱼ  (where i ≠ j)
   ```
3. **Depth-Connections**: Combines layer output with width-connected states
   ```
   output = β₀ · layer_output + Σᵢ₌₁ⁿ βᵢ · h'ᵢ
   ```
4. **Output Projection**: Final linear transformation

### Transformer Layer

Each Transformer layer contains:
- **Layer Normalization** (Pre-Norm style)
- **Multi-Head Self-Attention**
- **Hyper-Connection** (replaces residual connection after attention)
- **Layer Normalization**
- **Feed-Forward Network**
- **Hyper-Connection** (replaces residual connection after FFN)

## Model Configurations

### Comparison with Paper Models

The implementation supports configurations matching the paper's experiments:

| Model | Hidden Size | Layers | Heads | Expansion (n) | Parameters |
|-------|-------------|--------|-------|---------------|------------|
| Small | 768 | 12 | 12 | 4 | ~117M |
| Base | 1024 | 24 | 16 | 4 | ~350M |
| Large | 1536 | 32 | 24 | 4 | ~1B |

### OLMo-style Configuration (from paper)

```python
# OLMo-1B with DHC×4
model = TransformerWithHC(
    vocab_size=50257,
    hidden_size=2048,
    num_layers=16,
    num_heads=16,
    ffn_hidden_size=8192,
    expansion_rate=4,
    use_tanh=True,
)
```

## Training Example

```python
import torch
import torch.nn.functional as F
from hyper_connections import TransformerWithHC

# Create model
model = TransformerWithHC(
    vocab_size=50257,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    expansion_rate=4,
)

# Optimizer (from paper: AdamW)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1
)

# Training loop
model.train()
for batch in dataloader:
    input_ids, target_ids = batch
    
    # Forward pass
    logits = model(input_ids)
    
    # Compute loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1)
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping (optional)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
```

## Implementation Fidelity

This implementation faithfully follows the paper's architecture:

✅ **Correct formulation** of width-connections and depth-connections  
✅ **Learnable α and β parameters** as described  
✅ **Optional tanh activation** for DHC variant  
✅ **Support for static weights** (SHC variant)  
✅ **Configurable expansion rates** (n = 2, 4, 8, ...)  
✅ **Pre-Norm architecture** as used in paper's experiments  
✅ **Parameter initialization** following best practices  

## Paper Reference

```bibtex
@inproceedings{zhu2025hyperconnections,
  title={Hyper-Connections},
  author={Zhu, Defa and Huang, Hongzhi and Huang, Zihao and Zeng, Yutao and Mao, Yunyao and Wu, Banggu and Min, Qiyang and Zhou, Xun},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## Project Structure

```
cursor_manus_hc/
├── hyper_connections.py    # Main implementation
├── example_usage.py        # Usage examples
├── pyproject.toml          # Project configuration
├── README.md               # This file
└── HYPER-CONNECTIONS.pdf   # Original paper
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0.0

## License

This implementation is provided for research and educational purposes.

## Acknowledgments

This implementation is based on the paper "HYPER-CONNECTIONS" by the ByteDance Seed-Foundation-Model Team, published at ICLR 2025.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## FAQ

### Q: Which expansion rate should I use?

**A:** The paper shows that DHC×4 and DHC×8 perform best. DHC×4 is recommended as a good balance between performance and computational cost.

### Q: Should I use tanh activation?

**A:** Yes, the paper's best results (DHC) use tanh activation on the learned weights. The "DHC W/O tanh" variant is slightly worse.

### Q: What's the difference between DHC and SHC?

**A:** 
- **DHC (Dynamic Hyper-Connections)**: α and β are learnable parameters that are optimized during training
- **SHC (Static Hyper-Connections)**: α and β are fixed (non-learnable) values

DHC performs better but requires learning the connection weights.

### Q: How much more computation does this add?

**A:** The paper reports "negligible increase in computation and parameters." The main additions are:
- n projection matrices for hidden states
- α (n×n) and β (n+1) scalar parameters
- Additional forward pass computations for width/depth connections

For n=4, this is minimal compared to the attention and FFN computations.

### Q: Can I use this with pre-trained models?

**A:** This architecture is designed for training from scratch. Converting a pre-trained model with residual connections to use hyper-connections would require retraining.

---

**Happy Training! 🚀**
