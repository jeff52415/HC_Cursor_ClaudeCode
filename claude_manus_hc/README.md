# Transformer with Hyper-Connections

Implementation of **Hyper-Connections** from the paper:
> **"HYPER-CONNECTIONS"**
> Defa Zhu, Hongzhi Huang, Zihao Huang, et al.
> Published at ICLR 2025
> ByteDance Seed-Foundation-Model Team

## Overview

Hyper-connections are a novel alternative to residual connections in deep neural networks. They address common drawbacks like the seesaw effect between gradient vanishing and representation collapse.

### Key Features

- **Learnable Connection Strengths**: Network automatically adjusts connection weights
- **Multiple Information Pathways**: Expansion rate `n` creates parallel processing paths
- **Dynamic Adaptation**: Weights can depend on input (Dynamic Hyper-Connections)
- **Minimal Overhead**: <1% additional parameters compared to standard residual connections
- **Significant Improvements**:
  - 1.8× faster convergence (OLMoE-1B-7B)
  - +2.69% accuracy improvement (ViT-Large on ImageNet)
  - Lower loss across all model sizes (1B, 7B, MoE)

## Architecture

### Hyper-Connection Block

The core innovation replaces standard residual connections with:

```
HC = [[0,      β₁, β₂, ..., βₙ],
      [α₁,₀,  α₁,₁, α₁,₂, ..., α₁,ₙ],
      [α₂,₀,  α₂,₁, α₂,₂, ..., α₂,ₙ],
      [  ...    ...   ...  ...   ...],
      [αₙ,₀,  αₙ,₁, αₙ,₂, ..., αₙ,ₙ]]
```

Where:
- `n` is the expansion rate (typically 4)
- `β` weights control output contribution
- `α` weights control input mixing and residual connections

### Two Variants

1. **Static Hyper-Connections (SHC)**: Fixed learnable weights
2. **Dynamic Hyper-Connections (DHC)**: Weights depend on input (recommended)

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repo-url>
cd claude_manus_hc

# Install dependencies with uv
uv sync
```

### Using pip

```bash
# Clone the repository
git clone <repo-url>
cd claude_manus_hc

# Install dependencies
pip install torch numpy

# Or install in development mode
pip install -e .
```

## Usage

### Basic Example

```python
from hyper_connections import TransformerEncoderHyper
import torch

# Create model with hyper-connections
model = TransformerEncoderHyper(
    vocab_size=5000,
    dim=512,
    num_layers=6,
    num_heads=8,
    expansion_rate=4,  # n=4 recommended from paper
    max_seq_len=256,
    dropout=0.1,
    dynamic=True,      # Use Dynamic Hyper-Connections
    use_tanh=True
)

# Forward pass
input_ids = torch.randint(0, 5000, (8, 128))  # (batch, seq_len)
logits = model(input_ids)  # (batch, seq_len, vocab_size)
```

### Comparing DHC vs SHC

```python
# Dynamic Hyper-Connections (DHC)
model_dhc = TransformerEncoderHyper(
    vocab_size=1000,
    dim=256,
    num_layers=4,
    expansion_rate=4,
    dynamic=True  # Adaptive weights
)

# Static Hyper-Connections (SHC)
model_shc = TransformerEncoderHyper(
    vocab_size=1000,
    dim=256,
    num_layers=4,
    expansion_rate=4,
    dynamic=False  # Fixed weights
)
```

### Using Just the Hyper-Connection Block

```python
from hyper_connections import HyperConnection
import torch.nn as nn

# Create a hyper-connection
hc = HyperConnection(
    dim=512,
    expansion_rate=4,
    layer_id=0,
    dynamic=True
)

# Create a simple layer
layer = nn.Linear(512, 512)

# Forward pass
h = torch.randn(2, 10, 4, 512)  # (batch, seq, n, dim)
h_out = hc(h, layer)  # (batch, seq, n, dim)
```

## File Structure

```
claude_manus_hc/
├── README.md                    # This file
├── task_plan.md                 # Development plan
├── notes.md                     # Paper notes and findings
├── hyper_connection.py          # Core hyper-connection module
├── transformer_hyper.py         # Complete Transformer implementation
├── example_usage.py             # Usage examples
└── HYPER-CONNECTIONS.pdf        # Original paper
```

## Implementation Details

### Key Components

1. **hyper_connection.py**
   - `HyperConnection`: Core module implementing Equations 1-13 from paper
   - Supports both static and dynamic variants
   - Implements width-connections (input mixing) and depth-connections (output combination)

2. **transformer_hyper.py**
   - `TransformerEncoderHyper`: Complete Transformer with hyper-connections
   - `TransformerBlockHyper`: Single transformer block using hyper-connections
   - `MultiHeadAttention`: Standard multi-head self-attention
   - `FeedForward`: Position-wise feed-forward network

3. **example_usage.py**
   - Language modeling example
   - DHC vs SHC comparison
   - Effect of expansion rates
   - Architecture visualization

### Faithful to Paper

This implementation follows the paper specifications exactly:

- ✅ **Equation 1**: HC matrix structure (page 3)
- ✅ **Equations 2-5**: Forward pass computation (page 3)
- ✅ **Equations 10-13**: Dynamic parameter computation (page 4)
- ✅ **Equation 14**: Initialization strategy (page 4)
- ✅ **Algorithm 2-3**: PyTorch implementation (Appendix J, page 29)
- ✅ **Figure 8**: Complete Transformer architecture (Appendix A, page 14)
- ✅ **Section 4 Implementation**: Output scaling by √n

## Experimental Results (from Paper)

### Language Models (500B tokens)

| Model | Baseline | With HC | Improvement |
|-------|----------|---------|-------------|
| OLMo-1B | 2.811 loss | 2.781 loss | -1.1% |
| OLMo-7B | 2.581 loss | 2.559 loss | -0.9% |
| OLMoE-1B-7B | Standard | **1.8× faster** | 6 pts ARC-C |

### Vision Models (ImageNet)

| Model | Baseline Acc | With HC | Improvement |
|-------|--------------|---------|-------------|
| ViT-Base | 76.38% | 77.60% | +1.22% |
| ViT-Large | 77.25% | 79.94% | +2.69% |

### Recommended Settings

From ablation studies (Table 1, paper page 6):

- **Expansion rate**: `n=4` (best performance)
- **Variant**: Dynamic Hyper-Connections (DHC)
- **Activation**: Use `tanh` activation
- **Trainable components**: Both width (WC) and depth (B) connections

## Key Advantages

1. **Addresses Seesaw Effect**: Balances gradient vanishing vs representation collapse
2. **Learnable Architecture**: Network learns optimal connection patterns
3. **Dynamic Rearrangement**: Can learn sequential or parallel layer arrangements
4. **Minimal Cost**: Negligible parameter and computational overhead
5. **Broad Applicability**: Works for LLMs, vision models, and more

## Citation

```bibtex
@inproceedings{zhu2025hyperconnections,
  title={Hyper-Connections},
  author={Zhu, Defa and Huang, Hongzhi and Huang, Zihao and Zeng, Yutao and Mao, Yunyao and Wu, Banggu and Min, Qiyang and Zhou, Xun},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## References

- Paper: [arXiv:2409.19606](https://arxiv.org/abs/2409.19606)
- OLMo: [Groeneveld et al., 2024](https://arxiv.org/abs/2402.00838)
- OLMoE: [Muennighoff et al., 2024](https://arxiv.org/abs/2409.02060)

## License

This implementation is provided for research and educational purposes.

## Contact

For questions about the implementation, please open an issue or refer to the original paper.
