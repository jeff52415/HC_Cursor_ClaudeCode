# Hyper-Connections: Transformer Implementation

[![Paper](https://img.shields.io/badge/Paper-ICLR%202025-blue)](https://arxiv.org/abs/2409.19606v3)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A faithful, production-ready PyTorch implementation of **Hyper-Connections** from the ICLR 2025 paper by Zhu et al. (ByteDance).

## Overview

**Hyper-connections** are a novel architectural component that replaces traditional residual connections in deep neural networks. While standard residual connections (like Pre-Norm and Post-Norm) face a fundamental trade-off—the "seesaw effect" between gradient vanishing and representation collapse—hyper-connections elegantly solve both problems simultaneously.

Instead of maintaining a single hidden vector per layer, hyper-connections maintain **n parallel hidden vectors** (typically n=4) that can dynamically adjust connection strengths across both depth (between layers) and width (within layers). This allows the network to:

- **Learn optimal layer arrangements**: Networks can discover whether layers should operate sequentially, in parallel, or in hybrid configurations
- **Eliminate representation collapse**: Hidden vectors remain diverse and informative throughout the network depth
- **Maintain strong gradient flow**: Gradients propagate effectively without vanishing, enabling stable training of very deep networks
- **Achieve superior performance**: 1.8× faster convergence on MoE models, significant loss reductions on language models, and accuracy gains on vision tasks

The implementation adds **< 0.04% parameters** and **< 0.3% FLOPs** while delivering consistent improvements across architectures (Transformers, ViTs, MoE models) and scales (1B to 7B+ parameters).

## 🚀 Quick Start

### Installation (30 seconds)

```bash
# Using pip
pip install -r requirements.txt

# OR using uv (faster)
pip install uv && uv sync
```

### Create and Run Your First Model (2 minutes)

```python
from hyper_connections import TransformerWithHyperConnections
import torch

# Create model
model = TransformerWithHyperConnections(
    vocab_size=10000,
    dim=512,
    num_layers=6,
    expansion_rate=4,    # n=4 (recommended)
    dynamic=True         # Use Dynamic HC
)

# Forward pass
input_ids = torch.randint(0, 10000, (2, 128))
logits = model(input_ids)  # [2, 128, 10000]

# Generate text
prompt = torch.randint(0, 10000, (1, 20))
generated = model.generate(prompt, max_new_tokens=50)
```

### Train a Model (5 minutes)

```python
from train_example import configure_optimizer, get_cosine_schedule_with_warmup
import torch.nn.functional as F

# Setup optimizer (IMPORTANT: Use this function for proper weight decay)
optimizer = configure_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)

# Training loop
model.train()
for step in range(100):
    input_ids = torch.randint(0, 10000, (4, 32))
    target_ids = torch.randint(0, 10000, (4, 32))

    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, 10000), target_ids.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if (step + 1) % 10 == 0:
        print(f"Step {step + 1}, Loss: {loss.item():.4f}")
```

## 📚 What's Included

This repository contains a **complete** implementation with:

- ✅ **Core Implementation** - Faithful to paper (Algorithms 1-3, Equations 1-14)
- ✅ **Training Infrastructure** - Proper optimizer, scheduler, and training loop
- ✅ **Validation Suite** - Comprehensive tests verifying correctness
- ✅ **Documentation** - Detailed guides for users and Claude Code
- ✅ **Examples** - Training, generation, and comparison scripts
- ✅ **Visualization** - ASCII diagrams explaining the architecture

## 📁 Files

| File | Description | Lines |
|------|-------------|-------|
| `hyper_connections.py` | Core implementation | 650 |
| `train_example.py` | Training script | 250 |
| `compare_architectures.py` | Validation suite | 350 |
| `visualize_architecture.py` | Architecture visualization | 400 |
| `debug_example.py` | Debug mode demonstration | 100 |
| `IMPLEMENTATION.md` | Complete implementation guide | - |
| `CLAUDE.md` | Claude Code documentation | - |

## 🎯 Key Features

### Faithful Implementation

- **Algorithms 1-3**: Exact implementation from paper
- **All Equations**: Equations 1-14 implemented correctly
- **Critical Details**: Output scaling (√n), weight decay, initialization
- **Verified**: All tests passing, matches paper specifications

### Performance

Based on paper's experimental results:

| Model | Improvement | Notes |
|-------|-------------|-------|
| OLMo-1B | -0.03 loss | Validation set |
| OLMo-7B | -0.02 loss | Validation set |
| OLMoE-1B-7B | **1.8× faster** | Convergence speed |
| OLMoE-1B-7B | +6 points | ARC-Challenge @ 500B tokens |
| ViT-Large | +2.7% | ImageNet accuracy |

### Overhead

| Metric | Value |
|--------|-------|
| Parameters | < 0.04% increase |
| FLOPs | < 0.3% increase |
| Memory | ~15-30% increase (training) |

## 🔧 Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using uv (faster)

```bash
pip install uv
uv add torch numpy
uv sync
```

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| **[README.md](README.md)** | This file - overview and quick start |
| **[IMPLEMENTATION.md](IMPLEMENTATION.md)** | Complete implementation guide with verification |
| **[CLAUDE.md](CLAUDE.md)** | Claude Code documentation |

## 🧪 Testing

```bash
# Basic functionality test
python hyper_connections.py

# Full validation suite
python compare_architectures.py

# Training example
python train_example.py

# Architecture visualization
python visualize_architecture.py

# Debug mode (see tensor shapes at each layer)
python debug_example.py
```

## 💡 Usage Examples

### Training a Model

```python
from train_example import configure_optimizer, get_cosine_schedule_with_warmup

# Configure optimizer (handles weight decay correctly)
optimizer = configure_optimizer(model, lr=3e-4, weight_decay=0.1)

# Create scheduler with warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=1000, num_training_steps=10000
)

# Training loop
for input_ids, target_ids in dataloader:
    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### Comparing Architectures

```python
from compare_architectures import StandardTransformer, count_parameters

# Standard transformer
standard = StandardTransformer(vocab_size=10000, dim=512, num_layers=6)

# Hyper-connections transformer
hyper = TransformerWithHyperConnections(
    vocab_size=10000, dim=512, num_layers=6, expansion_rate=4, dynamic=True
)

# Compare
print(f"Standard:          {count_parameters(standard):,} params")
print(f"Hyper-connections: {count_parameters(hyper):,} params")
```

### Debug Mode (Visualize Tensor Shapes)

```python
# Enable debug mode to see tensor shapes at each operation
model = TransformerWithHyperConnections(
    vocab_size=1000,
    dim=128,
    num_layers=2,
    expansion_rate=4,
    debug=True  # <-- Enable debug output
)

# Forward pass will print detailed shape information
logits = model(input_ids)

# Example output:
# Layer 0 - ATTENTION BLOCK
# Input H shape: torch.Size([2, 8, 4, 128])
# After width_connection: mix_h shape: torch.Size([2, 8, 5, 128])
# After attention: h shape: torch.Size([2, 8, 128])
# ...
```

## 🔬 How It Works

### Standard Residual Connection
```
Input h → LayerNorm → Layer → Output
  │                             │
  └─────────────────────────────┘
            Residual (+)
```

### Hyper-Connections (n=4)
```
Input H = [h₁, h₂, h₃, h₄] (4 hidden vectors)
    │
    ├─ Width Connection: Mix vectors (α^T @ H)
    │
    ├─ Layer: Attention/FFN on mixed vector
    │
    └─ Depth Connection: Combine with residuals (β * output + residuals)
    │
Output H' = [h₁', h₂', h₃', h₄']
```

**See [visualize_architecture.py](visualize_architecture.py) for detailed diagrams!**

## 🎓 Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{zhu2025hyperconnections,
  title={Hyper-Connections},
  author={Zhu, Defa and Huang, Hongzhi and Huang, Zihao and Zeng, Yutao and Mao, Yunyao and Wu, Banggu and Min, Qiyang and Zhou, Xun},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## ⚙️ Configuration Options

### Model Parameters

```python
model = TransformerWithHyperConnections(
    vocab_size=10000,         # Vocabulary size
    dim=512,                  # Hidden dimension
    num_layers=6,             # Number of layers
    num_heads=8,              # Attention heads
    expansion_rate=4,         # n (2, 4, or 8)
    dynamic=True,             # Use Dynamic HC
    use_tanh=True,            # Stabilize training
    norm_type='rmsnorm',      # 'rmsnorm' or 'layernorm'
    causal=True,              # Causal attention mask
    max_seq_len=2048,         # Maximum sequence length
    dropout=0.1               # Dropout probability
)
```

### Recommended Settings

**For Language Models:**
```python
config = {
    'expansion_rate': 4,
    'dynamic': True,
    'use_tanh': True,
    'norm_type': 'rmsnorm'
}
```

**For Vision Models:**
```python
config = {
    'expansion_rate': 2,
    'dynamic': True,
    'use_tanh': True,
    'norm_type': 'layernorm'
}
```

## 🐛 Common Issues

### ImportError: No module named 'torch'
```bash
pip install torch  # or: uv add torch
```

### Loss is NaN
- Automatically handled by implementation
- Uses gradient clipping (1.0)
- Uses proper output scaling (√n)

### Out of Memory
```python
# Use smaller expansion rate
model = TransformerWithHyperConnections(expansion_rate=2)

# Or use activation checkpointing (see IMPLEMENTATION.md)
```

## 🎯 Verification

The implementation has been verified to match the paper:

✅ Algorithm 1 (Network) - Lines 428-447
✅ Algorithm 2 (HyperConnection) - Lines 79-178
✅ Algorithm 3 (Transformer) - Lines 309-327
✅ Equations 1-14 - All implemented correctly
✅ Output scaling by √n - Lines 414-425
✅ Weight decay configuration - train_example.py
✅ Initialization - Lines 54-58

**See [IMPLEMENTATION.md](IMPLEMENTATION.md) for detailed verification!**

## 📊 Expected Results

Based on the paper, you should expect:

- **Language Models**: -0.02 to -0.03 loss improvement
- **MoE Models**: 1.8× faster convergence to same quality
- **Vision Models**: +1-3% accuracy improvement
- **Training Stability**: No loss spikes (unlike baseline)

## 🚀 Next Steps

1. **Try it out**: Follow the quick start above (5 minutes)
2. **Understand it**: Run `python visualize_architecture.py`
3. **Validate it**: Run `python compare_architectures.py`
4. **Train it**: Use `train_example.py` as template
5. **Deep dive**: Read [IMPLEMENTATION.md](IMPLEMENTATION.md) for details

## 📝 License

This implementation is provided for research and educational purposes. See the original paper for licensing details of the method.

## 🙏 Acknowledgments

This implementation is based on the ICLR 2025 paper "HYPER-CONNECTIONS" by Zhu et al. All credit for the method goes to the original authors at ByteDance's Seed-Foundation-Model Team.

---

**Ready to get started?** Follow the quick start above!

**Need help?** Check [IMPLEMENTATION.md](IMPLEMENTATION.md) for comprehensive documentation.

**Want to verify?** Run the validation suite: `python compare_architectures.py`
