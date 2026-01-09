# Quick Start Guide

Get started with Hyper-Connections in 5 minutes!

## Installation (1 minute)

```bash
# Install uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup project
cd cursor_manus_hc
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Verify Installation (30 seconds)

```bash
python test_implementation.py
# Should see: "🎉 All tests passed successfully!"
```

## Basic Usage (2 minutes)

### Create a Model

```python
from hyper_connections import TransformerWithHC

# DHC×4 (Paper's best configuration)
model = TransformerWithHC(
    vocab_size=50257,      # GPT-2 vocab size
    hidden_size=768,       # Model dimension
    num_layers=12,         # Transformer layers
    num_heads=12,          # Attention heads
    expansion_rate=4,      # DHC×4
    use_tanh=True,         # DHC variant (recommended)
)

print(f"Parameters: {model.get_num_params():,}")
# Output: Parameters: 155,835,384
```

### Forward Pass

```python
import torch

# Create input (batch_size=2, seq_length=128)
input_ids = torch.randint(0, 50257, (2, 128))

# Forward pass
logits = model(input_ids)
# Shape: (2, 128, 50257)
```

### Training Step

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
model.train()
for batch in dataloader:
    input_ids, target_ids = batch
    
    # Forward
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1)
    )
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Configurations (1 minute)

### Try Different Expansion Rates

```python
# Smaller: DHC×2 (faster, less capacity)
model_small = TransformerWithHC(..., expansion_rate=2)

# Recommended: DHC×4 (best trade-off)
model_medium = TransformerWithHC(..., expansion_rate=4)

# Larger: DHC×8 (more capacity, slower)
model_large = TransformerWithHC(..., expansion_rate=8)
```

### Try Different Variants

```python
# Dynamic Hyper-Connections (DHC) - Default, Recommended
model_dhc = TransformerWithHC(
    ..., 
    use_tanh=True,      # With tanh
    static_weights=False # Learnable
)

# Static Hyper-Connections (SHC)
model_shc = TransformerWithHC(
    ...,
    static_weights=True  # Fixed weights
)

# DHC without tanh
model_no_tanh = TransformerWithHC(
    ...,
    use_tanh=False      # No tanh activation
)
```

## Model Sizes (30 seconds)

Choose based on your compute:

```python
# Small (~117M params) - Good for experiments
small_config = {
    'vocab_size': 50257,
    'hidden_size': 768,
    'num_layers': 12,
    'num_heads': 12,
    'expansion_rate': 4,
}

# Base (~350M params) - Research models
base_config = {
    'vocab_size': 50257,
    'hidden_size': 1024,
    'num_layers': 24,
    'num_heads': 16,
    'expansion_rate': 4,
}

# Large (~1B params) - Production models
large_config = {
    'vocab_size': 50257,
    'hidden_size': 1536,
    'num_layers': 32,
    'num_heads': 24,
    'expansion_rate': 4,
}

model = TransformerWithHC(**small_config)  # Pick one
```

## Key Features

### ✅ What Hyper-Connections Do

1. **Replace residual connections** with learnable connections
2. **Learn optimal connection strengths** automatically (α and β)
3. **Prevent gradient vanishing** without representation collapse
4. **1.8× faster convergence** (from paper)

### 🔧 When to Use

- ✅ Training new models from scratch
- ✅ Large-scale language model pre-training
- ✅ Experiments comparing to residual connections
- ✅ Research on connection architectures

### ⚠️ When NOT to Use

- ❌ Fine-tuning pre-trained models (architecture incompatible)
- ❌ Need for residual connection compatibility
- ❌ Extremely resource-constrained environments

## Common Patterns

### Pattern 1: Just the Hyper-Connection Block

```python
from hyper_connections import HyperConnection

# Use in your own architecture
hc = HyperConnection(hidden_size=768, expansion_rate=4)

# Apply after any layer
layer_output = attention_layer(x)  # or ffn_layer(x)
x = hc(layer_output, input_hidden=x)
```

### Pattern 2: Custom Transformer Layer

```python
from hyper_connections import TransformerLayerWithHC

# Use individual layers
layer = TransformerLayerWithHC(
    hidden_size=768,
    num_heads=12,
    ffn_hidden_size=3072,
    expansion_rate=4,
)

# Stack them yourself
layers = [layer for _ in range(12)]
```

### Pattern 3: Hybrid Architecture

```python
# Mix residual and hyper-connections
class HybridTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # First 6 layers: residual
        self.res_layers = nn.ModuleList([
            StandardTransformerLayer(...) 
            for _ in range(6)
        ])
        # Last 6 layers: hyper-connections
        self.hc_layers = nn.ModuleList([
            TransformerLayerWithHC(...) 
            for _ in range(6)
        ])
```

## Expected Results (from paper)

With hyper-connections, expect:
- ✅ **Faster convergence**: 1.8× compared to baseline
- ✅ **Lower perplexity**: Across multiple benchmarks
- ✅ **Better gradient flow**: No vanishing or collapse
- ✅ **Minimal overhead**: Negligible extra parameters

## Troubleshooting

### Import Error
```bash
# Activate venv first!
source .venv/bin/activate
```

### CUDA Out of Memory
```python
# Reduce batch size or sequence length
model = TransformerWithHC(..., max_seq_length=512)  # Default: 2048
```

### Slow Training
```python
# Use smaller expansion rate
model = TransformerWithHC(..., expansion_rate=2)  # Instead of 4 or 8
```

## More Information

- **Full examples**: `python example_usage.py`
- **Run tests**: `python test_implementation.py`
- **Documentation**: See `README.md`
- **Installation help**: See `INSTALL.md`
- **Paper**: See `HYPER-CONNECTIONS.pdf`

## API Reference (Quick)

### TransformerWithHC
```python
TransformerWithHC(
    vocab_size: int,              # Required
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    ffn_hidden_size: int = None,  # Defaults to 4*hidden_size
    max_seq_length: int = 2048,
    expansion_rate: int = 4,      # DHC×4
    dropout: float = 0.1,
    use_tanh: bool = True,        # DHC variant
    static_weights: bool = False, # DHC (not SHC)
)
```

### HyperConnection
```python
HyperConnection(
    hidden_size: int,             # Required
    expansion_rate: int = 4,
    use_tanh: bool = True,
    static_weights: bool = False,
)
```

### TransformerLayerWithHC
```python
TransformerLayerWithHC(
    hidden_size: int,             # Required
    num_heads: int,               # Required
    ffn_hidden_size: int,         # Required
    expansion_rate: int = 4,
    dropout: float = 0.1,
    use_tanh: bool = True,
    static_weights: bool = False,
)
```

---

**That's it! You're ready to use Hyper-Connections! 🚀**

For questions, see the full README.md or the paper.
