# Quick Start Guide

Get started with Hyper-Connections in 5 minutes!

## Installation

```bash
# Using uv (recommended)
uv add torch

# Or using pip
pip install torch>=2.0.0

# Or install all dependencies
pip install -r requirements.txt
```

## Basic Usage (3 steps)

### Step 1: Create a model

```python
import torch
from transformer_hc import TransformerWithHC

model = TransformerWithHC(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    expansion_rate=4,      # n=4 (recommended)
    dynamic=True,          # Use Dynamic HC
)
```

### Step 2: Configure optimizer

```python
from hyper_connections import configure_optimizers_for_hyper_connections

optimizer = configure_optimizers_for_hyper_connections(
    model,
    learning_rate=3e-4,
    weight_decay=0.01,
)
```

### Step 3: Train normally

```python
# Your training loop
for batch in dataloader:
    input_ids, labels = batch
    
    # Forward (use causal masking for language modeling)
    logits = model(input_ids, use_causal_mask=True)
    loss = compute_loss(logits, labels)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

That's it! The model will learn optimal connection patterns automatically.

## Complete Example

Run the provided example:

```bash
python example_usage.py
```

This will:
- Create a model with hyper-connections
- Train on a simple task
- Visualize learned connection patterns
- Generate sample text

## Verify Implementation

Run the test suite:

```bash
python test_implementation.py
```

All tests should pass, confirming:
- ✓ Correct initialization
- ✓ Proper output scaling
- ✓ Gradient flow
- ✓ Weight decay configuration

## Configuration Guide

### Choose Expansion Rate (n)

```python
# For language models (recommended)
expansion_rate=4

# For vision models
expansion_rate=2

# Don't use (reverts to seesaw effect)
expansion_rate=1  # ❌
```

### Choose HC Variant

```python
# Dynamic HC (best performance, recommended)
dynamic=True, use_tanh=True

# Static HC (simpler, faster)
dynamic=False

# Dynamic without tanh (experimental)
dynamic=True, use_tanh=False
```

### Model Size vs Memory

| Config | Params Increase | Memory Increase |
|--------|----------------|-----------------|
| n=2, Static | ~0.01% | ~10% |
| n=2, Dynamic | ~0.02% | ~10% |
| n=4, Static | ~0.01% | ~20% |
| n=4, Dynamic | ~0.04% | ~20% |
| n=8, Dynamic | ~0.08% | ~35% |

## Key Points to Remember

### ✅ DO

1. **Use n=4** for language models (best default)
2. **Use Dynamic HC** for best performance
3. **Configure weight decay properly** (use provided function)
4. **Scale output layers by √n** (done automatically)
5. **Train from scratch** for best results

### ❌ DON'T

1. **Don't use n=1** (defeats the purpose)
2. **Don't apply weight decay to B, Am, Ar** (handled automatically)
3. **Don't forget gradient clipping** (helps stability)
4. **Don't expect magic with tiny models** (benefits scale with size)
5. **Don't add HC to pre-trained models** (need to train from scratch)

## Expected Results

After training, you should see:

### Better Loss
- Language models: -0.02 to -0.04 validation loss
- Vision models: +1-3% accuracy

### Learned Patterns
- Non-uniform B (depth connections)
- Non-identity Ar (width connections)
- Layer-specific patterns

### Stable Training
- No gradient spikes (unlike some baselines)
- Smooth convergence
- Good generalization

## Troubleshooting

### "Out of memory"
- Reduce batch size
- Use smaller n (try n=2)
- Enable gradient checkpointing (requires modification)

### "Loss not improving"
- Check that n > 1
- Verify weight decay config
- Try learning rate tuning
- Ensure sufficient training steps

### "NaN in gradients"
- Check output scaling (should be automatic)
- Add gradient clipping (max_norm=1.0)
- Reduce learning rate

## File Guide

| File | Purpose |
|------|---------|
| `hyper_connections.py` | Core HC module |
| `transformer_hc.py` | Transformer architecture |
| `example_usage.py` | Training example |
| `test_implementation.py` | Validation tests |
| `ARCHITECTURE.md` | Visual architecture guide |
| `IMPLEMENTATION.md` | Detailed implementation docs |
| `README.md` | Project overview |
| `QUICKSTART.md` | This file |

## Next Steps

1. **Experiment with configurations**:
   ```python
   for n in [2, 4, 8]:
       model = TransformerWithHC(..., expansion_rate=n)
       # Train and compare
   ```

2. **Visualize connection patterns**:
   ```python
   for block in model.blocks:
       print(block.hc_attn.B)  # See learned patterns
   ```

3. **Compare with baseline**:
   - Train standard Pre-Norm model
   - Train same model with HC
   - Compare final loss/accuracy

4. **Read the paper**:
   - See `HYPER-CONNECTIONS.pdf`
   - Understand theoretical motivation
   - Learn about observed patterns

## Examples for Different Tasks

### Language Modeling

```python
model = TransformerWithHC(
    vocab_size=50000,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    ffn_dim=3072,
    max_seq_len=2048,
    expansion_rate=4,
    dynamic=True,
)
```

### Vision Transformer

```python
from transformer_hc import TransformerBlockWithHC

# Adapt for ViT (patches as tokens)
class ViTWithHC(nn.Module):
    def __init__(self):
        # Patch embedding
        # Add position embeddings
        # Use TransformerBlockWithHC for layers
        # Classification head
        pass
```

### Encoder-Decoder

```python
# Encoder
encoder = TransformerWithHC(...)

# Decoder (with cross-attention)
# Would need modification to add cross-attention
# Not implemented in current version
```

## Community & Support

- **Paper**: See `HYPER-CONNECTIONS.pdf`
- **Implementation**: All code in this repo
- **Issues**: File issues on GitHub (if available)
- **Citation**: See README.md

## Summary

Hyper-connections are:
- ✅ Easy to use (3 steps)
- ✅ Drop-in replacement for residual connections
- ✅ Minimal overhead (<0.04% params)
- ✅ Significant improvements (especially large-scale)
- ✅ Well-tested implementation

Just create model → configure optimizer → train!

The network will learn optimal connection patterns automatically.

---

**Ready to start?** Run `python example_usage.py` now!
