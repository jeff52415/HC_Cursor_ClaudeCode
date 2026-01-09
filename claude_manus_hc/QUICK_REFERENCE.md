# Quick Reference: Hyper-Connections Implementation

## Core Concepts

### What are Hyper-Connections?

Hyper-connections (HC) are an alternative to residual connections that:
- Learn optimal connection strengths between layers
- Create multiple parallel information pathways (expansion rate `n`)
- Can adapt dynamically based on input (Dynamic HC)
- Address gradient vanishing and representation collapse

### Key Difference from Residual Connections

**Residual Connection (Pre-Norm):**
```python
x = x + layer(norm(x))  # Fixed 1:1 weighting
```

**Hyper-Connection:**
```python
# Multiple hidden states with learnable weights
h₀ = Σᵢ αᵢ·hᵢ              # Width connection (input mixing)
output = layer(norm(h₀))  # Process mixed input
ĥᵢ = βᵢ·output + h'ᵢ      # Depth connection (weighted output)
```

## Quick Start

### 1. Basic Usage

```python
from transformer_hyper import TransformerEncoderHyper

# Create model
model = TransformerEncoderHyper(
    vocab_size=5000,
    dim=512,
    num_layers=6,
    expansion_rate=4,  # n=4 recommended
    dynamic=True       # Use Dynamic HC
)

# Forward pass
import torch
input_ids = torch.randint(0, 5000, (batch, seq_len))
logits = model(input_ids)
```

### 2. Using Just the HC Block

```python
from hyper_connection import HyperConnection

# Create HC module
hc = HyperConnection(dim=512, expansion_rate=4, dynamic=True)

# Use with any layer
layer = nn.Linear(512, 512)
h = torch.randn(batch, seq, 4, 512)  # (B, L, n, d)
h_out = hc(h, layer)
```

## Key Parameters

### HyperConnection

| Parameter | Default | Description | Paper Reference |
|-----------|---------|-------------|-----------------|
| `dim` | Required | Hidden dimension | Section 2.1 |
| `expansion_rate` | 4 | Number of hyper hidden vectors (n) | Table 1 |
| `layer_id` | 0 | Layer index for initialization | Eq. 14 |
| `dynamic` | True | Use dynamic (DHC) vs static (SHC) | Section 2.2 |
| `use_tanh` | True | Apply tanh activation | Eq. 11-13 |

### TransformerEncoderHyper

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | Required | Vocabulary size |
| `dim` | 512 | Model dimension |
| `num_layers` | 6 | Number of transformer blocks |
| `num_heads` | 8 | Number of attention heads |
| `expansion_rate` | 4 | HC expansion rate (n) |
| `max_seq_len` | 512 | Maximum sequence length |
| `dropout` | 0.1 | Dropout probability |
| `dynamic` | True | Use dynamic HC |

## Architecture Flow

### Transformer Block with HC

```
Input: H = [h₁, h₂, h₃, h₄]^T  (n=4 example)
│
├─ Attention Block ──────────────────┐
│  ├─ Width Connection                │
│  │   • Mix inputs: h₀ = Σ αᵢ·hᵢ    │
│  │   • Compute dynamic α, β         │
│  ├─ Layer Norm(h₀)                  │
│  ├─ Multi-Head Attention            │
│  └─ Depth Connection                │
│      • ĥᵢ = βᵢ·attn_out + h'ᵢ       │
│                                     │
├─ FFN Block ────────────────────────┤
│  ├─ Width Connection                │
│  ├─ Layer Norm                      │
│  ├─ Feed Forward Network            │
│  └─ Depth Connection                │
│                                     │
Output: Ĥ = [ĥ₁, ĥ₂, ĥ₃, ĥ₄]^T       │
```

### Final Output

```python
# Sum all hyper hidden vectors
h_final = Σᵢ ĥᵢ  # Row-wise sum

# Project to output
logits = output_layer(layer_norm(h_final))
```

## Implementation Checklist

When implementing HC in your model, ensure:

- [x] Input replicated n times: `H⁰ = [h⁰, h⁰, ..., h⁰]^T`
- [x] Width connections compute h₀ from all hᵢ
- [x] Depth connections weight output by β
- [x] Initialization matches Pre-Norm (Eq. 14)
- [x] Output layers scaled by √n
- [x] Final sum of hyper hidden vectors
- [x] Dynamic parameters use norm + linear + tanh
- [x] Static parameters exclude weight decay

## Equations Reference

### Core HC Computation (Eq. 2)

```
Ĥ = HC(T, H) = B^T·T(H^T·Am)^T + Ar^T·H
```

Where:
- `H ∈ R^(n×d)`: Hyper hidden matrix
- `Am ∈ R^(n×1)`: Width connection weights (input mixing)
- `Ar ∈ R^(n×n)`: Depth connection weights (residual)
- `B ∈ R^(1×n)`: Output weights
- `T`: Transformer layer (attention or FFN)

### Dynamic Parameters (Eq. 10-13)

```python
H̄ = LayerNorm(H)
B(H) = sβ ⊙ tanh(H̄·Wβ)^T + B
Am(H) = sα ⊙ tanh(H̄·Wm) + Am
Ar(H) = sα ⊙ tanh(H̄·Wr) + Ar
```

### Initialization (Eq. 14)

```
[[0,           1, 1, ..., 1],
 [e_{k mod n}, I_{n×n}]]
```

## Performance Tips

### From Paper Experiments

1. **Best Expansion Rate**: n=4 (Table 1, page 6)
   - n=1: Poor (worse than baseline)
   - n=2: Good (+1.4%)
   - n=4: Best (+1.9%) ✓
   - n=8: Similar to n=4

2. **Use Dynamic HC**: Outperforms Static HC (Table 2, page 7)

3. **Keep tanh activation**: Helps training stability

4. **Train both WC and B**: Critical for performance (Table 3, page 8)

### Computational Cost

- Parameter overhead: <1% (Table 7, page 15)
- FLOPs overhead: ~0.2% (Table 8, page 16)
- Memory overhead: ~26% during training (Table 9, page 16)
  - Can be reduced with activation recomputation

## Common Issues

### Issue: Output shape mismatch
**Solution**: Ensure input is expanded to (B, L, n, d) shape

### Issue: Initialization doesn't match Pre-Norm
**Solution**: Check Eq. 14 - use e_{k mod n} and identity matrix

### Issue: Poor performance with n=1
**Solution**: Use n≥2. Paper shows n=1 underperforms baseline

### Issue: Training unstable
**Solution**: Ensure output layers scaled by √n (Section 4)

## Testing Your Implementation

```python
def test_implementation():
    model = TransformerEncoderHyper(
        vocab_size=1000, dim=256, num_layers=4,
        expansion_rate=4, dynamic=True
    )

    # Test forward pass
    x = torch.randint(0, 1000, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 1000)

    # Test backward pass
    loss = logits.sum()
    loss.backward()

    # Check HC parameters exist
    for block in model.blocks:
        assert hasattr(block, 'attn_hyper_connection')
        assert hasattr(block, 'ffn_hyper_connection')

    print("✓ All tests passed!")
```

## File Organization

```
claude_manus_hc/
├── hyper_connection.py      # Core HC module ⭐
├── transformer_hyper.py     # Complete Transformer ⭐
├── example_usage.py         # Usage examples
├── README.md                # Full documentation
├── QUICK_REFERENCE.md       # This file
├── task_plan.md             # Implementation plan
├── notes.md                 # Paper notes
└── HYPER-CONNECTIONS.pdf    # Original paper
```

**Start with**: `example_usage.py` for practical examples
**Core modules**: `hyper_connection.py` and `transformer_hyper.py`

## Further Reading

- **Paper**: Section 2 (Method) - Core architecture
- **Code**: Algorithm 2-3 (Appendix J) - PyTorch implementation
- **Architecture**: Figure 8 (Appendix A) - Complete Transformer
- **Results**: Section 4 - Experimental results
