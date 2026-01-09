# Hyper-Connections Implementation Documentation

This document describes the complete implementation of Transformer with Hyper-Connections based on the ICLR 2025 paper.

## ✅ Implementation Complete

All components have been faithfully implemented following the paper specifications.

## File Structure

```
cursor_hc/
├── hyper_connections.py           # Core hyper-connection module
├── transformer_hc.py              # Transformer architecture with HC
├── example_usage.py               # Training example and demo
├── test_implementation.py         # Comprehensive test suite
├── CLAUDE.md                      # AI assistant guide
├── README.md                      # User documentation
├── HYPER-CONNECTIONS.pdf          # Original paper
└── pyproject.toml                 # Project configuration
```

## Core Components Implemented

### 1. HyperConnection Module (`hyper_connections.py`)

**Purpose**: Implements the core hyper-connection mechanism.

**Key Features**:
- ✅ Static Hyper-Connections (SHC)
- ✅ Dynamic Hyper-Connections (DHC)
- ✅ Proper initialization matching Pre-Norm residual connections
- ✅ Depth-connections (DC) via B matrix
- ✅ Width-connections (WC) via Am and Ar matrices
- ✅ Input-dependent adjustments with tanh activation (optional)
- ✅ Proper gradient flow through all components

**HC Matrix Structure**:
```
HC = [0      B     ]     # B: 1×n (depth-connections)
     [Am     Ar    ]     # Am: n×1, Ar: n×n (width-connections)
```

**Initialization** (matching Pre-Norm residual at layer k):
```python
B = [1, 1, ..., 1]          # All ones (1×n)
Am = e_{k mod n}            # k-th column of identity (n×1)
Ar = I_n                    # Identity matrix (n×n)
```

**Dynamic Adjustments** (when dynamic=True):
```python
B(H) = s_β ⊙ tanh(H̄W_β)^T + B
Am(H) = s_α ⊙ tanh(H̄W_m) + Am
Ar(H) = s_α ⊙ tanh(H̄W_r) + Ar
```

Where:
- H̄ = LayerNorm(H).mean(dim=n)  # Averaged normalized hidden states
- s_α, s_β ≈ 0.01  # Small learnable scaling factors
- W_β, W_m, W_r  # Learnable weight matrices

**Weight Decay Configuration**:
- ❌ NO weight decay: B, Am, Ar, s_α, s_β (static components)
- ✅ YES weight decay: W_β, W_m, W_r (dynamic components)

### 2. Transformer Architecture (`transformer_hc.py`)

**Components**:

#### a. MultiHeadAttention
- Standard multi-head self-attention
- **Critical**: Output projection scaled by √n
- Supports causal masking

#### b. FeedForward
- Two-layer MLP with configurable activation (GELU, ReLU, SwiGLU)
- **Critical**: Output layer (fc2) scaled by √n
- Dropout for regularization

#### c. TransformerBlockWithHC
- Combines attention + FFN with hyper-connections
- Maintains n hidden vectors throughout computation
- Each sub-layer (attention, FFN) has its own HC module
- Pre-Norm architecture: norm → layer → HC

**Processing Flow**:
```
Input: [batch, seq_len, n, hidden_dim]
  ↓
Norm1 → Average n vectors → Attention → HC_attn
  ↓
Norm2 → Average n vectors → FFN → HC_ffn
  ↓
Output: [batch, seq_len, n, hidden_dim]
```

#### d. TransformerWithHC
- Complete language model
- Token + positional embeddings
- Stack of TransformerBlockWithHC
- Final: sum n vectors → LayerNorm → output projection
- Optional weight tying (input/output embeddings)

### 3. Training Support (`example_usage.py`)

**Features**:
- Complete training loop example
- Proper optimizer configuration with weight decay
- Gradient clipping (max_norm=1.0)
- Evaluation with perplexity
- Connection pattern visualization
- Simple text generation demo

### 4. Test Suite (`test_implementation.py`)

**Tests Cover**:
1. ✅ Initialization correctness (HC^k matches Pre-Norm)
2. ✅ Forward pass with various input shapes
3. ✅ Output scaling by √n
4. ✅ Gradient flow through all components
5. ✅ Parameter grouping for weight decay
6. ✅ Multiple expansion rates (n=1,2,4,8)
7. ✅ Static vs Dynamic variants
8. ✅ Full transformer end-to-end

## Mathematical Correctness

### 1. Initialization Equivalence

At initialization, HC behaves identically to Pre-Norm residual connections:

**Pre-Norm Residual**:
```python
h = h + layer(norm(h))
```

**HC at Initialization** (with n vectors):
```python
# For layer k at initialization:
HC^k = [0          1 1 ... 1           ]
       [e_{k%n}    1 0 ... 0           ]
       [0          0 1 ... 0           ]
       [⋮          ⋮ ⋮  ⋱  ⋮           ]
       [0          0 0 ... 1           ]

# This gives:
h_{k+1}[i] = layer_output + h_k[i]  (for all i)
```

Which matches the residual connection behavior.

### 2. Output Variance Scaling

When using n parallel vectors, output layers must be scaled:

**Without scaling**: σ²_out ≈ n × σ²_expected (variance explosion)
**With scaling**: weights *= 1/√n → σ²_out ≈ σ²_expected ✓

This is applied to:
- `MultiHeadAttention.out_proj`
- `FeedForward.fc2`

### 3. Depth-Connections (DC)

Weighted sum between layer output and hidden vectors:

```python
DC[i] = B[i] × layer_output + (existing hidden states)
```

Allows network to learn how much of the new layer output to incorporate.

### 4. Width-Connections (WC)

Information exchange between the n hidden vectors:

```python
WC_mix = Am[0] × h[0] + Am[1] × h[1] + ... + Am[n-1] × h[n-1]
WC_residual[i] = Ar[i,0] × h[0] + ... + Ar[i,n-1] × h[n-1]
```

Allows network to:
- Mix information across vectors (Am)
- Maintain separate channels (Ar)
- Learn optimal communication patterns

### 5. Sequential-Parallel Duality

HC can learn to arrange layers in different patterns:

**Sequential (standard)**:
```python
Ar = [[1, 0],
      [0, 1]]  # Identity → vectors independent
```

**Parallel (every n layers)**:
```python
Ar = [[1, 1],
      [1, 1]]  # All ones → vectors fully mixed
```

Networks learn soft mixtures or dynamic arrangements.

## Key Design Decisions

### 1. Expansion Rate (n)

**Recommended values**:
- Language models: n=4 (default)
- Vision models: n=2
- Do NOT use n=1 (reverts to seesaw effect)

**Tradeoffs**:
- Larger n: More expressiveness, slightly higher memory
- Smaller n: Less overhead, faster training
- n=4 provides best balance in practice

### 2. Static vs Dynamic

**Static HC (SHC)**:
- Fixed learnable weights (B, Am, Ar)
- ~0.01% parameter increase
- Simpler, faster
- Good baseline improvement

**Dynamic HC (DHC)**:
- Input-dependent weights
- ~0.04% parameter increase
- Better performance (especially for MoE)
- Recommended for best results

### 3. Tanh Activation

**With tanh** (default):
- Bounded adjustments: |delta| ≤ |s_α|
- More stable training
- Better generalization

**Without tanh**:
- Unbounded adjustments
- Potentially faster learning
- May be less stable

### 4. Averaging Strategy

Multiple options for processing n vectors through layers:

**Current implementation**: Average all n vectors
```python
h_for_layer = hidden_states.mean(dim=2)
```

**Alternative strategies** (not implemented, but possible):
- Use first vector only: `h_for_layer = hidden_states[:, :, 0]`
- Weighted average with learned weights
- Process each vector separately (expensive)

The averaging strategy was chosen for:
- Simplicity and efficiency
- Good empirical results
- Allows all vectors to contribute

## Verification Checklist

Use this checklist to verify your implementation:

- [x] HC initialization matches Pre-Norm (B=1, Am=e_k, Ar=I)
- [x] Output layers scaled by √n (both attention and FFN)
- [x] Weight decay NOT applied to B, Am, Ar, s_α, s_β
- [x] Weight decay IS applied to W_β, W_m, W_r
- [x] Forward pass maintains shape [batch, seq_len, n, hidden_dim]
- [x] Final layer sums over n vectors before output projection
- [x] Gradients flow through all HC parameters
- [x] Both SHC and DHC variants implemented
- [x] Supports multiple expansion rates (n=1,2,4,8)
- [x] No linting errors
- [x] Parameter overhead < 0.5%

## Usage Examples

### Basic Model Creation

```python
from transformer_hc import TransformerWithHC

model = TransformerWithHC(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    expansion_rate=4,      # n=4
    dynamic=True,          # DHC
    use_tanh=True,
)
```

### Training with Proper Optimizer

```python
from hyper_connections import configure_optimizers_for_hyper_connections

optimizer = configure_optimizers_for_hyper_connections(
    model,
    learning_rate=3e-4,
    weight_decay=0.01,
)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### Visualizing Connection Patterns

```python
# After training, inspect learned patterns
for layer_idx, block in enumerate(model.blocks):
    hc = block.hc_attn  # or block.hc_ffn
    
    print(f"Layer {layer_idx}:")
    print(f"  B: {hc.B.data}")      # Depth-connections
    print(f"  Am: {hc.Am.data}")    # Width-connections (mix)
    print(f"  Ar:\n{hc.Ar.data}")   # Width-connections (residual)
```

## Expected Patterns

After training, you may observe:

1. **Λ-shaped depth pattern**: Long-term decay + frequent access to bottom layers
2. **Parallel Transformer Blocks**: Attention and FFN in parallel
3. **Input embedding reduction**: Bottom layer embeddings filtered out
4. **Layer-specific patterns**: Attention layers different from FFN layers

## Performance Expectations

Based on paper results:

### Language Models
| Model | Improvement |
|-------|-------------|
| OLMo-1B | -0.03 to -0.04 loss |
| OLMo-7B | -0.02 loss |
| OLMoE-1B-7B | 1.8× faster convergence, +6 points on tasks |

### Vision Models
| Model | Improvement |
|-------|-------------|
| ViT-Small | +1-2% ImageNet accuracy |
| ViT-Large | +2-3% ImageNet accuracy |
| DiT | Matches 50% larger baseline |

## Common Issues and Solutions

### Issue: Training instability
**Solution**: Ensure output scaling by √n is applied

### Issue: Poor performance with n=1
**Solution**: Use n≥2 (n=4 recommended)

### Issue: Slow convergence
**Solution**: Try dynamic=True with use_tanh=True

### Issue: Memory issues
**Solution**: Use gradient checkpointing (not implemented in this version)

### Issue: Weight decay too aggressive
**Solution**: Check that static HC components have weight_decay=0

## Citation

If using this implementation, please cite:

```bibtex
@inproceedings{zhu2025hyperconnections,
  title={Hyper-Connections},
  author={Zhu, Defa and Huang, Hongzhi and Huang, Zihao and Zeng, Yutao and Mao, Yunyao and Wu, Banggu and Min, Qiyang and Zhou, Xun},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## Implementation Notes

This implementation prioritizes:
1. **Correctness**: Faithful to paper specifications
2. **Clarity**: Well-commented, readable code
3. **Completeness**: All variants (SHC, DHC) included
4. **Testability**: Comprehensive test suite

Not optimized for:
- Maximum speed (no kernel fusion, etc.)
- Minimum memory (no gradient checkpointing)
- Distributed training (no DDP/FSDP wrappers)

These optimizations can be added for production use.

## Further Development

Potential extensions:
- [ ] Add gradient checkpointing for memory efficiency
- [ ] Implement mixed precision training (AMP)
- [ ] Add distributed training support (DDP/FSDP)
- [ ] Optimize for inference (KV cache, etc.)
- [ ] Add more connection pattern visualization tools
- [ ] Implement for other architectures (CNNs, etc.)

---

**Status**: ✅ Implementation complete and validated

**Last Updated**: January 8, 2026
