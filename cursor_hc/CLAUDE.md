# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **Hyper-Connections**, a novel alternative to residual connections for deep neural networks, as published at ICLR 2025. Hyper-connections address common drawbacks of residual connection variants (Pre-Norm and Post-Norm) such as the seesaw effect between gradient vanishing and representation collapse.

## Core Concepts

### What are Hyper-Connections?

Hyper-connections allow networks to:
- **Adjust connection strength** between features at different depths dynamically
- **Rearrange layers** by learning optimal layer arrangements (sequential, parallel, or hybrid)
- **Eliminate representation collapse** while maintaining gradient flow

### Key Components

1. **Expansion Rate (n)**: The number of parallel hidden vectors maintained. Typical values: n=2, 4, 8
   - n=1 does NOT work well (reverts to seesaw effect)
   - n=4 is the recommended default for most applications

2. **Depth-Connections (DC)**: Weighted connections between layer inputs and outputs across depths
   - Matrix representation: `DC = [B; diag(Ar)]` where B weights layer outputs, diag(Ar) weights inputs

3. **Width-Connections (WC)**: Information exchange between n hidden vectors within the same layer
   - Matrix representation: `WC = [Am, Ar]`

4. **Static vs Dynamic Hyper-Connections**:
   - **SHC**: Fixed learnable connection weights
   - **DHC**: Input-dependent connection weights (generally better performance)

## Architecture Details

### Hyper-Connection Matrix Structure

The hyper-connection matrix HC ∈ ℝ^(n+1)×(n+1) has the structure:
```
HC = [0      B     ]
     [Am     Ar    ]
```

Where:
- **B** ∈ ℝ^(1×n): Weights for layer output
- **Am** ∈ ℝ^(n×1): Weights for mixing inputs
- **Ar** ∈ ℝ^(n×n): Weights for residual connections

### Initialization Strategy

To match Pre-Norm residual connections at initialization:
```python
HC^k = [0_{1×1}    1_{1×n}     ]
       [e_{k mod n} e_{n×n}     ]
```

Where e_i is the i-th column of the identity matrix.

### Dynamic Hyper-Connections

DHC adds input-dependent adjustments:
```python
B(H) = s_β ⊙ tanh(H̄W_β)^T + B
Am(H) = s_α ⊙ tanh(H̄W_m) + Am
Ar(H) = s_α ⊙ tanh(H̄W_r) + Ar
```

Where H̄ = norm(H) and s_β, s_α are small learnable scaling factors (~0.01).

## Implementation Notes

### Standard Output Scaling

When using expansion rate n, scale the standard deviation of output layer weights by √n to maintain consistent output variance:
```python
output_layer.weight.data *= 1.0 / math.sqrt(n)
```

This applies to:
- Second linear layer of FFN
- Output projector of attention module

### Weight Decay Configuration

- **Static components** (B, Am, Ar): NO weight decay
- **Dynamic components** (W_β, W_m, W_r): YES weight decay

### Computational Overhead

- **Parameters**: Negligible increase (< 0.04% for most models)
- **FLOPs**: < 0.3% increase
- **Memory**: ~15-30% increase during training (can be optimized with recomputation)

### Connection Patterns Learned

Typical patterns observed in trained models:
1. **Λ-shaped pattern**: Long-term decay combined with frequent access to bottom layers
2. **Parallel Transformer Blocks (PTB)**: Attention and FFN layers operating in parallel
3. **Input embedding elimination**: Word embeddings removed from final output
4. **Attention layers**: Tend to have fewer long-term connections than FFN layers

## Common Development Tasks

### Training Models with Hyper-Connections

Replace residual connections with hyper-connections in transformer blocks:
```python
# Instead of: h = h + layer(norm(h))
# Use hyper-connections module that manages n hidden vectors
```

### Recommended Hyperparameters

**For Language Models (OLMo-style)**:
- Expansion rate: n=4 (default), try n=2 or n=8 for experimentation
- Use Dynamic Hyper-Connections (DHC) for best performance
- Include tanh activation in DHC
- Train with same learning rate and schedule as baseline

**For Vision Models (ViT-style)**:
- Expansion rate: n=2 typically sufficient
- Both SHC and DHC show improvements
- DHC shows larger gains at larger scales (ViT-Large)

### Expected Performance Improvements

**Language Models**:
- 1B dense models: ~0.03-0.04 loss reduction on validation
- 7B dense models: ~0.02 loss reduction
- MoE models: **1.8× faster convergence**, 6+ point improvements on downstream tasks

**Vision Models**:
- ImageNet classification: +1-3% accuracy improvement
- Image generation (DiT): Comparable to 50% larger baseline models

## Sequential-Parallel Duality

Hyper-connections can learn to arrange layers:

**Sequential** (standard residual):
```
HC = [0  1  1]
     [1  1  0]
     [0  0  1]
```

**Parallel** (every n layers):
```
HC_odd  = [0  1  0]    HC_even = [0  0  1]
          [1  1  1]              [0  1  0]
          [1  1  1]              [1  0  1]
```

Networks learn soft mixtures or dynamic arrangements between these extremes.

## Critical Implementation Details

### DO NOT Do These

1. **Never use n=1**: This reverts to the seesaw effect and performs worse than baseline
2. **Never skip output scaling**: Must scale by √n or output variance explodes
3. **Never apply weight decay to static components**: Breaks initialization equivalence
4. **Never use hyper-connections without training both B and WC**: Both are critical for performance

### Edge Cases

- **Representation collapse in Pre-Norm**: Hyper-connections reduce layer similarity (see Fig. 3 in paper)
- **Training instability**: Use QK-Norm for vision tasks, standard techniques work for LLMs
- **Memory constraints**: Implement activation checkpointing for hyper-connection outputs

## Testing and Validation

### Verification Checklist

1. Verify initialization matches Pre-Norm residual baseline
2. Check output scaling is applied correctly
3. Validate connection matrix patterns make sense (use visualization code from paper)
4. Monitor training stability (no spikes with DHC, unlike baseline)
5. Confirm downstream task improvements

### Debugging Tips

- Visualize connection matrices C^(0) to understand learned patterns
- Check cosine similarity between adjacent layer inputs (should be lower than Pre-Norm)
- Verify n hidden vectors are dissimilar (not all doing the same thing)
- Inspect dynamic weights to ensure they're being learned (not stuck at initialization)

## Architecture-Specific Notes

### Transformers

- Apply hyper-connections to BOTH attention and FFN blocks
- Each block gets its own HC module with separate parameters
- Final layer sums n hidden vectors before LayerNorm and unembedding

### Vision Transformers

- Same application as text transformers
- Particularly effective at larger scales (ViT-Large)
- Consider n=2 for computational efficiency

### Mixture-of-Experts

- Apply to both shared and expert layers
- Especially effective for MoE architectures (see OLMoE results)
- Can achieve 1.8× faster convergence

## References

For theoretical details, see the ICLR 2025 paper: "HYPER-CONNECTIONS" by Zhu et al.

Key sections:
- Section 2: Method description
- Section 3: Theoretical analysis (residual connections as non-trainable HC)
- Section 4.5: Visualization and learned patterns
- Appendix B: Computational cost analysis
- Appendix I-J: PyTorch implementation pseudocode
