# Hyper-Connections Architecture Guide

This document provides a visual and conceptual guide to understanding hyper-connections.

## Overview

Traditional residual connections:
```
h_{t+1} = h_t + layer(norm(h_t))
```

Hyper-connections with expansion rate n:
```
H_{t+1} = HC(layer_output, H_t)
where H_t ∈ ℝ^(batch × seq × n × dim)
```

## Visual Architecture

### 1. Standard Residual Connection (Pre-Norm)

```
Input: h_t [B, L, D]
  ↓
  ├─→ LayerNorm ──→ Layer ──→ [+] ──→ Output: h_{t+1} [B, L, D]
  └────────────────────────────↑
```

### 2. Hyper-Connection with n=2

```
Input: H_t [B, L, 2, D]
       h₀   h₁

  ↓ (average)
  
h_avg [B, L, D]
  ↓
  ├─→ LayerNorm ──→ Layer ──→ layer_output [B, L, D]
  ↓
  
┌─────────────────────────────────┐
│   Hyper-Connection Module       │
│                                  │
│   Depth-Connections (DC):       │
│   β₁ × layer_output              │
│   β₂ × layer_output              │
│                                  │
│   Width-Connections (WC):       │
│   α₀₀×h₀ + α₀₁×h₁  (mixing)     │
│   α₁₀×h₀ + α₁₁×h₁  (residual)   │
│   α₂₀×h₀ + α₂₁×h₁  (residual)   │
└─────────────────────────────────┘
  ↓
  
Output: H_{t+1} [B, L, 2, D]
        h'₀  h'₁
```

### 3. Complete Transformer Block with HC

```
Input: H [B, L, n, D]
  |
  |── Average over n ──→ h_avg [B, L, D]
  |                         ↓
  |                      LayerNorm
  |                         ↓
  |                   Multi-Head Attention
  |                         ↓
  |                   attn_output [B, L, D]
  |                         ↓
  └──────────────→ HC_attn (H, attn_output)
                          ↓
                     H' [B, L, n, D]
                          |
                          |── Average over n ──→ h'_avg [B, L, D]
                          |                         ↓
                          |                      LayerNorm
                          |                         ↓
                          |                    Feed-Forward
                          |                         ↓
                          |                   ffn_output [B, L, D]
                          |                         ↓
                          └──────────────→ HC_ffn (H', ffn_output)
                                                    ↓
                                             H'' [B, L, n, D]
```

## Hyper-Connection Matrix Decomposition

The HC matrix can be decomposed into depth and width connections:

### Matrix Structure (n=4 example)

```
        ┌─────────────────────────────────┐
        │   0     β₁    β₂    β₃    β₄    │  ← Output row
        │                                   │
        │  α₀₀   α₁₁   α₁₂   α₁₃   α₁₄    │  ← Input/residual rows
        │  α₀₁   α₂₁   α₂₂   α₂₃   α₂₄    │
        │  α₀₂   α₃₁   α₃₂   α₃₃   α₃₄    │
        │  α₀₃   α₄₁   α₄₂   α₄₃   α₄₄    │
        └─────────────────────────────────┘
         ↑      ↑                      ↑
         │      └──────────────────────┴─── B matrix (1×n)
         └─ Am (n×1)    Ar (n×n) ────────→
```

Decomposed:
```
B  = [β₁, β₂, β₃, β₄]           # Depth-connections (1×4)
Am = [α₀₀, α₀₁, α₀₂, α₀₃]ᵀ      # Width-connections: mixing (4×1)
Ar = [[α₁₁, α₁₂, α₁₃, α₁₄],    # Width-connections: residual (4×4)
      [α₂₁, α₂₂, α₂₃, α₂₄],
      [α₃₁, α₃₂, α₃₃, α₃₄],
      [α₄₁, α₄₂, α₄₃, α₄₄]]
```

### Initialization (Layer k, n=4)

```
k=0: Am = [1, 0, 0, 0]ᵀ    (e₀)
k=1: Am = [0, 1, 0, 0]ᵀ    (e₁)
k=2: Am = [0, 0, 1, 0]ᵀ    (e₂)
k=3: Am = [0, 0, 0, 1]ᵀ    (e₃)
k=4: Am = [1, 0, 0, 0]ᵀ    (e₀) -- cycles

B  = [1, 1, 1, 1]         (all ones)
Ar = I₄                    (identity)
```

This ensures that at initialization, HC behaves like Pre-Norm residual connections.

## Information Flow

### Forward Pass Through One Layer

```
Step 1: Prepare input for layer
  H [B, L, n, D] ──(average n)──→ h [B, L, D]

Step 2: Process through layer
  h ──(norm)──→ h_norm ──(layer)──→ layer_out [B, L, D]

Step 3: Apply hyper-connections
  
  Depth-connections (broadcast layer output):
    DC[i] = B[i] × layer_out    for i = 0...n-1
    Result: [B, L, n, D]
  
  Width-connections (mix hidden states):
    WC_mix = sum_j(Am[j] × H[:,:,j,:])
    Result: [B, L, D] → broadcast to [B, L, n, D]
    
    WC_res[i] = sum_j(Ar[i,j] × H[:,:,j,:])
    Result: [B, L, n, D]
  
  Combine:
    H_new[i] = DC[i] + WC_mix + WC_res[i]
    Result: [B, L, n, D]

Step 4: Output
  H_new [B, L, n, D]
```

## Dynamic Hyper-Connections

### Static HC (SHC)

Fixed learned matrices: B, Am, Ar

```
H_new = HC(layer_out, H; B, Am, Ar)
```

### Dynamic HC (DHC)

Input-dependent adjustments:

```
H̄ = LayerNorm(H).mean(dim=n)    [B, L, D]

δB  = tanh(H̄ · W_β)              [B, L, n]
δAm = tanh(H̄ · W_m)              [B, L, n]
δAr = tanh(H̄ · W_r)              [B, L, n, n]

B(H)  = s_β ⊙ δB  + B
Am(H) = s_α ⊙ δAm + Am
Ar(H) = s_α ⊙ δAr + Ar

H_new = HC(layer_out, H; B(H), Am(H), Ar(H))
```

Where s_α, s_β ≈ 0.01 are small learnable scalings.

## Connection Patterns

### Sequential Pattern (layers process in order)

```
Ar = [[1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]]  ← Identity

Effect: Each hidden vector independent
        h₀ → h'₀ (no mixing with h₁, h₂, h₃)
```

### Parallel Pattern (layers process together)

```
Ar = [[1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1]]  ← All ones

Effect: All hidden vectors fully mixed
        h'₀ = h'₁ = h'₂ = h'₃ (after normalization)
```

### Λ-shaped Pattern (observed after training)

```
Depth connections B over layers:

Layer 0:  [0.2, 0.8, 1.2, 0.3]  ← Early layers: moderate output
Layer 1:  [0.3, 1.0, 1.4, 0.4]
Layer 2:  [0.5, 1.2, 1.0, 0.6]  ← Middle layers: high output
Layer 3:  [0.4, 1.1, 0.9, 0.5]
Layer 4:  [0.3, 0.9, 0.7, 0.4]  ← Late layers: reduced output
Layer 5:  [0.2, 0.7, 0.5, 0.3]

Shape: Λ (inverted V) - peaks in middle, decays at ends
```

## Full Model Architecture

```
Input: token_ids [B, L]
  ↓
Token Embedding + Positional Embedding
  ↓
h [B, L, D]
  ↓
Expand to n vectors: H₀ = expand(h, n)  [B, L, n, D]
  ↓
┌─────────────────────────────────────┐
│ Transformer Block 0 (with HC)       │
│   - Attention + HC_attn             │
│   - FFN + HC_ffn                    │
└─────────────────────────────────────┘
  ↓ H₁ [B, L, n, D]
┌─────────────────────────────────────┐
│ Transformer Block 1 (with HC)       │
└─────────────────────────────────────┘
  ↓ H₂ [B, L, n, D]
  ⋮
  ↓ H_L [B, L, n, D]
  
Sum over n vectors: h_final = sum(H_L, dim=2)  [B, L, D]
  ↓
LayerNorm
  ↓
Output Projection (Linear)
  ↓
Logits [B, L, vocab_size]
```

## Key Differences from Residual Connections

| Aspect | Residual Connections | Hyper-Connections |
|--------|----------------------|-------------------|
| Hidden vectors | 1 | n (typically 4) |
| Connection strength | Fixed (h + x) | Learned (B, Am, Ar) |
| Layer arrangement | Sequential only | Can learn parallel |
| Depth mixing | Adjacent only | Can access any depth |
| Width mixing | No mixing | Learned mixing (Am, Ar) |
| Initialization | Identity | Equivalent to residual |
| Parameters | 0 | ~0.04% increase |
| Memory | 1× | ~1.15-1.30× |

## Computational Cost

### FLOPs (per layer)

**Standard Residual**:
```
FLOPs = FLOPs(attention) + FLOPs(FFN) + O(B×L×D)
                                         ↑ addition
```

**Hyper-Connection**:
```
FLOPs = FLOPs(attention) + FLOPs(FFN) + O(B×L×n×D) + O(B×L×n²×D)
                                         ↑ DC + WC mixing   ↑ WC residual

For n=4: ~0.3% FLOPs increase
```

### Memory (training)

**Activations**:
- Standard: B × L × D × num_layers
- HC: B × L × n × D × num_layers ≈ n× more

**Parameters**:
- Additional: O(n² × num_layers + D × n² × num_layers)
- Percentage: ~0.04% for DHC, ~0.01% for SHC

**Gradients**:
- Similar to activations: n× increase

**Total Training Memory**: ~15-30% increase (for n=4)

Can be reduced with gradient checkpointing (not implemented).

## When to Use Hyper-Connections

### ✅ Good Use Cases

1. **Pre-training large language models**
   - Especially MoE architectures (1.8× faster convergence)
   - Dense models show consistent improvements
   - Worth the small memory overhead

2. **Vision transformers**
   - ViT-style architectures
   - Larger models benefit more (ViT-Large > ViT-Small)

3. **When training from scratch**
   - HC learns optimal connections during training
   - Best results when trained end-to-end

4. **Research into architecture**
   - Study learned connection patterns
   - Understand layer interactions

### ❌ Less Suitable Use Cases

1. **Fine-tuning pre-trained models**
   - Can't add HC to existing checkpoints easily
   - Would need to retrain from scratch

2. **Extreme memory constraints**
   - ~20% memory increase may be prohibitive
   - Consider using n=2 instead of n=4

3. **Inference-only deployment**
   - Need to train with HC first
   - Inference overhead minimal but present

4. **Very small models**
   - Benefits less pronounced for tiny models
   - Overhead relatively larger

## Debug Checklist

When implementing or debugging HC:

1. **Check shapes**:
   - [ ] Hidden states: [B, L, n, D] throughout blocks
   - [ ] Layer output: [B, L, D] before HC
   - [ ] Final output: [B, L, D] after summing n vectors

2. **Check initialization**:
   - [ ] B = [1, 1, ..., 1]
   - [ ] Am[k % n] = 1, rest = 0
   - [ ] Ar = identity matrix

3. **Check scaling**:
   - [ ] Attention output layer scaled by √n
   - [ ] FFN output layer scaled by √n

4. **Check weight decay**:
   - [ ] B, Am, Ar: weight_decay = 0
   - [ ] W_β, W_m, W_r: weight_decay > 0

5. **Check gradients**:
   - [ ] All HC parameters have gradients
   - [ ] No NaN or Inf in gradients
   - [ ] Gradient magnitudes reasonable

6. **Check patterns**:
   - [ ] After training, B should be non-uniform
   - [ ] Ar should deviate from identity
   - [ ] Patterns should differ across layers

## Summary

Hyper-connections replace fixed residual connections with learned, adaptive connections:

- **Depth-connections**: Learn how much of new layer output to use
- **Width-connections**: Learn how hidden vectors communicate
- **Dynamic adjustments**: Adapt connections based on input
- **Minimal overhead**: <0.04% parameters, <0.3% FLOPs
- **Significant gains**: Especially for large-scale pre-training

The key insight: **Let the network learn the optimal connection structure rather than fixing it a priori.**

---

**Implementation**: See `hyper_connections.py` and `transformer_hc.py`

**Tests**: Run `python test_implementation.py`

**Example**: Run `python example_usage.py`
