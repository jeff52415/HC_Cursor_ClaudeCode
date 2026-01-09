# Notes: Hyper-Connections Architecture

## Key Paper Sections

### Section 2.1: Static Hyper-Connections (Page 3)

**Hyper Hidden Matrix:**
- Input h^(k-1) ∈ R^d is replicated n times to form H^0 = [h^0, h^0, ..., h^0]^T ∈ R^(n×d)
- For k-th layer: H^(k-1) = [h^(k-1)_1, h^(k-1)_2, ..., h^(k-1)_n]^T ∈ R^(n×d)
- Final output: sum rows of last hyper hidden matrix

**HC Matrix Structure (Eq 1):**
```
HC = [[0,      B],         where B = [β1, β2, ..., βn] ∈ R^(1×n)
      [Am,     Ar]]              Am ∈ R^(n×1), Ar ∈ R^(n×n)
```

**Forward Pass (Eq 2-5):**
1. Width connections: h0^T = Am^T * H  (weighted sum of inputs)
2. Layer computation: h'_0 = T(h0)
3. Map to hyper hidden: H' = Ar^T * H
4. Depth connections: Ĥ = B^T * (h'_0)^T + H'

**Decomposition:**
- Depth-connections (Eq 6): DC = [B; diag(Ar)] ∈ R^(2×n)
- Width-connections (Eq 7): WC = [Am, Ar] ∈ R^(n×(n+1))

### Section 2.2: Dynamic Hyper-Connections (Page 4)

**Dynamic Parameters (Eq 10-13):**
```python
H̄ = norm(H)
B(H) = sβ ⊙ tanh(H̄ * Wβ)^T + B
Am(H) = sα ⊙ tanh(H̄ * Wm) + Am
Ar(H) = sα ⊙ tanh(H̄ * Wr) + Ar
```

Where:
- Wβ ∈ R^(d×n), Wm ∈ R^(d×1), Wr ∈ R^(d×n×n)
- sβ, sα are small learnable scaling factors (init 0.01)
- All W matrices initialized to 0

### Section 2.3: Initialization (Page 4)

**Initialize to match Pre-Norm residual (Eq 14):**
```
[[0,           1, 1, ..., 1],
 [e_(k mod n), I_(n×n)]]
```
- e_(k mod n) is k-th column of identity (cycling)
- I_(n×n) is n×n identity matrix
- Dynamic parameters Wβ, Wm, Wr initialized to 0

### Section 3.1: Relation to Residual Connections (Page 4-5)

**Pre-Norm as HC (Eq 15, 32):**
```
HC_PreNorm = [[0, 1],
              [1, 1]]  with n=1
```

**Post-Norm as HC (Eq 16, 37):**
```
HC_PostNorm = [[0,                    1/√(σ_i² + σ_o² + 2σ_io)],
               [1,                    1/√(σ_i² + σ_o² + 2σ_io)]]  with n=1
```

### Implementation Details from Appendix J (Page 29)

**Algorithm 2: HyperConnection Module**
```python
class HyperConnection(nn.Module):
    def __init__(self, dim, rate, layer_id, dynamic):
        # Static parameters
        self.static_beta = ones(rate)
        self.static_alpha = [e_(layer_id mod rate), I_(rate×rate)]

        if dynamic:
            # Dynamic parameters
            self.dynamic_alpha_fn = zeros(dim, rate+1)
            self.dynamic_alpha_scale = ones(1) * 0.01
            self.dynamic_beta_fn = zeros(dim)
            self.dynamic_beta_scale = ones(1) * 0.01
            self.layer_norm = LayerNorm(dim)

    def width_connection(self, h):
        # Compute alpha and beta
        if dynamic:
            norm_h = self.layer_norm(h)
            dynamic_alpha = tanh(norm_h @ dynamic_alpha_fn) * scale
            alpha = dynamic_alpha + static_alpha
            dynamic_beta = tanh(norm_h @ dynamic_beta_fn) * scale
            beta = dynamic_beta + static_beta
        else:
            alpha, beta = static_alpha, static_beta

        # Width connection
        mix_h = alpha.T @ h
        return mix_h, beta

    def depth_connection(self, mix_h, h_o, beta):
        h = einsum("blh,bln->blnh", h_o, beta) + mix_h[..., 1:, :]
        return h
```

**Algorithm 3: Transformer Block**
```python
# Attention Block
mix_h, beta = atten_hyper_connection.width_connection(h)
h = attn_norm(mix_h[..., 0, :])
h = self_attention(h)
h = atten_hyper_connection.depth_connection(mix_h, dropout(h), beta)

# FFN Block
mix_h, beta = ffn_hyper_connection.width_connection(h)
h = ffn_norm(mix_h[..., 0, :])
h = ffn(h)
h = ffn_hyper_connection.depth_connection(mix_h, dropout(h), beta)
```

## Important Implementation Notes

1. **Normalization placement:** Applied BEFORE width connections in dynamic version
2. **Weight decay:** Static parameters don't use weight decay, dynamic do
3. **Output scaling:** Scale output layer weights by √n to maintain std
4. **Final pooling:** Sum all n hyper hidden vectors before final projection
5. **Expansion rate:** n=4 shows best results in experiments (Table 1)

## Experimental Results Summary

- OLMo-1B-DHC×4: 0.034 lower V2 loss vs baseline (Table 1)
- OLMo-7B-DHC×4: 0.022 lower V2 loss vs baseline (Table 5)
- OLMoE-1B-7B-DHC×4: 1.8× faster convergence (Figure 1)
- Vision: ViT-Large-DHC×2: 79.94% vs 77.25% baseline (Table 11)
