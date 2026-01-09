# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains two independent implementations of **Hyper-Connections** (ICLR 2025), a novel replacement for residual connections in deep neural networks:

- `cursor_hc/` - ❌ Implementation created with Cursor IDE (has architectural errors)
- `claude_hc/` - ✅ Implementation created with Claude Code (correct)

**IMPORTANT:** The Cursor implementation has a critical flaw - it does not correctly implement the (n+1) dimensional intermediate representation required by the paper. **Only use the Claude Code implementation** (`claude_hc/`) for research or production.

## Key Concepts

### What are Hyper-Connections?

Instead of a single hidden vector per layer, hyper-connections maintain **n parallel hidden vectors** (typically n=4) with learnable connection weights that can adjust across both:
- **Depth-connections**: Between layer inputs and outputs
- **Width-connections**: Between the n parallel vectors

This eliminates the "seesaw effect" between gradient vanishing and representation collapse that affects standard residual connections.

### Critical Parameters

- **expansion_rate (n)**: Number of parallel hidden vectors (recommended: n=4, valid: 2-8)
- **dynamic**: Use Dynamic HC (DHC) with input-dependent weights vs Static HC (SHC)
- **use_tanh**: Apply tanh activation in DHC for stability (recommended: True)

**NEVER use n=1** - this reverts to the problematic seesaw effect.

## Commands

### Setup and Installation

Both implementations use Python with PyTorch:

```bash
# Install dependencies for cursor_hc
cd cursor_hc
pip install -r requirements.txt

# Install dependencies for claude_hc
cd claude_hc
pip install -r requirements.txt

# Alternative: use uv (faster)
pip install uv
uv sync
```

### Testing

```bash
# Test cursor_hc implementation
cd cursor_hc
python test_implementation.py

# Test claude_hc implementation
cd claude_hc
python compare_architectures.py
```

### Running Examples

```bash
# cursor_hc: Basic usage example
cd cursor_hc
python example_usage.py

# claude_hc: Training example
cd claude_hc
python train_example.py

# claude_hc: Visualize architecture
cd claude_hc
python visualize_architecture.py

# claude_hc: Debug mode with tensor shapes
cd claude_hc
python debug_example.py
```

### Development Workflow

Both implementations include Jupyter notebooks for experimentation:
- `cursor_hc/run.ipynb`
- `claude_hc/run.ipynb`

## Architecture Differences

### cursor_hc Implementation

**Main files:**
- `hyper_connections.py` - Core HyperConnection module (12.8 KB)
- `transformer_hc.py` - Complete Transformer with HC (18.4 KB)
- `example_usage.py` - Usage demonstration
- `test_implementation.py` - Comprehensive test suite

**Key characteristics:**
- More verbose with extensive documentation
- Includes TransformerWithHC class for complete models
- Parameters named: `B`, `Am`, `Ar` (matching paper notation)
- Includes visualization helpers for connection matrices
- Has parameter grouping utilities for optimizer configuration

### claude_hc Implementation

**Main files:**
- `hyper_connections.py` - Core module + Transformer (23.3 KB, all-in-one)
- `train_example.py` - Training infrastructure
- `compare_architectures.py` - Validation suite
- `visualize_architecture.py` - Architecture diagrams
- `debug_example.py` - Debug mode demonstration

**Key characteristics:**
- More compact, single-file architecture
- Parameters named: `static_beta`, `static_alpha` (emphasizing static/dynamic split)
- Includes StandardTransformer for comparison
- Has built-in debug mode for tensor shape visualization
- More comprehensive training utilities (scheduler, optimizer config)

Both implementations are faithful to the paper and include all critical features.

## Critical Implementation Requirements

### Output Scaling by √n

Both implementations MUST scale output projection weights by √n to maintain stable variance:

```python
# In attention output projection and FFN second layer
if expansion_rate > 1:
    output_layer.weight.data *= 1.0 / math.sqrt(expansion_rate)
```

### Weight Decay Configuration

Static hyper-connection parameters (B, Am, Ar / static_beta, static_alpha) must NOT have weight decay:

```python
# cursor_hc approach - check parameter names
if 'B' in name or 'Am' in name or 'Ar' in name:
    no_decay_params.append(param)

# claude_hc approach - check parameter names
if 'static_alpha' in name or 'static_beta' in name:
    no_decay_params.append(param)
```

### Initialization

Both implementations initialize to match Pre-Norm residual connections:

```
HC^k = [0_{1×1}    1_{1×n}     ]
       [e_{k mod n} e_{n×n}     ]
```

Where k is the layer index and e_i is the i-th column of identity matrix.

## Critical Implementation Issue

### Cursor Implementation is Incorrect

The `cursor_hc/` implementation has a **fundamental architectural error**:

**Problem:** Uses `mean()` instead of `Am`-weighted mixing and never creates the (n+1) dimensional intermediate.

```python
# cursor_hc/transformer_hc.py (WRONG)
h_for_attn = hidden_states.mean(dim=2)  # ❌ Should use Am-weighted mix
```

**Correct approach** (claude_hc):
```python
# claude_hc/hyper_connections.py (CORRECT)
mix_h, beta = self.attn_hyper_connection.width_connection(H)  # Creates (n+1) vectors
h = self.attn_norm(mix_h[..., 0, :])  # First vector is Am-weighted mix
```

**Impact:** This breaks the algorithm's ability to learn layer arrangements and defeats the purpose of hyper-connections.

**Recommendation:** Do NOT use `cursor_hc/` implementation. Only work with `claude_hc/`.

## Working with the Claude Code Implementation

1. Read `claude_hc/CLAUDE.md` for implementation-specific details
2. Everything is in `hyper_connections.py` (TransformerWithHyperConnections)
3. Run `compare_architectures.py` to validate against standard Transformer
4. Use `configure_optimizer()` from `train_example.py` for proper weight decay
5. Enable `debug=True` in model constructor to see tensor shapes

## Common Tasks

### Adding a new feature

**Only modify the `claude_hc/` implementation**, as it is the only correct one.

1. Read the paper section relevant to your feature
2. Modify `claude_hc/hyper_connections.py`
3. Test with `compare_architectures.py`
4. Verify with `debug=True` mode

### Creating a new model variant

Use `claude_hc/` as the base implementation. It provides:
- All-in-one implementation in `hyper_connections.py`
- Debug mode for tensor inspection
- Complete training infrastructure in `train_example.py`

### Do NOT use cursor_hc

The `cursor_hc/` implementation is architecturally incorrect and should not be used as reference.

## Expected Performance

From the ICLR 2025 paper:

**Language Models:**
- 1B models: ~0.03-0.04 loss reduction
- 7B models: ~0.02 loss reduction
- MoE models: 1.8× faster convergence

**Vision Models:**
- ImageNet: +1-3% accuracy improvement
- Image generation: Comparable to 50% larger baseline

**Overhead:**
- Parameters: < 0.04% increase
- FLOPs: < 0.3% increase
- Memory: ~15-30% increase (training)

## Understanding the (n+1) Requirement

This is the most critical architectural detail:

### Why (n+1)?

The hyper-connection matrix HC is (n+1)×(n+1):
```
HC = [0      B     ]  where B: 1×n, Am: n×1, Ar: n×n
     [Am     Ar    ]
```

When applied to input `[h_layer; H]` (concatenation of layer input and n hidden vectors):
- Input is (n+1) dimensional: 1 + n
- HC is (n+1)×(n+1)
- Output is (n+1) dimensional: layer output + n residuals

### Width Connection Must Produce (n+1) Vectors

```python
# CORRECT (claude_hc)
alpha = [Am, Ar]  # Shape: (n, n+1)
mix_h = alpha^T @ H  # Shape: (n+1, D)
# mix_h[0] = Am^T @ H  (weighted mix)
# mix_h[1:] = Ar^T @ H  (residuals)
```

### Layer Operates on First Vector Only

The layer (attention/FFN) should operate on `mix_h[0]`, not on `mean(H)`:
```python
# CORRECT
h_mixed = mix_h[..., 0, :]  # First vector from width connection
h_normalized = norm(h_mixed)
output = layer(h_normalized)
```

### Depth Connection Combines Output with Residuals

```python
# CORRECT
H_new = beta * output + mix_h[..., 1:, :]  # Last n vectors are residuals
```

This architecture allows the network to learn which vectors to mix (via Am) and which to pass through (via Ar), enabling layer rearrangement.

## Important Reminders

1. **CRITICAL:** Only use `claude_hc/` implementation - `cursor_hc/` is architecturally incorrect
2. Never use n=1 (expansion_rate=1)
3. Always scale output by √n
4. Never apply weight decay to static HC parameters
5. The (n+1) intermediate representation is NOT optional - it's required by the algorithm
6. Test changes with the provided test suites before considering them complete
