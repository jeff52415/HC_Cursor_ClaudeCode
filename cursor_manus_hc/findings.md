# Findings: Hyper-Connections Architecture

## Key Architecture Components (from HYPER-CONNECTIONS.pdf)

### 1. Core Concept
- Alternative to residual connections that addresses gradient vanishing and representation collapse
- Allows network to autonomously learn optimal connection strengths
- Consists of two types: **depth-connections** and **width-connections**

### 2. Parameters
- **α (alpha)**: Width-connection weights - control lateral information exchange between hidden vectors
- **β (beta)**: Depth-connection weights - control integration of layer output with hidden states
- **n**: Expansion rate - number of intermediate hidden states (typically 2, 4, or 8)

### 3. Variants
- **DHC (Dynamic Hyper-Connections)**: α and β are learnable scalars or predicted by network
- **SHC (Static Hyper-Connections)**: Fixed connection weights
- Paper shows DHC performs best, especially DHC×4 and DHC×8

### 4. Architecture Details (Figure 2)
For n=2 expansion rate:
- Takes input hidden state
- Creates multiple hidden vectors (h_1, h_2, etc.)
- Width-connections: Exchange info between h_i states using α weights
- Depth-connections: Weighted sum of layer output with h_i using β weights
- Output combines all information

### 5. Formulas (need to extract exact math)
From paper description:
- Width-connections allow information exchange between hidden vectors at same depth
- Depth-connections perform weighted sum between layer output and hidden vectors
- Uses tanh activation in DHC (DHC W/O tanh is ablation)

### 6. Performance Results
- OLMo-1B-DHC×4 shows significant improvements over baseline
- Converges 1.8x faster
- Lower perplexity across multiple benchmarks
- Best results with n=4 or n=8

## Implementation Requirements
1. Hyper-connection block with configurable expansion rate n
2. Learnable α and β parameters
3. Width-connections between hidden states
4. Depth-connections integrating layer output
5. Integration with standard Transformer architecture
6. Support for both DHC and SHC variants
