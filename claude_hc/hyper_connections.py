"""
Hyper-Connections Implementation
Based on the ICLR 2025 paper: "HYPER-CONNECTIONS" by Zhu et al.

This module implements both Static Hyper-Connections (SHC) and
Dynamic Hyper-Connections (DHC) as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HyperConnection(nn.Module):
    """
    Hyper-Connection module that replaces residual connections.

    Args:
        dim: Hidden dimension (d_model)
        rate: Expansion rate (n) - number of parallel hidden vectors
        layer_id: Layer index for initialization (k)
        dynamic: Whether to use Dynamic Hyper-Connections (DHC)
        use_tanh: Whether to use tanh activation in dynamic variant
        norm_type: Type of normalization ('layernorm' or 'rmsnorm')
    """

    def __init__(
        self,
        dim: int,
        rate: int = 4,
        layer_id: int = 0,
        dynamic: bool = True,
        use_tanh: bool = True,
        norm_type: str = 'rmsnorm'
    ):
        super().__init__()

        assert rate >= 1, "Expansion rate must be at least 1"
        if rate == 1:
            print("Warning: rate=1 may not work well. Consider using rate>=2")

        self.dim = dim
        self.rate = rate
        self.layer_id = layer_id
        self.dynamic = dynamic
        self.use_tanh = use_tanh

        # Static parameters - initialize according to Eq. 14 in paper
        # B ∈ R^(1×n) - weights for layer output
        self.static_beta = nn.Parameter(torch.ones(rate))

        # Am ∈ R^(n×1) - weights for mixing inputs (column vector)
        # Ar ∈ R^(n×n) - weights for residual connections (matrix)
        init_alpha0 = torch.zeros(rate, 1)
        init_alpha0[layer_id % rate, 0] = 1.0

        # Combine Am and Ar into one matrix [Am, Ar] ∈ R^(n×(n+1))
        self.static_alpha = nn.Parameter(
            torch.cat([init_alpha0, torch.eye(rate)], dim=1)
        )

        # Dynamic parameters (if using DHC)
        if self.dynamic:
            # Normalization layer
            if norm_type == 'layernorm':
                self.layer_norm = nn.LayerNorm(dim)
            elif norm_type == 'rmsnorm':
                self.layer_norm = RMSNorm(dim)
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")

            # Dynamic weight matrices - initialized to 0
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, rate + 1))

            # Scaling factors - initialized to small values (0.01)
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)

    def get_alpha_beta(self, H):
        """
        Compute alpha and beta matrices (static or dynamic).

        Args:
            H: Hyper hidden matrix [B, L, N, D]

        Returns:
            alpha: [B, L, N, N+1] or [N, N+1]
            beta: [B, L, N] or [N]
        """
        if self.dynamic:
            # Normalize input
            norm_h = self.layer_norm(H)  # [B, L, N, D]

            # Compute dynamic alpha
            alpha_weight = torch.einsum('blnd,dk->blnk', norm_h, self.dynamic_alpha_fn)
            if self.use_tanh:
                alpha_weight = torch.tanh(alpha_weight)
            dynamic_alpha = alpha_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha + self.static_alpha.unsqueeze(0).unsqueeze(0)

            # Compute dynamic beta
            beta_weight = torch.einsum('blnd,d->bln', norm_h, self.dynamic_beta_fn)
            if self.use_tanh:
                beta_weight = torch.tanh(beta_weight)
            dynamic_beta = beta_weight * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta.unsqueeze(0).unsqueeze(0)

            return alpha, beta
        else:
            # Static case - return broadcasted tensors
            return self.static_alpha, self.static_beta

    def width_connection(self, H):
        """
        Perform width connections: compute weighted mix of hidden vectors.

        Args:
            H: Hyper hidden matrix [B, L, N, D]

        Returns:
            mix_h: Mixed hidden matrix [B, L, N+1, D]
            beta: Weights for depth connection [B, L, N] or [N]
        """
        alpha, beta = self.get_alpha_beta(H)

        # Width connection: WC^T @ H
        # alpha: [B, L, N, N+1] or [N, N+1]
        # H: [B, L, N, D]
        # mix_h: [B, L, N+1, D]

        if self.dynamic:
            # alpha: [B, L, N, N+1], H: [B, L, N, D]
            mix_h = torch.einsum('blnk,blnd->blkd', alpha, H)
        else:
            # alpha: [N, N+1], H: [B, L, N, D]
            mix_h = torch.einsum('nk,blnd->blkd', alpha, H)

        return mix_h, beta

    def depth_connection(self, mix_h, layer_output, beta):
        """
        Perform depth connections: combine layer output with residual.

        Args:
            mix_h: Mixed hidden matrix [B, L, N+1, D]
            layer_output: Output from transformer layer [B, L, D]
            beta: Weights for layer output [B, L, N] or [N]

        Returns:
            H_new: New hyper hidden matrix [B, L, N, D]
        """
        # Depth connection: B^T @ layer_output + H'
        # beta: [B, L, N] or [N]
        # layer_output: [B, L, D]
        # mix_h[..., 1:, :]: [B, L, N, D]

        if self.dynamic:
            # beta: [B, L, N], layer_output: [B, L, D]
            weighted_output = torch.einsum('bln,bld->blnd', beta, layer_output)
        else:
            # beta: [N], layer_output: [B, L, D]
            weighted_output = torch.einsum('n,bld->blnd', beta, layer_output)

        # Add residual connections from mix_h[:, :, 1:, :]
        H_new = weighted_output + mix_h[..., 1:, :]

        return H_new


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS normalization
        norm = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return self.weight * x / (norm + self.eps)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with optional causal masking"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal

        # Q, K, V projections
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, L, 3*D]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, H, L, L]

        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float('-inf'))

        # Apply additional mask if provided
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, H, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, D)  # [B, L, D]

        # Output projection
        out = self.out_proj(out)

        return out


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim

        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with Hyper-Connections.
    Replaces residual connections with hyper-connection modules.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_hidden_dim: int = None,
        dropout: float = 0.1,
        layer_id: int = 0,
        expansion_rate: int = 4,
        dynamic: bool = True,
        use_tanh: bool = True,
        causal: bool = True,
        norm_type: str = 'rmsnorm',
        debug: bool = False
    ):
        super().__init__()

        self.dim = dim
        self.layer_id = layer_id
        self.expansion_rate = expansion_rate
        self.debug = debug

        # Normalization layers
        if norm_type == 'layernorm':
            self.attn_norm = nn.LayerNorm(dim)
            self.ffn_norm = nn.LayerNorm(dim)
        elif norm_type == 'rmsnorm':
            self.attn_norm = RMSNorm(dim)
            self.ffn_norm = RMSNorm(dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # Attention and FFN modules
        self.attention = MultiHeadAttention(dim, num_heads, dropout, causal)
        self.ffn = FeedForward(dim, ffn_hidden_dim, dropout)

        # Hyper-connection modules for attention and FFN
        self.attn_hyper_connection = HyperConnection(
            dim=dim,
            rate=expansion_rate,
            layer_id=layer_id * 2,  # Attention layer
            dynamic=dynamic,
            use_tanh=use_tanh,
            norm_type=norm_type
        )

        self.ffn_hyper_connection = HyperConnection(
            dim=dim,
            rate=expansion_rate,
            layer_id=layer_id * 2 + 1,  # FFN layer
            dynamic=dynamic,
            use_tanh=use_tanh,
            norm_type=norm_type
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, H, mask=None):
        """
        Args:
            H: Hyper hidden matrix [B, L, N, D]
            mask: Optional attention mask [B, L]

        Returns:
            H: Updated hyper hidden matrix [B, L, N, D]
        """
        if self.debug:
            print(f"\n{'='*60}")
            print(f"Layer {self.layer_id} - ATTENTION BLOCK")
            print(f"{'='*60}")
            print(f"Input H shape: {H.shape}")

        # Attention block with hyper-connections
        mix_h, beta = self.attn_hyper_connection.width_connection(H)
        if self.debug:
            print(f"After width_connection:")
            print(f"  mix_h shape: {mix_h.shape}")
            print(f"  beta shape: {beta.shape if isinstance(beta, torch.Tensor) else 'static'}")

        h = self.attn_norm(mix_h[..., 0, :])  # Take first mixed vector
        if self.debug:
            print(f"After attn_norm (mix_h[..., 0, :]):")
            print(f"  h shape: {h.shape}")

        h = self.attention(h, mask)
        if self.debug:
            print(f"After attention:")
            print(f"  h shape: {h.shape}")

        h = self.dropout(h)
        if self.debug:
            print(f"After dropout:")
            print(f"  h shape: {h.shape}")

        H = self.attn_hyper_connection.depth_connection(mix_h, h, beta)
        if self.debug:
            print(f"After depth_connection:")
            print(f"  H shape: {H.shape}")

        if self.debug:
            print(f"\n{'='*60}")
            print(f"Layer {self.layer_id} - FFN BLOCK")
            print(f"{'='*60}")
            print(f"Input H shape: {H.shape}")

        # FFN block with hyper-connections
        mix_h, beta = self.ffn_hyper_connection.width_connection(H)
        if self.debug:
            print(f"After width_connection:")
            print(f"  mix_h shape: {mix_h.shape}")
            print(f"  beta shape: {beta.shape if isinstance(beta, torch.Tensor) else 'static'}")

        h = self.ffn_norm(mix_h[..., 0, :])  # Take first mixed vector
        if self.debug:
            print(f"After ffn_norm (mix_h[..., 0, :]):")
            print(f"  h shape: {h.shape}")

        h = self.ffn(h)
        if self.debug:
            print(f"After ffn:")
            print(f"  h shape: {h.shape}")

        h = self.dropout(h)
        if self.debug:
            print(f"After dropout:")
            print(f"  h shape: {h.shape}")

        H = self.ffn_hyper_connection.depth_connection(mix_h, h, beta)
        if self.debug:
            print(f"After depth_connection:")
            print(f"  H shape: {H.shape}")
            print(f"{'='*60}\n")

        return H


class TransformerWithHyperConnections(nn.Module):
    """
    Complete Transformer model with Hyper-Connections.

    This model can be used for language modeling or other sequence tasks.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_hidden_dim: int = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        expansion_rate: int = 4,
        dynamic: bool = True,
        use_tanh: bool = True,
        causal: bool = True,
        norm_type: str = 'rmsnorm',
        tie_weights: bool = True,
        debug: bool = False
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.expansion_rate = expansion_rate
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights
        self.debug = debug

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                layer_id=i,
                expansion_rate=expansion_rate,
                dynamic=dynamic,
                use_tanh=use_tanh,
                causal=causal,
                norm_type=norm_type,
                debug=debug
            )
            for i in range(num_layers)
        ])

        # Final normalization
        if norm_type == 'layernorm':
            self.final_norm = nn.LayerNorm(dim)
        elif norm_type == 'rmsnorm':
            self.final_norm = RMSNorm(dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # Output projection (unembedding)
        self.output_proj = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights if specified
        if tie_weights:
            self.output_proj.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Scale output layer weights by sqrt(expansion_rate) as per paper
        self._scale_output_weights()

    def _init_weights(self, module):
        """Initialize weights following standard practices"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'weight'):
                torch.nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _scale_output_weights(self):
        """
        Scale output layer weights by sqrt(expansion_rate).
        This is critical for hyper-connections (see paper Section 4 and Appendix B).
        """
        scale = math.sqrt(self.expansion_rate)

        # Scale output projection of attention and FFN in each block
        for block in self.blocks:
            block.attention.out_proj.weight.data /= scale
            block.ffn.fc2.weight.data /= scale

        # Note: final output_proj is NOT scaled if weights are tied
        # because token_embedding is used for input

    def forward(self, input_ids, mask=None):
        """
        Args:
            input_ids: Input token IDs [B, L]
            mask: Optional attention mask [B, L]

        Returns:
            logits: Output logits [B, L, vocab_size]
        """
        B, L = input_ids.shape

        if self.debug:
            print(f"\n{'#'*60}")
            print(f"TRANSFORMER FORWARD PASS")
            print(f"{'#'*60}")
            print(f"Input shape: {input_ids.shape}")

        # Token and position embeddings
        token_emb = self.token_embedding(input_ids)  # [B, L, D]

        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)  # [1, L]
        pos_emb = self.pos_embedding(positions)  # [1, L, D]

        # Initial hidden state
        h = token_emb + pos_emb  # [B, L, D]

        if self.debug:
            print(f"After embeddings:")
            print(f"  h shape: {h.shape}")

        # Replicate to create initial hyper hidden matrix
        # H0 = [h, h, ..., h]^T ∈ R^(n×d)
        H = h.unsqueeze(2).repeat(1, 1, self.expansion_rate, 1)  # [B, L, N, D]

        if self.debug:
            print(f"Initial hyper hidden matrix H0:")
            print(f"  H shape: {H.shape}")

        # Apply transformer blocks
        for block in self.blocks:
            H = block(H, mask)

        if self.debug:
            print(f"\n{'='*60}")
            print(f"FINAL PROCESSING")
            print(f"{'='*60}")
            print(f"After all blocks, H shape: {H.shape}")

        # Sum across hidden vectors (row-wise sum)
        h_final = H.sum(dim=2)  # [B, L, D]

        if self.debug:
            print(f"After sum(dim=2):")
            print(f"  h_final shape: {h_final.shape}")

        # Final normalization
        h_final = self.final_norm(h_final)

        if self.debug:
            print(f"After final_norm:")
            print(f"  h_final shape: {h_final.shape}")

        # Output projection
        logits = self.output_proj(h_final)  # [B, L, vocab_size]

        if self.debug:
            print(f"After output_proj:")
            print(f"  logits shape: {logits.shape}")
            print(f"{'#'*60}\n")

        return logits

    def generate(
        self,
        input_ids,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ):
        """
        Generate text autoregressively.

        Args:
            input_ids: Initial token IDs [B, L]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            generated: Generated token IDs [B, L + max_new_tokens]
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for next token
                logits = self.forward(input_ids)
                logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Truncate if exceeds max sequence length
                if input_ids.size(1) > self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len:]

        return input_ids


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage
    print("="*80)
    print("Transformer with Hyper-Connections - Example Usage")
    print("="*80)

    # Model configuration
    config = {
        'vocab_size': 10000,
        'dim': 512,
        'num_layers': 6,
        'num_heads': 8,
        'ffn_hidden_dim': 2048,
        'max_seq_len': 1024,
        'dropout': 0.1,
        'expansion_rate': 4,
        'dynamic': True,
        'use_tanh': True,
        'causal': True,
        'norm_type': 'rmsnorm',
        'tie_weights': True
    }

    # Create model
    model = TransformerWithHyperConnections(**config)

    print(f"\nModel Configuration:")
    print(f"  Vocab Size: {config['vocab_size']}")
    print(f"  Hidden Dim: {config['dim']}")
    print(f"  Num Layers: {config['num_layers']}")
    print(f"  Num Heads: {config['num_heads']}")
    print(f"  Expansion Rate: {config['expansion_rate']}")
    print(f"  Dynamic HC: {config['dynamic']}")
    print(f"  Use Tanh: {config['use_tanh']}")

    # Count parameters
    total_params = count_parameters(model)
    print(f"\nTotal Parameters: {total_params:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))

    print(f"\nTest Forward Pass:")
    print(f"  Input shape: {input_ids.shape}")

    logits = model(input_ids)
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {config['vocab_size']})")

    # Test generation
    print(f"\nTest Generation:")
    prompt = torch.randint(0, config['vocab_size'], (1, 10))
    print(f"  Prompt shape: {prompt.shape}")

    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=50)
    print(f"  Generated shape: {generated.shape}")
    print(f"  Expected: (1, 30)")

    print("\n" + "="*80)
    print("Implementation Complete!")
    print("="*80)
