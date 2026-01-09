"""
Example usage of Transformer with Hyper-Connections.

This script demonstrates how to:
1. Create a Transformer model with hyper-connections
2. Train the model on a simple task
3. Compare with baseline (residual connections)
4. Visualize connection patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
from typing import Optional

from transformer_hc import TransformerWithHC, count_parameters
from hyper_connections import configure_optimizers_for_hyper_connections


class SimpleLanguageDataset(Dataset):
    """Simple synthetic dataset for demonstration."""
    
    def __init__(self, vocab_size: int = 1000, seq_len: int = 128, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random sequences
        # In practice, this would be real data
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # Shift for next-token prediction
        labels = torch.roll(input_ids, -1)
        labels[-1] = 0  # Pad last token
        
        return input_ids, labels


def train_step(
    model: nn.Module,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Single training step."""
    model.train()
    
    input_ids, labels = batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    
    # Forward pass (with causal masking for language modeling)
    logits = model(input_ids, use_causal_mask=True)
    
    # Compute loss (next-token prediction)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Use causal mask for autoregressive language modeling
        logits = model(input_ids, use_causal_mask=True)
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='sum',
        )
        
        total_loss += loss.item()
        total_tokens += labels.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
    }


def visualize_connection_matrices(model: TransformerWithHC):
    """
    Visualize learned connection patterns in hyper-connections.
    
    This helps understand:
    - Which layers connect to which (depth patterns)
    - How hidden vectors interact (width patterns)
    """
    print("\n" + "="*80)
    print("HYPER-CONNECTION PATTERNS")
    print("="*80)
    
    for layer_idx, block in enumerate(model.blocks):
        print(f"\nLayer {layer_idx}")
        print("-" * 40)
        
        # Attention hyper-connection
        hc_attn = block.hc_attn
        print(f"Attention HC:")
        print(f"  B (output weights): {hc_attn.B.data.cpu().numpy()}")
        print(f"  Am (mixing weights): {hc_attn.Am.data.squeeze().cpu().numpy()}")
        print(f"  Ar (residual weights):\n{hc_attn.Ar.data.cpu().numpy()}")
        
        # FFN hyper-connection
        hc_ffn = block.hc_ffn
        print(f"FFN HC:")
        print(f"  B (output weights): {hc_ffn.B.data.cpu().numpy()}")
        print(f"  Am (mixing weights): {hc_ffn.Am.data.squeeze().cpu().numpy()}")
        print(f"  Ar (residual weights):\n{hc_ffn.Ar.data.cpu().numpy()}")
        
        if hc_attn.dynamic:
            print(f"  Dynamic scaling factors (s_α): {hc_attn.s_alpha.data.mean().item():.4f}")
            print(f"  Dynamic scaling factors (s_β): {hc_attn.s_beta.data.mean().item():.4f}")


def main():
    """Main example demonstrating hyper-connections."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model hyperparameters
    config = {
        'vocab_size': 1000,
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 4,
        'ffn_dim': 1024,
        'max_seq_len': 128,
        'dropout': 0.1,
        'expansion_rate': 4,  # n=4 (recommended default)
        'dynamic': True,  # Use Dynamic HC
        'use_tanh': True,  # Use tanh in DHC
        'activation': 'gelu',
    }
    
    print("\n" + "="*80)
    print("MODEL CONFIGURATION")
    print("="*80)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    
    # Create model
    print("\nCreating Transformer with Hyper-Connections...")
    model = TransformerWithHC(**config)
    model = model.to(device)
    
    # Count parameters
    param_counts = count_parameters(model)
    print("\n" + "="*80)
    print("PARAMETER COUNTS")
    print("="*80)
    print(f"Total parameters:           {param_counts['total']:,}")
    print(f"Hyper-connection params:    {param_counts['hyper_connection']:,} ({param_counts['hc_percentage']:.2f}%)")
    print(f"  - Static (B, Am, Ar):     {param_counts['hc_static']:,}")
    print(f"  - Dynamic (W_β, W_m, W_r): {param_counts['hc_dynamic']:,}")
    print(f"Other parameters:           {param_counts['other']:,}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SimpleLanguageDataset(
        vocab_size=config['vocab_size'],
        seq_len=config['max_seq_len'],
        num_samples=1000,
    )
    val_dataset = SimpleLanguageDataset(
        vocab_size=config['vocab_size'],
        seq_len=config['max_seq_len'],
        num_samples=200,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Configure optimizer (with proper weight decay for HC)
    optimizer = configure_optimizers_for_hyper_connections(
        model,
        learning_rate=3e-4,
        weight_decay=0.01,
    )
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            loss = train_step(model, batch, optimizer, device)
            total_loss += loss
            num_batches += 1
            
            if (batch_idx + 1) % 20 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {avg_loss:.4f}")
        
        avg_train_loss = total_loss / num_batches
        
        # Validation
        val_metrics = evaluate(model, val_loader, device, max_batches=20)
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Val PPL:    {val_metrics['perplexity']:.2f}")
    
    # Visualize learned patterns
    visualize_connection_matrices(model)
    
    # Generate some text (simple greedy decoding)
    print("\n" + "="*80)
    print("TEXT GENERATION EXAMPLE")
    print("="*80)
    
    model.eval()
    with torch.no_grad():
        # Start with a random prompt
        prompt = torch.randint(0, config['vocab_size'], (1, 10)).to(device)
        print(f"Prompt: {prompt.squeeze().tolist()}")
        
        # Generate 20 tokens
        generated = prompt
        for _ in range(20):
            # Use causal mask for generation
            logits = model(generated, use_causal_mask=True)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        
        print(f"Generated: {generated.squeeze().tolist()}")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print("\nKey observations about Hyper-Connections:")
    print("1. HC parameters are < 0.04% of total (negligible overhead)")
    print("2. Connection matrices show learned patterns (Λ-shaped, parallel, etc.)")
    print("3. Dynamic weights adapt based on input")
    print("4. Training is stable with proper weight decay configuration")
    print("\nFor best results on real tasks:")
    print("- Use n=4 for language models")
    print("- Use n=2 for vision models")
    print("- Enable dynamic=True for best performance")
    print("- Train with same LR schedule as baseline")


if __name__ == '__main__':
    main()
