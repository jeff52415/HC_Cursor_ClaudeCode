"""
Training Example for Transformer with Hyper-Connections

This script demonstrates how to train a language model using hyper-connections.
Based on the ICLR 2025 paper: "HYPER-CONNECTIONS" by Zhu et al.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from hyper_connections import TransformerWithHyperConnections, count_parameters


class SimpleTextDataset(Dataset):
    """Simple dataset for demonstration purposes"""

    def __init__(self, vocab_size=10000, num_samples=1000, seq_len=128):
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seq_len = seq_len

        # Generate random sequences
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return input and target (shifted by 1)
        return self.data[idx, :-1], self.data[idx, 1:]


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
):
    """
    Create a learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as a ratio of the initial learning rate
    """

    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def configure_optimizer(model, learning_rate, weight_decay):
    """
    Configure optimizer with proper weight decay settings.

    Following the paper's guidance:
    - Static HC parameters (B, Am, Ar): NO weight decay
    - Dynamic HC parameters (W_β, W_m, W_r): YES weight decay
    - Other parameters: standard weight decay
    """

    # Separate parameters into groups
    decay_params = []
    no_decay_params = []
    static_hc_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Static hyper-connection parameters (no weight decay)
        if 'static_alpha' in name or 'static_beta' in name:
            static_hc_params.append(param)
        # Bias and normalization parameters (no weight decay)
        elif 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        # All other parameters (with weight decay)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
        {'params': static_hc_params, 'weight_decay': 0.0},
    ]

    optimizer = optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

    return optimizer


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward pass
        logits = model(input_ids)

        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        # Update statistics
        total_loss += loss.item() * input_ids.size(0)
        total_tokens += input_ids.numel()

        # Log progress
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / total_tokens
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Avg Loss: {avg_loss:.4f}, "
                  f"LR: {current_lr:.6f}")

    avg_loss = total_loss / total_tokens
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward pass
        logits = model(input_ids)

        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1)
        )

        total_loss += loss.item() * input_ids.size(0)
        total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


def main():
    print("="*80)
    print("Training Transformer with Hyper-Connections")
    print("="*80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Hyperparameters
    config = {
        # Model architecture
        'vocab_size': 10000,
        'dim': 512,
        'num_layers': 6,
        'num_heads': 8,
        'ffn_hidden_dim': 2048,
        'max_seq_len': 512,
        'dropout': 0.1,

        # Hyper-connections settings
        'expansion_rate': 4,  # n = 4 (recommended)
        'dynamic': True,      # Use Dynamic HC
        'use_tanh': True,     # Use tanh activation

        # Other settings
        'causal': True,
        'norm_type': 'rmsnorm',
        'tie_weights': True,

        # Training settings
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'warmup_steps': 1000,
        'grad_clip': 1.0,

        # Data settings
        'seq_len': 128,
        'num_train_samples': 10000,
        'num_val_samples': 1000,
    }

    print("\nModel Configuration:")
    for key, value in config.items():
        if not key.startswith('num_') and key not in ['batch_size', 'seq_len']:
            print(f"  {key}: {value}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SimpleTextDataset(
        vocab_size=config['vocab_size'],
        num_samples=config['num_train_samples'],
        seq_len=config['seq_len']
    )

    val_dataset = SimpleTextDataset(
        vocab_size=config['vocab_size'],
        num_samples=config['num_val_samples'],
        seq_len=config['seq_len']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Batch size: {config['batch_size']}")

    # Create model
    print("\nCreating model...")
    model = TransformerWithHyperConnections(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ffn_hidden_dim=config['ffn_hidden_dim'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        expansion_rate=config['expansion_rate'],
        dynamic=config['dynamic'],
        use_tanh=config['use_tanh'],
        causal=config['causal'],
        norm_type=config['norm_type'],
        tie_weights=config['tie_weights']
    )

    model = model.to(device)

    total_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")

    # Create optimizer and scheduler
    print("\nSetting up optimizer and scheduler...")
    optimizer = configure_optimizer(
        model,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    num_training_steps = len(train_loader) * config['num_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=num_training_steps,
        min_lr_ratio=0.1
    )

    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Warmup steps: {config['warmup_steps']}")
    print(f"  Total training steps: {num_training_steps}")

    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)

    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            grad_clip=config['grad_clip']
        )

        # Evaluate
        val_loss, val_perplexity = evaluate(model, val_loader, device)

        print(f"\n  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Perplexity: {val_perplexity:.2f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  New best validation loss! Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, 'best_model.pt')

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("="*80)

    # Generation example
    print("\nGeneration Example:")
    print("-" * 80)

    model.eval()
    prompt = torch.randint(0, config['vocab_size'], (1, 20)).to(device)
    print(f"Prompt tokens: {prompt[0, :10].tolist()}")

    generated = model.generate(
        prompt,
        max_new_tokens=50,
        temperature=1.0,
        top_k=50
    )

    print(f"Generated tokens: {generated[0, 20:30].tolist()}")
    print(f"Total length: {generated.size(1)}")


if __name__ == "__main__":
    main()
