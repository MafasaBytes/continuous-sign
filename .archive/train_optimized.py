"""
Optimized training script for MobileNetV3-based sign language recognition.
Implements fixes for gradient vanishing and CTC blank collapse.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
import argparse
from datetime import datetime
import logging
from typing import Dict, Optional, Tuple
import random

from src.models.mobilenet_v3_fixed import create_mobilenet_v3_fixed_model
from src.data.mediapipe_dataset import MediaPipeFeatureDataset, collate_fn, build_vocabulary, Vocabulary
from src.utils.metrics import compute_wer, compute_ser


def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    log_file = output_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def monitor_gradients(model: nn.Module, logger: logging.Logger, step: int):
    """Monitor gradient flow through the model."""
    grad_stats = {}
    vanishing_layers = []
    exploding_layers = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()

            grad_stats[name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std
            }

            # Check for vanishing gradients
            if grad_norm < 1e-7:
                vanishing_layers.append(name)
            # Check for exploding gradients
            elif grad_norm > 100:
                exploding_layers.append(name)

    # Log warnings
    if vanishing_layers and step % 10 == 0:  # Log every 10 steps
        logger.warning(f"Step {step} - Vanishing gradients in: {', '.join(vanishing_layers[:5])}")
    if exploding_layers and step % 10 == 0:
        logger.warning(f"Step {step} - Exploding gradients in: {', '.join(exploding_layers[:5])}")

    return grad_stats


def check_output_distribution(log_probs: torch.Tensor, vocab_size: int, blank_id: int) -> Dict:
    """Check if output is uniform (indicating no learning)."""
    # Get probabilities for first sample
    probs = log_probs[:, 0, :].exp()  # [T, vocab_size]

    # Check entropy (uniform distribution has max entropy)
    entropy = -(probs * log_probs[:, 0, :]).sum(dim=-1).mean().item()
    max_entropy = np.log(vocab_size)

    # Check blank probability
    blank_prob = probs[:, blank_id].mean().item()

    # Check if distribution is nearly uniform
    prob_std = probs.std().item()
    is_uniform = prob_std < 0.01  # Very low std indicates uniform

    return {
        'entropy': entropy,
        'max_entropy': max_entropy,
        'entropy_ratio': entropy / max_entropy,
        'blank_prob': blank_prob,
        'is_uniform': is_uniform,
        'prob_std': prob_std
    }


def create_dataloaders(
    data_dir: Path,
    vocab: Vocabulary,
    batch_size: int = 4,
    num_workers: int = 0,
    num_train_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""

    # Training dataset
    train_dataset = MediaPipeFeatureDataset(
        data_dir=data_dir,
        annotation_file=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv"),
        vocabulary=vocab,
        split='train',
        augment=True
    )

    # Sample training data if requested
    if num_train_samples is not None and num_train_samples < len(train_dataset):
        from torch.utils.data import Subset
        logging.info(f"Sampling {num_train_samples} training samples from {len(train_dataset)}")
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        sampled_indices = indices[:num_train_samples]
        train_dataset = Subset(train_dataset, sampled_indices)

    # Validation dataset
    val_dataset = MediaPipeFeatureDataset(
        data_dir=data_dir,
        annotation_file=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/dev.SI5.corpus.csv"),
        vocabulary=vocab,
        split='dev',
        augment=False
    )

    # Test dataset
    test_dataset = MediaPipeFeatureDataset(
        data_dir=data_dir,
        annotation_file=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/test.SI5.corpus.csv"),
        vocabulary=vocab,
        split='test',
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def simple_greedy_decode(log_probs: torch.Tensor, input_lengths: torch.Tensor, blank_id: int = 0):
    """Simple greedy decoding without beam search."""
    batch_size = log_probs.size(1)
    predictions = []

    for b in range(batch_size):
        seq_len = input_lengths[b].item()
        probs = log_probs[:seq_len, b, :]  # [T, vocab_size]

        # Greedy decoding
        predicted = torch.argmax(probs, dim=-1).cpu().numpy()

        # Remove consecutive duplicates and blanks
        decoded = []
        prev = -1
        for p in predicted:
            if p != prev and p != blank_id:
                decoded.append(p)
            prev = p

        predictions.append(decoded)

    return predictions


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    max_grad_norm: float,
    accumulation_steps: int,
    vocab: Vocabulary,
    logger: logging.Logger,
    monitor_interval: int = 50
) -> Dict:
    """Train for one epoch with gradient monitoring."""
    model.train()
    total_loss = 0
    num_batches = 0
    global_step = epoch * len(dataloader)

    # Track gradient issues
    vanishing_count = 0
    uniform_output_count = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        # Forward pass with mixed precision
        with autocast():
            log_probs = model(features, input_lengths)

            # Check output distribution
            if batch_idx % monitor_interval == 0:
                dist_stats = check_output_distribution(
                    log_probs.detach(),
                    vocab_size=len(vocab),
                    blank_id=vocab.blank_id
                )
                if dist_stats['is_uniform']:
                    uniform_output_count += 1
                    logger.warning(f"Step {global_step}: Uniform output detected! "
                                 f"Entropy ratio: {dist_stats['entropy_ratio']:.3f}, "
                                 f"Blank prob: {dist_stats['blank_prob']:.3f}")

            # CTC loss
            loss = criterion(log_probs, labels, input_lengths, target_lengths)

        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Invalid loss at step {global_step}: {loss.item()}")
            continue

        # Normalize loss by accumulation steps
        loss = loss / accumulation_steps
        total_loss += loss.item()

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Monitor gradients
            if batch_idx % monitor_interval == 0:
                grad_stats = monitor_gradients(model, logger, global_step)

                # Count vanishing gradient occurrences
                for name, stats in grad_stats.items():
                    if stats['norm'] < 1e-7:
                        vanishing_count += 1

            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item() * accumulation_steps,
                'grad_norm': grad_norm.item(),
                'lr': optimizer.param_groups[0]['lr']
            })

        num_batches += 1
        global_step += 1

    # Log epoch statistics
    if uniform_output_count > 0:
        logger.warning(f"Epoch {epoch}: {uniform_output_count} batches with uniform outputs")
    if vanishing_count > 0:
        logger.warning(f"Epoch {epoch}: {vanishing_count} instances of vanishing gradients")

    return {
        'train_loss': total_loss / num_batches if num_batches > 0 else 0,
        'uniform_outputs': uniform_output_count,
        'vanishing_gradients': vanishing_count
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.CTCLoss,
    vocab: Vocabulary,
    device: torch.device
) -> Dict:
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            target_lengths = batch['target_lengths'].to(device)

            # Forward pass
            with autocast():
                log_probs = model(features, input_lengths)
                loss = criterion(log_probs, labels, input_lengths, target_lengths)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1

            # Decode predictions
            predictions = simple_greedy_decode(
                log_probs.cpu(),
                input_lengths.cpu(),
                blank_id=vocab.blank_id
            )

            # Convert to words
            for pred in predictions:
                pred_words = vocab.indices_to_words(pred)
                all_predictions.append(pred_words)

            # Convert targets to words
            labels_cpu = labels.cpu().numpy()
            target_lengths_cpu = target_lengths.cpu().numpy()
            start_idx = 0
            for length in target_lengths_cpu:
                target = labels_cpu[start_idx:start_idx+length]
                target_words = vocab.indices_to_words(target)
                all_targets.append(target_words)
                start_idx += length

    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    wer = compute_wer(all_targets, all_predictions)
    ser = compute_ser(all_targets, all_predictions)

    return {
        'val_loss': avg_loss,
        'wer': wer,
        'ser': ser
    }


def main(args):
    """Main training function."""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir) / f"mobilenet_v3_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Starting optimized training with args: {args}")
    logger.info(f"Device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Build vocabulary
    logger.info("Building vocabulary...")
    vocab = build_vocabulary(
        Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    )
    logger.info(f"Vocabulary size: {len(vocab)}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        vocab=vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_samples=args.num_train_samples
    )
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Create model
    logger.info("Creating optimized MobileNetV3 model...")
    model = create_mobilenet_v3_fixed_model(
        vocab_size=len(vocab),
        dropout=args.dropout
    )
    model = model.to(device)

    # Loss function - standard CTC
    criterion = nn.CTCLoss(blank=vocab.blank_id, reduction='mean', zero_infinity=True)

    # Optimizer with higher learning rate
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler - more aggressive
    steps_per_epoch = max(1, len(train_loader) // args.accumulation_steps)  # Ensure at least 1
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate * 10,  # Peak at 10x base LR
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # TensorBoard writer
    writer = SummaryWriter(output_dir / 'tensorboard')

    # Training configuration
    config = {
        'model': 'MobileNetV3SignLanguageFixed',
        'vocab_size': len(vocab),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        'max_grad_norm': args.max_grad_norm,
        'epochs': args.epochs,
        'seed': args.seed,
        'model_params': model.count_parameters()
    }

    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Configuration saved to {output_dir / 'config.json'}")

    # Training loop
    best_wer = float('inf')
    patience = args.early_stopping_patience
    patience_counter = 0
    no_improvement_epochs = 0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args.max_grad_norm,
            args.accumulation_steps, vocab, logger
        )

        # Step scheduler
        scheduler.step()

        # Validate
        val_metrics = validate(model, val_loader, criterion, vocab, device)

        # Log metrics
        logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"WER: {val_metrics['wer']:.2f}%")
        logger.info(f"SER: {val_metrics['ser']:.2f}%")
        logger.info(f"Uniform outputs: {train_metrics['uniform_outputs']}")
        logger.info(f"Vanishing gradients: {train_metrics['vanishing_gradients']}")

        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_metrics['train_loss'], epoch)
        writer.add_scalar('Loss/Val', val_metrics['val_loss'], epoch)
        writer.add_scalar('Metrics/WER', val_metrics['wer'], epoch)
        writer.add_scalar('Metrics/SER', val_metrics['ser'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Issues/UniformOutputs', train_metrics['uniform_outputs'], epoch)
        writer.add_scalar('Issues/VanishingGradients', train_metrics['vanishing_gradients'], epoch)

        # Save best model
        if val_metrics['wer'] < best_wer:
            best_wer = val_metrics['wer']
            patience_counter = 0
            no_improvement_epochs = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_wer': best_wer,
                'config': config
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            logger.info(f"New best model saved with WER: {best_wer:.2f}%")
        else:
            patience_counter += 1
            no_improvement_epochs += 1

            # Check if model is stuck
            if no_improvement_epochs >= 5 and val_metrics['wer'] > 90:
                logger.warning("Model appears stuck at high WER. Consider:")
                logger.warning("1. Increasing learning rate further")
                logger.warning("2. Reducing model complexity")
                logger.warning("3. Checking data loading")

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_wer': val_metrics['wer'],
                'config': config
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pth')

    # Final test evaluation
    logger.info(f"\n{'='*50}")
    logger.info("Final Test Evaluation")
    logger.info(f"{'='*50}")

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = validate(model, test_loader, criterion, vocab, device)
    logger.info(f"Test Loss: {test_metrics['val_loss']:.4f}")
    logger.info(f"Test WER: {test_metrics['wer']:.2f}%")
    logger.info(f"Test SER: {test_metrics['ser']:.2f}%")

    # Save final results
    results = {
        'best_val_wer': best_wer,
        'test_wer': test_metrics['wer'],
        'test_ser': test_metrics['ser'],
        'test_loss': test_metrics['val_loss'],
        'total_params': model.count_parameters(),
        'model_size_mb': model.count_parameters() * 4 / 1024 / 1024
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nTraining complete! Results saved to {output_dir}")
    logger.info(f"Model size: {results['model_size_mb']:.2f} MB")
    logger.info(f"Best WER: {best_wer:.2f}% (Target: < 25%)")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized training for Sign Language Recognition')

    # Data arguments
    parser.add_argument('--data_dir', type=str,
                        default='data/teacher_features/mediapipe_full',
                        help='Path to features directory')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/optimized',
                        help='Output directory for checkpoints')
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help='Number of training samples to use (default: all)')

    # Model arguments - UPDATED DEFAULTS
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='Dropout rate (default: 0.05 for better gradient flow)')

    # Training arguments - UPDATED DEFAULTS
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4, higher for breaking plateau)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5, reduced for less regularization)')
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                        help='Maximum gradient norm (default: 5.0, higher for more aggressive updates)')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Early stopping patience')

    args = parser.parse_args()
    main(args)