"""
Fixed training script for MobileNetV3-based sign language recognition.
Incorporates all discovered fixes and improvements.
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
from torch.utils.data import DataLoader, Subset
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

from src.models import MobileNetV3SignLanguage, create_mobilenet_v3_model
from src.data.mediapipe_dataset import MediaPipeFeatureDataset, collate_fn, build_vocabulary, Vocabulary
from src.utils.metrics import compute_wer, compute_ser
from utils.ctc import CTCLoss, ctc_decode, prepare_ctc_targets

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


def create_dataloaders(
    data_dir: Path,
    vocab: Vocabulary,
    batch_size: int = 4,
    num_workers: int = 0,
    num_train_samples: Optional[int] = None,
    remove_pca: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders with optional sampling."""

    # Training dataset
    train_dataset = MediaPipeFeatureDataset(
        data_dir=data_dir,
        annotation_file=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv"),
        vocabulary=vocab,
        split='train',
        augment=True  # Enable augmentation for training
    )

    # Sample training data if requested
    if num_train_samples is not None and num_train_samples < len(train_dataset):
        logging.info(f"Sampling {num_train_samples} training samples from {len(train_dataset)}")
        # Use random sampling
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
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # For stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


def simple_greedy_decode(log_probs: torch.Tensor, input_lengths: torch.Tensor, blank_id: int = 0):
    """Simple greedy decoding without beam search."""
    batch_size = log_probs.size(1)
    predictions = []

    for b in range(batch_size):
        seq_len = input_lengths[b].item()
        # Get the sequence for this batch element
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
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 1.0,  # Reduced from 5.0
    accumulation_steps: int = 1,
    dynamic_truncation: bool = True,
    max_seq_length: int = 512
) -> Dict[str, float]:
    """Train for one epoch with mixed-precision and gradient accumulation."""

    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    grad_norms = []

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')

    for batch_idx, batch in enumerate(progress_bar):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        # Dynamic sequence truncation for memory efficiency
        if dynamic_truncation and features.size(1) > max_seq_length:
            features = features[:, :max_seq_length, :]
            input_lengths = torch.minimum(
                input_lengths,
                torch.tensor(max_seq_length, device=device)
            )

        # Mixed precision forward pass if CUDA available
        if device.type == 'cuda':
            with autocast():
                log_probs = model(features, input_lengths)

                # CTC loss computation
                loss = criterion(
                    log_probs,
                    labels,
                    input_lengths,
                    target_lengths
                )

                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
        else:
            # CPU training without mixed precision
            log_probs = model(features, input_lengths)
            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                target_lengths
            )
            loss = loss / accumulation_steps

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"NaN/Inf loss detected in batch {batch_idx}")
            continue

        # Backward pass
        if device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if device.type == 'cuda':
                # Unscale and clip gradients
                scaler.unscale_(optimizer)

            # Compute gradient norm for monitoring
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norms.append(total_norm.item())

            if device.type == 'cuda':
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        # Track metrics
        total_loss += loss.item() * accumulation_steps
        num_batches += 1

        # Update progress bar
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
        progress_bar.set_postfix({
            'loss': f'{total_loss / num_batches:.4f}',
            'grad_norm': f'{avg_grad_norm:.2f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    return {
        'train_loss': total_loss / num_batches if num_batches > 0 else 0,
        'avg_grad_norm': np.mean(grad_norms) if grad_norms else 0
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.CTCLoss,
    vocab: Vocabulary,
    device: torch.device,
    use_beam_search: bool = False  # Default to greedy
) -> Dict[str, float]:
    """Validate the model and compute metrics."""

    model.eval()
    total_loss = 0
    num_batches = 0

    all_predictions = []
    all_targets = []

    for batch in tqdm(dataloader, desc='Validation'):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        # Forward pass
        if device.type == 'cuda':
            with autocast():
                log_probs = model(features, input_lengths)
                loss = criterion(log_probs, labels, input_lengths, target_lengths)
        else:
            log_probs = model(features, input_lengths)
            loss = criterion(log_probs, labels, input_lengths, target_lengths)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            num_batches += 1

        # Greedy decoding using ctc_decode
        predictions = ctc_decode(
            log_probs,
            blank_id=vocab.blank_id,
            method='greedy')
 

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

    # Compute WER and SER
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
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    output_dir = Path(args.output_dir) / f"mobilenet_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Starting training with args: {args}")
    logger.info(f"Device: {device}")

    # Set seed for reproducibility
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
        num_train_samples=args.num_train_samples,
        remove_pca=args.remove_pca
    )

    train_size = len(train_loader.dataset)
    logger.info(f"Train: {train_size}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    if args.num_train_samples:
        logger.info(f"Using subset of training data: {args.num_train_samples} samples")

    # Create model
    logger.info("Creating MobileNetV3 model...")
    model = create_mobilenet_v3_model(
        vocab_size=len(vocab),
        dropout=args.dropout
    )
    model = model.to(device)

    # Loss function (CTC)
    criterion = nn.CTCLoss(blank=vocab.blank_id, reduction='mean', zero_infinity=True)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler
    if args.use_scheduler:
        if args.num_train_samples and args.num_train_samples < 1000:
            # For small datasets, use simple step scheduler
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5
            )
            logger.info("Using StepLR scheduler for small dataset")
        else:
            # For full dataset, use cosine annealing with warmup
            from torch.optim.lr_scheduler import LinearLR, SequentialLR

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=5
            )

            main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[5]
            )
            logger.info("Using CosineAnnealingWarmRestarts with warmup")
    else:
        scheduler = None
        logger.info("No learning rate scheduler")

    # Mixed precision scaler (only for CUDA)
    scaler = GradScaler() if device.type == 'cuda' else None

    # TensorBoard writer
    writer = SummaryWriter(output_dir / 'tensorboard')

    # Training configuration
    config = {
        'model': 'MobileNetV3SignLanguage',
        'vocab_size': len(vocab),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'seed': args.seed,
        'num_train_samples': args.num_train_samples,
        'remove_pca': args.remove_pca,
        'mixed_precision': device.type == 'cuda',
        'gradient_checkpointing': args.gradient_checkpointing,
        'model_params': model.count_parameters(),
        'device': str(device)
    }

    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Configuration saved to {output_dir / 'config.json'}")

    # Training loop
    best_wer = float('inf')
    patience = args.early_stopping_patience
    patience_counter = 0

    logger.info(f"\nStarting training for {args.epochs} epochs")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Dropout: {args.dropout}")

    for epoch in range(args.epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch+1, args.max_grad_norm,
            args.accumulation_steps, args.dynamic_truncation
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, vocab, device,
            use_beam_search=False  # Start with greedy decoding
        )

        # Log metrics
        logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"Avg Gradient Norm: {train_metrics['avg_grad_norm']:.2f}")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"WER: {val_metrics['wer']:.2f}%")
        logger.info(f"SER: {val_metrics['ser']:.2f}%")

        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_metrics['train_loss'], epoch)
        writer.add_scalar('Loss/Val', val_metrics['val_loss'], epoch)
        writer.add_scalar('Metrics/WER', val_metrics['wer'], epoch)
        writer.add_scalar('Metrics/SER', val_metrics['ser'], epoch)
        writer.add_scalar('Training/GradNorm', train_metrics['avg_grad_norm'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_metrics['wer'] < best_wer:
            best_wer = val_metrics['wer']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_wer': best_wer,
                'config': config
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            logger.info(f"New best model saved with WER: {best_wer:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'wer': val_metrics['wer'],
                'config': config
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')

    # Final test evaluation
    logger.info("\n" + "="*50)
    logger.info("Final Test Evaluation")
    logger.info("="*50)

    # Load best model
    if (output_dir / 'best_model.pth').exists():
        checkpoint = torch.load(output_dir / 'best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1}")

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
    parser = argparse.ArgumentParser(description='Train MobileNetV3 for Sign Language Recognition')

    # Data arguments
    parser.add_argument('--data_dir', type=str,
                        default='data/teacher_features/mediapipe_full',
                        help='Path to features directory')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/student',
                        help='Output directory for checkpoints')
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help='Number of training samples to use (default: all)')

    # Model arguments
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1 for better initial convergence)')
    parser.add_argument('--remove_pca', action='store_true', default=True,
                        help='Remove PCA reduction to preserve modality boundaries')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4, reduced for stability)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for regularization')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping (default: 1.0)')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')

    # Scheduler arguments
    parser.add_argument('--use_scheduler', action='store_true', default=False,
                        help='Use learning rate scheduler')

    # Memory optimization arguments
    parser.add_argument('--dynamic_truncation', action='store_true', default=True,
                        help='Enable dynamic sequence truncation')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing to save memory')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Force CPU training even if CUDA is available')

    args = parser.parse_args()
    main(args)