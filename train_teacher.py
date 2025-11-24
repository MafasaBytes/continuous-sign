"""
Training script for I3D Teacher model on full PHOENIX-2014 dataset.
Implements optimized training strategy validated through overfitting test.

Features:
- Learning rate warmup
- Cosine annealing scheduler  
- Gradient accumulation
- Mixed-precision training
- Early stopping
- Comprehensive validation
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
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
from typing import Dict, Optional, Tuple, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.i3d_teacher import create_i3d_teacher
from src.models.hierarchical_teacher import create_hierarchical_teacher
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


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, 
                 base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self, epoch: int):
        """Update learning rate."""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.min_lr + (self.base_lr - self.min_lr) * (epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def decode_predictions(log_probs: torch.Tensor, vocab: Vocabulary, use_simple_greedy: bool = False) -> List[str]:
    """
    Decode CTC predictions.
    
    Args:
        log_probs: [T, B, V] log probabilities
        vocab: Vocabulary object
        use_simple_greedy: If True, use simple greedy without filtering (for early training)
    """
    T, B, V = log_probs.shape
    decoded = []
    
    for b in range(B):
        sequence_log_probs = log_probs[:, b, :]
        max_log_probs, max_indices = torch.max(sequence_log_probs, dim=1)
        
        words = []
        word_scores = []
        prev_idx = -1
        
        for t in range(T):
            idx = max_indices[t].item()
            score = max_log_probs[t].item()
            
            if idx != prev_idx and idx != vocab.blank_id:
                if idx in vocab.idx2word:
                    words.append(vocab.idx2word[idx])
                    word_scores.append(score)
            prev_idx = idx
        
        # Simple greedy - no filtering (use during early training)
        if use_simple_greedy:
            decoded.append(' '.join(words[:50]))  # Cap at 50 words
            continue
        
        # Adaptive filtering for later training
        if len(words) > 2:
            mean_score = sum(word_scores) / len(word_scores) if word_scores else -10.0
            std_score = (sum((s - mean_score) ** 2 for s in word_scores) / len(word_scores)) ** 0.5 if word_scores else 1.0
            
            # More lenient thresholds
            if len(words) <= 5:
                threshold = mean_score - 0.8 * std_score  # Increased from 0.5
            elif len(words) <= 15:
                threshold = mean_score - 0.6 * std_score  # Increased from 0.3
            else:
                threshold = mean_score - 0.4 * std_score  # Increased from 0.2
            
            filtered_words = []
            for i, (word, score) in enumerate(zip(words, word_scores)):
                # Keep first 2 words always
                if i < 2:
                    filtered_words.append(word)
                elif score >= threshold:
                    filtered_words.append(word)
                else:
                    # Check if remaining words are above threshold
                    if i < len(words) - 1:
                        remaining_scores = word_scores[i:]
                        remaining_avg = sum(remaining_scores) / len(remaining_scores)
                        if remaining_avg >= threshold:
                            filtered_words.append(word)
                        else:
                            break
                    else:
                        break
            words = filtered_words[:50]
        
        decoded.append(' '.join(words))
    
    return decoded


def train_epoch(model, dataloader, criterion, optimizer, scaler, device,
                accumulation_steps: int = 1, max_grad_norm: float = 1.0,
                use_amp: bool = True) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(pbar):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        feature_lengths = batch['input_lengths'].to(device)
        label_lengths = batch['target_lengths'].to(device)
        
        # Forward pass with mixed precision
        if use_amp:
            with autocast():
                log_probs = model(features, feature_lengths)
                loss = criterion(log_probs, labels, feature_lengths, label_lengths)
                loss = loss / accumulation_steps
        else:
            log_probs = model(features, feature_lengths)
            loss = criterion(log_probs, labels, feature_lengths, label_lengths)
            loss = loss / accumulation_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (i + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
            
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})
        
        # Decode predictions for WER (on subset for speed)
        # Note: Using simple greedy for training WER (faster, less critical)
        if i % 50 == 0:
            with torch.no_grad():
                predictions = decode_predictions(log_probs, dataloader.dataset.vocabulary, use_simple_greedy=True)
                targets = [' '.join(words) for words in batch['words']]
                all_predictions.extend(predictions)
                all_targets.extend(targets)
    
    avg_loss = total_loss / len(dataloader)
    wer = compute_wer(all_targets, all_predictions) if all_targets else 100.0
    
    return avg_loss, wer


def validate(model, dataloader, criterion, vocab, device) -> Dict:
    """Validate model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            feature_lengths = batch['input_lengths'].to(device)
            label_lengths = batch['target_lengths'].to(device)
            
            # Forward pass
            log_probs = model(features, feature_lengths)
            loss = criterion(log_probs, labels, feature_lengths, label_lengths)
            
            total_loss += loss.item()
            
            # Decode predictions (use simple greedy for consistency)
            predictions = decode_predictions(log_probs, vocab, use_simple_greedy=True)
            targets = [' '.join(words) for words in batch['words']]
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    avg_loss = total_loss / len(dataloader)
    wer = compute_wer(all_targets, all_predictions)
    ser = compute_ser(all_targets, all_predictions)
    
    return {
        'val_loss': avg_loss,
        'wer': wer,
        'ser': ser,
        'predictions': all_predictions[:10],  # Save first 10 for inspection
        'targets': all_targets[:10]
    }


def plot_training_curves(train_losses, val_losses, val_wers, learning_rates, 
                         best_wer, output_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, train_losses, label='Train', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('CTC Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # WER curve
    axes[0, 1].plot(epochs, val_wers, linewidth=2, color='orange')
    axes[0, 1].axhline(y=best_wer, color='green', linestyle='--', 
                       label=f'Best: {best_wer:.2f}%')
    axes[0, 1].axhline(y=25, color='red', linestyle='--', 
                       label='Target: 25%')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('WER (%)')
    axes[0, 1].set_title('Validation WER')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(epochs, learning_rates, linewidth=2, color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss vs WER
    axes[1, 1].scatter(val_losses, val_wers, alpha=0.6, s=30)
    axes[1, 1].set_xlabel('Validation Loss')
    axes[1, 1].set_ylabel('WER (%)')
    axes[1, 1].set_title('Loss vs WER Correlation')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def main(args):
    """Main training function."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info(f"Training I3D Teacher Model")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Build vocabulary
    logger.info("Building vocabulary...")
    annotation_file = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    vocab = build_vocabulary(annotation_file)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = MediaPipeFeatureDataset(
        data_dir=Path(args.data_dir),
        annotation_file=annotation_file,
        vocabulary=vocab,
        split='train',
        augment=True
    )
    
    val_dataset = MediaPipeFeatureDataset(
        data_dir=Path(args.data_dir),
        annotation_file=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/dev.SI5.corpus.csv"),
        vocabulary=vocab,
        split='dev',
        augment=False
    )
    
    test_dataset = MediaPipeFeatureDataset(
        data_dir=Path(args.data_dir),
        annotation_file=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/test.SI5.corpus.csv"),
        vocabulary=vocab,
        split='test',
        augment=False
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
    )
    
    # Create model
    if args.model_type == 'hierarchical':
        logger.info("Creating Hierarchical Teacher model...")
        model = create_hierarchical_teacher(
            vocab_size=len(vocab),
            dropout=args.dropout
        )
    else:
        logger.info("Creating I3D Teacher model...")
        model = create_i3d_teacher(
            vocab_size=len(vocab),
            dropout=args.dropout
        )
    model = model.to(device)
    
    total_params = model.count_parameters()
    logger.info(f"Model parameters: {total_params:,}")
    logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Loss, optimizer, scheduler
    criterion = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.epochs,
        args.learning_rate, args.min_lr
    )
    
    # Note: For full training, model needs more time to learn
    # Overfitting test succeeded, so be patient with full dataset
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.01)  # Increased min_delta
    scaler = GradScaler(enabled=args.use_amp)
    
    # TensorBoard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Training loop
    logger.info("\n" + "="*70)
    logger.info("Starting Training")
    logger.info("="*70)
    
    best_wer = float('inf')
    train_losses, val_losses, val_wers, val_sers, learning_rates = [], [], [], [], []
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Update learning rate
        current_lr = scheduler.step(epoch)
        learning_rates.append(current_lr)
        logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Train
        train_loss, train_wer = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            args.accumulation_steps, args.max_grad_norm, args.use_amp
        )
        train_losses.append(train_loss)
        
        logger.info(f"Train Loss: {train_loss:.4f} | Train WER: {train_wer:.2f}%")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, vocab, device)
        val_losses.append(val_metrics['val_loss'])
        val_wers.append(val_metrics['wer'])
        val_sers.append(val_metrics['ser'])
        
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f} | "
                   f"Val WER: {val_metrics['wer']:.2f}% | "
                   f"Val SER: {val_metrics['ser']:.2f}%")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
        writer.add_scalar('WER/train', train_wer, epoch)
        writer.add_scalar('WER/val', val_metrics['wer'], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Save best model
        if val_metrics['wer'] < best_wer:
            best_wer = val_metrics['wer']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'wer': best_wer,
                'vocab_size': len(vocab)
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            logger.info(f"New best model saved! WER: {best_wer:.2f}%")
            
            # Show sample predictions
            logger.info("\nSample predictions:")
            for pred, target in zip(val_metrics['predictions'][:3], 
                                   val_metrics['targets'][:3]):
                logger.info(f"  Target: {target}")
                logger.info(f"  Pred:   {pred}")
        
        # Plot training curves
        if (epoch + 1) % 5 == 0:
            plot_training_curves(
                train_losses, val_losses, val_wers,
                learning_rates, best_wer, output_dir
            )
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'wer': val_metrics['wer']
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Early stopping
        if early_stopping(val_metrics['val_loss']):
            logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation on test set
    logger.info("\n" + "="*70)
    logger.info("Final Test Evaluation")
    logger.info("="*70)
    
    checkpoint = torch.load(output_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, vocab, device)
    logger.info(f"Test Loss: {test_metrics['val_loss']:.4f}")
    logger.info(f"Test WER: {test_metrics['wer']:.2f}%")
    logger.info(f"Test SER: {test_metrics['ser']:.2f}%")
    
    # Save results
    results = {
        'best_val_wer': best_wer,
        'test_wer': test_metrics['wer'],
        'test_ser': test_metrics['ser'],
        'total_params': total_params,
        'model_size_mb': total_params * 4 / 1024 / 1024,
        'training_epochs': len(train_losses),
        'config': vars(args)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final plots
    plot_training_curves(
        train_losses, val_losses, val_wers,
        learning_rates, best_wer, output_dir
    )
    
    logger.info(f"\n{'='*70}")
    logger.info("Training Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Best Validation WER: {best_wer:.2f}%")
    logger.info(f"Test WER: {test_metrics['wer']:.2f}%")
    logger.info(f"Target WER: < 25% {'ACHIEVED' if test_metrics['wer'] < 25 else 'NOT YET'}")
    logger.info(f"Results saved to: {output_dir}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train I3D Teacher for Sign Language Recognition')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/teacher_features/mediapipe_full')
    parser.add_argument('--output_dir', type=str, default='checkpoints/teacher')
    
    # Model
    parser.add_argument('--model_type', type=str, default='i3d',
                       choices=['i3d', 'hierarchical'],
                       help='Model architecture: i3d (baseline) or hierarchical (multi-scale, better generalization)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate (0.5 for strong regularization - prevents overfitting)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                       help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                       help='Minimum learning rate for cosine annealing')
    parser.add_argument('--warmup_epochs', type=int, default=30,
                       help='Number of warmup epochs')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                       help='Weight decay (L2 regularization - 0.001 for strong reg)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Gradient clipping threshold')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--patience', type=int, default=100,
                       help='Early stopping patience (increased for full training)')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    main(args)

