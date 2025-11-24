"""Training script for efficient hybrid model."""

import sys
from pathlib import Path
# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import time
from datetime import datetime
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from teacher.configs.mediapipe_pca1024 import get_default_config
from teacher.models.efficient_hybrid import create_efficient_model
from teacher.data.mediapipe_dataset import MediaPipeFeatureDataset, collate_fn, build_vocabulary
from teacher.trainers.ctc_trainer import train_epoch, validate
from teacher.utils.seed import set_seed
from teacher.utils.schedule import get_warmup_lr, get_decaying_blank_penalty


def build_loaders(config, logger):
    """Build train and validation data loaders."""
    train_dataset = MediaPipeFeatureDataset(
        npz_dir=config['train_npz'],
        annotation_csv=config['train_csv'],
        vocabulary=config['vocab'],
        max_length=None,
        normalize=True
    )
    val_dataset = MediaPipeFeatureDataset(
        npz_dir=config['dev_npz'],
        annotation_csv=config['dev_csv'],
        vocabulary=config['vocab'],
        max_length=None,
        normalize=True,
        feature_mean=getattr(train_dataset, 'feature_mean', None),
        feature_std=getattr(train_dataset, 'feature_std', None)
    )
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=max(8, os.cpu_count()//4),
        pin_memory=True,
        persistent_workers=True  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=max(4, os.cpu_count()//4),
        pin_memory=True,
        persistent_workers=True
    )
    return train_loader, val_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Train the efficient hybrid model')
    parser.add_argument('--hidden_dim', type=int, default=768,
                       help='Hidden dimension (default: 768)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs (default: 150)')
    parser.add_argument('--decode_method', type=str, choices=['greedy', 'beam_search'],
                       default='greedy', help='Decoding method')
    parser.add_argument('--beam_width', type=int, default=10,
                       help='Beam width for beam search')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='checkpoints/efficient_hybrid',
                       help='Directory to save checkpoints')
    return parser.parse_args()


def main():
    args = parse_args()

    # Get base config and override with command line args
    config = get_default_config()
    config['hidden_dim'] = args.hidden_dim
    config['batch_size'] = args.batch_size
    config['decode_method'] = args.decode_method
    config['beam_width'] = args.beam_width
    config['checkpoint_dir'] = args.checkpoint_dir
    config['experiment_name'] = f'efficient_hybrid_{args.hidden_dim}'

    set_seed(config['seed'])
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config['log_dir'], f'{config["experiment_name"]}_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration:")
    logger.info(f"  Hidden dimension: {config['hidden_dim']}")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Decoding method: {config['decode_method']}")

    # Build vocabulary
    logger.info("Building vocabulary...")
    vocab, _ = build_vocabulary([config['train_csv']])
    config['num_classes'] = len(vocab)
    config['vocab'] = vocab
    logger.info(f"Vocabulary size: {config['num_classes']}")

    # Create data loaders
    train_loader, val_loader = build_loaders(config, logger)

    # Create efficient hybrid model
    logger.info("Creating Efficient Hybrid Model...")
    model = create_efficient_model(
        model_type='hybrid',
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        dropout=0.3
    ).to(device)

    # Log model details
    total_params = model.count_parameters()
    logger.info(f"Model parameters: {total_params:,}")
    logger.info(f"Model size: {total_params * 4 / (1024**2):.2f} MB")

    # Loss function with regularization
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler - cosine annealing with warmup
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=warmup_epochs/args.epochs,  # Warmup percentage
        anneal_strategy='cos',
        div_factor=25,  # Initial lr = max_lr/25
        final_div_factor=1000  # Final lr = max_lr/1000
    )

    # Training loop
    best_wer = float('inf')
    training_history = []

    logger.info("Starting training...")
    logger.info("="*60)

    for epoch in range(1, args.epochs + 1):
        # Adjust regularization over time
        if epoch <= 10:
            # Phase 1: Warmup with low regularization
            blank_penalty = -2.0
            temperature = 1.5
            time_mask_prob = 0.0
            dropout_rate = 0.2
        elif epoch <= 50:
            # Phase 2: Main training with moderate regularization
            blank_penalty = -1.0
            temperature = 1.2
            time_mask_prob = 0.1
            dropout_rate = 0.3
        elif epoch <= 100:
            # Phase 3: Fine-tuning with higher regularization
            blank_penalty = -0.5
            temperature = 1.0
            time_mask_prob = 0.15
            dropout_rate = 0.35
        else:
            # Phase 4: Final refinement
            blank_penalty = 0.0
            temperature = 1.0
            time_mask_prob = 0.05
            dropout_rate = 0.4

        # Update dropout in model
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate

        # Training
        epoch_start = time.time()
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger,
            blank_penalty=blank_penalty,
            time_mask_prob=time_mask_prob,
            temperature=temperature,
            gradient_clip=5.0
        )

        # Step scheduler after each batch (OneCycleLR requires this)
        for _ in range(len(train_loader)):
            scheduler.step()

        # Validation
        val_metrics = validate(
            model, val_loader, criterion, vocab, device, logger,
            epoch=epoch,
            blank_penalty=0.0,  # No penalty during validation
            temperature=1.0,    # No temperature during validation
            decode_method=config['decode_method'],
            beam_width=config['beam_width']
        )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        logger.info(f"Epoch {epoch}/{args.epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f} | "
                   f"Grad Norm: {train_metrics['gradient_norm']:.2f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f} | "
                   f"WER: {val_metrics['val_wer']:.2f}% | "
                   f"Blank Ratio: {val_metrics['blank_ratio']:.1f}%")
        logger.info(f"  Unique Predictions: {val_metrics['unique_nonblank_predictions']} | "
                   f"CTC Too Short: {val_metrics['ctc_too_short_ratio']:.1f}%")

        # Save best model
        if val_metrics['val_wer'] < best_wer:
            best_wer = val_metrics['val_wer']
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_wer': best_wer,
                'config': config,
                'val_metrics': val_metrics
            }, checkpoint_path)
            logger.info(f"  [SAVED] New best WER: {best_wer:.2f}%")

        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_metrics['train_loss'],
            'val_loss': val_metrics['val_loss'],
            'val_wer': val_metrics['val_wer'],
            'lr': current_lr
        })

        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_wer': best_wer,
                'config': config
            }, checkpoint_path)

        # Early stopping check
        if epoch > 50 and val_metrics['val_wer'] > 80:
            logger.warning(f"WER still above 80% after {epoch} epochs. Consider:")
            logger.warning("  1. Check data loading and preprocessing")
            logger.warning("  2. Verify vocabulary mapping")
            logger.warning("  3. Adjust learning rate or model capacity")

    # Save final results
    logger.info("="*60)
    logger.info("Training completed!")
    logger.info(f"Best WER: {best_wer:.2f}%")

    # Save training history
    history_path = os.path.join(config['checkpoint_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    logger.info(f"Results saved to {config['checkpoint_dir']}")


if __name__ == "__main__":
    main()