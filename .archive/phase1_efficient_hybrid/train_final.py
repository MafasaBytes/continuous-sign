"""Final working training script with all fixes applied."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import json
from tqdm import tqdm

from teacher.configs.mediapipe_pca1024 import get_default_config
from teacher.models.efficient_hybrid import create_efficient_model
from teacher.data.mediapipe_dataset import MediaPipeFeatureDataset, collate_fn, build_vocabulary
from teacher.utils.seed import set_seed


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Simple training epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')
    for batch in progress_bar:
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        # Forward
        log_probs = model(features, input_lengths)

        # Loss
        loss = criterion(log_probs, labels, input_lengths, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / max(num_batches, 1)


def validate(model, dataloader, criterion, device, epoch):
    """Simple validation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} - Validation')
        for batch in progress_bar:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            target_lengths = batch['target_lengths'].to(device)

            # Forward
            log_probs = model(features, input_lengths)

            # Loss
            loss = criterion(log_probs, labels, input_lengths, target_lengths)

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / max(num_batches, 1)


def main():
    # Configuration
    config = get_default_config()
    config['batch_size'] = 8  # Small batch size
    config['hidden_dim'] = 768
    config['checkpoint_dir'] = 'checkpoints/final'

    set_seed(42)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Build vocabulary
    logger.info("Building vocabulary...")
    vocab, _ = build_vocabulary([config['train_csv']])
    config['vocab'] = vocab
    config['num_classes'] = len(vocab)
    logger.info(f"Vocabulary size: {config['num_classes']}")

    # Data loaders
    train_dataset = MediaPipeFeatureDataset(
        npz_dir=config['train_npz'],
        annotation_csv=config['train_csv'],
        vocabulary=vocab,
        max_length=None,
        normalize=True
    )
    val_dataset = MediaPipeFeatureDataset(
        npz_dir=config['dev_npz'],
        annotation_csv=config['dev_csv'],
        vocabulary=vocab,
        max_length=None,
        normalize=True,
        feature_mean=getattr(train_dataset, 'feature_mean', None),
        feature_std=getattr(train_dataset, 'feature_std', None)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Model
    model = create_efficient_model(
        model_type='hybrid',
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        dropout=0.35  # Moderate dropout
    ).to(device)

    total_params = model.count_parameters()
    logger.info(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Conservative learning rate with strong weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,  # Lower learning rate
        weight_decay=1e-3  # Strong weight decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # Training loop
    best_val_loss = float('inf')
    history = []
    patience = 0
    max_patience = 15

    logger.info("Starting training...")
    logger.info("="*60)

    for epoch in range(1, 51):  # 50 epochs max
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, criterion, device, epoch)

        # Step scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        logger.info(f"Epoch {epoch}/50 | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}")
        logger.info(f"  Gap: {val_loss - train_loss:.4f}")

        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
            'gap': val_loss - train_loss
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(config['checkpoint_dir'], 'best.pt'))
            logger.info(f"  [SAVED] Best model (val_loss: {val_loss:.4f})")
            patience = 0
        else:
            patience += 1

        # Early stopping
        if patience >= max_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

        # Save history
        with open(os.path.join(config['checkpoint_dir'], 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        # Detect severe overfitting
        if epoch > 10 and val_loss - train_loss > 1.0:
            logger.warning(f"Severe overfitting detected! Gap: {val_loss - train_loss:.4f}")
            # Increase regularization
            for g in optimizer.param_groups:
                g['weight_decay'] = min(5e-3, g['weight_decay'] * 2)
            logger.info(f"  Increased weight decay to {optimizer.param_groups[0]['weight_decay']:.2e}")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()