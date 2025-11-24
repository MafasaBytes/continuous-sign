"""
Training script for BiLSTM-CTC model with CNN features.

Research-grade implementation following best practices for reproducibility
and rigorous evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import random
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import h5py
import tqdm.auto as tqdm

# Import dataset and utilities
from experiments.cnn_feature_dataset import (
    CNNFeatureDataset,
    collate_fn,
    build_vocabulary
)


class CNNBiLSTM(nn.Module):
    """
    Two-layer BiLSTM for sign language recognition with CNN features.

    Architecture optimized for 1024-D CNN features with proper regularization
    to prevent overfitting and CTC collapse.
    """

    def __init__(self,
                 input_dim: int = 1024,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 num_classes: int = 1295,
                 dropout_input: float = 0.5,
                 dropout_lstm: float = 0.5):
        """
        Initialize model.

        Args:
            input_dim: CNN feature dimension (1024 for GoogLeNet)
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of BiLSTM layers
            num_classes: Vocabulary size (including blank)
            dropout_input: Dropout rate for input
            dropout_lstm: Dropout rate between LSTM layers
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Input projection with normalization
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_input)
        )

        # First BiLSTM layer
        self.lstm1 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0  # Handle dropout separately
        )
        self.dropout1 = nn.Dropout(dropout_lstm)

        # Second BiLSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * 2,  # Bidirectional input
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.dropout2 = nn.Dropout(dropout_lstm)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, num_classes)

        # Initialize weights
        self._initialize_weights()

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: [B, T, 1024] CNN features
            lengths: [B] actual sequence lengths

        Returns:
            [T, B, C] log probabilities for CTC loss
        """
        batch_size, max_len, _ = features.shape

        # Input projection
        x = self.input_projection(features)  # [B, T, hidden_dim]

        # Pack sequences for efficiency
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # First BiLSTM
        out1, _ = self.lstm1(packed)
        out1, _ = pad_packed_sequence(out1, batch_first=True)
        out1 = self.dropout1(out1)  # [B, T, hidden_dim*2]

        # Second BiLSTM (pack again)
        packed2 = pack_padded_sequence(
            out1, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out2, _ = self.lstm2(packed2)
        out2, _ = pad_packed_sequence(out2, batch_first=True)
        out2 = self.dropout2(out2)  # [B, T, hidden_dim*2]

        # Output projection
        logits = self.output_projection(out2)  # [B, T, num_classes]

        # Log softmax for CTC
        log_probs = F.log_softmax(logits, dim=-1)

        # Transpose for CTC: [B, T, C] -> [T, B, C]
        log_probs = log_probs.transpose(0, 1)

        return log_probs

    def _initialize_weights(self):
        """Initialize weights to prevent CTC collapse."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-hidden weights
                nn.init.orthogonal_(param)
            elif 'bias' in name and 'lstm' in name:
                # LSTM biases
                param.data.zero_()
                # Set forget gate bias to 1.0
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
            elif 'output_projection.weight' in name:
                # Output layer - smaller initialization
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'output_projection.bias' in name:
                param.data.zero_()
                # Add negative bias to blank to discourage blank collapse at start
                param.data[0] = -3.0


def compute_wer(predictions: List[List[int]],
                 targets: List[List[int]]) -> float:
    """
    Compute Word Error Rate.

    Args:
        predictions: List of predicted sequences
        targets: List of target sequences

    Returns:
        WER as percentage
    """
    total_errors = 0
    total_words = 0

    for pred, target in zip(predictions, targets):
        # Simple WER using edit distance
        errors = edit_distance(pred, target)
        total_errors += errors
        total_words += len(target)

    if total_words == 0:
        return 0.0

    return (total_errors / total_words) * 100


def edit_distance(seq1: List[int], seq2: List[int]) -> int:
    """Compute edit distance between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]


def decode_predictions(log_probs: torch.Tensor,
                       lengths: torch.Tensor,
                       blank_idx: int = 0) -> List[List[int]]:
    """
    Decode CTC output using greedy decoding.

    Args:
        log_probs: [T, B, C] log probabilities
        lengths: [B] sequence lengths
        blank_idx: Index of blank token

    Returns:
        List of decoded sequences
    """
    batch_size = log_probs.size(1)
    predictions = []

    for b in range(batch_size):
        seq_len = lengths[b].item()
        # Get predictions for this sequence
        seq_log_probs = log_probs[:seq_len, b, :]
        # Greedy decoding
        _, pred_indices = seq_log_probs.max(dim=-1)
        pred_indices = pred_indices.cpu().numpy()

        # Remove blanks and repeated tokens
        decoded = []
        prev = blank_idx
        for idx in pred_indices:
            if idx != blank_idx and idx != prev:
                decoded.append(idx)
            prev = idx

        predictions.append(decoded)

    return predictions


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.CTCLoss,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epoch: int,
                logger: logging.Logger,
                time_mask_prob: float = 0.0,
                time_mask_width: int = 0) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_samples = 0
    total_gradient_norm = 0
    batch_times = []

    for batch_idx, batch in enumerate(dataloader):
        start_time = time.time()

        # Move to device
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        # Optional time masking augmentation (SpecAugment-like on time axis)
        if time_mask_prob > 0 and time_mask_width > 0:
            with torch.no_grad():
                B, T, D = features.shape
                for i in range(B):
                    seq_len = int(input_lengths[i].item())
                    if seq_len <= 0:
                        continue
                    # With given probability, apply one time mask per sequence
                    if random.random() < time_mask_prob:
                        width = min(time_mask_width, max(1, seq_len // 10))
                        start = random.randint(0, max(0, seq_len - width))
                        features[i, start:start+width, :] = 0.0

        # Forward pass
        log_probs = model(features, input_lengths)

        # CTC loss
        loss = criterion(
            log_probs,
            labels,
            input_lengths,
            target_lengths
        )

        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}")
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # Logging
        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)
        total_gradient_norm += grad_norm
        batch_times.append(time.time() - start_time)

        # Log every 50 batches
        if batch_idx % 50 == 0:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            logger.info(f"Epoch {epoch} Batch {batch_idx}/{len(dataloader)}: "
                       f"Loss={loss.item():.4f}, "
                       f"AvgLoss={avg_loss:.4f}, "
                       f"GradNorm={grad_norm:.4f}")

    # Epoch statistics
    metrics = {
        'train_loss': total_loss / total_samples if total_samples > 0 else 0,
        'gradient_norm': total_gradient_norm / len(dataloader),
        'batch_time': np.mean(batch_times),
        'total_time': sum(batch_times)
    }

    return metrics


def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.CTCLoss,
             vocab: Dict[str, int],
             device: torch.device,
             logger: logging.Logger) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_targets = []

    # Tracking metrics (decoded-level)
    blank_predictions = 0
    total_predictions = 0
    unique_predictions_set = set()
    unique_nonblank_predictions_set = set()
    too_short_count = 0
    total_sequences = 0

    # Frame-level metrics (pre-decoding argmax over classes)
    from collections import Counter
    frame_blank_frames = 0
    frame_total_frames = 0
    frame_token_counts = Counter()

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            target_lengths = batch['target_lengths'].to(device)

            # Forward pass
            log_probs = model(features, input_lengths)

            # Frame-level argmax (before collapsing), per sequence length
            B = features.size(0)
            for b in range(B):
                seq_len = input_lengths[b].item()
                frame_preds = log_probs[:seq_len, b, :].argmax(dim=-1)  # [seq_len]
                frame_blank_frames += (frame_preds == 0).sum().item()
                frame_total_frames += seq_len
                frame_token_counts.update(frame_preds.cpu().tolist())

            # CTC loss
            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                target_lengths
            )

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * features.size(0)
                total_samples += features.size(0)

            # Decode predictions
            predictions = decode_predictions(log_probs, input_lengths)

            # Convert labels to list format
            for b in range(len(predictions)):
                target_len = target_lengths[b].item()
                target = labels[b, :target_len].cpu().tolist()
                all_targets.append(target)
                all_predictions.append(predictions[b])

                # Track metrics
                for pred_idx in predictions[b]:
                    if pred_idx == 0:  # Blank token
                        blank_predictions += 1
                    unique_predictions_set.add(pred_idx)
                    if pred_idx != 0:
                        unique_nonblank_predictions_set.add(pred_idx)
                    total_predictions += 1

                # Check CTC constraint T >= 2L + 1
                T = input_lengths[b].item()
                L = target_len
                if T < (2 * L + 1):
                    too_short_count += 1
                total_sequences += 1

    # Compute WER
    wer = compute_wer(all_predictions, all_targets)

    # Compute additional metrics
    # Use frame-level blank ratio (decoded removes blanks)
    frame_blank_ratio = (frame_blank_frames / frame_total_frames) if frame_total_frames > 0 else 0.0
    too_short_ratio = (too_short_count / total_sequences) if total_sequences > 0 else 0.0
    if frame_total_frames > 0 and len(frame_token_counts) > 0:
        most_common_token, most_common_count = frame_token_counts.most_common(1)[0]
        frame_top_token_ratio = most_common_count / frame_total_frames
    else:
        most_common_token, frame_top_token_ratio = None, 0.0

    metrics = {
        'val_loss': total_loss / total_samples if total_samples > 0 else 0,
        'val_wer': wer,
        'blank_ratio': frame_blank_ratio * 100,
        'unique_predictions': len(unique_predictions_set),
        'unique_nonblank_predictions': len(unique_nonblank_predictions_set),
        'ctc_too_short_ratio': too_short_ratio * 100,
        'frame_top_token_ratio': frame_top_token_ratio * 100,
        'frame_top_token_id': int(most_common_token) if most_common_token is not None else -1,
        'frame_unique_predictions': len(frame_token_counts),
    }

    return metrics


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    # Configuration
    config = {
        # Data
        'train_h5': 'data/features_cnn/train_features.h5',
        'dev_h5': 'data/features_cnn/dev_features.h5',
        'test_h5': 'data/features_cnn/test_features.h5',

        # Model
        'input_dim': 1024,
        'hidden_dim': 512,
        'num_layers': 2,
        'dropout_input': 0.3,
        'dropout_lstm': 0.5,

        # Training
        'batch_size': 16,
        'learning_rate': 1e-3,  # Increased from 1e-4 to escape mode collapse (21 unique preds)
        'weight_decay': 1e-4,
        'num_epochs': 500,
        'gradient_clip': 10.0,  # Relaxed to allow more exploration
        # Simple regularization/augmentation
        'time_mask_prob': 0.15,   # probability to apply a time mask per sequence
        'time_mask_width': 12,    # max mask width in frames

        # Scheduler
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'min_lr': 1e-6,

        # Early stopping
        'early_stopping_patience': 150,
        'min_delta': 0.001,

        # Reproducibility
        'seed': 42,

        # Paths
        'checkpoint_dir': 'checkpoints/cnn_bilstm',
        'log_dir': 'logs/cnn_bilstm'
    }

    # Set seed
    set_seed(config['seed'])

    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config['log_dir'], f'training_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting training with CNN features")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Build vocabulary
    logger.info("Building vocabulary...")
    vocab, idx2sign = build_vocabulary([config['train_h5'], config['dev_h5']])
    config['num_classes'] = len(vocab)
    logger.info(f"Vocabulary size: {config['num_classes']}")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = CNNFeatureDataset(
        h5_file=config['train_h5'],
        vocabulary=vocab
    )

    # Reuse train normalization statistics for validation
    val_dataset = CNNFeatureDataset(
        h5_file=config['dev_h5'],
        vocabulary=vocab,
        feature_mean=getattr(train_dataset, 'feature_mean', None),
        feature_std=getattr(train_dataset, 'feature_std', None)
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
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

    # Create model
    logger.info("Creating model...")
    model = CNNBiLSTM(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout_input=config['dropout_input'],
        dropout_lstm=config['dropout_lstm']
    )
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Model size: {num_params * 4 / 1024**2:.2f} MB (FP32)")

    # Loss function
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience'],
        min_lr=config['min_lr']
    )

    # Training loop
    best_wer = float('inf')
    patience_counter = 0
    training_history = []

    logger.info("Starting training...")

    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, logger,
            time_mask_prob=config['time_mask_prob'],
            time_mask_width=config['time_mask_width']
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, vocab,
            device, logger
        )

        # Update scheduler
        scheduler.step(val_metrics['val_loss'])
        current_lr = optimizer.param_groups[0]['lr']

        # Epoch summary
        epoch_time = time.time() - epoch_start
        logger.info(f"\nEpoch {epoch}/{config['num_epochs']} Summary:")
        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  Val WER: {val_metrics['val_wer']:.2f}%")
        logger.info(f"  Blank Ratio: {val_metrics['blank_ratio']:.2f}%")
        logger.info(f"  Unique Predictions: {val_metrics['unique_predictions']}")
        logger.info(f"  Unique Non-Blank Predictions: {val_metrics['unique_nonblank_predictions']}")
        logger.info(f"  CTC Too-Short Ratio: {val_metrics['ctc_too_short_ratio']:.2f}%")
        logger.info(f"  Frame Blank Ratio: {val_metrics['blank_ratio']:.2f}%")
        logger.info(f"  Frame Unique Predictions: {val_metrics['frame_unique_predictions']}")
        logger.info(f"  Frame Top Token: id={val_metrics['frame_top_token_id']} ratio={val_metrics['frame_top_token_ratio']:.2f}%")
        logger.info(f"  Learning Rate: {current_lr:.2e}")
        logger.info(f"  Epoch Time: {epoch_time:.2f}s")

        # Save metrics (convert tensors to floats)
        epoch_metrics = {
            'epoch': epoch,
            **{k: float(v) if isinstance(v, torch.Tensor) else v for k, v in train_metrics.items()},
            **{k: float(v) if isinstance(v, torch.Tensor) else v for k, v in val_metrics.items()},
            'lr': float(current_lr) if isinstance(current_lr, torch.Tensor) else current_lr,
            'epoch_time': float(epoch_time)
        }
        training_history.append(epoch_metrics)

        # Save training history
        history_file = os.path.join(config['log_dir'], f'history_{timestamp}.json')
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)

        # Check for improvement
        if val_metrics['val_wer'] < best_wer - config['min_delta']:
            best_wer = val_metrics['val_wer']
            patience_counter = 0

            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_wer': best_wer,
                'config': config,
                'vocab': vocab
            }

            best_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"  Saved best model (WER: {best_wer:.2f}%)")
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= config['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break

        # Sanity checks
        if epoch == 1:
            if val_metrics['val_wer'] == 100.0:
                logger.warning("WARNING: Model not learning (100% WER after epoch 1)")
                logger.warning("Check CTC constraint: T >= 2*L+1")
                logger.warning(f"Blank ratio: {val_metrics['blank_ratio']:.2f}%")
                logger.warning(f"Unique predictions: {val_metrics['unique_predictions']}")
            elif val_metrics['val_wer'] < 85.0:
                logger.info("GOOD: Model is learning faster than expected!")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'wer': val_metrics['val_wer'],
                'config': config
            }
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"  Saved checkpoint at epoch {epoch}")

    logger.info(f"\nTraining completed!")
    logger.info(f"Best WER: {best_wer:.2f}%")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Best model: {best_path}")

    # Final test evaluation (optional)
    if os.path.exists(config['test_h5']):
        logger.info("\nEvaluating on test set...")
        test_dataset = CNNFeatureDataset(
            h5_file=config['test_h5'],
            vocabulary=vocab,
            feature_mean=getattr(train_dataset, 'feature_mean', None),
            feature_std=getattr(train_dataset, 'feature_std', None)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2
        )

        # Load best model
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Test
        test_metrics = validate(
            model, test_loader, criterion, vocab,
            device, logger
        )

        logger.info(f"Test WER: {test_metrics['val_wer']:.2f}%")
        logger.info(f"Test Loss: {test_metrics['val_loss']:.4f}")

        # Save test results
        test_results = {
            'test_wer': test_metrics['val_wer'],
            'test_loss': test_metrics['val_loss'],
            'best_val_wer': best_wer,
            'config': config
        }

        test_file = os.path.join(config['log_dir'], f'test_results_{timestamp}.json')
        with open(test_file, 'w') as f:
            json.dump(test_results, f, indent=2)


if __name__ == "__main__":
    main()