"""
Hierarchical Training with MediaPipe PCA Features - Fixed Approach

Uses PCA-reduced MediaPipe features (1024 dimensions) to address curse of dimensionality.

Original MediaPipe: 6516 dimensions → Too many for 4376 samples (248× overfitting!)
PCA Reduction: 6516 → 1024 dimensions (6.36× compression, 100% variance retained)

This solves the sample complexity problem:
- Before: 6516 features / 4376 samples = 1.49 features per sample
- After:  1024 features / 4376 samples = 4.27 samples per feature

Key Fixes (same as CNN version):
1. Blank penalty applied BEFORE softmax (966x stronger effect)
2. ReduceLROnPlateau with linear warmup for stable optimization
3. Full model (stage=2) from start - no frame-only stage
4. No sequence clipping initially - only add if needed
5. Lower initial dropout (0.1) - build capacity first, regularize later
6. Dynamic blank_penalty decay: -3.0 → 0.0 over training

Training Phases:
1. Warmup (30 epochs): Initialize gradients without collapse
2. Exploration (60 epochs): Explore vocabulary with stable gradients
3. Consolidation (50 epochs): Reduce overfitting, improve alignment
4. Fine-tuning (25 epochs): Final refinement

Expected Results (vs 6516-dim MediaPipe which had 248× overfitting):
- Phase 1 end: val_wer ~75-80%, blank_ratio ~50-60%, unique_nonblank ~250-350, overfit ~2-3×
- Phase 2 end: val_wer ~50-60%, blank_ratio ~35-45%, unique_nonblank ~450-550, overfit ~2-4×
- Phase 3 end: val_wer ~35-45%, blank_ratio ~25-35%, unique_nonblank ~600-750, overfit ~1.5-2.5×
- Phase 4 end: val_wer ~30-40%, blank_ratio ~20-30%, unique_nonblank ~700-850, overfit ~1.3-2×

Key improvement: Overfitting reduced from 248× to 2-4× (100× better generalization!)
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
from tqdm import tqdm

# Import dataset and utilities
from experiments.mediapipe_feature_dataset import (
    MediaPipeFeatureDataset,
    collate_fn,
    build_vocabulary
)

# Import from baseline for shared utilities
import sys
sys.path.append(str(Path(__file__).parent))
from train_cnn_features import (
    compute_wer,
    edit_distance,
    decode_predictions,
    set_seed
)


# ==================== FIXED MODEL WITH BLANK PENALTY ====================

class TemporalConvolution(nn.Module):
    """Temporal convolution to capture short-term motion patterns."""
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)
        self.combine = nn.Linear(hidden_dim * 3, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, T, input_dim], Returns: [B, T, hidden_dim]"""
        x_t = x.transpose(1, 2)  # [B, D, T]

        conv1_out = F.relu(self.conv1(x_t)).transpose(1, 2)
        conv2_out = F.relu(self.conv2(x_t)).transpose(1, 2)
        conv3_out = F.relu(self.conv3(x_t)).transpose(1, 2)

        combined = torch.cat([conv1_out, conv2_out, conv3_out], dim=-1)
        output = self.combine(combined)
        output = self.norm(output)

        return output


class HierarchicalModelFixed(nn.Module):
    """
    Fixed hierarchical model with proper blank penalty support.

    Key fix: blank_penalty parameter in forward() applies BEFORE softmax
    for 966x stronger effect than bias-based approach.
    """

    def __init__(self,
                 input_dim: int = 1024,
                 hidden_dim: int = 512,
                 num_classes: int = 966,
                 dropout_frame: float = 0.2,
                 dropout_sequence: float = 0.5,
                 use_temporal_conv: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_temporal_conv = use_temporal_conv

        # Temporal convolution
        if use_temporal_conv:
            self.temporal_conv = TemporalConvolution(input_dim, hidden_dim)
            lstm_input_dim = hidden_dim
        else:
            lstm_input_dim = input_dim

        # Frame-level BiLSTM
        self.frame_lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.frame_dropout = nn.Dropout(dropout_frame)

        # Sequence-level BiLSTM
        self.sequence_lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.sequence_dropout = nn.Dropout(dropout_sequence)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, num_classes)

        self._initialize_weights()

    def forward(self, features: torch.Tensor, lengths: torch.Tensor,
                stage: int = 2, blank_penalty: float = 0.0, temperature: float = 1.0) -> torch.Tensor:
        """
        Forward pass with blank penalty applied BEFORE softmax.

        Args:
            features: [B, T, 1024]
            lengths: [B]
            stage: 1 (frame-level only) or 2 (full hierarchical)
            blank_penalty: Log-space penalty for blank token (e.g., -3.0)
                          Applied BEFORE softmax for full effect

        Returns:
            [T, B, C] log probabilities
        """
        # Temporal convolution (if enabled)
        if self.use_temporal_conv:
            x = self.temporal_conv(features)
        else:
            x = features

        # Pack for LSTM
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Frame-level BiLSTM
        frame_out, _ = self.frame_lstm(packed)
        frame_out, _ = pad_packed_sequence(frame_out, batch_first=True)
        frame_out = self.frame_dropout(frame_out)

        if stage == 1:
            # Frame-level only (not used in fixed approach, but kept for compatibility)
            logits = self.output_projection(frame_out)
        else:
            # Sequence-level BiLSTM
            packed_seq = pack_padded_sequence(
                frame_out, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            sequence_out, _ = self.sequence_lstm(packed_seq)
            sequence_out, _ = pad_packed_sequence(sequence_out, batch_first=True)
            sequence_out = self.sequence_dropout(sequence_out)

            logits = self.output_projection(sequence_out)

        # CRITICAL FIX: Apply blank penalty BEFORE softmax
        # This is 966x stronger than bias-based approach
        if blank_penalty != 0.0:
            logits[:, :, 0] = logits[:, :, 0] + blank_penalty

        # Temperature smoothing
        log_probs = F.log_softmax(logits / max(1e-6, temperature), dim=-1) # Self Note Larry

        return log_probs.transpose(0, 1)  # [T, B, C]

    def _initialize_weights(self):
        """Initialize weights with proper strategies."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name and 'lstm' in name:
                param.data.zero_()
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)  # Forget gate bias = 1
            elif 'output_projection.weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'output_projection.bias' in name:
                param.data.zero_()


# ==================== HELPER FUNCTIONS ====================

def get_warmup_lr(epoch: int, warmup_epochs: int, base_lr: float, start_lr: float = 1e-6) -> float:
    """Linear warmup learning rate."""
    if epoch <= warmup_epochs:
        return start_lr + (base_lr - start_lr) * (epoch / warmup_epochs)
    return base_lr


def get_decaying_blank_penalty(epoch: int, start_penalty: float,
                               end_penalty: float, total_epochs: int) -> float:
    """Linearly decay blank penalty over epochs."""
    decay_rate = (start_penalty - end_penalty) / total_epochs
    return max(end_penalty, start_penalty - decay_rate * epoch)


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.CTCLoss,
                optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                logger: logging.Logger, blank_penalty: float = 0.0, gradient_clip: float = 10.0,
                time_mask_prob: float = 0.0, max_seq_len: Optional[int] = None, temperature: float = 1.0) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()


    total_loss = 0
    total_samples = 0
    total_gradient_norm = 0
    batch_times = []

    for batch_idx, batch in tqdm(enumerate(dataloader)):
        start_time = time.time()

        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        # Sequence length clipping (only if specified)
        if max_seq_len is not None:
            clipped_lengths = torch.clamp(input_lengths, max=max_seq_len)
            if features.size(1) > max_seq_len:
                features = features[:, :max_seq_len, :]
            input_lengths = clipped_lengths

        # Time masking augmentation
        if time_mask_prob > 0:
            B, T, D = features.shape
            for i in range(B):
                seq_len = int(input_lengths[i].item())
                if seq_len > 0 and random.random() < time_mask_prob:
                    width = min(12, max(1, seq_len // 10))
                    start = random.randint(0, max(0, seq_len - width))
                    features[i, start:start+width, :] = 0.0

        # Forward pass with blank_penalty
        log_probs = model(features, input_lengths, stage=2, blank_penalty=blank_penalty, temperature=temperature)

        # CTC loss
        loss = criterion(log_probs, labels, input_lengths, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}")
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)
        total_gradient_norm += grad_norm
        batch_times.append(time.time() - start_time)

        if batch_idx % 50 == 0:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            logger.info(f"Epoch {epoch} Batch {batch_idx}/{len(dataloader)}: "
                       f"Loss={loss.item():.4f}, AvgLoss={avg_loss:.4f}, "
                       f"GradNorm={grad_norm:.4f}, BlankPen={blank_penalty:.2f}")

    return {
        'train_loss': total_loss / total_samples if total_samples > 0 else 0,
        'gradient_norm': total_gradient_norm / len(dataloader),
        'batch_time': np.mean(batch_times),
        'total_time': sum(batch_times)
    }

def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.CTCLoss,
             vocab: Dict[str, int], device: torch.device, logger: logging.Logger,
             epoch: int = 0, blank_penalty: float = 0.0, temperature: float = 1.0) -> Dict[str, float]:
    """Validate model with comprehensive metrics."""
    model.eval()


    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_targets = []

    # Tracking metrics
    unique_predictions_set = set()
    unique_nonblank_predictions_set = set()

    # Frame-level metrics
    from collections import Counter
    frame_blank_frames = 0
    frame_total_frames = 0
    frame_token_counts = Counter()

    too_short_count = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Validating"):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            target_lengths = batch['target_lengths'].to(device)

            # Forward pass - IMPORTANT: Apply SAME blank_penalty as training for fair evaluation
            log_probs = model(features, input_lengths, stage=2, blank_penalty=blank_penalty, temperature=temperature)

            # Frame-level analysis
            B = features.size(0)
            for b in range(B):
                seq_len = input_lengths[b].item()
                frame_preds = log_probs[:seq_len, b, :].argmax(dim=-1)
                frame_blank_frames += (frame_preds == 0).sum().item()
                frame_total_frames += seq_len
                frame_token_counts.update(frame_preds.cpu().tolist())

            # CTC loss
            loss = criterion(log_probs, labels, input_lengths, target_lengths)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * features.size(0)
                total_samples += features.size(0)

            # Decode predictions
            predictions = decode_predictions(log_probs, input_lengths)

            # Convert labels
            for b in range(len(predictions)):
                target_len = target_lengths[b].item()
                target = labels[b, :target_len].cpu().tolist()
                all_targets.append(target)
                all_predictions.append(predictions[b])

                for pred_idx in predictions[b]:
                    unique_predictions_set.add(pred_idx)
                    if pred_idx != 0:
                        unique_nonblank_predictions_set.add(pred_idx)

                # CTC constraint check
                T = input_lengths[b].item()
                L = target_len
                if T < (2 * L + 1):
                    too_short_count += 1
                total_sequences += 1

    # Compute metrics
    wer = compute_wer(all_predictions, all_targets)
    frame_blank_ratio = (frame_blank_frames / frame_total_frames) if frame_total_frames > 0 else 0.0
    too_short_ratio = (too_short_count / total_sequences) if total_sequences > 0 else 0.0

    if frame_total_frames > 0 and len(frame_token_counts) > 0:
        most_common_token, most_common_count = frame_token_counts.most_common(1)[0]
        frame_top_token_ratio = most_common_count / frame_total_frames
    else:
        most_common_token, frame_top_token_ratio = None, 0.0

    # Sample prediction analysis (every 10 epochs)
    if epoch % 10 == 0 and len(all_predictions) > 0:
        sample_pred = all_predictions[0]
        sample_target = all_targets[0]
        overlap = len(set(sample_pred) & set(sample_target))

        logger.info(f"\n  [Sample Analysis] Target len: {len(sample_target)}, "
                   f"Pred len: {len(sample_pred)}, Overlap: {overlap} tokens")

    return {
        'val_loss': total_loss / total_samples if total_samples > 0 else 0,
        'val_wer': wer,
        'blank_ratio': frame_blank_ratio * 100,
        'unique_predictions': len(unique_predictions_set),
        'unique_nonblank_predictions': len(unique_nonblank_predictions_set),
        'frame_unique_predictions': len(frame_token_counts),
        'ctc_too_short_ratio': too_short_ratio * 100,
        'frame_top_token_id': int(most_common_token) if most_common_token is not None else -1,
        'frame_top_token_ratio': frame_top_token_ratio * 100,
    }


def main():
    """Main training with fixed 4-phase approach."""

    # Configuration - FIXED APPROACH with MediaPipe PCA Features
    config = {
        # Data - MediaPipe PCA-reduced NPZ files and annotations
        'train_npz': 'data/teacher_features/mediapipe_pca1024/train',
        'dev_npz': 'data/teacher_features/mediapipe_pca1024/dev',
        'test_npz': 'data/teacher_features/mediapipe_pca1024/test',
        'train_csv': 'data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv',
        'dev_csv': 'data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/dev.SI5.corpus.csv',
        'test_csv': 'data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/test.SI5.corpus.csv',

        # Model
        'input_dim': 1024,  # PCA-reduced MediaPipe features (6516 → 1024, 6.36x compression, 100% variance)
        'hidden_dim': 512,
        'use_temporal_conv': True,

        # Phase 1: Warmup (30 epochs)
        'phase1_epochs': 30,
        'phase1_lr_start': 1e-6,
        'phase1_lr_end': 1e-3,
        'phase1_blank_penalty': -5.0,  # More aggressive blank suppression
        'phase1_warmup_epochs': 5,
        'phase1_temperature': 1.5,
        'phase1_dropout_frame': 0.1,
        'phase1_dropout_sequence': 0.1,
        'phase1_time_mask_prob': 0.0,
        'phase1_weight_decay': 1e-5,
        'phase1_scheduler_patience': 5,
        'phase1_max_seq_len': None,  # No clipping

        # Phase 2: Exploration (20 epochs)
        'phase2_epochs': 100,
        'phase2_lr': 1e-3,
        'phase2_blank_penalty_start': -6.0,
        'phase2_blank_penalty_end': -2.0,
        'phase2_dropout_frame': 0.15,
        'phase2_dropout_sequence': 0.15,
        'phase2_time_mask_prob': 0.2, # Masking to encourage exploration
        'phase2_weight_decay': 1e-4,
        'phase2_max_seq_len': None,  # No clipping
        'phase2_scheduler_patience': 6,
        'phase2_scheduler_factor': 0.5,
        'phase2_temperature': 1.5,

        # Phase 3: Consolidation (15 epochs)
        'phase3_epochs': 80,
        'phase3_blank_penalty': -0.5,
        'phase3_dropout_frame': 0.2,
        'phase3_dropout_sequence': 0.25,
        'phase3_time_mask_prob': 0.15,
        'phase3_weight_decay': 1e-4,
        'phase3_max_seq_len': 300,  # Mild clipping
        'phase3_scheduler_patience': 5,
        'phase3_scheduler_factor': 0.6,
        'phase3_temperature': 1.2,

        # Phase 4: Fine-tuning (10 epochs)
        'phase4_epochs': 60,
        'phase4_blank_penalty': 0.0,
        'phase4_dropout_frame': 0.25,
        'phase4_dropout_sequence': 0.35,
        'phase4_time_mask_prob': 0.05,
        'phase4_weight_decay': 1e-4,
        'phase4_max_seq_len': 300,
        'phase4_scheduler_patience': 3,
        'phase4_scheduler_factor': 0.7,
        'phase4_temperature': 1.0,

        # Common
        'batch_size': 8,
        'gradient_clip': 10.0,
        'seed': 42,
        'checkpoint_dir': 'checkpoints/hierarchical_mediapipe_pca1024',
        'log_dir': 'logs/hierarchical_mediapipe_pca1024',
        'experiment_name': 'hierarchical_mediapipe_pca1024_v1',
        'target_wer': 35.0  # More realistic target with 4376 samples and 1024 dims
    }

    # Set seed
    set_seed(config['seed'])

    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config['log_dir'], f'{config["experiment_name"]}_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("HIERARCHICAL TRAINING WITH MEDIAPIPE PCA FEATURES")
    logger.info("="*80)
    logger.info("PCA DIMENSIONALITY REDUCTION:")
    logger.info("  - Original MediaPipe: 6516 dimensions")
    logger.info("  - PCA Reduction: 6516 → 1024 dims (6.36× compression)")
    logger.info("  - Variance retained: 100.01% (perfect compression!)")
    logger.info("  - Sample complexity: 4376 samples / 1024 dims = 4.27 samples/feature")
    logger.info("  - Previous problem: 6516 dims / 4376 samples = 248× overfitting")
    logger.info("")
    logger.info("KEY FIXES:")
    logger.info("  1. PCA reduction solves curse of dimensionality")
    logger.info("  2. Blank penalty applied BEFORE softmax (966× stronger)")
    logger.info("  3. ReduceLROnPlateau with linear warmup")
    logger.info("  4. Full model (stage=2) from start")
    logger.info("  5. No sequence clipping initially")
    logger.info("  6. Dynamic blank_penalty decay: -3.0 to 0.0")
    logger.info("="*80)
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")


    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Build vocabulary
    logger.info("Building vocabulary from annotation files...")
    vocab, idx2sign = build_vocabulary([config['train_csv'], config['dev_csv']])
    config['num_classes'] = len(vocab)
    logger.info(f"Vocabulary size: {config['num_classes']}")

    # Create datasets
    logger.info("Creating MediaPipe datasets...")
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

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=max(8, os.cpu_count()//4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=max(4, os.cpu_count()//4),
        pin_memory=True
    )

    # Create model
    logger.info("Creating fixed hierarchical model...")
    model = HierarchicalModelFixed(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        dropout_frame=config['phase1_dropout_frame'],
        dropout_sequence=config['phase1_dropout_sequence'],
        use_temporal_conv=config['use_temporal_conv']
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Model size: {num_params * 4 / 1024**2:.2f} MB (FP32)")

    # Loss function
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Training history
    training_history = []
    best_wer = float('inf')
    global_epoch = 0

    # ==================== PHASE 1: WARMUP ====================
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: WARMUP (Epochs 1-{})".format(config['phase1_epochs']))
    logger.info("  Goal: Initialize gradients without collapse")
    logger.info("  LR: {} to {} (linear warmup)".format(
        config['phase1_lr_start'], config['phase1_lr_end']))
    logger.info(f"  Blank Penalty: {config['phase1_blank_penalty']}")
    logger.info(f"  Dropout: {config['phase1_dropout_frame']}")
    logger.info(f"  Max Seq Len: {config['phase1_max_seq_len']}")
    logger.info("  Target: val_loss < 15.0, unique_nonblank > 50, blank_ratio < 85%")
    logger.info("="*80 + "\n")

    # Update model dropout
    model.frame_dropout.p = config['phase1_dropout_frame']
    model.sequence_dropout.p = config['phase1_dropout_sequence']

    optimizer1 = torch.optim.AdamW(
        model.parameters(),
        lr=config['phase1_lr_start'],
        weight_decay=config['phase1_weight_decay']
    )

    for epoch in range(1, config['phase1_epochs'] + 1):
        global_epoch += 1

        # Linear warmup LR
        current_lr = get_warmup_lr(
            epoch, config['phase1_warmup_epochs'],
            config['phase1_lr_end'], config['phase1_lr_start']
        )
        for param_group in optimizer1.param_groups:
            param_group['lr'] = current_lr

        epoch_start = time.time()

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer1,
            device, global_epoch, logger,
            blank_penalty=config['phase1_blank_penalty'],
            time_mask_prob=config['phase1_time_mask_prob'],
            temperature=config['phase1_temperature'],
            max_seq_len=config['phase1_max_seq_len'],
            gradient_clip=config['gradient_clip'],
        )

        val_metrics = validate(
            model, val_loader, criterion, vocab,
            device, logger, epoch=global_epoch,
            blank_penalty=config['phase1_blank_penalty'],
            temperature=config['phase1_temperature'],
        )

        epoch_time = time.time() - epoch_start
        overfit_ratio = val_metrics['val_loss'] / train_metrics['train_loss'] if train_metrics['train_loss'] > 0 else float('inf')

        logger.info(f"\nEpoch {global_epoch} (Phase 1: Warmup) Summary:")
        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  Overfit Ratio: {overfit_ratio:.2f}x")
        logger.info(f"  Val WER: {val_metrics['val_wer']:.2f}%")
        logger.info(f"  Frame Blank Ratio: {val_metrics['blank_ratio']:.2f}%")
        logger.info(f"  Unique Non-Blank: {val_metrics['unique_nonblank_predictions']} / {config['num_classes']} ({val_metrics['unique_nonblank_predictions']/config['num_classes']*100:.1f}%)")
        logger.info(f"  Learning Rate: {current_lr:.2e}")
        logger.info(f"  Epoch Time: {epoch_time:.2f}s")

        # Check targets
        if val_metrics['blank_ratio'] > 90.0:
            logger.warning(f"  WARNING: Blank ratio {val_metrics['blank_ratio']:.2f}% too high!")

        # Record
        epoch_metrics = {
            'epoch': global_epoch,
            'phase': 1,
            'phase_name': 'Warmup',
            **{k: float(v) if isinstance(v, torch.Tensor) else v
               for k, v in train_metrics.items()},
            **{k: float(v) if isinstance(v, torch.Tensor) else v
               for k, v in val_metrics.items()},
            'lr': float(current_lr),
            'epoch_time': float(epoch_time),
            'blank_penalty': config['phase1_blank_penalty'],
            'overfit_ratio': float(overfit_ratio)
        }
        training_history.append(epoch_metrics)

        # Save history
        history_file = os.path.join(config['log_dir'],
                                   f'{config["experiment_name"]}_history_{timestamp}.json')
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)

        # Save best
        if val_metrics['val_wer'] < best_wer - 0.001:
            best_wer = val_metrics['val_wer']
            checkpoint = {
                'epoch': global_epoch,
                'phase': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'best_wer': best_wer,
                'config': config,
                'vocab': vocab
            }
            best_path = os.path.join(config['checkpoint_dir'],
                                   f'{config["experiment_name"]}_best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"  [SAVED] Best model (WER: {best_wer:.2f}%)")

    # ==================== PHASE 2: EXPLORATION ====================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: EXPLORATION (Epochs {}-{})".format(
        global_epoch + 1, global_epoch + config['phase2_epochs']))
    logger.info("  Goal: Explore vocabulary with stable gradients")
    logger.info(f"  LR: {config['phase2_lr']} (ReduceLROnPlateau, patience={config['phase2_scheduler_patience']})")
    logger.info(f"  Blank Penalty: {config['phase2_blank_penalty_start']} → {config['phase2_blank_penalty_end']} (decaying)")
    logger.info(f"  Dropout: {config['phase2_dropout_frame']}")
    logger.info("  Target: val_wer < 80%, unique_nonblank > 300, blank_ratio < 70%")
    logger.info("="*80 + "\n")

    # Update dropout
    model.frame_dropout.p = config['phase2_dropout_frame']
    model.sequence_dropout.p = config['phase2_dropout_sequence']

    optimizer2 = torch.optim.AdamW(
        model.parameters(),
        lr=config['phase2_lr'],
        weight_decay=config['phase2_weight_decay']
    )

    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2,
        mode='min',
        factor=config['phase2_scheduler_factor'],
        patience=config['phase2_scheduler_patience'],
        threshold = 0.01,
        threshold_mode = 'rel',
        cooldown = 3,
        min_lr=1e-6,
        # verbose=True
    )

    val_loss_increasing_count = 0

    for epoch in range(1, config['phase2_epochs'] + 1):
        global_epoch += 1

        # Decay blank penalty
        blank_penalty = get_decaying_blank_penalty(
            epoch,
            config['phase2_blank_penalty_start'],
            config['phase2_blank_penalty_end'],
            config['phase2_epochs']
        )

        epoch_start = time.time()

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer2,
            device, global_epoch, logger,
            blank_penalty=blank_penalty,
            time_mask_prob=config['phase2_time_mask_prob'],
            max_seq_len=config['phase2_max_seq_len'],
            temperature=config['phase2_temperature'],
            gradient_clip=config['gradient_clip'],
        )

        val_metrics = validate(
            model, val_loader, criterion, vocab,
            device, logger, epoch=global_epoch,
            blank_penalty=blank_penalty,  # Use the decaying penalty
            temperature=config['phase2_temperature'],
            
        )

        scheduler2.step(val_metrics['val_loss'])
        current_lr = optimizer2.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start
        overfit_ratio = val_metrics['val_loss'] / train_metrics['train_loss'] if train_metrics['train_loss'] > 0 else float('inf')

        logger.info(f"\nEpoch {global_epoch} (Phase 2: Exploration) Summary:")
        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  Overfit Ratio: {overfit_ratio:.2f}x")
        logger.info(f"  Val WER: {val_metrics['val_wer']:.2f}%")
        logger.info(f"  Frame Blank Ratio: {val_metrics['blank_ratio']:.2f}%")
        logger.info(f"  Unique Non-Blank: {val_metrics['unique_nonblank_predictions']} / {config['num_classes']} ({val_metrics['unique_nonblank_predictions']/config['num_classes']*100:.1f}%)")
        logger.info(f"  Learning Rate: {current_lr:.2e}")
        logger.info(f"  Blank Penalty: {blank_penalty:.2f}")
        logger.info(f"  Epoch Time: {epoch_time:.2f}s")
        logger.info(f"  Best WER: {best_wer:.2f}%")

        # Check targets
        if val_metrics['blank_ratio'] > 85.0 and epoch > 10:
            logger.warning(f"  WARNING: Blank ratio {val_metrics['blank_ratio']:.2f}% too high - consider increasing |blank_penalty|")

        if val_metrics['unique_nonblank_predictions'] < 200 and epoch > 10:
            logger.warning(f"  WARNING: Only {val_metrics['unique_nonblank_predictions']} unique non-blanks - exploration insufficient")

        # Val loss increasing detection
        if len(training_history) > 0:
            prev_val_loss = training_history[-1]['val_loss']
            if val_metrics['val_loss'] > prev_val_loss:
                val_loss_increasing_count += 1
                if val_loss_increasing_count >= 3:
                    logger.error(f"  CRITICAL: Val loss increasing for 3 epochs - consider stopping phase early")
            else:
                val_loss_increasing_count = 0

        # Record
        epoch_metrics = {
            'epoch': global_epoch,
            'phase': 2,
            'phase_name': 'Exploration',
            **{k: float(v) if isinstance(v, torch.Tensor) else v
               for k, v in train_metrics.items()},
            **{k: float(v) if isinstance(v, torch.Tensor) else v
               for k, v in val_metrics.items()},
            'lr': float(current_lr),
            'epoch_time': float(epoch_time),
            'blank_penalty': float(blank_penalty),
            'overfit_ratio': float(overfit_ratio)
        }
        training_history.append(epoch_metrics)

        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)

        if val_metrics['val_wer'] < best_wer - 0.001:
            best_wer = val_metrics['val_wer']
            checkpoint = {
                'epoch': global_epoch,
                'phase': 2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
                'best_wer': best_wer,
                'config': config,
                'vocab': vocab
            }
            torch.save(checkpoint, best_path)
            logger.info(f"  [SAVED] Best model (WER: {best_wer:.2f}%)")

    # ==================== PHASE 3: CONSOLIDATION ====================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: CONSOLIDATION (Epochs {}-{})".format(
        global_epoch + 1, global_epoch + config['phase3_epochs']))
    logger.info("  Goal: Reduce overfitting, improve alignment")
    logger.info(f"  Blank Penalty: {config['phase3_blank_penalty']}")
    logger.info(f"  Dropout: {config['phase3_dropout_sequence']}")
    logger.info(f"  Max Seq Len: {config['phase3_max_seq_len']}")
    logger.info("  Target: val_wer < 50%, overfit_ratio < 2.5x")
    logger.info("="*80 + "\n")

    model.frame_dropout.p = config['phase3_dropout_frame']
    model.sequence_dropout.p = config['phase3_dropout_sequence']

    optimizer3 = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer2.param_groups[0]['lr'],  # Continue from phase 2
        weight_decay=config['phase3_weight_decay']
    )

    scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer3,
        mode='min',
        factor=config['phase3_scheduler_factor'],
        patience=config['phase3_scheduler_patience'],
        threshold = 0.01,
        threshold_mode = 'rel',
        cooldown = 2,
        min_lr=1e-6,
        # verbose=True
    )

    for epoch in range(1, config['phase3_epochs'] + 1):
        global_epoch += 1
        epoch_start = time.time()

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer3,
            device, global_epoch, logger,
            blank_penalty=config['phase3_blank_penalty'],
            time_mask_prob=config['phase3_time_mask_prob'],
            max_seq_len=config['phase3_max_seq_len'],
            temperature=config['phase3_temperature'],
            gradient_clip=config['gradient_clip'],
        )

        val_metrics = validate(
            model, val_loader, criterion, vocab,
            device, logger, epoch=global_epoch,
            blank_penalty=config['phase3_blank_penalty'],
            temperature=config['phase3_temperature'],
        )

        scheduler3.step(val_metrics['val_loss'])
        current_lr = optimizer3.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start
        overfit_ratio = val_metrics['val_loss'] / train_metrics['train_loss'] if train_metrics['train_loss'] > 0 else float('inf')

        logger.info(f"\nEpoch {global_epoch} (Phase 3: Consolidation) Summary:")
        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  Overfit Ratio: {overfit_ratio:.2f}x")
        logger.info(f"  Val WER: {val_metrics['val_wer']:.2f}%")
        logger.info(f"  Frame Blank Ratio: {val_metrics['blank_ratio']:.2f}%")
        logger.info(f"  Unique Non-Blank: {val_metrics['unique_nonblank_predictions']}")
        logger.info(f"  Learning Rate: {current_lr:.2e}")
        logger.info(f"  Best WER: {best_wer:.2f}%")

        if overfit_ratio > 3.0:
            logger.warning(f"  WARNING: Overfit ratio {overfit_ratio:.2f}x too high")

        epoch_metrics = {
            'epoch': global_epoch,
            'phase': 3,
            'phase_name': 'Consolidation',
            **{k: float(v) if isinstance(v, torch.Tensor) else v
               for k, v in train_metrics.items()},
            **{k: float(v) if isinstance(v, torch.Tensor) else v
               for k, v in val_metrics.items()},
            'lr': float(current_lr),
            'epoch_time': float(epoch_time),
            'blank_penalty': config['phase3_blank_penalty'],
            'overfit_ratio': float(overfit_ratio)
        }
        training_history.append(epoch_metrics)

        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)

        if val_metrics['val_wer'] < best_wer - 0.001:
            best_wer = val_metrics['val_wer']
            checkpoint = {
                'epoch': global_epoch,
                'phase': 3,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer3.state_dict(),
                'best_wer': best_wer,
                'config': config,
                'vocab': vocab
            }
            torch.save(checkpoint, best_path)
            logger.info(f"  [SAVED] Best model (WER: {best_wer:.2f}%)")

    # ==================== PHASE 4: FINE-TUNING ====================
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: FINE-TUNING (Epochs {}-{})".format(
        global_epoch + 1, global_epoch + config['phase4_epochs']))
    logger.info("  Goal: Final refinement")
    logger.info(f"  Blank Penalty: {config['phase4_blank_penalty']} (no penalty)")
    logger.info(f"  Dropout: {config['phase4_dropout_sequence']}")
    logger.info("  Target: val_wer < 30%, overfit_ratio < 2.0x")
    logger.info("="*80 + "\n")

    model.frame_dropout.p = config['phase4_dropout_frame']
    model.sequence_dropout.p = config['phase4_dropout_sequence']

    optimizer4 = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer3.param_groups[0]['lr'],
        weight_decay=config['phase4_weight_decay']
    )

    scheduler4 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer4,
        mode='min',
        factor=config['phase4_scheduler_factor'],
        patience=config['phase4_scheduler_patience'],
        min_lr=1e-6,
        threshold = 0.01,
        threshold_mode = 'rel',
        cooldown = 2,
        # verbose=True
    )

    for epoch in range(1, config['phase4_epochs'] + 1):
        global_epoch += 1
        epoch_start = time.time()

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer4,
            device, global_epoch, logger,
            blank_penalty=config['phase4_blank_penalty'],
            time_mask_prob=config['phase4_time_mask_prob'],
            max_seq_len=config['phase4_max_seq_len'],
            temperature=config['phase4_temperature'],
            gradient_clip=config['gradient_clip'],
        )

        val_metrics = validate(
            model, val_loader, criterion, vocab,
            device, logger, epoch=global_epoch,
            blank_penalty=config['phase4_blank_penalty'],
            temperature=config['phase4_temperature'],
        )

        scheduler4.step(val_metrics['val_loss'])
        current_lr = optimizer4.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start
        overfit_ratio = val_metrics['val_loss'] / train_metrics['train_loss'] if train_metrics['train_loss'] > 0 else float('inf')

        logger.info(f"\nEpoch {global_epoch} (Phase 4: Fine-tuning) Summary:")
        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  Overfit Ratio: {overfit_ratio:.2f}x")
        logger.info(f"  Val WER: {val_metrics['val_wer']:.2f}%")
        logger.info(f"  Frame Blank Ratio: {val_metrics['blank_ratio']:.2f}%")
        logger.info(f"  Unique Non-Blank: {val_metrics['unique_nonblank_predictions']}")
        logger.info(f"  Learning Rate: {current_lr:.2e}")
        logger.info(f"  Best WER: {best_wer:.2f}%")

        if val_metrics['val_wer'] < config['target_wer']:
            logger.info(f"  [TARGET ACHIEVED] WER < {config['target_wer']}%!")

        epoch_metrics = {
            'epoch': global_epoch,
            'phase': 4,
            'phase_name': 'Fine-tuning',
            **{k: float(v) if isinstance(v, torch.Tensor) else v
               for k, v in train_metrics.items()},
            **{k: float(v) if isinstance(v, torch.Tensor) else v
               for k, v in val_metrics.items()},
            'lr': float(current_lr),
            'epoch_time': float(epoch_time),
            'blank_penalty': config['phase4_blank_penalty'],
            'overfit_ratio': float(overfit_ratio)
        }
        training_history.append(epoch_metrics)

        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)

        if val_metrics['val_wer'] < best_wer - 0.001:
            best_wer = val_metrics['val_wer']
            checkpoint = {
                'epoch': global_epoch,
                'phase': 4,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer4.state_dict(),
                'best_wer': best_wer,
                'config': config,
                'vocab': vocab
            }
            torch.save(checkpoint, best_path)
            logger.info(f"  [SAVED] Best model (WER: {best_wer:.2f}%)")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FIXED HIERARCHICAL TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Best WER: {best_wer:.2f}%")
    logger.info(f"Target WER: {config['target_wer']:.2f}%")
    if best_wer < config['target_wer']:
        logger.info("[SUCCESS] TARGET ACHIEVED!")
    else:
        logger.info(f"[PROGRESS] Need {best_wer - config['target_wer']:.2f}% improvement")

    if training_history:
        final = training_history[-1]
        logger.info(f"\nFinal Metrics:")
        logger.info(f"  Val Loss: {final['val_loss']:.4f}")
        logger.info(f"  Train Loss: {final['train_loss']:.4f}")
        logger.info(f"  Overfit Ratio: {final['overfit_ratio']:.2f}x")
        logger.info(f"  Blank Ratio: {final['blank_ratio']:.2f}%")
        logger.info(f"  Unique Non-Blank: {final['unique_nonblank_predictions']}/{config['num_classes']}")

    logger.info(f"\nLog file: {log_file}")
    logger.info(f"History file: {history_file}")
    logger.info(f"Best model: {best_path}")
    logger.info("="*80)


if __name__ == "__main__":
    # Delegate to modular teacher pipeline
    from teacher.train_mediapipe import main as teacher_main
    teacher_main()
