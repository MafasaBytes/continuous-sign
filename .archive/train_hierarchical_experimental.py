"""
Experimental Hierarchical Training with Data-Driven Approach

Based on HONEST_ANALYSIS.md recommendations:
1. Start with 2-stage (simpler than 3-stage)
2. Search hyperparameters (don't guess)
3. Record everything
4. Iterate based on empirical results

This is a hypothesis to test, not a proven solution.
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
from collections import defaultdict

# Import dataset and utilities
from experiments.cnn_feature_dataset import (
    CNNFeatureDataset,
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


class TemporalConvolution(nn.Module):
    """
    Temporal convolution to capture short-term motion patterns.
    Hypothesis: CNN features lack temporal info, this compensates.
    """
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        # Multi-scale temporal convolutions
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)
        
        # Combine multi-scale features
        self.combine = nn.Linear(hidden_dim * 3, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, T, input_dim], Returns: [B, T, hidden_dim]"""
        x_t = x.transpose(1, 2)  # [B, D, T]
        
        conv1_out = F.relu(self.conv1(x_t)).transpose(1, 2)  # [B, T, hidden_dim]
        conv2_out = F.relu(self.conv2(x_t)).transpose(1, 2)
        conv3_out = F.relu(self.conv3(x_t)).transpose(1, 2)
        
        combined = torch.cat([conv1_out, conv2_out, conv3_out], dim=-1)
        output = self.combine(combined)
        output = self.norm(output)
        
        return output


class HierarchicalModel(nn.Module):
    """
    Simplified 2-stage hierarchical model.
    
    Stage 1: Frame-level (exploration)
    Stage 2: Sequence-level (refinement)
    
    Hypothesis: This structure helps with CTC collapse and vocabulary exploration.
    """
    
    def __init__(self,
                 input_dim: int = 1024,
                 hidden_dim: int = 512,
                 num_classes: int = 966,
                 dropout_stage1: float = 0.2,
                 dropout_stage2: float = 0.5,
                 use_temporal_conv: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_temporal_conv = use_temporal_conv
        
        # Temporal convolution (hypothesis: helps with motion)
        if use_temporal_conv:
            self.temporal_conv = TemporalConvolution(input_dim, hidden_dim)
            lstm_input_dim = hidden_dim
        else:
            lstm_input_dim = input_dim
        
        # Stage 1: Frame-level BiLSTM
        self.frame_lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.frame_dropout = nn.Dropout(dropout_stage1)
        
        # Stage 2: Sequence-level BiLSTM
        self.sequence_lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.sequence_dropout = nn.Dropout(dropout_stage2)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, num_classes)
        
        self._initialize_weights()
    
    def forward(self, features: torch.Tensor, lengths: torch.Tensor, 
                stage: int = 2) -> torch.Tensor:
        """
        Forward pass with stage control.
        
        Args:
            features: [B, T, 1024]
            lengths: [B]
            stage: 1 (frame-level only) or 2 (full)
        
        Returns:
            [T, B, C] log probabilities
        """
        # Temporal convolution (if enabled)
        if self.use_temporal_conv:
            x = self.temporal_conv(features)  # [B, T, hidden_dim]
        else:
            x = features  # [B, T, input_dim]
        
        # Pack for LSTM
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Stage 1: Frame-level BiLSTM
        frame_out, _ = self.frame_lstm(packed)
        frame_out, _ = pad_packed_sequence(frame_out, batch_first=True)
        frame_out = self.frame_dropout(frame_out)  # [B, T, hidden_dim*2]
        
        if stage == 1:
            # Early stage: use frame-level output
            logits = self.output_projection(frame_out)
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs.transpose(0, 1)
        
        # Stage 2: Sequence-level BiLSTM
        packed_seq = pack_padded_sequence(
            frame_out, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        sequence_out, _ = self.sequence_lstm(packed_seq)
        sequence_out, _ = pad_packed_sequence(sequence_out, batch_first=True)
        sequence_out = self.sequence_dropout(sequence_out)  # [B, T, hidden_dim*2]
        
        # Final output
        logits = self.output_projection(sequence_out)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs.transpose(0, 1)  # [T, B, C]
    
    def _initialize_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name and 'lstm' in name:
                param.data.zero_()
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)  # Forget gate
            elif 'output_projection.weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'output_projection.bias' in name:
                param.data.zero_()
                # Blank bias will be set dynamically during training


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.CTCLoss,
                optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                logger: logging.Logger, stage: int = 2, blank_bias: float = 0.0,
                time_mask_prob: float = 0.0) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_samples = 0
    total_gradient_norm = 0
    batch_times = []
    
    # Set blank bias dynamically
    with torch.no_grad():
        model.output_projection.bias[0] = blank_bias
    
    for batch_idx, batch in enumerate(dataloader):
        start_time = time.time()
        
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        
        # Time masking augmentation
        if time_mask_prob > 0:
            B, T, D = features.shape
            for i in range(B):
                seq_len = int(input_lengths[i].item())
                if seq_len > 0 and random.random() < time_mask_prob:
                    width = min(12, max(1, seq_len // 10))
                    start = random.randint(0, max(0, seq_len - width))
                    features[i, start:start+width, :] = 0.0
        
        # Forward pass
        log_probs = model(features, input_lengths, stage=stage)
        
        # CTC loss
        loss = criterion(log_probs, labels, input_lengths, target_lengths)
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)
        total_gradient_norm += grad_norm
        batch_times.append(time.time() - start_time)
        
        if batch_idx % 50 == 0:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            logger.info(f"Epoch {epoch} Batch {batch_idx}/{len(dataloader)}: "
                       f"Loss={loss.item():.4f}, AvgLoss={avg_loss:.4f}, "
                       f"GradNorm={grad_norm:.4f}, Stage={stage}")
    
    return {
        'train_loss': total_loss / total_samples if total_samples > 0 else 0,
        'gradient_norm': total_gradient_norm / len(dataloader),
        'batch_time': np.mean(batch_times),
        'total_time': sum(batch_times)
    }


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.CTCLoss,
             vocab: Dict[str, int], device: torch.device, logger: logging.Logger,
             stage: int = 2) -> Dict[str, float]:
    """Validate model with comprehensive metrics."""
    model.eval()
    
    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    # Tracking metrics
    blank_predictions = 0
    total_predictions = 0
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
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            target_lengths = batch['target_lengths'].to(device)
            
            # Forward pass
            log_probs = model(features, input_lengths, stage=stage)
            
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
                    if pred_idx == 0:
                        blank_predictions += 1
                    unique_predictions_set.add(pred_idx)
                    if pred_idx != 0:
                        unique_nonblank_predictions_set.add(pred_idx)
                    total_predictions += 1
                
                # CTC constraint
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
    """Main training with 2-stage approach and hyperparameter search."""
    
    # Configuration - DATA-DRIVEN: We'll search these
    config = {
        # Data
        'train_h5': 'data/features_cnn/train_features.h5',
        'dev_h5': 'data/features_cnn/dev_features.h5',
        'test_h5': 'data/features_cnn/test_features.h5',
        
        # Model
        'input_dim': 1024,
        'hidden_dim': 512,
        'use_temporal_conv': True,  # Hypothesis to test
        
        # Stage 1: Exploration
        'stage1_epochs': 20,  # Start conservative
        'stage1_lr': 1e-3,
        'stage1_blank_bias': -3.0,  # Will search: [-5.0, -4.0, -3.0, -2.0, -1.0]
        'stage1_dropout': 0.2,
        'stage1_time_mask_prob': 0.15,
        'stage1_weight_decay': 1e-4,
        
        # Stage 2: Refinement
        'stage2_epochs': 30,  # Start conservative
        'stage2_lr': 5e-4,
        'stage2_blank_bias': 0.0,  # Neutral
        'stage2_dropout': 0.5,
        'stage2_time_mask_prob': 0.0,  # No augmentation
        'stage2_weight_decay': 1e-4,
        
        # Common
        'batch_size': 16,
        'gradient_clip': 5.0,
        'seed': 42,
        'checkpoint_dir': 'checkpoints/hierarchical_experimental',
        'log_dir': 'logs/hierarchical_experimental',
        'experiment_name': 'hierarchical_2stage_v1'
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
    logger.info("EXPERIMENTAL HIERARCHICAL TRAINING")
    logger.info("Based on HONEST_ANALYSIS.md - This is a hypothesis to test")
    logger.info("="*80)
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
    logger.info("Creating hierarchical model...")
    model = HierarchicalModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        dropout_stage1=config['stage1_dropout'],
        dropout_stage2=config['stage2_dropout'],
        use_temporal_conv=config['use_temporal_conv']
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Model size: {num_params * 4 / 1024**2:.2f} MB (FP32)")
    
    # Loss function
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Training history for analysis
    training_history = []
    best_wer = float('inf')
    global_epoch = 0
    
    # STAGE 1: Exploration
    logger.info("\n" + "="*80)
    logger.info("STAGE 1: EXPLORATION (Epochs 1-{})".format(config['stage1_epochs']))
    logger.info(f"  Learning Rate: {config['stage1_lr']}")
    logger.info(f"  Blank Bias: {config['stage1_blank_bias']} (HYPOTHESIS: discourages blank)")
    logger.info(f"  Dropout: {config['stage1_dropout']}")
    logger.info(f"  Time Mask Prob: {config['stage1_time_mask_prob']}")
    logger.info("="*80 + "\n")
    
    optimizer1 = torch.optim.AdamW(
        model.parameters(),
        lr=config['stage1_lr'],
        weight_decay=config['stage1_weight_decay']
    )
    
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    for epoch in range(1, config['stage1_epochs'] + 1):
        global_epoch += 1
        epoch_start = time.time()
        
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer1,
            device, global_epoch, logger, stage=1,
            blank_bias=config['stage1_blank_bias'],
            time_mask_prob=config['stage1_time_mask_prob']
        )
        
        val_metrics = validate(
            model, val_loader, criterion, vocab,
            device, logger, stage=1
        )
        
        scheduler1.step(val_metrics['val_loss'])
        current_lr = optimizer1.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        logger.info(f"\nEpoch {global_epoch} (Stage 1) Summary:")
        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  Val WER: {val_metrics['val_wer']:.2f}%")
        logger.info(f"  Frame Blank Ratio: {val_metrics['blank_ratio']:.2f}%")
        logger.info(f"  Unique Non-Blank Predictions: {val_metrics['unique_nonblank_predictions']}")
        logger.info(f"  Frame Unique Predictions: {val_metrics['frame_unique_predictions']}")
        logger.info(f"  Learning Rate: {current_lr:.2e}")
        logger.info(f"  Epoch Time: {epoch_time:.2f}s")
        
        # Record everything
        epoch_metrics = {
            'epoch': global_epoch,
            'stage': 1,
            **{k: float(v) if isinstance(v, torch.Tensor) else v 
               for k, v in train_metrics.items()},
            **{k: float(v) if isinstance(v, torch.Tensor) else v 
               for k, v in val_metrics.items()},
            'lr': float(current_lr),
            'epoch_time': float(epoch_time),
            'blank_bias': config['stage1_blank_bias']
        }
        training_history.append(epoch_metrics)
        
        # Save history
        history_file = os.path.join(config['log_dir'], 
                                   f'{config["experiment_name"]}_history_{timestamp}.json')
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Check improvement
        if val_metrics['val_wer'] < best_wer - 0.001:
            best_wer = val_metrics['val_wer']
            checkpoint = {
                'epoch': global_epoch,
                'stage': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'best_wer': best_wer,
                'config': config,
                'vocab': vocab
            }
            best_path = os.path.join(config['checkpoint_dir'], 
                                   f'{config["experiment_name"]}_best_stage1.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"  Saved best model (WER: {best_wer:.2f}%)")
    
    # STAGE 2: Refinement
    logger.info("\n" + "="*80)
    logger.info("STAGE 2: REFINEMENT (Epochs {}-{})".format(
        global_epoch + 1, global_epoch + config['stage2_epochs']))
    logger.info(f"  Learning Rate: {config['stage2_lr']}")
    logger.info(f"  Blank Bias: {config['stage2_blank_bias']} (neutral)")
    logger.info(f"  Dropout: {config['stage2_dropout']} (higher for regularization)")
    logger.info(f"  Time Mask Prob: {config['stage2_time_mask_prob']} (none)")
    logger.info("="*80 + "\n")
    
    optimizer2 = torch.optim.AdamW(
        model.parameters(),
        lr=config['stage2_lr'],
        weight_decay=config['stage2_weight_decay']
    )
    
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    for epoch in range(1, config['stage2_epochs'] + 1):
        global_epoch += 1
        epoch_start = time.time()
        
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer2,
            device, global_epoch, logger, stage=2,
            blank_bias=config['stage2_blank_bias'],
            time_mask_prob=config['stage2_time_mask_prob']
        )
        
        val_metrics = validate(
            model, val_loader, criterion, vocab,
            device, logger, stage=2
        )
        
        scheduler2.step(val_metrics['val_loss'])
        current_lr = optimizer2.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        logger.info(f"\nEpoch {global_epoch} (Stage 2) Summary:")
        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  Val WER: {val_metrics['val_wer']:.2f}%")
        logger.info(f"  Frame Blank Ratio: {val_metrics['blank_ratio']:.2f}%")
        logger.info(f"  Unique Non-Blank Predictions: {val_metrics['unique_nonblank_predictions']}")
        logger.info(f"  Frame Unique Predictions: {val_metrics['frame_unique_predictions']}")
        logger.info(f"  Learning Rate: {current_lr:.2e}")
        logger.info(f"  Epoch Time: {epoch_time:.2f}s")
        
        # Record everything
        epoch_metrics = {
            'epoch': global_epoch,
            'stage': 2,
            **{k: float(v) if isinstance(v, torch.Tensor) else v 
               for k, v in train_metrics.items()},
            **{k: float(v) if isinstance(v, torch.Tensor) else v 
               for k, v in val_metrics.items()},
            'lr': float(current_lr),
            'epoch_time': float(epoch_time),
            'blank_bias': config['stage2_blank_bias']
        }
        training_history.append(epoch_metrics)
        
        # Save history
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Check improvement
        if val_metrics['val_wer'] < best_wer - 0.001:
            best_wer = val_metrics['val_wer']
            checkpoint = {
                'epoch': global_epoch,
                'stage': 2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
                'best_wer': best_wer,
                'config': config,
                'vocab': vocab
            }
            best_path = os.path.join(config['checkpoint_dir'], 
                                   f'{config["experiment_name"]}_best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"  Saved best model (WER: {best_wer:.2f}%)")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"Best WER: {best_wer:.2f}%")
    logger.info(f"Log file: {log_file}")
    logger.info(f"History file: {history_file}")
    logger.info(f"Best model: {best_path}")
    logger.info("\nNEXT STEPS:")
    logger.info("1. Analyze history JSON to see what actually happened")
    logger.info("2. Compare with baseline (~95-97% WER)")
    logger.info("3. If better than ~85% WER, refine hyperparameters")
    logger.info("4. If not, try different blank bias values or architecture")
    logger.info("="*80)


if __name__ == "__main__":
    main()

