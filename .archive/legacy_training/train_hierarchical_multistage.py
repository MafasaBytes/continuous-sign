"""
Multi-Stage Hierarchical Training (4 Stages) - PRINCIPLED APPROACH

Core Principle: "Explore with Constraints"
- Force vocabulary exploration AND prevent overfitting SIMULTANEOUSLY from start
- NO learning rate schedulers (they lock model into bad minima)
- Aggressive blank bias from start (force vocabulary exploration)
- Moderate dropout from start (prevent memorization)
- Sequence clipping from start (constrain to useful information)

Stages:
1. Aggressive Exploration (30 epochs): Force vocab exploration with regularization
2. Consolidation (25 epochs): Consolidate knowledge, switch to full model
3. Generalization (25 epochs): Improve generalization, reduce overfitting
4. Fine-tuning (20 epochs): Final refinement

Target: <25% WER with stable val_loss and >800 unique predictions
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
import glob

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

# Import model from experimental version
from train_hierarchical_experimental import (
    TemporalConvolution,
    HierarchicalModel
)

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.CTCLoss,
                optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                logger: logging.Logger, stage: int = 2, blank_bias: float = 0.0,
                time_mask_prob: float = 0.0, max_seq_len: Optional[int] = None) -> Dict[str, float]:
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
        
        # Sequence length clipping (constrain to useful information)
        if max_seq_len is not None:
            # Clip sequences that are too long to prevent overfitting on noise
            clipped_lengths = torch.clamp(input_lengths, max=max_seq_len)
            # Also clip features if needed
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
                    unique_predictions_set.add(pred_idx)
                    if pred_idx != 0:
                        unique_nonblank_predictions_set.add(pred_idx)
                
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
    """Main training with 4-stage principled approach: Explore with Constraints."""
    
    # Configuration - 4 STAGES with principled "Explore with Constraints" approach
    config = {
        # Data
        'train_h5': 'data/features_cnn/train_features.h5',
        'dev_h5': 'data/features_cnn/dev_features.h5',
        'test_h5': 'data/features_cnn/test_features.h5',

        # Model
        'input_dim': 1024,
        'hidden_dim': 512,
        'use_temporal_conv': True,

        # Stage 1: Aggressive Exploration (30 epochs)
        # Goal: Force vocabulary exploration while preventing memorization
        # Target: >500 unique non-blank predictions (50% vocab coverage)
        'stage1_epochs': 30,
        'stage1_lr': 5e-4,  # Moderate LR, CONSTANT (no scheduler)
        'stage1_blank_bias': -8.0,  # Very aggressive - force non-blank predictions
        'stage1_dropout': 0.25,  # Moderate dropout from start - prevent memorization
        'stage1_time_mask_prob': 0.15,  # Moderate augmentation
        'stage1_weight_decay': 1e-4,  # Moderate regularization
        'stage1_use_stage': 1,  # Frame-level only (simpler model for exploration)
        'stage1_max_seq_len': 250,  # Clip from start - constrain to useful info

        # Stage 2: Consolidation (25 epochs)
        # Goal: Consolidate vocabulary knowledge, switch to full model
        # Target: >700 unique predictions (70% vocab), val_loss/train_loss < 2.0
        'stage2_epochs': 25,
        'stage2_lr': 3e-4,  # Slightly reduced, CONSTANT
        'stage2_blank_bias': -5.0,  # Still aggressive
        'stage2_dropout': 0.30,  # Moderate-high dropout
        'stage2_time_mask_prob': 0.10,  # Light augmentation
        'stage2_weight_decay': 1e-4,
        'stage2_use_stage': 2,  # Full hierarchical model
        'stage2_max_seq_len': 250,  # Keep clipping

        # Stage 3: Generalization (25 epochs)
        # Goal: Improve generalization, reduce overfitting
        # Target: val_loss/train_loss < 1.5, WER < 50%
        'stage3_epochs': 25,
        'stage3_lr': 1e-4,  # Reduced, CONSTANT
        'stage3_blank_bias': -3.0,  # Moderate penalty
        'stage3_dropout': 0.35,  # High dropout
        'stage3_time_mask_prob': 0.05,  # Very light augmentation
        'stage3_weight_decay': 1e-4,
        'stage3_use_stage': 2,  # Full model
        'stage3_max_seq_len': 250,  # Keep clipping

        # Stage 4: Fine-tuning (20 epochs)
        # Goal: Final refinement
        # Target: WER < 25%, val_loss/train_loss < 1.3
        'stage4_epochs': 20,
        'stage4_lr': 5e-5,  # Low LR, CONSTANT
        'stage4_blank_bias': -1.0,  # Weak penalty
        'stage4_dropout': 0.40,  # High dropout
        'stage4_time_mask_prob': 0.0,  # No augmentation
        'stage4_weight_decay': 1e-4,
        'stage4_use_stage': 2,  # Full model
        'stage4_max_seq_len': 250,  # Keep clipping

        # Common
        'batch_size': 16,
        'gradient_clip': 5.0,
        'seed': 42,
        'checkpoint_dir': 'checkpoints/hierarchical_multistage',
        'log_dir': 'logs/hierarchical_multistage',
        'experiment_name': 'hierarchical_4stage_principled',
        'target_wer': 25.0  # Target
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
    logger.info("MULTI-STAGE HIERARCHICAL TRAINING (4 STAGES) - PRINCIPLED APPROACH")
    logger.info("Strategy: EXPLORE WITH CONSTRAINTS")
    logger.info("  - Force vocabulary exploration AND prevent overfitting from START")
    logger.info("  - NO schedulers (constant LR per stage)")
    logger.info("  - Aggressive blank bias from start (-8.0)")
    logger.info("  - Moderate dropout from start (0.25+)")
    logger.info("  - Sequence clipping from start (250)")
    logger.info("Target: <25% WER with stable val_loss and >800 unique predictions")
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
        dropout_stage1=0.1,  # Will be updated dynamically
        dropout_stage2=0.5,  # Will be updated dynamically
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
    
    # Define all 4 stages - Principled "Explore with Constraints" approach
    stages = [
        {
            'name': 'Stage 1: Aggressive Exploration',
            'epochs': config['stage1_epochs'],
            'lr': config['stage1_lr'],
            'blank_bias': config['stage1_blank_bias'],
            'dropout': config['stage1_dropout'],
            'time_mask_prob': config['stage1_time_mask_prob'],
            'weight_decay': config['stage1_weight_decay'],
            'use_stage': config['stage1_use_stage'],
            'max_seq_len': config.get('stage1_max_seq_len'),
            'description': 'Force vocab exploration (blank_bias=-8.0) with regularization (dropout=0.25, seq_len=250)',
            'target_metrics': 'Target: >500 unique non-blank predictions (50% vocab coverage)'
        },
        {
            'name': 'Stage 2: Consolidation',
            'epochs': config['stage2_epochs'],
            'lr': config['stage2_lr'],
            'blank_bias': config['stage2_blank_bias'],
            'dropout': config['stage2_dropout'],
            'time_mask_prob': config['stage2_time_mask_prob'],
            'weight_decay': config['stage2_weight_decay'],
            'use_stage': config['stage2_use_stage'],
            'max_seq_len': config.get('stage2_max_seq_len'),
            'description': 'Consolidate vocabulary, switch to full model (blank_bias=-5.0, dropout=0.30)',
            'target_metrics': 'Target: >700 unique predictions (70% vocab), val_loss/train_loss < 2.0'
        },
        {
            'name': 'Stage 3: Generalization',
            'epochs': config['stage3_epochs'],
            'lr': config['stage3_lr'],
            'blank_bias': config['stage3_blank_bias'],
            'dropout': config['stage3_dropout'],
            'time_mask_prob': config['stage3_time_mask_prob'],
            'weight_decay': config['stage3_weight_decay'],
            'use_stage': config['stage3_use_stage'],
            'max_seq_len': config.get('stage3_max_seq_len'),
            'description': 'Improve generalization, reduce overfitting (blank_bias=-3.0, dropout=0.35)',
            'target_metrics': 'Target: val_loss/train_loss < 1.5, WER < 50%'
        },
        {
            'name': 'Stage 4: Fine-tuning',
            'epochs': config['stage4_epochs'],
            'lr': config['stage4_lr'],
            'blank_bias': config['stage4_blank_bias'],
            'dropout': config['stage4_dropout'],
            'time_mask_prob': config['stage4_time_mask_prob'],
            'weight_decay': config['stage4_weight_decay'],
            'use_stage': config['stage4_use_stage'],
            'max_seq_len': config.get('stage4_max_seq_len'),
            'description': 'Final refinement (blank_bias=-1.0, dropout=0.40)',
            'target_metrics': 'Target: WER < 25%, val_loss/train_loss < 1.3'
        }
    ]
    
    # Update model dropout dynamically
    def update_dropout(dropout_frame, dropout_sequence):
        """Update dropout rates dynamically."""
        model.frame_dropout.p = dropout_frame
        model.sequence_dropout.p = dropout_sequence
    
    # Train each stage
    for stage_num, stage_config in enumerate(stages, 1):
        stage_name = stage_config['name']
        stage_epochs = stage_config['epochs']
        stage_lr = stage_config['lr']
        stage_blank_bias = stage_config['blank_bias']
        stage_dropout = stage_config['dropout']
        stage_time_mask = stage_config['time_mask_prob']
        stage_weight_decay = stage_config['weight_decay']
        stage_use_stage = stage_config['use_stage']
        stage_max_seq_len = stage_config.get('max_seq_len')
        
        logger.info("\n" + "="*80)
        logger.info(f"{stage_name}")
        logger.info(f"Epochs: {global_epoch+1}-{global_epoch+stage_epochs}")
        logger.info(f"Description: {stage_config['description']}")
        logger.info(f"Target Metrics: {stage_config['target_metrics']}")
        logger.info(f"  Learning Rate: {stage_lr} (CONSTANT - no scheduler)")
        logger.info(f"  Blank Bias: {stage_blank_bias}")
        logger.info(f"  Dropout: {stage_dropout}")
        logger.info(f"  Time Mask Prob: {stage_time_mask}")
        logger.info(f"  Max Seq Len: {stage_max_seq_len}")
        logger.info(f"  Use Stage: {stage_use_stage} (1=frame-only, 2=full)")
        logger.info("="*80 + "\n")
        
        # Update dropout based on stage
        if stage_use_stage == 1:
            # Frame-level only: update frame dropout, keep sequence dropout high (not used)
            model.frame_dropout.p = stage_dropout
            model.sequence_dropout.p = 0.5  # Not used in stage 1, but set anyway
        else:
            # Full model: update both, but sequence dropout is more important
            model.frame_dropout.p = max(0.2, stage_dropout * 0.6)  # Keep frame dropout moderate
            model.sequence_dropout.p = stage_dropout  # This is the main regularization
        
        # Optimizer - NO SCHEDULER (constant LR per stage)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=stage_lr,
            weight_decay=stage_weight_decay
        )
        
        for epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            epoch_start = time.time()
            
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer,
                device, global_epoch, logger, stage=stage_use_stage,
                blank_bias=stage_blank_bias,
                time_mask_prob=stage_time_mask,
                max_seq_len=stage_max_seq_len
            )
            
            val_metrics = validate(
                model, val_loader, criterion, vocab,
                device, logger, stage=stage_use_stage
            )

            # No scheduler - constant LR per stage
            current_lr = stage_lr
            
            epoch_time = time.time() - epoch_start
            
            # Calculate overfitting ratio
            overfit_ratio = val_metrics['val_loss'] / train_metrics['train_loss'] if train_metrics['train_loss'] > 0 else float('inf')
            
            logger.info(f"\nEpoch {global_epoch} (Stage {stage_num}: {stage_name}) Summary:")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Overfit Ratio: {overfit_ratio:.2f}x (val/train)")
            logger.info(f"  Val WER: {val_metrics['val_wer']:.2f}%")
            logger.info(f"  Frame Blank Ratio: {val_metrics['blank_ratio']:.2f}%")
            logger.info(f"  Unique Non-Blank Predictions: {val_metrics['unique_nonblank_predictions']} / {config['num_classes']} ({val_metrics['unique_nonblank_predictions']/config['num_classes']*100:.1f}%)")
            logger.info(f"  Frame Unique Predictions: {val_metrics['frame_unique_predictions']}")
            logger.info(f"  Learning Rate: {current_lr:.2e}")
            logger.info(f"  Epoch Time: {epoch_time:.2f}s")
            logger.info(f"  Best WER: {best_wer:.1f}%")
            
            # Stage-specific target monitoring
            vocab_coverage = val_metrics['unique_nonblank_predictions'] / config['num_classes'] * 100

            # Stage 1: Focus on vocabulary exploration
            if stage_num == 1:
                if val_metrics['unique_nonblank_predictions'] < 500:
                    logger.warning(f"  WARNING: Stage 1 vocabulary exploration below target ({val_metrics['unique_nonblank_predictions']}/500)")
                else:
                    logger.info(f"  [GOOD] Stage 1 vocabulary exploration on track ({val_metrics['unique_nonblank_predictions']}/500)")

            # Stage 2: Check vocabulary consolidation and overfitting
            elif stage_num == 2:
                if val_metrics['unique_nonblank_predictions'] < 700:
                    logger.warning(f"  WARNING: Stage 2 vocabulary below target ({val_metrics['unique_nonblank_predictions']}/700)")
                else:
                    logger.info(f"  [GOOD] Stage 2 vocabulary on track ({val_metrics['unique_nonblank_predictions']}/700)")
                if overfit_ratio > 2.0:
                    logger.warning(f"  WARNING: Stage 2 overfitting ratio above target ({overfit_ratio:.2f}x vs 2.0x)")
                else:
                    logger.info(f"  [GOOD] Stage 2 overfitting ratio acceptable ({overfit_ratio:.2f}x)")

            # Stage 3: Check generalization
            elif stage_num == 3:
                if overfit_ratio > 1.5:
                    logger.warning(f"  WARNING: Stage 3 overfitting ratio above target ({overfit_ratio:.2f}x vs 1.5x)")
                else:
                    logger.info(f"  [GOOD] Stage 3 overfitting ratio on track ({overfit_ratio:.2f}x)")
                if val_metrics['val_wer'] > 50:
                    logger.warning(f"  WARNING: Stage 3 WER above target ({val_metrics['val_wer']:.2f}% vs 50%)")
                else:
                    logger.info(f"  [GOOD] Stage 3 WER on track ({val_metrics['val_wer']:.2f}% vs 50%)")

            # Stage 4: Check final targets
            elif stage_num == 4:
                if overfit_ratio > 1.3:
                    logger.warning(f"  WARNING: Stage 4 overfitting ratio above target ({overfit_ratio:.2f}x vs 1.3x)")
                else:
                    logger.info(f"  [GOOD] Stage 4 overfitting ratio on track ({overfit_ratio:.2f}x)")
                if val_metrics['val_wer'] > 25:
                    logger.warning(f"  WARNING: Stage 4 WER above target ({val_metrics['val_wer']:.2f}% vs 25%)")
                else:
                    logger.info(f"  [EXCELLENT] Stage 4 WER achieved target! ({val_metrics['val_wer']:.2f}% vs 25%)")

            # General overfitting detection
            if overfit_ratio > 2.5:
                logger.error(f"  CRITICAL: SEVERE OVERFITTING: Ratio {overfit_ratio:.2f}x - Model may have collapsed")
            
            # Record everything
            epoch_metrics = {
                'epoch': global_epoch,
                'stage': stage_num,
                'stage_name': stage_name,
                **{k: float(v) if isinstance(v, torch.Tensor) else v 
                   for k, v in train_metrics.items()},
                **{k: float(v) if isinstance(v, torch.Tensor) else v 
                   for k, v in val_metrics.items()},
                'lr': float(current_lr),
                'epoch_time': float(epoch_time),
                'blank_bias': stage_blank_bias,
                'dropout': stage_dropout
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
                    'stage': stage_num,
                    'stage_name': stage_name,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_wer': best_wer,
                    'config': config,
                    'vocab': vocab
                }
                best_path = os.path.join(config['checkpoint_dir'],
                                       f'{config["experiment_name"]}_best.pt')
                torch.save(checkpoint, best_path)
                logger.info(f"  [OK] Saved best model (WER: {best_wer:.2f}%)")
            
            # Check target
            if val_metrics['val_wer'] < config['target_wer']:
                logger.info(f"  [TARGET] TARGET ACHIEVED! WER < {config['target_wer']}%")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE - 4-STAGE PRINCIPLED APPROACH")
    logger.info("="*80)
    logger.info(f"Best WER: {best_wer:.2f}%")
    logger.info(f"Target WER: {config['target_wer']:.2f}%")
    if best_wer < config['target_wer']:
        logger.info(f"[SUCCESS] TARGET ACHIEVED!")
    else:
        logger.info(f"[FAIL] Target not achieved (need {best_wer - config['target_wer']:.2f}% improvement)")

    # Get final metrics from last epoch
    if training_history:
        final_metrics = training_history[-1]
        logger.info(f"\nFinal Metrics:")
        logger.info(f"  Val Loss: {final_metrics['val_loss']:.4f}")
        logger.info(f"  Train Loss: {final_metrics['train_loss']:.4f}")
        logger.info(f"  Overfit Ratio: {final_metrics['val_loss']/final_metrics['train_loss']:.2f}x")
        logger.info(f"  Unique Non-Blank Predictions: {final_metrics['unique_nonblank_predictions']}/{config['num_classes']}")
        logger.info(f"  Vocabulary Coverage: {final_metrics['unique_nonblank_predictions']/config['num_classes']*100:.1f}%")

    logger.info(f"\nLog file: {log_file}")
    logger.info(f"History file: {history_file}")
    logger.info(f"Best model: {best_path}")
    logger.info("\nKEY INSIGHTS FROM THIS APPROACH:")
    logger.info("1. NO SCHEDULER: Constant LR per stage prevents locking into bad minima")
    logger.info("2. AGGRESSIVE BLANK BIAS (-8.0): Forces vocabulary exploration from start")
    logger.info("3. MODERATE DROPOUT (0.25+): Prevents memorization from start")
    logger.info("4. SEQUENCE CLIPPING (250): Constrains to useful info from start")
    logger.info("\nNEXT STEPS:")
    logger.info("1. Check if vocabulary exploration happened in Stage 1 (target: >500 predictions)")
    logger.info("2. Check if val_loss stayed stable (not exploding like before)")
    logger.info("3. Check overfitting ratio progression across stages")
    logger.info("4. If successful, this validates the 'Explore with Constraints' principle")
    logger.info("="*80)


if __name__ == "__main__":
    main()

