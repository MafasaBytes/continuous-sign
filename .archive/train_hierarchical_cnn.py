"""
Hierarchical Multi-Stage Training for Sign Language Recognition

Addresses CTC collapse and vocabulary exploration through:
1. Hierarchical temporal modeling (frame → segment → sequence)
2. Multi-stage curriculum learning
3. Progressive regularization
4. Temporal convolution to compensate for CNN's lack of motion info
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


class TemporalConvolution(nn.Module):
    """
    Temporal convolution to capture short-term motion patterns.
    Compensates for CNN features' lack of temporal information.
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
        """
        Args:
            x: [B, T, input_dim]
        Returns:
            [B, T, hidden_dim]
        """
        # Transpose for Conv1d: [B, T, D] -> [B, D, T]
        x_t = x.transpose(1, 2)
        
        # Multi-scale convolutions
        conv1_out = F.relu(self.conv1(x_t))  # [B, hidden_dim, T]
        conv2_out = F.relu(self.conv2(x_t))
        conv3_out = F.relu(self.conv3(x_t))
        
        # Transpose back: [B, hidden_dim, T] -> [B, T, hidden_dim]
        conv1_out = conv1_out.transpose(1, 2)
        conv2_out = conv2_out.transpose(1, 2)
        conv3_out = conv3_out.transpose(1, 2)
        
        # Concatenate and combine
        combined = torch.cat([conv1_out, conv2_out, conv3_out], dim=-1)
        output = self.combine(combined)
        output = self.norm(output)
        
        return output


class SegmentPooling(nn.Module):
    """
    Temporal pooling to group frames into sign segments.
    Reduces sequence length for higher-level modeling.
    """
    def __init__(self, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.AvgPool1d(kernel_size, stride, padding=kernel_size//2)
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D]
            lengths: [B] original lengths
        Returns:
            pooled: [B, T', D]
            new_lengths: [B] new lengths
        """
        # Transpose for pooling: [B, T, D] -> [B, D, T]
        x_t = x.transpose(1, 2)
        
        # Pool
        pooled_t = self.pool(x_t)  # [B, D, T']
        
        # Transpose back: [B, D, T'] -> [B, T', D]
        pooled = pooled_t.transpose(1, 2)
        
        # Update lengths (approximate)
        new_lengths = (lengths.float() / self.stride).long() + 1
        new_lengths = torch.clamp(new_lengths, max=pooled.size(1))
        
        return pooled, new_lengths


class AttentionModule(nn.Module):
    """
    Attention mechanism to focus on important segments.
    """
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, T_q, D]
            key: [B, T_k, D]
            value: [B, T_k, D]
        Returns:
            [B, T_q, D]
        """
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(scores, dim=-1)
        
        # Weighted sum
        output = torch.bmm(attn, V)
        return output


class HierarchicalSignLanguageModel(nn.Module):
    """
    Hierarchical model for sign language recognition.
    
    Architecture:
    1. Temporal Convolution (short-term patterns)
    2. Frame-level BiLSTM (local temporal)
    3. Segment Pooling (frame → segment)
    4. Segment-level BiLSTM (sign-level patterns)
    5. Attention (focus on important segments)
    6. Sequence-level BiLSTM (sentence-level)
    7. Output projection
    """
    
    def __init__(self,
                 input_dim: int = 1024,
                 hidden_dim: int = 512,
                 num_classes: int = 966,
                 dropout_stage1: float = 0.2,
                 dropout_stage2: float = 0.4,
                 dropout_stage3: float = 0.6,
                 use_segment_pooling: bool = True,
                 use_attention: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_segment_pooling = use_segment_pooling
        self.use_attention = use_attention
        
        # Stage 1: Temporal Convolution (compensate for CNN's lack of temporal info)
        self.temporal_conv = TemporalConvolution(input_dim, hidden_dim)
        
        # Stage 1: Frame-level BiLSTM
        self.frame_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.frame_dropout = nn.Dropout(dropout_stage1)
        
        # Stage 2: Segment Pooling
        if use_segment_pooling:
            self.segment_pool = SegmentPooling(kernel_size=5, stride=3)
        
        # Stage 2: Segment-level BiLSTM
        self.segment_lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # Bidirectional input
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.segment_dropout = nn.Dropout(dropout_stage2)
        
        # Stage 2: Attention
        if use_attention:
            self.attention = AttentionModule(hidden_dim * 2)
        
        # Stage 3: Sequence-level BiLSTM
        self.sequence_lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.sequence_dropout = nn.Dropout(dropout_stage3)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, features: torch.Tensor, lengths: torch.Tensor, 
                stage: int = 3) -> torch.Tensor:
        """
        Forward pass with stage control.
        
        Args:
            features: [B, T, 1024] CNN features
            lengths: [B] sequence lengths
            stage: Training stage (1, 2, or 3)
        
        Returns:
            [T, B, C] log probabilities for CTC
        """
        batch_size, max_len, _ = features.shape
        
        # Stage 1: Temporal Convolution + Frame-level BiLSTM
        x = self.temporal_conv(features)  # [B, T, hidden_dim]
        
        # Pack for LSTM
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Frame-level BiLSTM
        frame_out, _ = self.frame_lstm(packed)
        frame_out, _ = pad_packed_sequence(frame_out, batch_first=True)
        frame_out = self.frame_dropout(frame_out)  # [B, T, hidden_dim*2]
        
        if stage == 1:
            # Early stage: use frame-level output
            logits = self.output_projection(frame_out)  # [B, T, num_classes]
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs.transpose(0, 1)  # [T, B, C]
        
        # Stage 2: Segment Pooling + Segment-level BiLSTM
        if self.use_segment_pooling:
            segment_features, segment_lengths = self.segment_pool(frame_out, lengths)
        else:
            segment_features = frame_out
            segment_lengths = lengths
        
        # Pack for segment LSTM
        packed_seg = pack_padded_sequence(
            segment_features, segment_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        segment_out, _ = self.segment_lstm(packed_seg)
        segment_out, _ = pad_packed_sequence(segment_out, batch_first=True)
        segment_out = self.segment_dropout(segment_out)  # [B, T', hidden_dim*2]
        
        # Attention (segment attends to frames)
        if self.use_attention and stage >= 2:
            # Interpolate frame_out to match segment_out length for attention
            # For simplicity, use segment_out as both query and key
            segment_out = self.attention(segment_out, segment_out, segment_out)
        
        if stage == 2:
            # Mid stage: use segment-level output
            logits = self.output_projection(segment_out)
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs.transpose(0, 1)  # [T', B, C]
        
        # Stage 3: Sequence-level BiLSTM
        packed_seq = pack_padded_sequence(
            segment_out, segment_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        sequence_out, _ = self.sequence_lstm(packed_seq)
        sequence_out, _ = pad_packed_sequence(sequence_out, batch_first=True)
        sequence_out = self.sequence_dropout(sequence_out)  # [B, T', hidden_dim*2]
        
        # Final output
        logits = self.output_projection(sequence_out)  # [B, T', num_classes]
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Transpose for CTC: [B, T', C] -> [T', B, C]
        return log_probs.transpose(0, 1)
    
    def _initialize_weights(self):
        """Initialize weights to prevent CTC collapse."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name and 'lstm' in name:
                param.data.zero_()
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)  # Forget gate bias
            elif 'output_projection.weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'output_projection.bias' in name:
                param.data.zero_()
                # Blank bias will be set dynamically during training


def compute_wer(predictions: List[List[int]], targets: List[List[int]]) -> float:
    """Compute Word Error Rate."""
    total_errors = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
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


def decode_predictions(log_probs: torch.Tensor, lengths: torch.Tensor, 
                       blank_idx: int = 0) -> List[List[int]]:
    """Decode CTC output using greedy decoding."""
    batch_size = log_probs.size(1)
    predictions = []
    
    for b in range(batch_size):
        seq_len = lengths[b].item()
        seq_log_probs = log_probs[:seq_len, b, :]
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


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.CTCLoss,
                optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                logger: logging.Logger, stage: int = 3, blank_bias: float = 0.0,
                time_mask_prob: float = 0.0) -> Dict[str, float]:
    """Train for one epoch with stage-specific settings."""
    model.train()
    
    total_loss = 0
    total_samples = 0
    total_gradient_norm = 0
    batch_times = []
    
    # Set blank bias dynamically
    if hasattr(model, 'output_projection'):
        with torch.no_grad():
            model.output_projection.bias[0] = blank_bias
    
    for batch_idx, batch in enumerate(dataloader):
        start_time = time.time()
        
        # Move to device
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        
        # Time masking augmentation (Stage 1 only)
        if time_mask_prob > 0 and stage == 1:
            features = apply_time_masking(features, input_lengths, time_mask_prob)
        
        # Forward pass
        log_probs = model(features, input_lengths, stage=stage)
        
        # Adjust target lengths if using segment pooling (approximate)
        if stage >= 2 and model.use_segment_pooling:
            # Approximate: segment length ≈ frame length / 3
            adjusted_lengths = (input_lengths.float() / 3).long() + 1
            adjusted_lengths = torch.clamp(adjusted_lengths, max=log_probs.size(0))
        else:
            adjusted_lengths = input_lengths
        
        # CTC loss
        loss = criterion(
            log_probs,
            labels,
            adjusted_lengths,
            target_lengths
        )
        
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
        
        if batch_idx % 50 == 0:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            logger.info(f"Epoch {epoch} Batch {batch_idx}/{len(dataloader)}: "
                       f"Loss={loss.item():.4f}, AvgLoss={avg_loss:.4f}, "
                       f"GradNorm={grad_norm:.4f}, Stage={stage}")
    
    metrics = {
        'train_loss': total_loss / total_samples if total_samples > 0 else 0,
        'gradient_norm': total_gradient_norm / len(dataloader),
        'batch_time': np.mean(batch_times),
        'total_time': sum(batch_times)
    }
    
    return metrics


def apply_time_masking(features: torch.Tensor, lengths: torch.Tensor, 
                      mask_prob: float = 0.2) -> torch.Tensor:
    """Apply time masking augmentation."""
    batch_size, max_len, feat_dim = features.shape
    masked_features = features.clone()
    
    for b in range(batch_size):
        seq_len = lengths[b].item()
        num_masks = int(seq_len * mask_prob)
        
        if num_masks > 0:
            mask_indices = torch.randperm(seq_len)[:num_masks]
            masked_features[b, mask_indices, :] = 0.0
    
    return masked_features


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.CTCLoss,
             vocab: Dict[str, int], device: torch.device, logger: logging.Logger,
             stage: int = 3) -> Dict[str, float]:
    """Validate model."""
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
    frame_unique_predictions_set = set()
    frame_blank_count = 0
    frame_total_count = 0
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
            
            # Adjust lengths if needed
            if stage >= 2 and model.use_segment_pooling:
                adjusted_lengths = (input_lengths.float() / 3).long() + 1
                adjusted_lengths = torch.clamp(adjusted_lengths, max=log_probs.size(0))
            else:
                adjusted_lengths = input_lengths
            
            # CTC loss
            loss = criterion(
                log_probs,
                labels,
                adjusted_lengths,
                target_lengths
            )
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * features.size(0)
                total_samples += features.size(0)
            
            # Decode predictions
            predictions = decode_predictions(log_probs, adjusted_lengths)
            
            # Frame-level analysis
            for b in range(len(predictions)):
                seq_len = adjusted_lengths[b].item()
                seq_log_probs = log_probs[:seq_len, b, :]
                _, frame_preds = seq_log_probs.max(dim=-1)
                frame_preds = frame_preds.cpu().numpy()
                
                for pred_idx in frame_preds:
                    frame_unique_predictions_set.add(pred_idx)
                    if pred_idx == 0:
                        frame_blank_count += 1
                    frame_total_count += 1
            
            # Convert labels to list format
            for b in range(len(predictions)):
                target_len = target_lengths[b].item()
                target = labels[b, :target_len].cpu().tolist()
                all_targets.append(target)
                all_predictions.append(predictions[b])
                
                # Track metrics
                for pred_idx in predictions[b]:
                    if pred_idx == 0:
                        blank_predictions += 1
                    unique_predictions_set.add(pred_idx)
                    if pred_idx != 0:
                        unique_nonblank_predictions_set.add(pred_idx)
                    total_predictions += 1
                
                # Check CTC constraint
                T = adjusted_lengths[b].item()
                L = target_len
                if T < (2 * L + 1):
                    too_short_count += 1
                total_sequences += 1
    
    # Compute WER
    wer = compute_wer(all_predictions, all_targets)
    
    # Compute metrics
    blank_ratio = blank_predictions / total_predictions if total_predictions > 0 else 0
    frame_blank_ratio = frame_blank_count / frame_total_count if frame_total_count > 0 else 0
    too_short_ratio = (too_short_count / total_sequences) if total_sequences > 0 else 0.0
    
    # Top token analysis
    if frame_total_count > 0:
        # Count frame predictions
        frame_counts = {}
        # This is approximate - would need to recompute from log_probs
        top_token_id = 0  # Default to blank
        top_token_ratio = frame_blank_ratio
    else:
        top_token_id = 0
        top_token_ratio = 0.0
    
    metrics = {
        'val_loss': total_loss / total_samples if total_samples > 0 else 0,
        'val_wer': wer,
        'blank_ratio': blank_ratio * 100,
        'frame_blank_ratio': frame_blank_ratio * 100,
        'unique_predictions': len(unique_predictions_set),
        'unique_nonblank_predictions': len(unique_nonblank_predictions_set),
        'frame_unique_predictions': len(frame_unique_predictions_set),
        'ctc_too_short_ratio': too_short_ratio * 100,
        'frame_top_token_id': top_token_id,
        'frame_top_token_ratio': top_token_ratio * 100,
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
    """Main training function with multi-stage curriculum."""
    # Configuration
    config = {
        # Data
        'train_h5': 'data/features_cnn/train_features.h5',
        'dev_h5': 'data/features_cnn/dev_features.h5',
        'test_h5': 'data/features_cnn/test_features.h5',
        
        # Model
        'input_dim': 1024,
        'hidden_dim': 512,
        'dropout_stage1': 0.2,
        'dropout_stage2': 0.4,
        'dropout_stage3': 0.6,
        
        # Training stages
        'stage1_epochs': 30,  # Feature learning & exploration
        'stage2_epochs': 30,  # Segment modeling
        'stage3_epochs': 40,  # Sequence refinement
        
        # Stage 1: Exploration
        'stage1_lr': 1e-3,
        'stage1_blank_bias': -3.0,
        'stage1_time_mask_prob': 0.2,
        'stage1_weight_decay': 1e-4,
        
        # Stage 2: Segment modeling
        'stage2_lr': 5e-4,
        'stage2_blank_bias': -1.0,
        'stage2_time_mask_prob': 0.1,
        'stage2_weight_decay': 1e-4,
        
        # Stage 3: Refinement
        'stage3_lr': 1e-4,
        'stage3_blank_bias': 0.0,
        'stage3_time_mask_prob': 0.0,
        'stage3_weight_decay': 1e-4,
        
        # Common
        'batch_size': 16,
        'gradient_clip': 5.0,
        'seed': 42,
        'checkpoint_dir': 'checkpoints/hierarchical_cnn',
        'log_dir': 'logs/hierarchical_cnn'
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
    
    logger.info("Starting Hierarchical Multi-Stage Training")
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
    model = HierarchicalSignLanguageModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        dropout_stage1=config['dropout_stage1'],
        dropout_stage2=config['dropout_stage2'],
        dropout_stage3=config['dropout_stage3']
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Model size: {num_params * 4 / 1024**2:.2f} MB (FP32)")
    
    # Loss function
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Training loop for each stage
    best_wer = float('inf')
    training_history = []
    global_epoch = 0
    
    for stage in [1, 2, 3]:
        stage_epochs = config[f'stage{stage}_epochs']
        stage_lr = config[f'stage{stage}_lr']
        stage_blank_bias = config[f'stage{stage}_blank_bias']
        stage_time_mask = config[f'stage{stage}_time_mask_prob']
        stage_weight_decay = config[f'stage{stage}_weight_decay']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE {stage}: Epochs {global_epoch+1}-{global_epoch+stage_epochs}")
        logger.info(f"Learning Rate: {stage_lr}")
        logger.info(f"Blank Bias: {stage_blank_bias}")
        logger.info(f"Time Mask Prob: {stage_time_mask}")
        logger.info(f"{'='*60}\n")
        
        # Optimizer for this stage
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=stage_lr,
            weight_decay=stage_weight_decay
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        for epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            epoch_start = time.time()
            
            # Train
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer,
                device, global_epoch, logger, stage=stage,
                blank_bias=stage_blank_bias,
                time_mask_prob=stage_time_mask
            )
            
            # Validate
            val_metrics = validate(
                model, val_loader, criterion, vocab,
                device, logger, stage=stage
            )
            
            # Update scheduler
            scheduler.step(val_metrics['val_loss'])
            current_lr = optimizer.param_groups[0]['lr']
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            logger.info(f"\nEpoch {global_epoch} (Stage {stage}) Summary:")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Val WER: {val_metrics['val_wer']:.2f}%")
            logger.info(f"  Frame Blank Ratio: {val_metrics['frame_blank_ratio']:.2f}%")
            logger.info(f"  Unique Non-Blank Predictions: {val_metrics['unique_nonblank_predictions']}")
            logger.info(f"  Frame Unique Predictions: {val_metrics['frame_unique_predictions']}")
            logger.info(f"  Learning Rate: {current_lr:.2e}")
            logger.info(f"  Epoch Time: {epoch_time:.2f}s")
            
            # Save metrics
            epoch_metrics = {
                'epoch': global_epoch,
                'stage': stage,
                **{k: float(v) if isinstance(v, torch.Tensor) else v 
                   for k, v in train_metrics.items()},
                **{k: float(v) if isinstance(v, torch.Tensor) else v 
                   for k, v in val_metrics.items()},
                'lr': float(current_lr) if isinstance(current_lr, torch.Tensor) else current_lr,
                'epoch_time': float(epoch_time)
            }
            training_history.append(epoch_metrics)
            
            # Save training history
            history_file = os.path.join(config['log_dir'], f'history_{timestamp}.json')
            with open(history_file, 'w') as f:
                json.dump(training_history, f, indent=2)
            
            # Check for improvement
            if val_metrics['val_wer'] < best_wer - 0.001:
                best_wer = val_metrics['val_wer']
                
                # Save best model
                checkpoint = {
                    'epoch': global_epoch,
                    'stage': stage,
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
            
            # Save checkpoint every 10 epochs
            if global_epoch % 10 == 0:
                checkpoint = {
                    'epoch': global_epoch,
                    'stage': stage,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'wer': val_metrics['val_wer'],
                    'config': config
                }
                checkpoint_path = os.path.join(config['checkpoint_dir'], 
                                              f'checkpoint_epoch_{global_epoch}.pt')
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"  Saved checkpoint at epoch {global_epoch}")
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best WER: {best_wer:.2f}%")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Best model: {best_path}")


if __name__ == "__main__":
    main()

