"""Training functions compatible with EfficientHybridModel."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
# Removed unused imports
import numpy as np


def train_epoch_efficient(model, dataloader, criterion, optimizer, device, epoch, logger,
                          blank_penalty=0.0, time_mask_prob=0.0, gradient_clip=5.0):
    """Train for one epoch with EfficientHybridModel."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_gradient_norm = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')

    for batch_idx, batch in enumerate(progress_bar):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        # Apply time masking augmentation
        if time_mask_prob > 0 and random.random() < 0.5:
            batch_size, seq_len, _ = features.shape
            for i in range(batch_size):
                if random.random() < time_mask_prob:
                    width = min(12, max(1, seq_len // 10))
                    start = random.randint(0, max(0, seq_len - width))
                    features[i, start:start+width, :] = 0.0

        # Forward pass - EfficientHybridModel only expects features and lengths
        log_probs = model(features, input_lengths)

        # Apply blank penalty if specified (positive penalty discourages blanks)
        if blank_penalty > 0:
            # Create a copy to avoid in-place modification
            blank_penalty_tensor = torch.zeros_like(log_probs)
            blank_penalty_tensor[:, :, 0] = -blank_penalty
            log_probs = log_probs + blank_penalty_tensor

        # CTC Loss
        loss = criterion(log_probs, labels, input_lengths, target_lengths)

        # Skip invalid losses
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}")
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        # Accumulate metrics
        total_loss += float(loss.item()) * features.size(0)
        total_samples += features.size(0)
        total_gradient_norm += float(grad_norm)
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item(), 'grad_norm': grad_norm.item()})

        # Clear cache periodically
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_gradient_norm = total_gradient_norm / num_batches if num_batches > 0 else 0

    return {
        'train_loss': avg_loss,
        'gradient_norm': avg_gradient_norm
    }


def validate_efficient(model, dataloader, criterion, vocab, device, logger, epoch=0,
                       decode_method='greedy', beam_width=10):
    """Validate with EfficientHybridModel."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    num_batches = 0

    all_predictions = []
    all_targets = []
    total_blank_predictions = 0
    total_predictions = 0
    unique_predictions = set()

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} - Validation')

        for batch in progress_bar:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            target_lengths = batch['target_lengths'].to(device)

            # Forward pass
            log_probs = model(features, input_lengths)

            # CTC Loss
            loss = criterion(log_probs, labels, input_lengths, target_lengths)

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += float(loss.item()) * features.size(0)
                total_samples += features.size(0)
                num_batches += 1

            # Decode predictions
            if decode_method == 'greedy':
                predictions = decode_greedy(log_probs, input_lengths)
            else:
                predictions = decode_beam_search(log_probs, input_lengths, beam_width)

            # Convert to text
            for pred, target_len in zip(predictions, target_lengths):
                # Remove blanks and convert to list
                pred_filtered = [p for p in pred if p != 0]

                # Track statistics
                total_predictions += len(pred)
                total_blank_predictions += (pred == 0).sum() if hasattr(pred, 'sum') else pred.count(0)
                if pred_filtered:
                    unique_predictions.add(tuple(pred_filtered))

                # Store for WER calculation
                all_predictions.append(pred_filtered)

                # Get target
                target_idx = len(all_targets)
                if target_idx < len(batch['labels']):
                    target = batch['labels'][target_idx][:target_len].tolist()
                    all_targets.append([t for t in target if t != 0])
                else:
                    all_targets.append([])

            progress_bar.set_postfix({'loss': loss.item()})

    # Calculate metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')

    # Calculate WER
    total_errors = 0
    total_words = 0
    for pred, target in zip(all_predictions, all_targets):
        errors = calculate_simple_wer(pred, target)
        total_errors += errors
        total_words += max(len(target), 1)

    wer = (total_errors / total_words * 100) if total_words > 0 else 100.0
    blank_ratio = (total_blank_predictions / total_predictions * 100) if total_predictions > 0 else 0

    return {
        'val_loss': avg_loss,
        'val_wer': wer,
        'blank_ratio': blank_ratio,
        'unique_nonblank_predictions': len(unique_predictions),
        'ctc_too_short_ratio': 0.0  # Not tracking this for simplicity
    }


def decode_greedy(log_probs, lengths):
    """Simple greedy decoding."""
    # log_probs: [T, B, C]
    T, B, C = log_probs.shape
    predictions = []

    for b in range(B):
        length = min(lengths[b].item(), T)
        # Get most likely class at each timestep
        pred = torch.argmax(log_probs[:length, b, :], dim=-1)
        predictions.append(pred.cpu().numpy())

    return predictions


def decode_beam_search(log_probs, lengths, beam_width=10):
    """Simple beam search decoding (placeholder - use greedy for now)."""
    # For simplicity, fall back to greedy
    return decode_greedy(log_probs, lengths)


def calculate_simple_wer(pred, target):
    """Calculate simple word error rate using edit distance."""
    if len(target) == 0:
        return len(pred)
    if len(pred) == 0:
        return len(target)

    # Simple Levenshtein distance
    dp = [[0] * (len(target) + 1) for _ in range(len(pred) + 1)]

    for i in range(len(pred) + 1):
        dp[i][0] = i
    for j in range(len(target) + 1):
        dp[0][j] = j

    for i in range(1, len(pred) + 1):
        for j in range(1, len(target) + 1):
            if pred[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[len(pred)][len(target)]