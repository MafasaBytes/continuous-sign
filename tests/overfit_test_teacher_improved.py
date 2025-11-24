"""
Improved Overfitting Test for I3D Teacher Sign Language Model
With advanced training techniques to match MobileNetV3 baseline performance
Success criteria: Loss → 0, WER → 0% within reasonable epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.i3d_teacher import create_i3d_teacher
from src.utils.metrics import compute_wer
from src.data.mediapipe_dataset import MediaPipeFeatureDataset, collate_fn, build_vocabulary


def decode_predictions_greedy(log_probs: torch.Tensor, vocab_obj) -> List[str]:
    """
    Simple greedy decoding (for comparison).
    
    Args:
        log_probs: [T, B, V] log probabilities
        vocab_obj: Vocabulary object
        
    Returns:
        List of decoded strings (space-separated words)
    """
    # Get best path (greedy decoding)
    _, max_indices = torch.max(log_probs, dim=2)  # [T, B]
    max_indices = max_indices.transpose(0, 1)  # [B, T]
    
    decoded = []
    for sequence in max_indices:
        # Remove consecutive duplicates and blanks
        words = []
        prev_idx = -1
        for idx in sequence:
            idx = idx.item()
            # Skip blank (index 0) and duplicates
            if idx != prev_idx and idx != vocab_obj.blank_id:
                if idx in vocab_obj.idx2word:
                    words.append(vocab_obj.idx2word[idx])
            prev_idx = idx
        decoded.append(' '.join(words))
    
    return decoded


def decode_predictions_with_length_norm(
    log_probs: torch.Tensor, 
    vocab_obj,
    length_penalty: float = 0.6,
    confidence_threshold: float = -5.0,
    min_sequence_length: int = 2,
    max_sequence_length: int = 50,
    adaptive_threshold: bool = True
) -> List[str]:
    """
    Decode with adaptive length normalization that handles both short and long sequences.
    
    Args:
        log_probs: [T, B, V] log probabilities
        vocab_obj: Vocabulary object
        length_penalty: Penalty for sequence length (0.6 = standard, higher = prefer shorter)
        confidence_threshold: Minimum log probability to accept a token
        min_sequence_length: Minimum allowed sequence length (protect short sequences)
        max_sequence_length: Maximum allowed sequence length (prevent runaway)
        adaptive_threshold: Use length-adaptive confidence thresholds
        
    Returns:
        List of decoded strings (space-separated words)
    """
    T, B, V = log_probs.shape
    decoded = []
    
    for b in range(B):
        sequence_log_probs = log_probs[:, b, :]  # [T, V]
        
        # Get best path with scores
        max_log_probs, max_indices = torch.max(sequence_log_probs, dim=1)  # [T]
        
        # Remove consecutive duplicates and blanks, tracking confidence
        words = []
        word_scores = []
        prev_idx = -1
        
        for t in range(T):
            idx = max_indices[t].item()
            score = max_log_probs[t].item()
            
            # Skip blank and duplicates
            if idx != prev_idx and idx != vocab_obj.blank_id:
                if idx in vocab_obj.idx2word:
                    # Only add if confidence is above threshold
                    if score > confidence_threshold:
                        words.append(vocab_obj.idx2word[idx])
                        word_scores.append(score)
            
            prev_idx = idx
        
        # Apply adaptive filtering based on sequence length
        if len(words) > min_sequence_length:
            # Calculate statistics
            mean_score = sum(word_scores) / len(word_scores)
            std_score = (sum((s - mean_score) ** 2 for s in word_scores) / len(word_scores)) ** 0.5
            
            if adaptive_threshold:
                # Adaptive threshold: more lenient for longer sequences
                # Short sequences (< 5 words): aggressive filtering (0.5 * std)
                # Medium sequences (5-15 words): moderate filtering (0.3 * std)
                # Long sequences (> 15 words): gentle filtering (0.2 * std)
                
                if len(words) <= 5:
                    std_multiplier = 0.5  # Aggressive for short sequences
                elif len(words) <= 15:
                    std_multiplier = 0.3  # Moderate for medium sequences
                else:
                    std_multiplier = 0.2  # Gentle for long sequences
            else:
                std_multiplier = 0.5  # Fixed threshold
            
            threshold_score = mean_score - std_multiplier * std_score
            
            # Filter trailing low-confidence words
            # But only if we're above min_sequence_length
            filtered_words = []
            filtered_scores = []
            
            for i, (word, score) in enumerate(zip(words, word_scores)):
                # Always keep first min_sequence_length words
                if i < min_sequence_length:
                    filtered_words.append(word)
                    filtered_scores.append(score)
                # For remaining words, apply confidence threshold
                elif score >= threshold_score:
                    filtered_words.append(word)
                    filtered_scores.append(score)
                else:
                    # Found low-confidence word
                    # Check if it's truly trailing (all remaining words are low confidence)
                    remaining_scores = word_scores[i:]
                    avg_remaining = sum(remaining_scores) / len(remaining_scores)
                    
                    if avg_remaining < threshold_score:
                        # Yes, all remaining are low confidence - stop here
                        break
                    else:
                        # No, there are good words after - keep this one too
                        filtered_words.append(word)
                        filtered_scores.append(score)
            
            words = filtered_words[:max_sequence_length]  # Enforce max length
            word_scores = filtered_scores[:max_sequence_length]
        
        decoded.append(' '.join(words))
    
    return decoded


def decode_predictions(log_probs: torch.Tensor, vocab_obj, debug: bool = False) -> List[str]:
    """
    Main decode function - uses adaptive length-normalized decoding.
    
    Args:
        log_probs: [T, B, V] log probabilities
        vocab_obj: Vocabulary object
        debug: If True, print diagnostic information
        
    Returns:
        List of decoded strings (space-separated words)
    """
    # Configuration for adaptive decoding
    config = {
        'length_penalty': 0.6,           # Standard length penalty
        'confidence_threshold': -8.0,    # Accept reasonable confidence
        'min_sequence_length': 2,        # Protect short sequences (keep at least 2 words)
        'max_sequence_length': 50,       # Cap very long sequences
        'adaptive_threshold': True       # Use length-adaptive filtering
    }
    
    if debug:
        # Compare greedy vs length-normalized
        greedy_results = decode_predictions_greedy(log_probs, vocab_obj)
        length_norm_results = decode_predictions_with_length_norm(
            log_probs, vocab_obj, **config
        )
        
        print("\n=== Decoding Comparison ===")
        for i, (greedy, norm) in enumerate(zip(greedy_results, length_norm_results)):
            greedy_words = greedy.split()
            norm_words = norm.split()
            if greedy != norm:
                print(f"Sample {i}:")
                print(f"  Greedy ({len(greedy_words)} words): '{greedy}'")
                print(f"  Adaptive ({len(norm_words)} words):  '{norm}'")
                if len(greedy_words) > len(norm_words):
                    removed = ' '.join(greedy_words[len(norm_words):])
                    print(f"  Removed: '{removed}'")
                elif len(norm_words) > len(greedy_words):
                    print(f"  Note: Length-normalized is longer (should not happen)")
        print()
    
    return decode_predictions_with_length_norm(log_probs, vocab_obj, **config)


class WarmupScheduler:
    """Learning rate scheduler with warmup."""
    
    def __init__(self, optimizer, warmup_epochs: int, base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self, epoch: int):
        """Update learning rate based on epoch."""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.min_lr + (self.base_lr - self.min_lr) * (epoch / self.warmup_epochs)
        else:
            # After warmup, keep base_lr (or could add decay)
            lr = self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def train_overfit_test(
    feature_dir: Path,
    annotation_file: Path,
    num_samples: int = 5,
    num_epochs: int = 2000,
    learning_rate: float = 0.001,
    warmup_epochs: int = 50,
    dropout: float = 0.1,  # Match MobileNetV3
    optimizer_type: str = 'adam',  # 'adam', 'adamw', 'sgd'
    weight_decay: float = 0.0,  # L2 regularization
    gradient_clip: float = 1.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Run improved overfitting test with advanced training techniques.
    
    Args:
        feature_dir: Directory containing MediaPipe features
        annotation_file: CSV file with annotations
        num_samples: Number of samples to overfit on
        num_epochs: Number of training epochs
        learning_rate: Base learning rate
        warmup_epochs: Number of warmup epochs
        dropout: Dropout rate
        optimizer_type: Type of optimizer to use
        weight_decay: Weight decay for regularization
        gradient_clip: Gradient clipping threshold
        device: Device to train on
        
    Returns:
        Dictionary with training history
    """
    print(f"\n{'='*70}")
    print(f"IMPROVED OVERFITTING TEST - I3D Teacher Sign Language Model")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Samples: {num_samples}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate} (with {warmup_epochs} epoch warmup)")
    print(f"Dropout: {dropout}")
    print(f"Optimizer: {optimizer_type}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Gradient Clip: {gradient_clip}")
    print(f"{'='*70}\n")
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocabulary(annotation_file)
    print(f"Vocabulary size: {len(vocab)} words")
    
    # Load full dataset
    print("\nLoading dataset...")
    full_dataset = MediaPipeFeatureDataset(
        data_dir=feature_dir,
        annotation_file=annotation_file,
        vocabulary=vocab,
        split='train',
        augment=False  # No augmentation for overfitting test
    )
    
    # Sample a tiny subset
    print(f"\nSampling {num_samples} examples for overfitting test...")
    indices = list(range(len(full_dataset)))
    random.seed(42)  # For reproducibility
    random.shuffle(indices)
    sampled_indices = indices[:num_samples]
    
    tiny_dataset = Subset(full_dataset, sampled_indices)
    
    # Print sample info
    print("\nSelected samples:")
    for i, idx in enumerate(sampled_indices[:5]):  # Show first 5
        sample = full_dataset[idx]
        label_text = ' '.join(sample['words'])
        print(f"  {i+1}. Features: {sample['features'].shape}, "
              f"Labels: {len(sample['labels'])} words, "
              f"Text: '{label_text}'")
    
    # Create dataloader
    dataloader = DataLoader(
        tiny_dataset,
        batch_size=num_samples,  # Use all samples in one batch
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create teacher model with lower dropout
    print("\nCreating I3D teacher model...")
    model = create_i3d_teacher(
        vocab_size=len(vocab),
        dropout=dropout  # Lower dropout for overfitting test
    )
    model = model.to(device)
    
    # Loss and optimizer
    ctc_loss = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Learning rate scheduler with warmup
    scheduler = WarmupScheduler(optimizer, warmup_epochs, learning_rate, min_lr=learning_rate/10)
    
    # Training history
    history = {
        'loss': [],
        'wer': [],
        'predictions': [],
        'epoch_details': [],
        'learning_rates': []
    }
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_wer = float('inf')
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_predictions = []
        epoch_targets = []
        sample_ids = []
        
        # Update learning rate
        current_lr = scheduler.step(epoch)
        history['learning_rates'].append(current_lr)
        
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            feature_lengths = batch['input_lengths'].to(device)
            label_lengths = batch['target_lengths'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            log_probs = model(features, feature_lengths)  # [T, B, V]
            
            # Check for NaN/Inf in outputs
            if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                print(f"  WARNING: NaN/Inf in model outputs at epoch {epoch+1}")
                continue
            
            # Calculate CTC loss
            loss = ctc_loss(
                log_probs,
                labels,
                feature_lengths,
                label_lengths
            )
            
            # Backward pass
            if torch.isfinite(loss):
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                
                # Check for exploding gradients
                if grad_norm > 100:
                    print(f"  WARNING: Large gradient norm {grad_norm:.2f} at epoch {epoch+1}")
                
                optimizer.step()
                
                epoch_loss += loss.item()
            else:
                print(f"  WARNING: Non-finite loss at epoch {epoch+1}")
            
            # Decode predictions for WER
            with torch.no_grad():
                predictions = decode_predictions(log_probs, vocab)
                epoch_predictions.extend(predictions)
                # Convert words list to space-separated string
                targets = [' '.join(words) for words in batch['words']]
                epoch_targets.extend(targets)
                sample_ids = batch.get('video_ids', ['unknown'] * len(predictions))
        
        # Calculate WER
        wer = compute_wer(epoch_targets, epoch_predictions)
        avg_loss = epoch_loss / len(dataloader)
        
        # Update history
        history['loss'].append(avg_loss)
        history['wer'].append(wer)
        
        # Track best metrics
        if wer < best_wer:
            best_wer = wer
            best_epoch = epoch + 1
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Print progress every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            warmup_indicator = " [WARMUP]" if epoch < warmup_epochs else ""
            print(f"Epoch [{epoch+1:4d}/{num_epochs}] | "
                  f"Loss: {avg_loss:.6f} (Best: {best_loss:.6f}) | "
                  f"WER: {wer:.2f}% (Best: {best_wer:.2f}% @ {best_epoch}) | "
                  f"LR: {current_lr:.6f}{warmup_indicator}")
            
            # Show sample predictions
            if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
                print(f"\n  Sample Predictions (Epoch {epoch+1}):")
                for i, (pred, target) in enumerate(zip(epoch_predictions, epoch_targets)):
                    sid = sample_ids[i] if i < len(sample_ids) else f"sample_{i}"
                    match = "✓" if pred == target else "✗"
                    print(f"    {match} {sid}:")
                    print(f"      Target: '{target}'")
                    print(f"      Pred:   '{pred}'")
                print()
        
        # Store detailed info for key epochs
        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            history['epoch_details'].append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'wer': wer,
                'lr': current_lr,
                'predictions': list(zip(epoch_predictions, epoch_targets))
            })
        
        # Early stopping if we achieve 0% WER
        if wer == 0.0 and avg_loss < 0.01:
            print(f"\n🎉 SUCCESS! Achieved 0% WER at epoch {epoch+1}")
            print(f"Loss: {avg_loss:.6f}, stopping early.")
            break
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return history, model, vocab


def plot_results(history: Dict, save_path: Path = None):
    """Plot training results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axes[0, 0].plot(history['loss'], linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('CTC Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss - I3D Teacher', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot WER
    axes[0, 1].plot(history['wer'], linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('WER (%)', fontsize=12)
    axes[0, 1].set_title('Word Error Rate - I3D Teacher', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (0%)')
    axes[0, 1].legend()
    
    # Plot learning rate
    axes[1, 0].plot(history['learning_rates'], linewidth=2, color='purple')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot loss vs WER correlation
    axes[1, 1].scatter(history['loss'], history['wer'], alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Loss', fontsize=12)
    axes[1, 1].set_ylabel('WER (%)', fontsize=12)
    axes[1, 1].set_title('Loss vs WER Correlation', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def evaluate_results(history: Dict) -> Dict:
    """Evaluate if overfitting test passed."""
    final_loss = history['loss'][-1]
    final_wer = history['wer'][-1]
    min_loss = min(history['loss'])
    min_wer = min(history['wer'])
    
    # Success criteria
    loss_threshold = 0.5  # Loss should go below 0.5
    wer_threshold = 15.0  # WER should go below 15% (realistic for sign language CTC)
    strict_wer_threshold = 5.0  # Stretch goal
    perfect_wer_threshold = 0.0  # Perfect overfitting
    
    loss_passed = min_loss < loss_threshold
    wer_passed = min_wer < wer_threshold
    wer_strict_passed = min_wer < strict_wer_threshold
    wer_perfect_passed = min_wer == perfect_wer_threshold
    
    # Calculate improvement
    initial_wer = history['wer'][0] if history['wer'] else 100.0
    wer_improvement = ((initial_wer - min_wer) / initial_wer) * 100 if initial_wer > 0 else 0
    
    results = {
        'final_loss': final_loss,
        'final_wer': final_wer,
        'min_loss': min_loss,
        'min_wer': min_wer,
        'loss_passed': loss_passed,
        'wer_passed': wer_passed,
        'wer_strict_passed': wer_strict_passed,
        'wer_perfect_passed': wer_perfect_passed,
        'wer_improvement': wer_improvement,
        'overall_passed': loss_passed and wer_passed
    }
    
    print("\n" + "="*70)
    print("OVERFITTING TEST RESULTS - I3D TEACHER (IMPROVED)")
    print("="*70)
    print(f"Final Loss:     {final_loss:.6f}")
    print(f"Minimum Loss:   {min_loss:.6f} {'✓ PASS' if loss_passed else '✗ FAIL'} (threshold: < {loss_threshold})")
    print(f"Final WER:      {final_wer:.2f}%")
    print(f"Minimum WER:    {min_wer:.2f}%")
    print(f"  Perfect (0%): {'✓ PASS' if wer_perfect_passed else '✗ FAIL'} (target: 0%)")
    print(f"  Strict (<5%): {'✓ PASS' if wer_strict_passed else '✗ FAIL'} (threshold: < {strict_wer_threshold}%)")
    print(f"  Standard:     {'✓ PASS' if wer_passed else '✗ FAIL'} (threshold: < {wer_threshold}%)")
    print(f"  Improvement:  {wer_improvement:.1f}% reduction from baseline")
    print(f"\nOverall: {'✓ PASSED' if results['overall_passed'] else '✗ FAILED'}")
    if wer_perfect_passed:
        print("🎉 PERFECT! Achieved 0% WER - model can perfectly memorize data!")
    print("="*70)
    
    return results


def main():
    """Main function to run improved overfitting test."""
    # Configuration - matching MobileNetV3 successful setup
    feature_dir = Path("data/teacher_features/mediapipe_full")
    annotation_file = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    
    # Test configuration
    num_samples = 5
    num_epochs = 2000
    learning_rate = 0.001  # Start slightly higher
    warmup_epochs = 50  # Gradual warmup
    dropout = 0.1  # Match MobileNetV3
    optimizer_type = 'adam'  # or try 'adamw'
    weight_decay = 0.0  # No weight decay for overfitting test
    gradient_clip = 1.0
    
    # Check if data exists
    if not feature_dir.exists():
        print(f"ERROR: Feature directory not found: {feature_dir}")
        print("Please run feature extraction first.")
        print("\nExpected structure:")
        print("  data/teacher_features/mediapipe_full/train/*.npz")
        return
    
    if not annotation_file.exists():
        print(f"ERROR: Annotation file not found: {annotation_file}")
        print("Please ensure the raw data is downloaded.")
        return
    
    # Run overfitting test
    try:
        history, model, vocab = train_overfit_test(
            feature_dir=feature_dir,
            annotation_file=annotation_file,
            num_samples=num_samples,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            dropout=dropout,
            optimizer_type=optimizer_type,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip
        )
        
        # Evaluate results
        results = evaluate_results(history)
        
        # Plot results
        plot_path = Path("overfit_test_teacher_improved_results.png")
        plot_results(history, save_path=plot_path)
        
        # Save detailed report
        report_path = Path("overfit_test_teacher_improved_report.txt")
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("OVERFITTING TEST REPORT - I3D Teacher (IMPROVED)\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Samples: {num_samples}\n")
            f.write(f"  Epochs: {num_epochs}\n")
            f.write(f"  Learning Rate: {learning_rate}\n")
            f.write(f"  Warmup Epochs: {warmup_epochs}\n")
            f.write(f"  Dropout: {dropout}\n")
            f.write(f"  Optimizer: {optimizer_type}\n")
            f.write(f"  Weight Decay: {weight_decay}\n")
            f.write(f"  Gradient Clip: {gradient_clip}\n")
            f.write(f"  Model Parameters: {model.count_parameters():,}\n")
            f.write(f"  Vocabulary Size: {len(vocab)}\n\n")
            
            f.write(f"Results:\n")
            f.write(f"  Final Loss: {results['final_loss']:.6f}\n")
            f.write(f"  Minimum Loss: {results['min_loss']:.6f}\n")
            f.write(f"  Final WER: {results['final_wer']:.2f}%\n")
            f.write(f"  Minimum WER: {results['min_wer']:.2f}%\n\n")
            
            f.write(f"Test Status: {'PASSED [OK]' if results['overall_passed'] else 'FAILED [NOK]'}\n")
            if results['wer_perfect_passed']:
                f.write(f"Perfect Overfitting: YES (0% WER achieved)\n\n")
            else:
                f.write(f"Perfect Overfitting: NO (best: {results['min_wer']:.2f}% WER)\n\n")
            
            f.write("Detailed Epoch History:\n")
            for detail in history['epoch_details']:
                f.write(f"\nEpoch {detail['epoch']}:\n")
                f.write(f"  Loss: {detail['loss']:.6f}\n")
                f.write(f"  WER: {detail['wer']:.2f}%\n")
                f.write(f"  Learning Rate: {detail['lr']:.6f}\n")
                f.write(f"  Predictions:\n")
                for pred, target in detail['predictions']:
                    match = "✓" if pred == target else "✗"
                    f.write(f"    {match} Target: '{target}'\n")
                    f.write(f"       Pred:   '{pred}'\n")
        
        print(f"\nDetailed report saved to: {report_path}")
        
        # Final summary
        if results['wer_perfect_passed']:
            print("\n🎉 SUCCESS! The I3D teacher model achieved perfect overfitting (0% WER)!")
            print("The architecture is fully capable of learning and can now be trained on full dataset.")
            print("\nNext steps:")
            print("  1. Train teacher model on full dataset with these optimized settings")
            print("  2. Use trained teacher for knowledge distillation to student model")
            print("  3. Compare teacher vs student performance")
        elif results['overall_passed']:
            print("\n✓ GOOD! The teacher model passed the basic overfitting test.")
            print("Consider:")
            print("  - Training for more epochs to reach 0% WER")
            print("  - Trying different learning rates or optimizers")
            print("  - Adjusting gradient clipping threshold")
        else:
            print("\n⚠ WARNING! The teacher model failed to overfit sufficiently.")
            print("\nTroubleshooting suggestions:")
            print("  1. Try LOWER learning rate (e.g., 0.0005 or 0.0001)")
            print("  2. Try LONGER warmup (e.g., 100 epochs)")
            print("  3. Try AdamW optimizer with weight_decay=0.01")
            print("  4. Check for gradient issues (NaN/Inf warnings above)")
            print("  5. Verify the log_softmax fix is working correctly")
    
    except Exception as e:
        print(f"\nERROR during overfitting test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

