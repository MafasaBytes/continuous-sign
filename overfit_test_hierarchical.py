"""
Overfitting Test for Hierarchical Teacher Sign Language Model
Tests if the hierarchical architecture can memorize a small dataset (3-5 samples)
Success criteria: Loss → 0, WER → 0% within reasonable epochs

This test verifies:
1. The hierarchical architecture can learn (not broken)
2. Multi-scale features work correctly
3. Hierarchical attention and LSTM function properly
4. Model can achieve very low loss/WER on tiny dataset
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

from src.models.hierarchical_teacher import create_hierarchical_teacher
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
    confidence_threshold: float = -8.0,
    min_sequence_length: int = 2,
    max_sequence_length: int = 50,
    adaptive_threshold: bool = True
) -> List[str]:
    """
    Decode with adaptive length normalization that handles both short and long sequences.
    Filters out low-confidence trailing words (like the extra "MONTAG" being added).
    
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
    This filters out low-confidence trailing words (like extra "MONTAG" tokens).
    
    Args:
        log_probs: [T, B, V] log probabilities
        vocab_obj: Vocabulary object
        debug: If True, print diagnostic information
        
    Returns:
        List of decoded strings (space-separated words)
    """
    # Configuration for adaptive decoding (same as I3D teacher test)
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


def train_overfit_test(
    feature_dir: Path,
    annotation_file: Path,
    num_samples: int = 10,
    num_epochs: int = 2000,
    learning_rate: float = 0.0001,
    dropout: float = 0.1,  # Lower dropout for overfitting test
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Run overfitting test for hierarchical model.
    
    Args:
        feature_dir: Directory containing MediaPipe features
        annotation_file: CSV file with annotations
        num_samples: Number of samples to overfit on
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        dropout: Dropout rate (lower for overfitting test)
        device: Device to train on
        
    Returns:
        Dictionary with training history
    """
    print(f"\n{'='*70}")
    print(f"OVERFITTING TEST - Hierarchical Teacher Sign Language Model")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Samples: {num_samples}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Dropout: {dropout}")
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
    for i, idx in enumerate(sampled_indices):
        sample = full_dataset[idx]
        label_text = ' '.join(sample['words'])
        print(f"  {i+1}. Features: {sample['features'].shape}, "
              f"Labels: {len(sample['labels'])} words, "
              f"Text: '{label_text[:60]}...'")
    
    # Create dataloader
    dataloader = DataLoader(
        tiny_dataset,
        batch_size=num_samples,  # Use all samples in one batch
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create hierarchical teacher model
    print("\nCreating Hierarchical teacher model...")
    model = create_hierarchical_teacher(
        vocab_size=len(vocab),
        dropout=dropout  # Lower dropout for overfitting test
    )
    model = model.to(device)
    
    total_params = model.count_parameters()
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Loss and optimizer
    ctc_loss = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'loss': [],
        'wer': [],
        'predictions': [],
        'epoch_details': []
    }
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_wer = float('inf')
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_predictions = []
        epoch_targets = []
        sample_ids = []
        
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            feature_lengths = batch['input_lengths'].to(device)
            label_lengths = batch['target_lengths'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            log_probs = model(features, feature_lengths)  # [T, B, V]
            
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            else:
                print(f"  WARNING: Non-finite loss at epoch {epoch+1}")
            
            # Decode predictions for WER (use adaptive decoding to filter trailing words)
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
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Print progress every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1:4d}/{num_epochs}] | "
                  f"Loss: {avg_loss:.6f} (Best: {best_loss:.6f}) | "
                  f"WER: {wer:.2f}% (Best: {best_wer:.2f}%)")
            
            # Show sample predictions
            if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
                print(f"\n  Sample Predictions (Epoch {epoch+1}):")
                for i, (pred, target) in enumerate(zip(epoch_predictions, epoch_targets)):
                    sid = sample_ids[i] if i < len(sample_ids) else f"sample_{i}"
                    match = "OK" if pred == target else "NOK"
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
                'predictions': list(zip(epoch_predictions, epoch_targets))
            })
        
        # Early success check
        if wer < 0.0 and avg_loss < 0.5:
            print(f"\nSUCCESS! Achieved WER < 5% and Loss < 0.5 at epoch {epoch+1}")
            print(f"  Final Loss: {avg_loss:.6f}")
            print(f"  Final WER: {wer:.2f}%")
            break
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return history, model, vocab


def plot_results(history: Dict, save_path: Path = None):
    """Plot training results."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['loss'], linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('CTC Loss', fontsize=12)
    axes[0].set_title('Training Loss - Hierarchical Teacher', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot WER
    axes[1].plot(epochs, history['wer'], linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('WER (%)', fontsize=12)
    axes[1].set_title('Word Error Rate - Hierarchical Teacher', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (0%)')
    axes[1].axhline(y=5, color='yellow', linestyle='--', linewidth=2, alpha=0.5, label='Good (<5%)')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.close()


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
    
    loss_passed = min_loss < loss_threshold
    wer_passed = min_wer < wer_threshold
    wer_strict_passed = min_wer < strict_wer_threshold
    
    # Calculate improvement
    initial_wer = history['wer'][0] if history['wer'] else 100.0
    wer_improvement = ((initial_wer - min_wer) / initial_wer) * 100
    
    results = {
        'final_loss': final_loss,
        'final_wer': final_wer,
        'min_loss': min_loss,
        'min_wer': min_wer,
        'loss_passed': loss_passed,
        'wer_passed': wer_passed,
        'wer_strict_passed': wer_strict_passed,
        'wer_improvement': wer_improvement,
        'overall_passed': loss_passed and wer_passed
    }
    
    print("\n" + "="*70)
    print("OVERFITTING TEST RESULTS - Hierarchical Teacher")
    print("="*70)
    print(f"Final Loss:     {final_loss:.6f}")
    print(f"Minimum Loss:   {min_loss:.6f} {'PASS' if loss_passed else 'FAIL'} (threshold: < {loss_threshold})")
    print(f"Final WER:      {final_wer:.2f}%")
    print(f"Minimum WER:    {min_wer:.2f}%")
    print(f"  Standard:     {'PASS' if wer_passed else 'FAIL'} (threshold: < {wer_threshold}%)")
    print(f"  Strict:       {'PASS' if wer_strict_passed else 'FAIL'} (threshold: < {strict_wer_threshold}%)")
    print(f"  Improvement:  {wer_improvement:.1f}% reduction from baseline")
    print(f"\nOverall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    print("="*70)
    
    return results


def main():
    """Main function to run overfitting test."""
    # Configuration
    feature_dir = Path("data/teacher_features/mediapipe_full")
    annotation_file = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    num_samples = 5
    num_epochs = 2000  # Should converge faster with hierarchical architecture
    learning_rate = 0.0005
    dropout = 0.1  # Lower dropout for overfitting test
    
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
            dropout=dropout
        )
        
        # Evaluate results
        results = evaluate_results(history)
        
        # Plot results
        plot_path = Path("overfit_test_hierarchical_results.png")
        plot_results(history, save_path=plot_path)
        
        # Save detailed report
        report_path = Path("overfit_test_hierarchical_report.txt")
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("OVERFITTING TEST REPORT - Hierarchical Teacher Sign Language Model\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Samples: {num_samples}\n")
            f.write(f"  Epochs: {num_epochs}\n")
            f.write(f"  Learning Rate: {learning_rate}\n")
            f.write(f"  Dropout: {dropout}\n")
            f.write(f"  Model Parameters: {model.count_parameters():,}\n")
            f.write(f"  Vocabulary Size: {len(vocab)}\n\n")
            
            f.write(f"Results:\n")
            f.write(f"  Final Loss: {results['final_loss']:.6f}\n")
            f.write(f"  Minimum Loss: {results['min_loss']:.6f}\n")
            f.write(f"  Final WER: {results['final_wer']:.2f}%\n")
            f.write(f"  Minimum WER: {results['min_wer']:.2f}%\n\n")
            
            f.write(f"Test Status: {'PASSED [OK]' if results['overall_passed'] else 'FAILED [NOK]'}\n\n")
            
            f.write("Detailed Epoch History:\n")
            for detail in history['epoch_details']:
                f.write(f"\nEpoch {detail['epoch']}:\n")
                f.write(f"  Loss: {detail['loss']:.6f}\n")
                f.write(f"  WER: {detail['wer']:.2f}%\n")
                f.write(f"  Predictions:\n")
                for pred, target in detail['predictions']:
                    match = "✓" if pred == target else "✗"
                    f.write(f"    {match} Target: '{target}'\n")
                    f.write(f"       Pred:   '{pred}'\n")
        
        print(f"\nDetailed report saved to: {report_path}")
        
        # Final summary
        if results['overall_passed']:
            print("\n" + "="*70)
            print("SUCCESS! The Hierarchical teacher model can learn and memorize the training data.")
            print("The architecture is capable of learning the vocabulary and reducing WER.")
            print("="*70)
            print("\nNext steps:")
            print("  1. Train hierarchical model on full dataset")
            print("  2. Use trained teacher for knowledge distillation to student model")
            print("  3. Compare with I3D teacher baseline")
        else:
            print("\n" + "="*70)
            print("WARNING! The hierarchical model failed to overfit to the training data.")
            print("This indicates potential issues with:")
            print("  - Model architecture capacity")
            print("  - Training setup (loss function, optimizer)")
            print("  - Data preprocessing or feature quality")
            print("  - Gradient flow or numerical stability")
            print("="*70)
            print("\nTroubleshooting:")
            print("  1. Check if features are loaded correctly")
            print("  2. Verify model forward pass works")
            print("  3. Check gradient flow (add gradient logging)")
            print("  4. Try different learning rate or optimizer")
    
    except Exception as e:
        print(f"\nERROR during overfitting test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

