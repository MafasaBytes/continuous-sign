"""
Diagnostic script to identify root causes of 93% WER in sign language recognition.
Analyzes model behavior, data quality, and training dynamics.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

def diagnose_model_outputs():
    """Check if model outputs are degenerate (all same class, all blank, etc.)"""
    print("\n" + "="*80)
    print("1. MODEL OUTPUT ANALYSIS")
    print("="*80)

    # Load checkpoint
    ckpt_path = Path("checkpoints/student/mobilenet_v3_20251117_163006/best_model.pth")
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Analyze output projection weights
    if 'output_proj.weight' in checkpoint['model_state_dict']:
        output_weights = checkpoint['model_state_dict']['output_proj.weight']
        print(f"Output projection shape: {output_weights.shape}")

        # Check for dead neurons
        weight_norms = torch.norm(output_weights, dim=1)
        dead_outputs = (weight_norms < 1e-6).sum().item()
        print(f"Dead output neurons: {dead_outputs}/{output_weights.shape[0]} ({dead_outputs/output_weights.shape[0]*100:.1f}%)")

        # Check weight statistics
        print(f"Weight stats: mean={output_weights.mean():.4f}, std={output_weights.std():.4f}")
        print(f"Weight range: [{output_weights.min():.4f}, {output_weights.max():.4f}]")

        # Check if weights are collapsed (all similar)
        weight_variance = output_weights.var(dim=0).mean()
        print(f"Cross-class weight variance: {weight_variance:.6f}")
        if weight_variance < 1e-4:
            print("WARNING: Weights appear collapsed (very low variance across classes)")

        # Check bias terms
        if 'output_proj.bias' in checkpoint['model_state_dict']:
            output_bias = checkpoint['model_state_dict']['output_proj.bias']
            blank_bias = output_bias[0].item()  # CTC blank is typically index 0
            other_bias_mean = output_bias[1:].mean().item()
            print(f"Blank token bias: {blank_bias:.4f}")
            print(f"Other tokens mean bias: {other_bias_mean:.4f}")
            if blank_bias > other_bias_mean + 2:
                print("WARNING: Model heavily biased toward blank token")

def diagnose_training_history():
    """Analyze training logs for convergence issues"""
    print("\n" + "="*80)
    print("2. TRAINING DYNAMICS ANALYSIS")
    print("="*80)

    log_path = Path("checkpoints/student/mobilenet_v3_20251117_163006/training.log")
    if not log_path.exists():
        print(f"Training log not found at {log_path}")
        return

    # Parse training log
    with open(log_path, 'r') as f:
        lines = f.readlines()

    train_losses = []
    val_losses = []
    wer_scores = []

    for line in lines:
        if "Train Loss:" in line:
            loss = float(line.split("Train Loss:")[1].strip())
            train_losses.append(loss)
        elif "Val Loss:" in line:
            loss = float(line.split("Val Loss:")[1].strip())
            val_losses.append(loss)
        elif "WER:" in line:
            wer = float(line.split("WER:")[1].strip().replace('%', ''))
            wer_scores.append(wer)

    print(f"Epochs trained: {len(train_losses)}")
    if train_losses:
        print(f"Initial train loss: {train_losses[0]:.2f}")
        print(f"Final train loss: {train_losses[-1]:.2f}")

        # Check for loss explosion
        if train_losses[0] > 50:
            print("WARNING: Initial loss very high - possible initialization issue")

        # Check for plateau
        if len(train_losses) > 5:
            recent_change = abs(train_losses[-1] - train_losses[-5])
            if recent_change < 0.01:
                print("WARNING: Training loss plateaued")

    if wer_scores:
        print(f"Best WER achieved: {min(wer_scores):.2f}%")
        if all(wer >= 90 for wer in wer_scores):
            print("CRITICAL: Model never achieved WER below 90% - fundamental issue")

def diagnose_vocabulary():
    """Check vocabulary quality and alignment"""
    print("\n" + "="*80)
    print("3. VOCABULARY ANALYSIS")
    print("="*80)

    # Load vocabulary
    vocab_path = Path("checkpoints/student/mobilenet_v3_20251117_163006/vocabulary.json")
    vocab = {}
    if vocab_path.exists():
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
            vocab = vocab_data.get('word2idx', {})

        print(f"Vocabulary size: {len(vocab)}")
        print(f"Has blank token: {'<blank>' in vocab}")
        print(f"Blank token index: {vocab.get('<blank>', 'Not found')}")

        # Check for special tokens that shouldn't be in vocabulary
        special_tokens = [w for w in vocab if any(x in w for x in ['__', 'loc-', 'cl-', 'IX', 'WG'])]
        if special_tokens:
            print(f"WARNING: Found {len(special_tokens)} special tokens in vocab: {special_tokens[:5]}")

    # Check annotation file
    annotation_path = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    if annotation_path.exists():
        df = pd.read_csv(annotation_path, sep='|', on_bad_lines='skip')
        if 'annotation' in df.columns:
            # Count actual vocabulary usage
            all_words = []
            for ann in df['annotation'].dropna():
                words = str(ann).split()
                all_words.extend([w for w in words if not any(x in w for x in ['__', 'loc-', 'cl-'])])

            word_freq = Counter(all_words)
            print(f"Unique words in annotations: {len(word_freq)}")
            print(f"Most common words: {word_freq.most_common(10)}")

            # Check if vocabulary covers frequent words
            if vocab:
                uncovered = [w for w, c in word_freq.most_common(50) if w not in vocab]
                if uncovered:
                    print(f"WARNING: Common words missing from vocab: {uncovered[:5]}")

def diagnose_feature_quality():
    """Check if features are properly extracted and normalized"""
    print("\n" + "="*80)
    print("4. FEATURE QUALITY ANALYSIS")
    print("="*80)

    # Check a sample feature file
    feature_dir = Path("data/teacher_features/mediapipe_full")
    if feature_dir.exists():
        feature_files = list(feature_dir.glob("*.npy"))
        if feature_files:
            # Load first feature
            sample = np.load(feature_files[0])
            print(f"Sample feature shape: {sample.shape}")
            print(f"Feature statistics: mean={sample.mean():.4f}, std={sample.std():.4f}")
            print(f"Feature range: [{sample.min():.4f}, {sample.max():.4f}]")

            # Check for NaN or inf
            if np.isnan(sample).any():
                print("CRITICAL: NaN values found in features!")
            if np.isinf(sample).any():
                print("CRITICAL: Inf values found in features!")

            # Check if features are properly normalized
            if abs(sample.mean()) > 10 or sample.std() > 100:
                print("WARNING: Features may not be properly normalized")

            # Check temporal dimension
            if len(sample.shape) == 2:
                print(f"Temporal dimension: {sample.shape[0]} frames")
                if sample.shape[0] < 10:
                    print("WARNING: Very short sequence - may cause training issues")
        else:
            print("CRITICAL: No feature files found!")

def diagnose_ctc_alignment():
    """Diagnose CTC-specific issues"""
    print("\n" + "="*80)
    print("5. CTC ALIGNMENT ANALYSIS")
    print("="*80)

    # Common CTC issues
    print("Common CTC failure modes to check:")
    print("1. Input length < Target length: Model cannot produce enough outputs")
    print("2. Blank token dominance: Model outputs mostly blanks")
    print("3. Gradient vanishing: Long sequences cause gradient issues")
    print("4. Label repetition: Same label repeated without blanks")

    # Load sample data to check length ratios
    annotation_path = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    if annotation_path.exists():
        df = pd.read_csv(annotation_path, sep='|', on_bad_lines='skip')

        # Estimate typical sequence lengths
        if 'annotation' in df.columns:
            label_lengths = []
            for ann in df['annotation'].dropna()[:100]:  # Sample 100
                words = str(ann).split()
                # Filter out special tokens
                words = [w for w in words if not any(x in w for x in ['__', 'loc-', 'cl-'])]
                label_lengths.append(len(words))

            if label_lengths:
                print(f"\nLabel length statistics:")
                print(f"  Mean: {np.mean(label_lengths):.1f} words")
                print(f"  Max: {max(label_lengths)} words")
                print(f"  Min: {min(label_lengths)} words")

                # Typical frame rates
                typical_fps = 25
                typical_sign_duration = 0.5  # seconds per sign
                expected_frames = np.mean(label_lengths) * typical_sign_duration * typical_fps
                print(f"\nExpected frames for average sequence: ~{expected_frames:.0f}")
                print("If actual frame count is much lower, CTC alignment will fail!")

def suggest_fixes():
    """Provide prioritized recommendations"""
    print("\n" + "="*80)
    print("PRIORITIZED ACTION PLAN")
    print("="*80)

    print("""
IMMEDIATE FIXES (Do These First):

1. **Fix CTC Blank Token Initialization** [HIGHEST PRIORITY]
   - Ensure blank token is at index 0 in vocabulary
   - Initialize output bias for blank token to small positive value (~0.1)
   - Other classes should start at 0 or small negative values

2. **Verify Input/Output Length Compatibility** [CRITICAL]
   - CTC requires: input_length >= target_length
   - Add this check in your collate_fn:
     ```python
     # Ensure input is at least as long as target
     min_input_len = target_lengths.max() * 2  # 2x for safety
     if input_lengths.min() < min_input_len:
         # Pad or upsample input
     ```

3. **Simplify Architecture for Debugging** [HIGH]
   - Temporarily remove MobileNetV3 blocks
   - Use simple Linear -> BiLSTM -> Linear
   - Get this working first (WER < 50%), then add complexity

DIAGNOSTIC EXPERIMENTS:

4. **Overfit on Single Sample** [ESSENTIAL]
   - Train on just 1-5 samples until loss approaches 0
   - If this fails, there's a fundamental bug

5. **Check Gradient Flow** [IMPORTANT]
   - Add gradient logging:
     ```python
     for name, param in model.named_parameters():
         if param.grad is not None:
             writer.add_histogram(f'grad/{name}', param.grad, epoch)
     ```

6. **Visualize Model Predictions** [REVEALING]
   - Decode actual predictions and print them
   - Are they all blank? All same token? Random?

MEDIUM-TERM FIXES:

7. **Data Quality**
   - Remove sequences shorter than 10 frames
   - Remove sequences with > 20 words (too long for CTC)
   - Balance dataset by sequence length

8. **Training Strategy**
   - Start with shorter sequences, gradually increase
   - Use curriculum learning
   - Reduce learning rate (try 1e-4 or 5e-5)

9. **Architecture Adjustments**
   - Increase BiLSTM hidden dim (try 256 or 512)
   - Add more BiLSTM layers (2-3 total)
   - Remove downsampling in temporal dimension

VALIDATION CHECKS:

10. **Verify Feature Extraction**
    - Plot MediaPipe landmarks on video
    - Ensure hands/face are properly tracked
    - Check for frame drops or misalignment
""")

if __name__ == "__main__":
    print("SIGN LANGUAGE RECOGNITION DIAGNOSTIC REPORT")
    print("=" * 80)

    diagnose_model_outputs()
    diagnose_training_history()
    diagnose_vocabulary()
    diagnose_feature_quality()
    diagnose_ctc_alignment()
    suggest_fixes()

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)