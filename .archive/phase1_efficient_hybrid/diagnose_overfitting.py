"""Diagnostic script to identify and fix overfitting issues."""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add project to path
import sys
sys.path.append(str(Path(__file__).parent))

from teacher.models.efficient_hybrid import EfficientHybridModel
from teacher.models.lightweight_hybrid import LightweightHybridModel
from teacher.data.mediapipe_dataset import MediaPipeFeatureDataset, build_vocabulary


def analyze_model_capacity():
    """Compare model sizes for the dataset."""
    print("\n" + "="*60)
    print("MODEL CAPACITY ANALYSIS")
    print("="*60)

    # Dataset size
    train_samples = 5672
    val_samples = 540
    vocab_size = 1122
    input_dim = 1024

    print(f"\nDataset Statistics:")
    print(f"  Training samples: {train_samples:,}")
    print(f"  Validation samples: {val_samples:,}")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Input dimension: {input_dim:,}")

    # Original model
    print("\n1. Original EfficientHybridModel (OVERFITTING):")
    model_orig = EfficientHybridModel(
        input_dim=input_dim,
        hidden_dim=768,
        num_classes=vocab_size,
        dropout=0.3
    )
    params_orig = model_orig.count_parameters()
    print(f"   Parameters: {params_orig:,}")
    print(f"   Params per sample: {params_orig/train_samples:.1f}")
    print(f"   Model size: {params_orig * 4 / (1024**2):.2f} MB")
    print(f"   Status: SEVERELY OVERPARAMETERIZED!")

    # Recommended lightweight model
    print("\n2. Lightweight Model (RECOMMENDED):")
    model_light = LightweightHybridModel(
        input_dim=input_dim,
        hidden_dim=256,
        num_classes=vocab_size,
        dropout=0.5
    )
    params_light = model_light.count_parameters()
    print(f"   Parameters: {params_light:,}")
    print(f"   Params per sample: {params_light/train_samples:.1f}")
    print(f"   Model size: {params_light * 4 / (1024**2):.2f} MB")
    print(f"   Reduction: {(1 - params_light/params_orig)*100:.1f}%")

    # Ultra-light version
    print("\n3. Ultra-Light Model (MINIMAL):")
    model_ultra = LightweightHybridModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_classes=vocab_size,
        dropout=0.6
    )
    params_ultra = model_ultra.count_parameters()
    print(f"   Parameters: {params_ultra:,}")
    print(f"   Params per sample: {params_ultra/train_samples:.1f}")
    print(f"   Model size: {params_ultra * 4 / (1024**2):.2f} MB")
    print(f"   Reduction: {(1 - params_ultra/params_orig)*100:.1f}%")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("\nBased on dataset size, recommended parameters:")
    print(f"  - Target: 100-500 params/sample")
    print(f"  - Ideal range: {train_samples * 100:,} - {train_samples * 500:,} parameters")
    print(f"  - Current model has {params_orig/train_samples:.0f} params/sample (WAY TOO HIGH!)")
    print(f"  - Lightweight model has {params_light/train_samples:.0f} params/sample (GOOD)")


def check_data_distribution():
    """Analyze data distribution and potential issues."""
    print("\n" + "="*60)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*60)

    # Load annotations
    train_csv = Path("data/raw_data/phoenix-2014-multisigner/annotations/manual/train.corpus.csv")
    dev_csv = Path("data/raw_data/phoenix-2014-multisigner/annotations/manual/dev.corpus.csv")

    if not train_csv.exists():
        print("  [ERROR] Training annotations not found")
        return

    import pandas as pd
    train_df = pd.read_csv(train_csv, sep='|')
    dev_df = pd.read_csv(dev_csv, sep='|')

    print(f"\n1. Sample Distribution:")
    print(f"   Train samples: {len(train_df)}")
    print(f"   Dev samples: {len(dev_df)}")
    print(f"   Train/Val ratio: {len(train_df)/len(dev_df):.1f}:1")

    # Analyze sequence lengths
    train_lengths = [len(ann.split()) for ann in train_df['annotation']]
    dev_lengths = [len(ann.split()) for ann in dev_df['annotation']]

    print(f"\n2. Sequence Length Statistics:")
    print(f"   Train: mean={np.mean(train_lengths):.1f}, std={np.std(train_lengths):.1f}, "
          f"max={max(train_lengths)}, min={min(train_lengths)}")
    print(f"   Dev: mean={np.mean(dev_lengths):.1f}, std={np.std(dev_lengths):.1f}, "
          f"max={max(dev_lengths)}, min={min(dev_lengths)}")

    # Check for overlap
    train_ids = set(train_df['id'])
    dev_ids = set(dev_df['id'])
    overlap = train_ids & dev_ids
    if overlap:
        print(f"\n  [WARNING] Data leakage detected! {len(overlap)} samples in both train and dev!")
    else:
        print(f"\n3. Data Split: [OK] No overlap detected")

    # Vocabulary analysis
    train_vocab = Counter()
    dev_vocab = Counter()
    for ann in train_df['annotation']:
        train_vocab.update(ann.split())
    for ann in dev_df['annotation']:
        dev_vocab.update(ann.split())

    print(f"\n4. Vocabulary Statistics:")
    print(f"   Unique signs in train: {len(train_vocab)}")
    print(f"   Unique signs in dev: {len(dev_vocab)}")
    print(f"   Signs only in dev: {len(set(dev_vocab) - set(train_vocab))}")
    print(f"   Most common train signs: {train_vocab.most_common(5)}")


def suggest_hyperparameters():
    """Suggest optimal hyperparameters."""
    print("\n" + "="*60)
    print("HYPERPARAMETER RECOMMENDATIONS")
    print("="*60)

    train_samples = 5672
    vocab_size = 1122

    print("\n1. Model Architecture:")
    print(f"   - Hidden dimension: 128-256 (current: 768 - TOO LARGE)")
    print(f"   - LSTM layers: 1-2 (current: 3 - TOO DEEP)")
    print(f"   - Dropout: 0.5-0.6 (current: 0.3 - TOO LOW)")

    print("\n2. Training Configuration:")
    print(f"   - Batch size: 4-8 with gradient accumulation")
    print(f"   - Learning rate: 1e-4 to 5e-4 (start low)")
    print(f"   - Weight decay: 1e-3 to 5e-3 (strong L2)")
    print(f"   - Gradient clipping: 1.0 (aggressive)")

    print("\n3. CTC Regularization:")
    print(f"   - Blank penalty: 0.5 to 1.0 (POSITIVE values)")
    print(f"   - Temperature: 1.2 to 1.5 during training")
    print(f"   - Label smoothing: 0.1 to 0.2")

    print("\n4. Data Augmentation:")
    print(f"   - Time masking: 20-30% of frames")
    print(f"   - Feature dropout: 10-20% of features")
    print(f"   - Speed perturbation: +/-10-15%")
    print(f"   - MixUp: alpha=0.2")

    print("\n5. Training Strategy:")
    print(f"   - Use curriculum learning: start with short sequences")
    print(f"   - Implement gradient accumulation for stable training")
    print(f"   - Use cosine annealing with warmup")
    print(f"   - Early stop based on validation loss gap, not just WER")


def analyze_existing_checkpoint():
    """Analyze existing checkpoint for issues."""
    print("\n" + "="*60)
    print("CHECKPOINT ANALYSIS")
    print("="*60)

    checkpoint_paths = [
        "checkpoints/improved/best_model.pth",
        "checkpoints/efficient_hybrid/best_model.pth",
    ]

    for path in checkpoint_paths:
        if Path(path).exists():
            print(f"\nAnalyzing: {path}")
            checkpoint = torch.load(path, map_location='cpu')

            if 'val_metrics' in checkpoint:
                metrics = checkpoint['val_metrics']
                print(f"  Best WER: {metrics.get('val_wer', 'N/A'):.2f}%")
                print(f"  Val Loss: {metrics.get('val_loss', 'N/A'):.4f}")
                print(f"  Blank Ratio: {metrics.get('blank_ratio', 'N/A'):.1f}%")

            if 'epoch' in checkpoint:
                print(f"  Saved at epoch: {checkpoint['epoch']}")

            if 'config' in checkpoint:
                config = checkpoint['config']
                print(f"  Batch size: {config.get('batch_size', 'N/A')}")
                print(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
                print(f"  Hidden dim: {config.get('hidden_dim', 'N/A')}")

            # Analyze weight statistics
            if 'model_state_dict' in checkpoint:
                weights = checkpoint['model_state_dict']
                weight_norms = []
                for name, param in weights.items():
                    if 'weight' in name and param.dim() >= 2:
                        weight_norms.append(torch.norm(param).item())

                if weight_norms:
                    print(f"  Weight norm statistics:")
                    print(f"    Mean: {np.mean(weight_norms):.4f}")
                    print(f"    Std: {np.std(weight_norms):.4f}")
                    print(f"    Max: {np.max(weight_norms):.4f}")

                    if np.max(weight_norms) > 100:
                        print("    [WARNING] Very large weight norms detected - sign of instability!")
            break


def main():
    """Run all diagnostics."""
    print("\n" + "="*80)
    print(" OVERFITTING DIAGNOSIS FOR SIGN LANGUAGE RECOGNITION MODEL")
    print("="*80)

    # Run diagnostics
    analyze_model_capacity()
    check_data_distribution()
    suggest_hyperparameters()
    analyze_existing_checkpoint()

    # Final recommendations
    print("\n" + "="*80)
    print("IMMEDIATE ACTION ITEMS")
    print("="*80)
    print("""
1. SWITCH TO LIGHTWEIGHT MODEL:
   python teacher/train_efficient.py \\
     --model lightweight \\
     --hidden_dim 256 \\
     --dropout 0.5 \\
     --batch_size 4

2. USE FIXED CONFIGURATION:
   python teacher/train.py \\
     --config configs/teacher/fix_overfitting.yaml

3. MONITOR OVERFITTING METRICS:
   - Track train/val loss gap (should be < 0.5)
   - Watch blank ratio (should be 10-30%)
   - Monitor gradient norms (should be < 10)

4. DATA IMPROVEMENTS:
   - Augment training data aggressively
   - Consider synthetic data generation
   - Use pseudo-labeling on unlabeled data

5. ENSEMBLE APPROACH:
   - Train multiple small models
   - Use different random seeds
   - Average predictions
    """)


if __name__ == "__main__":
    main()