"""
Verification Script: Test I3D Teacher Training with MediaPipe Features
=======================================================================

This script verifies that the normalization fix resolves the NaN/Inf issue.

Tests:
1. Dataset loads correctly with normalize=False
2. First batch processes without NaN/Inf
3. Forward pass completes successfully
4. Backward pass and gradient computation work
5. 10 training iterations complete without issues
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.i3d_teacher import create_i3d_teacher
from src.data.dataset import MediaPipeFeatureDataset, collate_fn, build_vocabulary
from torch.utils.data import DataLoader


def test_training_loop():
    """Test the training loop with the fix."""
    
    print("=" * 80)
    print("VERIFICATION: I3D Teacher Training with MediaPipe Features")
    print("=" * 80)
    
    # Configuration
    feature_dir = Path("data/teacher_features/mediapipe_full")
    annotation_file = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {device}")
    
    # Check data exists
    if not feature_dir.exists():
        print(f"\n[ERROR] Feature directory not found: {feature_dir}")
        print("Please extract features first.")
        return False
    
    if not annotation_file.exists():
        print(f"\n[ERROR] Annotation file not found: {annotation_file}")
        return False
    
    # Build vocabulary
    print("\n[1/6] Building vocabulary...")
    vocab = build_vocabulary(annotation_file)
    print(f"      Vocabulary size: {len(vocab)} words")
    
    # Create dataset with normalize=False (THE FIX)
    print("\n[2/6] Creating dataset with normalize=False...")
    dataset = MediaPipeFeatureDataset(
        data_dir=feature_dir,
        annotation_file=annotation_file,
        vocabulary=vocab,
        split='train',
        augment=False,
        normalize=False,  # THE CRITICAL FIX
        max_seq_length=256
    )
    print(f"      Dataset size: {len(dataset)} samples")
    print(f"      Normalization: {dataset.normalize}")  # Should be False
    
    if dataset.normalize:
        print("      [WARNING] Dataset normalization is enabled! This will cause issues.")
    else:
        print("      [OK] Dataset normalization is disabled (correct)")
    
    # Create dataloader
    print("\n[3/6] Creating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    print(f"      Batch size: 2")
    print(f"      Number of batches: {len(dataloader)}")
    
    # Create model
    print("\n[4/6] Creating I3D teacher model...")
    model = create_i3d_teacher(vocab_size=len(vocab), dropout=0.3)
    model = model.to(device)
    print(f"      Parameters: {model.count_parameters():,}")
    
    # Setup training
    criterion = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # Test first batch
    print("\n[5/6] Testing first batch...")
    batch = next(iter(dataloader))
    
    features = batch['features'].to(device)
    labels = batch['labels'].to(device)
    input_lengths = batch['input_lengths'].to(device)
    target_lengths = batch['target_lengths'].to(device)
    
    print(f"      Raw features shape: {features.shape}")
    print(f"      Raw features mean: {features.mean():.4f}, std: {features.std():.4f}")
    print(f"      Raw features range: [{features.min():.4f}, {features.max():.4f}]")
    
    # Apply per-sample normalization (same as training loop)
    features_mean = features.mean(dim=(1, 2), keepdim=True)
    features_std = features.std(dim=(1, 2), keepdim=True) + 1e-6
    features = (features - features_mean) / features_std
    
    print(f"      After normalization mean: {features.mean():.4f}, std: {features.std():.4f}")
    print(f"      After normalization range: [{features.min():.4f}, {features.max():.4f}]")
    
    # Check for NaN/Inf
    if torch.isnan(features).any() or torch.isinf(features).any():
        print("      [FAIL] Features contain NaN/Inf after normalization!")
        return False
    else:
        print("      [OK] Features are clean after normalization")
    
    # Forward pass
    model.train()
    log_probs = model(features, input_lengths)
    
    print(f"      Log probs shape: {log_probs.shape}")
    print(f"      Log probs mean: {log_probs.mean():.4f}, std: {log_probs.std():.4f}")
    
    if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
        print("      [FAIL] Model output contains NaN/Inf!")
        return False
    else:
        print("      [OK] Model output is clean")
    
    # Compute loss
    loss = criterion(log_probs, labels, input_lengths, target_lengths)
    print(f"      CTC Loss: {loss.item():.6f}")
    
    if torch.isnan(loss) or torch.isinf(loss):
        print("      [FAIL] Loss is NaN/Inf!")
        return False
    else:
        print("      [OK] Loss is valid")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"      Gradient norm: {grad_norm:.4f}")
    
    has_nan_grad = any(
        torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
        for p in model.parameters() if p.grad is not None
    )
    
    if has_nan_grad:
        print("      [FAIL] Gradients contain NaN/Inf!")
        return False
    else:
        print("      [OK] Gradients are clean")
    
    optimizer.step()
    print("      [OK] Optimizer step completed")
    
    # Test 10 iterations
    print("\n[6/6] Running 10 training iterations...")
    model.train()
    
    for i, batch in enumerate(dataloader):
        if i >= 10:
            break
        
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        
        # Normalize
        features_mean = features.mean(dim=(1, 2), keepdim=True)
        features_std = features.std(dim=(1, 2), keepdim=True) + 1e-6
        features = (features - features_mean) / features_std
        
        # Check features
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"      [FAIL] Iteration {i+1}: Features have NaN/Inf")
            return False
        
        # Forward
        log_probs = model(features, input_lengths)
        
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            print(f"      [FAIL] Iteration {i+1}: Model output has NaN/Inf")
            return False
        
        # Loss
        loss = criterion(log_probs, labels, input_lengths, target_lengths)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"      [FAIL] Iteration {i+1}: Loss is NaN/Inf (value: {loss.item()})")
            return False
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        has_nan_grad = any(
            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
            for p in model.parameters() if p.grad is not None
        )
        
        if has_nan_grad:
            print(f"      [FAIL] Iteration {i+1}: Gradients have NaN/Inf")
            return False
        
        optimizer.step()
        
        print(f"      Iteration {i+1:2d}/10: Loss={loss.item():.4f}, GradNorm={grad_norm:.2f} [OK]")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE: ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe normalization fix resolves the NaN/Inf issue.")
    print("Training with full dataset should now work correctly.")
    print("\nNext steps:")
    print("  1. Start full training: python -m src.training.train_teacher")
    print("  2. Monitor the first epoch for any issues")
    print("  3. Check logs for 'First batch diagnostics' message")
    print("\n" + "=" * 80)
    
    return True


def main():
    """Run verification."""
    try:
        success = test_training_loop()
        if success:
            print("\n[SUCCESS] Verification passed - ready for full training")
            sys.exit(0)
        else:
            print("\n[FAILURE] Verification failed - check errors above")
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Verification failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

