"""
Example: Using Pre-trained Weights for Teacher Model

This script demonstrates how to programmatically use pre-trained weights
for the teacher model training.
"""

import torch
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.i3d_teacher import create_i3d_teacher
from src.models.pretrained_loader import (
    load_pretrained_i3d_weights,
    freeze_layers,
    unfreeze_layers,
    progressive_unfreezing_schedule
)


def example_1_load_from_checkpoint():
    """
    Example 1: Load pre-trained weights from an existing checkpoint
    """
    print("=" * 70)
    print("Example 1: Load from Existing Checkpoint")
    print("=" * 70)
    
    # Create teacher model
    vocab_size = 1232
    model = create_i3d_teacher(vocab_size, dropout=0.3)
    
    print(f"\nModel created with {model.count_parameters():,} parameters")
    
    # Load pre-trained weights from your existing checkpoint
    checkpoint_path = Path("checkpoints/teacher/i3d_teacher_20251119_093842/best_i3d.pth")
    
    if checkpoint_path.exists():
        print(f"\nLoading pre-trained weights from: {checkpoint_path}")
        model, num_loaded = load_pretrained_i3d_weights(
            model,
            checkpoint_path,
            freeze_backbone=True,
            freeze_until_layer='mixed_4f'
        )
        print(f"[OK] Loaded {num_loaded} parameters")
        
        # Check trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = model.count_parameters()
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    else:
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("   Train a baseline model first or use a different path")
    
    print()


def example_2_transfer_learning():
    """
    Example 2: Transfer learning with layer freezing
    """
    print("=" * 70)
    print("Example 2: Transfer Learning with Layer Freezing")
    print("=" * 70)
    
    vocab_size = 1232
    model = create_i3d_teacher(vocab_size, dropout=0.3)
    
    checkpoint_path = Path("checkpoints/teacher/i3d_teacher_20251119_093842/best_i3d.pth")
    
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found, skipping example")
        return
    
    # Load weights without freezing
    print("\n1. Loading weights without freezing...")
    model, _ = load_pretrained_i3d_weights(
        model,
        checkpoint_path,
        freeze_backbone=False
    )
    
    total = model.count_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   All layers trainable: {trainable:,} parameters")
    
    # Freeze different amounts
    print("\n2. Freezing strategies:")
    
    # Strategy 1: Freeze everything except classifier
    num_frozen = freeze_layers(model, freeze_until='mixed_5c')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   a) Freeze until mixed_5c: {trainable:,} trainable ({100*trainable/total:.1f}%)")
    
    # Strategy 2: Unfreeze all, then freeze less
    unfreeze_layers(model)
    num_frozen = freeze_layers(model, freeze_until='mixed_4f')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   b) Freeze until mixed_4f: {trainable:,} trainable ({100*trainable/total:.1f}%)")
    
    # Strategy 3: Unfreeze all, then freeze only stem
    unfreeze_layers(model)
    num_frozen = freeze_layers(model, freeze_until='mixed_3b')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   c) Freeze until mixed_3b: {trainable:,} trainable ({100*trainable/total:.1f}%)")
    
    print()


def example_3_progressive_unfreezing():
    """
    Example 3: Progressive unfreezing schedule
    """
    print("=" * 70)
    print("Example 3: Progressive Unfreezing Schedule")
    print("=" * 70)
    
    vocab_size = 1232
    model = create_i3d_teacher(vocab_size, dropout=0.3)
    
    checkpoint_path = Path("checkpoints/teacher/i3d_teacher_20251119_093842/best_i3d.pth")
    
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found, skipping example")
        return
    
    # Load and freeze
    print("\nLoading pre-trained weights and freezing backbone...")
    model, _ = load_pretrained_i3d_weights(
        model,
        checkpoint_path,
        freeze_backbone=True,
        freeze_until_layer='mixed_4f'
    )
    
    # Simulate progressive unfreezing over 50 epochs
    print("\nProgressive unfreezing schedule (50 epochs, 4 stages):")
    print("-" * 70)
    
    total_epochs = 50
    check_epochs = [0, 5, 15, 25, 35, 45, 49]
    
    for epoch in check_epochs:
        trainable = progressive_unfreezing_schedule(
            model, epoch, total_epochs, num_stages=4
        )
        stage = int((epoch / total_epochs) * 4)
        print(f"  Epoch {epoch:2d} (Stage {stage+1}/4): {trainable:,} trainable parameters")
    
    print()


def example_4_practical_workflow():
    """
    Example 4: Complete practical workflow
    """
    print("=" * 70)
    print("Example 4: Complete Practical Workflow")
    print("=" * 70)
    
    print("\n[*] Recommended workflow for your situation:")
    print()
    
    print("1. Load existing checkpoint with frozen backbone")
    print("   -> Start with proven initialization (even if WER is high)")
    print()
    
    print("2. Train with top layers only (feature extraction)")
    print("   -> Quick test to validate approach (~20 epochs)")
    print("   -> Expected: WER 50-60%")
    print()
    
    print("3. If successful, use progressive unfreezing")
    print("   -> Gradually unfreeze deeper layers (~50 epochs)")
    print("   -> Expected: WER 25-35%")
    print()
    
    print("4. Use improved teacher for distillation")
    print("   -> Much better soft targets for student")
    print("   -> Expected student WER: 30-45% (vs 60-70% with bad teacher)")
    print()
    
    print("[*] Quick start commands:")
    print()
    print("   # Test approach (fast)")
    print("   scripts/train_teacher_pretrained.bat feature-extraction")
    print()
    print("   # Full training (best results)")
    print("   scripts/train_teacher_pretrained.bat progressive")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("PRE-TRAINED WEIGHTS USAGE EXAMPLES")
    print("=" * 70)
    print()
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Run examples
    try:
        example_1_load_from_checkpoint()
        example_2_transfer_learning()
        example_3_progressive_unfreezing()
        example_4_practical_workflow()
        
        print("=" * 70)
        print("[OK] All examples completed!")
        print("=" * 70)
        print()
        print("For more details, see:")
        print("   - docs/PRETRAINED_WEIGHTS_GUIDE.md")
        print("   - QUICKSTART_PRETRAINED.md")
        print()
        
    except Exception as e:
        print(f"\n[ERROR] Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

