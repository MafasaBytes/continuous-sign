"""
Test script to compare Teacher (I3D) and Student (MobileNetV3) models.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from src.models import MobileNetV3SignLanguage, create_mobilenet_v3_model
from src.models.i3d_teacher import I3DTeacher, create_i3d_teacher


def compare_models():
    """Compare teacher and student models."""

    print("="*70)
    print("TEACHER-STUDENT MODEL COMPARISON")
    print("="*70)

    vocab_size = 1232  # RWTH-PHOENIX

    # Create Teacher model
    print("\n[TEACHER] I3D Model:")
    print("-"*50)
    teacher = create_i3d_teacher(vocab_size=vocab_size)
    teacher_params = teacher.count_parameters()
    teacher_size_mb = teacher_params * 4 / (1024 * 1024)

    # Create Student model
    print("\n[STUDENT] MobileNetV3 Model:")
    print("-"*50)
    student = create_mobilenet_v3_model(vocab_size=vocab_size)
    student_params = student.count_parameters()
    student_size_mb = student_params * 4 / (1024 * 1024)

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print(f"\nModel Parameters:")
    print(f"  Teacher (I3D):        {teacher_params:,} params")
    print(f"  Student (MobileNetV3): {student_params:,} params")
    print(f"  Compression Ratio:     {teacher_params / student_params:.2f}x")

    print(f"\nModel Size:")
    print(f"  Teacher: {teacher_size_mb:.2f} MB")
    print(f"  Student: {student_size_mb:.2f} MB")
    print(f"  Size Reduction: {(1 - student_size_mb/teacher_size_mb)*100:.1f}%")

    print(f"\nExpected Performance:")
    print(f"  Teacher WER (after training): ~20-30%")
    print(f"  Student WER (baseline):       ~40-50%")
    print(f"  Student WER (distilled):      <25% (target)")

    print(f"\nTraining Strategy:")
    print(f"  Phase I:  Train student baseline")
    print(f"  Phase II: Train teacher on sign language")
    print(f"  Phase III: Distill teacher -> student")

    print(f"\nKnowledge Distillation Config (from proposal):")
    print(f"  Temperature: 3.0")
    print(f"  Loss: 0.7 * Soft + 0.3 * Hard")
    print(f"  Expected improvement: 15-20% WER reduction")

    # Test forward pass compatibility
    print(f"\n" + "="*70)
    print("TESTING FORWARD PASS COMPATIBILITY")
    print("="*70)

    batch_size = 2
    seq_length = 100
    feature_dim = 6516

    dummy_input = torch.randn(batch_size, seq_length, feature_dim)
    dummy_lengths = torch.tensor([100, 90])

    with torch.no_grad():
        # Teacher forward
        teacher_output = teacher(dummy_input, dummy_lengths)
        print(f"Teacher output shape: {teacher_output.shape}")

        # Student forward
        student_output = student(dummy_input, dummy_lengths)
        print(f"Student output shape: {student_output.shape}")

        # Check compatibility
        if teacher_output.shape == student_output.shape:
            print("[PASS] Output shapes match - ready for distillation!")
        else:
            print("[FAIL] Output shape mismatch")

    # Memory requirements
    print(f"\n" + "="*70)
    print("MEMORY REQUIREMENTS")
    print("="*70)

    teacher_memory = teacher_size_mb + 100  # Rough estimate with activations
    student_memory = student_size_mb + 50
    distillation_memory = teacher_memory + student_memory

    print(f"Teacher training:     ~{teacher_memory:.0f} MB")
    print(f"Student training:     ~{student_memory:.0f} MB")
    print(f"Distillation training: ~{distillation_memory:.0f} MB")
    print(f"Fits in 8GB VRAM: {'[PASS]' if distillation_memory < 8000 else '[FAIL]'}")

    # Training recommendations
    print(f"\n" + "="*70)
    print("TRAINING RECOMMENDATIONS")
    print("="*70)

    print("\nOption 1: Quick Baseline (1-2 days)")
    print("  1. Train MobileNetV3 student directly")
    print("  2. Expected WER: ~40-50%")
    print("  3. Command: python src/training/train.py")

    print("\nOption 2: Full Distillation (3-5 days) [RECOMMENDED]")
    print("  1. Train I3D teacher first (~1-2 days)")
    print("  2. Distill to MobileNetV3 (~1-2 days)")
    print("  3. Expected WER: <25%")
    print("  4. Commands:")
    print("     python src/training/train_teacher.py")
    print("     python src/training/train_distillation.py")

    print("\nOption 3: Progressive (Parallel)")
    print("  1. Start training student baseline")
    print("  2. Meanwhile train teacher")
    print("  3. Fine-tune student with distillation")
    print("  4. Best of both worlds")

    return {
        'teacher_params': teacher_params,
        'student_params': student_params,
        'compression_ratio': teacher_params / student_params
    }


if __name__ == "__main__":
    results = compare_models()

    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Teacher is {results['compression_ratio']:.1f}x larger than student")
    print(f"Knowledge distillation will transfer teacher's knowledge")
    print(f"to the efficient student model for deployment.")
    print("\nReady to proceed with training!")