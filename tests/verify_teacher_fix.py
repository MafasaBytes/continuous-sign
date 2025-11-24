"""
Quick verification script to ensure I3D Teacher model fix is working correctly.
This script checks that the model outputs valid log probabilities for CTC loss.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.i3d_teacher import create_i3d_teacher
from src.models.mobilenet_v3 import create_mobilenet_v3_model


def verify_model_output(model, model_name: str, vocab_size: int):
    """Verify model outputs valid log probabilities."""
    
    print(f"\n{'='*70}")
    print(f"Verifying {model_name}")
    print(f"{'='*70}")
    
    # Create dummy input
    batch_size = 2
    seq_length = 50
    feature_dim = 6516
    
    dummy_input = torch.randn(batch_size, seq_length, feature_dim)
    dummy_lengths = torch.tensor([50, 45])
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input, dummy_lengths)
    
    # Check output shape
    expected_shape = (seq_length, batch_size, vocab_size)
    print(f"\n1. Output Shape Check:")
    print(f"   Expected: {expected_shape}")
    print(f"   Got:      {output.shape}")
    print(f"   Status:   {'✓ PASS' if output.shape == expected_shape else '✗ FAIL'}")
    
    # Check if values are log probabilities (should be negative)
    print(f"\n2. Log Probability Range Check:")
    print(f"   Min value: {output.min().item():.4f}")
    print(f"   Max value: {output.max().item():.4f}")
    print(f"   Mean value: {output.mean().item():.4f}")
    
    is_negative = output.max().item() <= 0.0
    print(f"   All negative: {'✓ PASS' if is_negative else '✗ FAIL (should be negative)'}")
    
    # Check if values sum to ~1 in probability space
    print(f"\n3. Probability Sum Check:")
    probs = torch.exp(output)  # Convert log probs to probs
    prob_sums = probs.sum(dim=-1)  # Sum over vocabulary
    print(f"   Probability sums (should be ~1.0):")
    print(f"   Min: {prob_sums.min().item():.4f}")
    print(f"   Max: {prob_sums.max().item():.4f}")
    print(f"   Mean: {prob_sums.mean().item():.4f}")
    
    is_normalized = (prob_sums.min() > 0.99) and (prob_sums.max() < 1.01)
    print(f"   Properly normalized: {'✓ PASS' if is_normalized else '✗ FAIL'}")
    
    # Check for NaN or Inf
    print(f"\n4. Numerical Stability Check:")
    has_nan = torch.isnan(output).any()
    has_inf = torch.isinf(output).any()
    print(f"   Contains NaN: {has_nan.item()} {'✗ FAIL' if has_nan else '✓ PASS'}")
    print(f"   Contains Inf: {has_inf.item()} {'✗ FAIL' if has_inf else '✓ PASS'}")
    
    # Overall verdict
    all_passed = (
        output.shape == expected_shape and
        is_negative and
        is_normalized and
        not has_nan and
        not has_inf
    )
    
    print(f"\n{'='*70}")
    print(f"OVERALL: {'✓✓✓ ALL CHECKS PASSED ✓✓✓' if all_passed else '✗✗✗ SOME CHECKS FAILED ✗✗✗'}")
    print(f"{'='*70}")
    
    return all_passed


def compare_models():
    """Compare I3D Teacher and MobileNetV3 outputs."""
    
    vocab_size = 1232  # PHOENIX dataset vocab size
    
    print("\n" + "="*70)
    print("MODEL OUTPUT VERIFICATION")
    print("="*70)
    print("\nThis script verifies that models output valid log probabilities for CTC.")
    print("Both models should pass all checks for correct CTC training.")
    
    # Test MobileNetV3 (known to work)
    print("\n\n" + "🔍 Testing MobileNetV3 (Baseline - should pass)")
    mobilenet = create_mobilenet_v3_model(vocab_size, dropout=0.1)
    mobilenet_passed = verify_model_output(mobilenet, "MobileNetV3", vocab_size)
    
    # Test I3D Teacher (recently fixed)
    print("\n\n" + "🔍 Testing I3D Teacher (Recently Fixed - should pass)")
    teacher = create_i3d_teacher(vocab_size, dropout=0.1)
    teacher_passed = verify_model_output(teacher, "I3D Teacher", vocab_size)
    
    # Final summary
    print("\n\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"MobileNetV3:  {'✓ PASS' if mobilenet_passed else '✗ FAIL'}")
    print(f"I3D Teacher:  {'✓ PASS' if teacher_passed else '✗ FAIL'}")
    
    if mobilenet_passed and teacher_passed:
        print("\n🎉 SUCCESS! Both models output valid log probabilities.")
        print("The I3D Teacher fix is working correctly.")
        print("\nYou can now proceed with overfitting tests:")
        print("  python overfit_test_teacher_improved.py")
    elif teacher_passed and not mobilenet_passed:
        print("\n⚠️ Unexpected: Teacher passed but MobileNetV3 failed.")
        print("This might indicate an issue with the test itself.")
    elif not teacher_passed and mobilenet_passed:
        print("\n❌ ISSUE: I3D Teacher still failing checks.")
        print("\nTroubleshooting:")
        print("  1. Verify log_softmax was added to forward() method")
        print("  2. Check line ~568 in src/models/i3d_teacher.py")
        print("  3. Ensure you have: log_probs = F.log_softmax(logits, dim=-1)")
        print("  4. Make sure you're using the updated model file")
    else:
        print("\n❌ ISSUE: Both models failing.")
        print("Something unexpected happened. Check the error messages above.")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        compare_models()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nVerification failed. Check the error messages above.")

