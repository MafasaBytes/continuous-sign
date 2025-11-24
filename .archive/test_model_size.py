"""
Test script to verify MobileNetV3 model meets size requirements.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from src.models import MobileNetV3SignLanguage, create_mobilenet_v3_model


def test_model_size():
    """Test model size and parameter count."""

    print("="*60)
    print("Testing MobileNetV3 Sign Language Model")
    print("="*60)

    # Create model with RWTH-PHOENIX vocab size
    vocab_size = 1232
    model = create_mobilenet_v3_model(vocab_size=vocab_size, dropout=0.6)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate model size
    # Float32: 4 bytes per parameter
    model_size_mb_fp32 = total_params * 4 / (1024 * 1024)

    # Float16 (with mixed precision): 2 bytes per parameter
    model_size_mb_fp16 = total_params * 2 / (1024 * 1024)

    # INT8 (with quantization): 1 byte per parameter
    model_size_mb_int8 = total_params * 1 / (1024 * 1024)

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print(f"\nModel Size:")
    print(f"  FP32 (baseline): {model_size_mb_fp32:.2f} MB")
    print(f"  FP16 (mixed precision): {model_size_mb_fp16:.2f} MB")
    print(f"  INT8 (quantized): {model_size_mb_int8:.2f} MB")

    print(f"\nResearch Proposal Targets:")
    print(f"  Target size: < 100 MB")
    print(f"  Target params: 3-5M (optimal for dataset)")

    # Check if meets requirements
    meets_size = model_size_mb_fp32 < 100
    meets_params = 3_000_000 <= total_params <= 5_000_000

    print(f"\nValidation:")
    print(f"  Meets size requirement (<100MB): {'[PASS]' if meets_size else '[FAIL]'} ({model_size_mb_fp32:.2f} MB)")
    print(f"  Meets param range (3-5M): {'[PASS]' if meets_params else '[FAIL]'} ({total_params/1e6:.2f}M)")

    # Test forward pass
    print(f"\nTesting Forward Pass:")
    batch_size = 4
    seq_length = 100
    feature_dim = 6516  # Full MediaPipe features (no PCA)

    dummy_input = torch.randn(batch_size, seq_length, feature_dim)
    dummy_lengths = torch.tensor([100, 90, 80, 70])

    try:
        with torch.no_grad():
            output = model(dummy_input, dummy_lengths)
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: [T={seq_length}, B={batch_size}, V={vocab_size}]")
        print(f"  Forward pass: [PASS]")
    except Exception as e:
        print(f"  Forward pass failed: {e}")

    # Memory estimation
    print(f"\nMemory Requirements:")
    # Rough estimation for batch_size=4
    activation_memory_mb = (batch_size * seq_length * 384 * 4) / (1024 * 1024)  # Intermediate activations
    model_memory_mb = model_size_mb_fp32
    total_memory_mb = activation_memory_mb + model_memory_mb

    print(f"  Model memory: {model_memory_mb:.2f} MB")
    print(f"  Activation memory (batch=4): ~{activation_memory_mb:.2f} MB")
    print(f"  Total GPU memory needed: ~{total_memory_mb:.2f} MB")
    print(f"  Fits in 8GB VRAM: {'[PASS]' if total_memory_mb < 8000 else '[FAIL]'}")

    # Compare with old model
    print(f"\nComparison with Old EfficientHybridModel:")
    print(f"  Old model: 36M params (~144 MB)")
    print(f"  New model: {total_params/1e6:.2f}M params (~{model_size_mb_fp32:.2f} MB)")
    print(f"  Reduction: {(1 - total_params/36e6)*100:.1f}% smaller")
    print(f"  Efficiency gain: {36e6/total_params:.1f}x")

    # Per-module parameter count
    print(f"\nParameter Distribution:")
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_params[name] = params
        if params > 0:
            percentage = (params / total_params) * 100
            print(f"  {name:20s}: {params:>10,} ({percentage:5.1f}%)")

    print("\n" + "="*60)
    print("Model validation complete!")
    print("="*60)

    # Return results for potential CI/CD
    return {
        'total_params': total_params,
        'size_mb': model_size_mb_fp32,
        'meets_requirements': meets_size and meets_params
    }


if __name__ == "__main__":
    results = test_model_size()

    # Exit code for CI/CD
    if results['meets_requirements']:
        print("\n[SUCCESS] All requirements met! Ready for training.")
        sys.exit(0)
    else:
        print("\n[WARNING] Some requirements not met. Please adjust model architecture.")
        sys.exit(1)