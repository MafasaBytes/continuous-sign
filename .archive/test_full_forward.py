"""
Test full forward pass through the model.
"""

import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.models.mobilenet_v3 import MobileNetV3SignLanguage

def test_forward():
    """Test complete forward pass."""

    # Create model
    model = MobileNetV3SignLanguage(vocab_size=973, dropout=0.3)
    model.eval()

    # Create dummy input
    batch_size = 1
    seq_length = 176
    feature_dim = 6516

    x = torch.randn(batch_size, seq_length, feature_dim)
    input_lengths = torch.tensor([seq_length])

    print(f"Input shape: {x.shape}")
    print(f"Input lengths: {input_lengths}")

    # Forward pass
    with torch.no_grad():
        output = model(x, input_lengths)

    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: [T={seq_length}, B={batch_size}, vocab_size=973]")

    # Check if output is valid
    if output.shape[0] < 12:  # Minimum for 12 labels
        print("\n[ERROR] Output temporal dimension too small for CTC!")
        print(f"  Output has {output.shape[0]} frames")
        print(f"  Need at least 12 frames for 12 labels")

    # Check for NaN or Inf
    if torch.isnan(output).any():
        print("\n[ERROR] Output contains NaN values!")
    if torch.isinf(output).any():
        print("\n[ERROR] Output contains Inf values!")

    # Check output range (should be log probabilities)
    print(f"\nOutput statistics:")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")
    print(f"  Mean: {output.mean().item():.4f}")

if __name__ == "__main__":
    test_forward()