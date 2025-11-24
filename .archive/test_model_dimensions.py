"""
Debug script to trace where temporal dimension is lost.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.models.mobilenet_v3 import MobileNetV3SignLanguage

def trace_forward_pass():
    """Trace the forward pass to see where dimensions change."""

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

    # Manually trace through the model
    with torch.no_grad():
        # Extract features
        pose = x[:, :, :99]
        hands = x[:, :, 99:225]
        face = x[:, :, 225:1629]
        temporal = x[:, :, 1629:]

        print(f"After feature split:")
        print(f"  Pose: {pose.shape}")
        print(f"  Hands: {hands.shape}")
        print(f"  Face: {face.shape}")
        print(f"  Temporal: {temporal.shape}")

        # Encode modalities
        pose_enc = model.pose_encoder(pose)
        hands_enc = model.hands_encoder(hands)
        face_enc = model.face_encoder(face)
        temporal_enc = model.temporal_encoder(temporal)

        print(f"After encoding:")
        print(f"  Pose encoded: {pose_enc.shape}")
        print(f"  Hands encoded: {hands_enc.shape}")
        print(f"  Face encoded: {face_enc.shape}")
        print(f"  Temporal encoded: {temporal_enc.shape}")

        # Concatenate
        x = torch.cat([pose_enc, hands_enc, face_enc, temporal_enc], dim=-1)
        print(f"After concatenation: {x.shape}")

        # Transpose for Conv1d
        x = x.transpose(1, 2)
        print(f"After transpose: {x.shape}")

        # Stem
        x = model.stem(x)
        print(f"After stem (stride=2): {x.shape}")

        # MobileNet blocks
        for i, block in enumerate(model.blocks):
            x_before = x.shape
            x = block(x)
            if x.shape != x_before:
                print(f"After block {i}: {x_before} -> {x.shape}")

        print(f"After all blocks: {x.shape}")

        # Upsampling
        x_before = x.shape
        x = model.temporal_upsample(x)
        print(f"After temporal_upsample: {x_before} -> {x.shape}")

        x = model.temporal_proj(x)
        print(f"After temporal_proj: {x.shape}")

        # Check final temporal dimension
        current_T = x.size(2)
        print(f"\nFinal temporal dimension: {current_T}")
        print(f"Original temporal dimension: {seq_length}")
        print(f"Reduction factor: {seq_length / current_T:.2f}x")

if __name__ == "__main__":
    trace_forward_pass()