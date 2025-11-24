"""
Diagnostic script to check data loading and model compatibility.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from src.data.dataset import MediaPipeFeatureDataset, build_vocabulary, Vocabulary
from src.models.mobilenet_v3 import create_mobilenet_v3_model
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    # Build vocabulary
    annotation_file = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    vocab = build_vocabulary(annotation_file)
    print(f"\n[OK] Vocabulary size: {len(vocab)}")

    # Create dataset
    data_dir = Path("data/teacher_features/mediapipe_full")
    dataset = MediaPipeFeatureDataset(
        data_dir=data_dir,
        annotation_file=annotation_file,
        vocabulary=vocab,
        split='train',
        augment=False,
        normalize=False
    )
    print(f"[OK] Dataset loaded with {len(dataset)} samples")

    # Test loading a few samples
    print("\nTesting data loading:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"  Sample {i}:")
        print(f"    Video ID: {sample['video_id']}")
        print(f"    Feature shape: {sample['features'].shape}")
        print(f"    Labels shape: {sample['labels'].shape}")
        print(f"    Words: {sample['words'][:5]}")  # First 5 words

    # Check feature dimensions
    if len(dataset) > 0:
        sample = dataset[0]
        input_dim = sample['features'].shape[-1]  # Feature dimension
        print(f"\n[OK] Input feature dimension: {input_dim}")

        # Try to create model
        try:
            model = create_mobilenet_v3_model(
                vocab_size=len(vocab),
                dropout=0.3
            )
            print(f"[OK] Model created successfully")

            # Test forward pass
            batch_features = sample['features'].unsqueeze(0)  # Add batch dimension [1, T, 6516]
            input_lengths = torch.tensor([sample['features'].shape[0]])

            with torch.no_grad():
                output = model(batch_features, input_lengths)
                print(f"[OK] Forward pass successful")
                print(f"  Output shape: {output.shape}")
                print(f"  Expected: (batch=1, time_steps, num_classes={len(vocab)})")

                # Check for downsampling issues
                time_reduction = sample['features'].shape[0] / output.shape[1]
                print(f"\n[WARNING]  Time reduction factor: {time_reduction:.2f}x")

                if time_reduction > 2:
                    print("  [ERROR] PROBLEM: Too much downsampling! This will cause CTC alignment issues.")
                    print("     CTC needs output_length >= target_length/2 typically")

                # Check if output length is sufficient for CTC
                target_length = len(sample['labels'])
                output_length = output.shape[1]
                print(f"  Target length (labels): {target_length}")
                print(f"  Output length (model): {output_length}")

                if output_length < target_length:
                    print("  [ERROR] CRITICAL: Output sequence shorter than target! CTC will fail.")
                elif output_length < target_length * 2:
                    print("  [WARNING]  WARNING: Output might be too short for good CTC alignment.")
                else:
                    print("  [OK] Output length sufficient for CTC")

        except Exception as e:
            print(f"[ERROR] Model creation/forward pass failed: {e}")

    # Check NPZ file structure
    print("\nChecking NPZ file structure:")
    sample_npz = list(Path("data/teacher_features/mediapipe_full/train").glob("*.npz"))[:1]
    if sample_npz:
        data = np.load(sample_npz[0])
        print(f"  NPZ keys: {list(data.keys())}")
        for key in data.keys():
            print(f"    {key}: shape {data[key].shape}, dtype {data[key].dtype}")

if __name__ == "__main__":
    main()