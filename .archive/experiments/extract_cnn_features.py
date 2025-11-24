"""
Feature Extraction from Pre-trained GoogLeNet

This script extracts 1024-D CNN features from video frames using GoogLeNet.

Strategy:
1. Start with PyTorch's pre-trained GoogLeNet (ImageNet weights)
2. Extract features from 'avgpool' layer (before final FC)
3. Save features to HDF5 for efficient training

Later: Can be replaced with exact RWTH Caffe weights via conversion.
"""

import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import csv
from pathlib import Path


class GoogLeNetFeatureExtractor:
    """Extract 1024-D features from GoogLeNet avgpool layer"""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load pre-trained GoogLeNet
        self.model = models.googlenet(pretrained=True)

        # Remove final FC layers (we want features before classification)
        # GoogLeNet structure: ... -> avgpool -> dropout -> fc
        # We want output after avgpool (1024-D)
        self.feature_extractor = nn.Sequential(
            *list(self.model.children())[:-2]  # Remove dropout and fc
        )

        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # RWTH preprocessing (from net.prototxt):
        # - Input: 224×224 (center crop from 256×256)
        # - Mean subtraction from training data
        # For now, use ImageNet normalization (close approximation)
        self.transform = transforms.Compose([
            transforms.Resize(256),  # Resize to 256
            transforms.CenterCrop(224),  # Center crop to 224×224
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])

    def extract_frame_feature(self, frame_path):
        """Extract 1024-D feature from single frame"""
        try:
            # Load image
            img = Image.open(frame_path).convert('RGB')

            # Preprocess
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(img_tensor)
                # Output shape: [1, 1024, 1, 1]
                features = features.squeeze()  # Shape: [1024]

            return features.cpu().numpy()

        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            return None

    def extract_video_features(self, frame_pattern):
        """
        Extract features from all frames in a video

        Args:
            frame_pattern: Glob pattern like "path/to/video/*.png"

        Returns:
            features: numpy array of shape [T, 1024]
        """
        frame_paths = sorted(glob.glob(frame_pattern))

        if len(frame_paths) == 0:
            print(f"Warning: No frames found for pattern {frame_pattern}")
            return None

        features_list = []
        for frame_path in frame_paths:
            feat = self.extract_frame_feature(frame_path)
            if feat is not None:
                features_list.append(feat)

        if len(features_list) == 0:
            return None

        # Stack to [T, 1024]
        features = np.stack(features_list, axis=0)
        return features


def load_annotations(split='train'):
    """
    Load RWTH-PHOENIX annotations

    Returns:
        List of dicts with keys: id, folder, signer, annotation
    """
    anno_file = f"data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/{split}.SI5.corpus.csv"

    samples = []
    with open(anno_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            samples.append(row)

    print(f"Loaded {len(samples)} samples from {split} set")
    return samples


def extract_features_for_split(split='train', output_dir='data/features_cnn', batch_size=1):
    """
    Extract CNN features for entire data split

    Args:
        split: 'train', 'dev', or 'test'
        output_dir: Directory to save HDF5 files
        batch_size: Currently only supports 1 (video-level processing)
    """
    print(f"\n{'='*60}")
    print(f"Extracting features for {split} set")
    print(f"{'='*60}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize feature extractor
    extractor = GoogLeNetFeatureExtractor()

    # Load annotations
    samples = load_annotations(split)

    # Prepare HDF5 file
    h5_file = os.path.join(output_dir, f'{split}_features.h5')

    features_dict = {}
    failed_samples = []

    # Extract features for each video
    for sample in tqdm(samples, desc=f"Processing {split}"):
        video_id = sample['id']
        folder = sample['folder']
        annotation = sample['annotation']

        # Build frame pattern
        base_path = f"data/raw_data/phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px/{split}"
        frame_pattern = os.path.join(base_path, folder)

        # Extract features
        features = extractor.extract_video_features(frame_pattern)

        if features is not None:
            features_dict[video_id] = {
                'features': features,
                'annotation': annotation,
                'folder': folder,
                'num_frames': len(features)
            }
        else:
            failed_samples.append(video_id)
            print(f"Failed to extract features for {video_id}")

    # Save to HDF5
    print(f"\nSaving features to {h5_file}...")
    with h5py.File(h5_file, 'w') as f:
        for video_id, data in tqdm(features_dict.items(), desc="Saving to HDF5"):
            grp = f.create_group(video_id)
            grp.create_dataset('features', data=data['features'], compression='gzip')
            grp.attrs['annotation'] = data['annotation']
            grp.attrs['folder'] = data['folder']
            grp.attrs['num_frames'] = data['num_frames']

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary for {split} set:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Successful: {len(features_dict)}")
    print(f"  Failed: {len(failed_samples)}")
    if failed_samples:
        print(f"  Failed IDs: {failed_samples[:5]}..." if len(failed_samples) > 5 else f"  Failed IDs: {failed_samples}")
    print(f"  Output: {h5_file}")
    print(f"{'='*60}\n")

    return features_dict, failed_samples


def validate_features(h5_file):
    """Validate extracted features"""
    print(f"\nValidating {h5_file}...")

    with h5py.File(h5_file, 'r') as f:
        video_ids = list(f.keys())
        print(f"Total videos: {len(video_ids)}")

        # Check first video
        if len(video_ids) > 0:
            sample_id = video_ids[0]
            sample_grp = f[sample_id]
            features = sample_grp['features'][:]

            print(f"\nSample video: {sample_id}")
            print(f"  Feature shape: {features.shape}")
            print(f"  Feature dtype: {features.dtype}")
            print(f"  Annotation: {sample_grp.attrs['annotation']}")
            print(f"  Num frames: {sample_grp.attrs['num_frames']}")
            print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
            print(f"  Feature mean: {features.mean():.3f}")
            print(f"  Feature std: {features.std():.3f}")

        # Statistics
        total_frames = 0
        feature_dims = []
        for vid_id in video_ids:
            feats = f[vid_id]['features'][:]
            total_frames += len(feats)
            feature_dims.append(feats.shape[1])

        print(f"\nDataset statistics:")
        print(f"  Total frames: {total_frames}")
        print(f"  Avg frames per video: {total_frames / len(video_ids):.1f}")
        print(f"  Feature dimension: {feature_dims[0]} (all should be 1024)")
        print(f"  Dimension check: {all(d == 1024 for d in feature_dims)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract CNN features from RWTH-PHOENIX dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'dev', 'test'],
                        help='Data split to process')
    parser.add_argument('--output_dir', type=str, default='data/features_cnn',
                        help='Output directory for HDF5 files')
    parser.add_argument('--validate_only', action='store_true',
                        help='Only validate existing features')

    args = parser.parse_args()

    if args.validate_only:
        h5_file = os.path.join(args.output_dir, f'{args.split}_features.h5')
        if os.path.exists(h5_file):
            validate_features(h5_file)
        else:
            print(f"Error: {h5_file} not found")
    else:
        # Extract features
        features_dict, failed = extract_features_for_split(
            split=args.split,
            output_dir=args.output_dir
        )

        # Validate
        h5_file = os.path.join(args.output_dir, f'{args.split}_features.h5')
        validate_features(h5_file)

        print("\n" + "="*60)
        print("Feature extraction complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run for other splits:")
        print(f"   python experiments/extract_cnn_features.py --split dev")
        print(f"   python experiments/extract_cnn_features.py --split test")
        print("\n2. Train BiLSTM model on extracted features")
        print("\n3. (Optional) Convert RWTH Caffe model for exact weights")
        print("="*60)
