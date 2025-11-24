"""
PyTorch Dataset for MediaPipe Features

Loads pre-extracted MediaPipe Holistic features from NPZ files for BiLSTM training.
Features include landmarks, velocities, accelerations, and spatial relationships.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.vocabulary import should_exclude_token


class MediaPipeFeatureDataset(Dataset):
    """
    Dataset that loads pre-extracted MediaPipe features from NPZ files

    Features extracted by MediaPipe Holistic with temporal and spatial enhancements
    """

    def __init__(self, npz_dir, annotation_csv, vocabulary, max_length=None,
                 normalize=True, feature_mean=None, feature_std=None,
                 suppress_oov_warnings: bool = False):
        """
        Args:
            npz_dir: Directory containing .npz feature files
            annotation_csv: Path to Phoenix annotation CSV file (e.g., train.corpus.csv)
            vocabulary: Dictionary mapping sign labels to indices
            max_length: Maximum sequence length (for truncation)
            normalize: Whether to normalize features
            feature_mean: Feature mean for normalization (if None, computed from data)
            feature_std: Feature std for normalization (if None, computed from data)
        """
        self.npz_dir = Path(npz_dir)
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.normalize = normalize
        self.suppress_oov_warnings = suppress_oov_warnings

        # Load annotations
        self.annotations = self._load_annotations(annotation_csv)

        # Get list of available NPZ files
        self.samples = []
        for video_id, annotation in self.annotations.items():
            npz_path = self.npz_dir / f"{video_id}.npz"
            if npz_path.exists():
                self.samples.append({
                    'video_id': video_id,
                    'npz_path': npz_path,
                    'annotation': annotation
                })

        print(f"Found {len(self.samples)}/{len(self.annotations)} samples with both features and annotations")

        if not self.samples:
            raise ValueError(f"No samples found! Check that {npz_dir} contains .npz files matching {annotation_csv}")

        # Compute normalization stats if needed
        if normalize and feature_mean is None:
            print(f"Computing feature statistics from {npz_dir}...")
            self.feature_mean, self.feature_std = self._compute_stats()
        else:
            self.feature_mean = feature_mean
            self.feature_std = feature_std

        if normalize and self.feature_mean is not None:
            print(f"Feature normalization: mean={self.feature_mean.mean():.4f}, std={self.feature_std.mean():.4f}")

    def _load_annotations(self, annotation_csv):
        """Load annotations from Phoenix corpus CSV file."""
        df = pd.read_csv(annotation_csv, sep='|')
        annotations = {}
        for _, row in df.iterrows():
            video_id = row['id']  # Phoenix uses 'id' column for video ID
            annotation = row['annotation']  # Space-separated sign labels
            annotations[video_id] = annotation
        return annotations

    def _compute_stats(self, max_samples=500):
        """Compute feature statistics for normalization."""
        all_features = []
        num_samples = min(max_samples, len(self.samples))

        print(f"Computing stats from {num_samples} samples...")
        for i in range(num_samples):
            data = np.load(self.samples[i]['npz_path'])
            features = data['features']
            all_features.append(features)

        if not all_features:
            return None, None

        all_features = np.concatenate(all_features, axis=0)
        mean = np.mean(all_features, axis=0).astype(np.float32)
        std = np.std(all_features, axis=0).astype(np.float32) + 1e-8
        return mean, std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample['video_id']
        npz_path = sample['npz_path']
        annotation = sample['annotation']

        # Load MediaPipe features from NPZ
        data = np.load(npz_path)
        features = data['features']  # Shape: [T, 6516]
        detection_masks = data['detection_masks']  # Shape: [T, 543]

        # Truncate if needed
        if self.max_length is not None and len(features) > self.max_length:
            features = features[:self.max_length]
            detection_masks = detection_masks[:self.max_length]

        # Normalize features
        if self.normalize and self.feature_mean is not None:
            features = (features - self.feature_mean) / self.feature_std

        # Parse annotation to label indices
        signs = [s for s in annotation.split() if not should_exclude_token(s)]
        labels = []
        oov_count = 0
        for sign in signs:
            if sign in self.vocabulary:
                labels.append(self.vocabulary[sign])
            else:
                oov_count += 1

        if (not getattr(self, 'suppress_oov_warnings', False)) and oov_count > 0 and oov_count < 10:
            print(f"Warning: {oov_count} OOV tokens in {video_id}")

        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        labels_tensor = torch.LongTensor(labels)

        return {
            'video_id': video_id,
            'features': features_tensor,
            'labels': labels_tensor,
            'input_lengths': len(features),
            'target_lengths': len(labels),
            'detection_masks': torch.BoolTensor(detection_masks)
        }


def collate_fn(batch):
    """
    Collate function for DataLoader.
    Pads sequences to same length in batch.
    """
    # Sort by sequence length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x['input_lengths'], reverse=True)

    # Get max lengths
    max_input_len = max(x['input_lengths'] for x in batch)
    max_target_len = max(x['target_lengths'] for x in batch)
    feature_dim = batch[0]['features'].shape[1]  # 6516 for MediaPipe

    # Initialize tensors
    batch_size = len(batch)
    features_padded = torch.zeros(batch_size, max_input_len, feature_dim)
    labels_padded = torch.zeros(batch_size, max_target_len).long()
    input_lengths = torch.zeros(batch_size).long()
    target_lengths = torch.zeros(batch_size).long()
    detection_masks_padded = torch.zeros(batch_size, max_input_len, batch[0]['detection_masks'].shape[1]).bool()

    video_ids = []

    # Fill tensors
    for i, sample in enumerate(batch):
        input_len = sample['input_lengths']
        target_len = sample['target_lengths']

        features_padded[i, :input_len, :] = sample['features']
        labels_padded[i, :target_len] = sample['labels']
        input_lengths[i] = input_len
        target_lengths[i] = target_len
        detection_masks_padded[i, :input_len, :] = sample['detection_masks']
        video_ids.append(sample['video_id'])

    return {
        'video_ids': video_ids,
        'features': features_padded,
        'labels': labels_padded,
        'input_lengths': input_lengths,
        'target_lengths': target_lengths,
        'detection_masks': detection_masks_padded
    }


def build_vocabulary(annotation_csvs):
    """
    Build vocabulary from annotation CSV files

    Args:
        annotation_csvs: List of annotation CSV files (e.g., [train.corpus.csv, dev.corpus.csv])

    Returns:
        vocab: Dict mapping sign -> index
        idx2sign: Dict mapping index -> sign
    """
    sign_set = set()

    for csv_file in annotation_csvs:
        df = pd.read_csv(csv_file, sep='|')
        for annotation in df['annotation']:
            signs = [s for s in annotation.split() if not should_exclude_token(s)]
            sign_set.update(signs)

    # Sort for consistency
    signs_sorted = sorted(sign_set)

    # Create vocabulary (reserve 0 for CTC blank)
    vocab = {sign: idx + 1 for idx, sign in enumerate(signs_sorted)}
    vocab['<BLANK>'] = 0  # CTC blank token

    # Reverse mapping
    idx2sign = {idx: sign for sign, idx in vocab.items()}

    print(f"Vocabulary size: {len(vocab)} (including blank)")
    print(f"Sample signs: {list(vocab.keys())[1:11]}")  # Skip <BLANK>

    return vocab, idx2sign


# Example usage
if __name__ == "__main__":
    # Build vocabulary
    annotation_csvs = [
        'data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv',
        'data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/dev.SI5.corpus.csv'
    ]

    print("Building vocabulary...")
    vocab, idx2sign = build_vocabulary(annotation_csvs)

    # Create dataset
    train_dataset = MediaPipeFeatureDataset(
        npz_dir='data/teacher_features/mediapipe_full/train',
        annotation_csv=annotation_csvs[0],
        vocabulary=vocab,
        max_length=None,
        normalize=True
    )

    print(f"\nDataset created with {len(train_dataset)} samples")

    # Test loading one sample
    sample = train_dataset[0]
    print(f"\nSample 0:")
    print(f"  Video ID: {sample['video_id']}")
    print(f"  Features shape: {sample['features'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")
    print(f"  Input length: {sample['input_lengths']}")
    print(f"  Target length: {sample['target_lengths']}")
    print(f"  Detection masks shape: {sample['detection_masks'].shape}")
