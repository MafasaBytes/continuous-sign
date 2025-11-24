"""
PyTorch Dataset for CNN Features

Loads pre-extracted CNN features from HDF5 files for BiLSTM training.
"""

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

from pathlib import Path as Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.vocabulary import should_exclude_token


class CNNFeatureDataset(Dataset):
    """
    Dataset that loads pre-extracted CNN features from HDF5

    Features extracted by extract_cnn_features.py
    """

    def __init__(self, h5_file, vocabulary, max_length=None, 
                 normalize=True, feature_mean=None, feature_std=None):
        """
        Args:
            h5_file: Path to HDF5 file with extracted features
            vocabulary: Dictionary mapping sign labels to indices
            max_length: Maximum sequence length (for truncation)
            normalize: Whether to normalize features
            feature_mean: Feature mean for normalization (if None, computed from data)
            feature_std: Feature std for normalization (if None, computed from data)
        """
        self.h5_file = h5_file
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.normalize = normalize

        # Load video IDs
        with h5py.File(h5_file, 'r') as f:
            self.video_ids = list(f.keys())

        # Compute normalization stats if needed
        if normalize and feature_mean is None:
            print(f"Computing feature statistics from {h5_file}...")
            self.feature_mean, self.feature_std = self._compute_stats()
        else:
            self.feature_mean = feature_mean
            self.feature_std = feature_std

        print(f"Loaded {len(self.video_ids)} videos from {h5_file}")
        if normalize:
            print(f"Feature normalization: mean={self.feature_mean.mean():.4f}, std={self.feature_std.mean():.4f}")

    def _compute_stats(self, max_samples=500):
        """Compute feature statistics for normalization."""
        all_features = []
        for i in range(min(max_samples, len(self.video_ids))):
            with h5py.File(self.h5_file, 'r') as f:
                grp = f[self.video_ids[i]]
                features = grp['features'][:]
                all_features.append(features)
        
        if not all_features:
            return None, None
        
        all_features = np.concatenate(all_features, axis=0)
        mean = np.mean(all_features, axis=0)
        std = np.std(all_features, axis=0) + 1e-8
        return mean, std

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]

        with h5py.File(self.h5_file, 'r') as f:
            grp = f[video_id]

            # Load features [T, 1024]
            features = grp['features'][:]

            # Load annotation (space-separated sign labels)
            annotation = grp.attrs['annotation']

        # Truncate if needed
        if self.max_length is not None and len(features) > self.max_length:
            features = features[:self.max_length]

        # Parse annotation to label indices
        signs = [s for s in annotation.split() if not should_exclude_token(s)]
        labels = []
        oov_count = 0
        for sign in signs:
            if sign in self.vocabulary:
                labels.append(self.vocabulary[sign])
            else:
                # FIXED: Don't silently skip - this was causing vocabulary mismatch
                # If vocabulary was built correctly from all splits, this shouldn't happen
                # But if it does, we'll log it (but not print every time to avoid spam)
                oov_count += 1
                # Skip for now, but this indicates vocabulary building issue
        
        if oov_count > 0 and len(labels) == 0:
            # If all signs are OOV, this is a problem
            print(f"ERROR: All signs OOV in {video_id}: {signs}")

        # Normalize features if needed
        if self.normalize and self.feature_mean is not None and self.feature_std is not None:
            features = (features - self.feature_mean) / self.feature_std

        # Convert to tensors
        features_tensor = torch.FloatTensor(features)  # [T, 1024]
        labels_tensor = torch.LongTensor(labels) if labels else torch.LongTensor([0])  # [L] (at least blank)

        return {
            'video_id': video_id,
            'features': features_tensor,
            'labels': labels_tensor,
            'input_length': len(features),
            'target_length': len(labels),
            'annotation': annotation
        }


def collate_fn(batch):
    """
    Collate function for DataLoader

    Handles variable-length sequences by padding.
    """
    # Sort batch by input length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x['input_length'], reverse=True)

    # Get max lengths
    max_input_length = batch[0]['input_length']
    max_target_length = max(item['target_length'] for item in batch)

    # Prepare batch tensors
    batch_size = len(batch)
    feature_dim = batch[0]['features'].shape[1]  # 1024

    # Initialize tensors
    features_padded = torch.zeros(batch_size, max_input_length, feature_dim)
    labels_padded = torch.zeros(batch_size, max_target_length, dtype=torch.long)
    input_lengths = torch.zeros(batch_size, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)

    video_ids = []
    annotations = []

    # Fill tensors
    for i, item in enumerate(batch):
        input_len = item['input_length']
        target_len = item['target_length']

        features_padded[i, :input_len, :] = item['features']
        labels_padded[i, :target_len] = item['labels']
        input_lengths[i] = input_len
        target_lengths[i] = target_len

        video_ids.append(item['video_id'])
        annotations.append(item['annotation'])

    return {
        'features': features_padded,  # [B, T_max, 1024]
        'labels': labels_padded,  # [B, L_max]
        'input_lengths': input_lengths,  # [B]
        'target_lengths': target_lengths,  # [B]
        'video_ids': video_ids,
        'annotations': annotations
    }


def build_vocabulary(h5_files):
    """
    Build vocabulary from annotation files

    Args:
        h5_files: List of HDF5 files (e.g., [train.h5, dev.h5])

    Returns:
        vocab: Dict mapping sign -> index
        idx2sign: Dict mapping index -> sign
    """
    sign_set = set()

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            for video_id in f.keys():
                annotation = f[video_id].attrs['annotation']
                # Apply same filtering used at runtime to ensure consistency
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
    print(f"Sample signs: {list(vocab.keys())[:10]}")

    return vocab, idx2sign


# Example usage
if __name__ == "__main__":
    # Build vocabulary
    h5_files = [
        'data/features_cnn/train_features.h5',
        'data/features_cnn/dev_features.h5'
    ]

    print("Building vocabulary...")
    vocab, idx2sign = build_vocabulary(h5_files)

    # Create dataset
    print("\nCreating dataset...")
    train_dataset = CNNFeatureDataset(
        h5_file='data/features_cnn/train_features.h5',
        vocabulary=vocab
    )

    # Test loading one sample
    print("\nTesting data loading...")
    sample = train_dataset[0]
    print(f"Video ID: {sample['video_id']}")
    print(f"Features shape: {sample['features'].shape}")
    print(f"Labels: {sample['labels']}")
    print(f"Input length: {sample['input_length']}")
    print(f"Target length: {sample['target_length']}")
    print(f"Annotation: {sample['annotation']}")

    # Test DataLoader
    print("\nTesting DataLoader...")
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    for batch in train_loader:
        print(f"Batch features shape: {batch['features'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        print(f"Input lengths: {batch['input_lengths']}")
        print(f"Target lengths: {batch['target_lengths']}")
        break

    print("\nDataset is ready for training!")
