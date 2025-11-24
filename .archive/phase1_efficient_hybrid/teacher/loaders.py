"""Data loaders for teacher model training."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import os
import sys

# Add parent directory to path to allow imports from utils
# MUST be before any utils imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.vocabulary import Vocabulary, filter_annotation


class SignLanguageDataset(Dataset):
    """Dataset for sign language recognition using MediaPipe features."""
    
    def __init__(self, 
                 features_dir: Path,
                 annotations_file: Path,
                 vocabulary: Vocabulary,
                 split: str = 'train',
                 max_length: Optional[int] = None,
                 normalize: bool = True):
        """
        Initialize dataset.
        
        Args:
            features_dir: Directory containing MediaPipe feature files (.npz)
            annotations_file: CSV file with annotations
            vocabulary: Vocabulary object
            split: 'train', 'dev', or 'test'
            max_length: Maximum sequence length (None for no limit)
            normalize: Whether to normalize features
        """
        self.features_dir = Path(features_dir) / split
        self.vocabulary = vocabulary
        self.split = split
        self.max_length = max_length
        self.normalize = normalize
        
        # Load annotations
        self.annotations_df = pd.read_csv(annotations_file, sep='|')
        self.annotations_df = self.annotations_df[self.annotations_df['id'].notna()]
        
        # Filter to available features
        available_features = set(f.stem for f in self.features_dir.glob('*.npz'))
        self.annotations_df = self.annotations_df[
            self.annotations_df['id'].isin(available_features)
        ]
        
        # Compute normalization stats if needed
        self.feature_mean = None
        self.feature_std = None
        if normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self, sample_size: int = 100):
        """Compute feature normalization statistics."""
        print(f"Computing normalization statistics from {sample_size} samples...")
        
        feature_files = list(self.features_dir.glob('*.npz'))[:sample_size]
        all_features = []
        
        for feature_file in feature_files:
            data = np.load(feature_file, allow_pickle=True)
            features = data['features']
            all_features.append(features)
        
        if all_features:
            all_features = np.concatenate(all_features, axis=0)
            self.feature_mean = np.mean(all_features, axis=0, keepdims=True)
            self.feature_std = np.std(all_features, axis=0, keepdims=True) + 1e-8
            print(f"Normalization stats computed: mean shape {self.feature_mean.shape}, "
                  f"std shape {self.feature_std.shape}")
    
    def __len__(self) -> int:
        return len(self.annotations_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        row = self.annotations_df.iloc[idx]
        video_id = row['id']
        annotation = row['annotation']
        
        # Load features
        feature_path = self.features_dir / f"{video_id}.npz"
        data = np.load(feature_path, allow_pickle=True)
        features = data['features'].astype(np.float32)
        
        # Normalize if needed
        if self.normalize and self.feature_mean is not None:
            features = (features - self.feature_mean) / self.feature_std
        
        # Truncate if needed
        if self.max_length and len(features) > self.max_length:
            features = features[:self.max_length]
        
        # Filter excluded tokens from annotation (removes __ON__, loc-, IX, etc.)
        cleaned_annotation = filter_annotation(annotation)
        
        # Encode cleaned annotation
        target_indices = self.vocabulary.encode(cleaned_annotation, filter_excluded=False)
        
        return {
            'features': torch.from_numpy(features),
            'target': torch.tensor(target_indices, dtype=torch.long),
            'video_id': video_id,
            'annotation': cleaned_annotation,  # Store cleaned annotation (without excluded tokens)
            'original_annotation': annotation,  # Keep original for reference if needed
            'sequence_length': len(features)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    # Pad sequences
    max_seq_len = max(item['features'].shape[0] for item in batch)
    feature_dim = batch[0]['features'].shape[1]
    batch_size = len(batch)
    
    # Pad features
    features_padded = torch.zeros(batch_size, max_seq_len, feature_dim)
    sequence_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Pad targets
    max_target_len = max(len(item['target']) for item in batch)
    targets_padded = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    video_ids = []
    annotations = []
    
    for i, item in enumerate(batch):
        seq_len = item['features'].shape[0]
        features_padded[i, :seq_len] = item['features']
        sequence_lengths[i] = seq_len
        
        target_len = len(item['target'])
        targets_padded[i, :target_len] = item['target']
        target_lengths[i] = target_len
        
        video_ids.append(item['video_id'])
        annotations.append(item['annotation'])
    
    return {
        'features': features_padded,
        'sequence_lengths': sequence_lengths,
        'targets': targets_padded,
        'target_lengths': target_lengths,
        'video_ids': video_ids,
        'annotations': annotations
    }


def create_dataloaders(features_dir: Path,
                       annotations_dir: Path,
                       vocabulary: Vocabulary,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       max_length: Optional[int] = None,
                       normalize: bool = True) -> Dict[str, DataLoader]:
    """
    Create data loaders for train, dev, and test sets.
    
    Args:
        features_dir: Directory containing MediaPipe features
        annotations_dir: Directory containing annotation CSV files
        vocabulary: Vocabulary object
        batch_size: Batch size
        num_workers: Number of worker processes
        max_length: Maximum sequence length
        normalize: Whether to normalize features
    
    Returns:
        Dictionary with 'train', 'dev', 'test' DataLoaders
    """
    loaders = {}
    
    for split in ['train', 'dev', 'test']:
        annotations_file = annotations_dir / f'{split}.corpus.csv'
        
        if not annotations_file.exists():
            print(f"Warning: {annotations_file} not found, skipping {split} split")
            continue
        
        dataset = SignLanguageDataset(
            features_dir=features_dir,
            annotations_file=annotations_file,
            vocabulary=vocabulary,
            split=split,
            max_length=max_length,
            normalize=normalize
        )
        
        shuffle = (split == 'train')
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        
        loaders[split] = loader
        print(f"Created {split} loader: {len(dataset)} samples")
    
    return loaders

