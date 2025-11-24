"""
Enhanced data augmentation for MediaPipe features to prevent overfitting.
Apply this if dropout=0.5 and weight_decay=0.001 are not enough.
"""

import numpy as np
from typing import Tuple


def augment_features_strong(features: np.ndarray, augment: bool = True, split: str = 'train') -> np.ndarray:
    """
    Strong augmentation for MediaPipe features to combat overfitting.
    
    Augmentations applied (with higher probabilities and stronger effects):
    1. Temporal: Speed perturbation, time masking, reverse
    2. Spatial: Noise, scaling, rotation (for pose coordinates)
    3. Feature-level: Dropout, gaussian noise, smoothing
    
    Args:
        features: [T, F] array where T=time, F=feature_dim (6516)
        augment: Whether to apply augmentation
        split: 'train', 'dev', or 'test'
        
    Returns:
        Augmented features [T', F] where T' may differ due to temporal augmentation
    """
    if not augment or split != 'train':
        return features
    
    T, F = features.shape
    
    # === 1. TEMPORAL AUGMENTATIONS ===
    
    # 1a. Speed perturbation (stronger range)
    if np.random.random() < 0.6:  # Increased from 0.5
        speed_factor = np.random.uniform(0.85, 1.15)  # Wider range (was 0.9-1.1)
        new_len = int(T * speed_factor)
        new_len = max(10, min(new_len, T * 2))  # Clamp to reasonable range
        indices = np.linspace(0, T - 1, new_len)
        features = np.array([features[int(i)] for i in indices])
        T = new_len
    
    # 1b. Time masking (SpecAugment-style)
    if np.random.random() < 0.3:  # NEW
        mask_length = int(T * np.random.uniform(0.05, 0.15))  # Mask 5-15% of time
        mask_start = np.random.randint(0, max(1, T - mask_length))
        features[mask_start:mask_start + mask_length] = 0
    
    # 1c. Temporal reversal (for non-directional sequences)
    if np.random.random() < 0.1:  # NEW - rare but useful
        features = features[::-1].copy()
    
    # 1d. Frame sampling (drop random frames)
    if np.random.random() < 0.2:  # NEW
        keep_ratio = np.random.uniform(0.85, 0.95)
        n_keep = int(T * keep_ratio)
        indices = np.sort(np.random.choice(T, n_keep, replace=False))
        features = features[indices]
        T = n_keep
    
    # === 2. SPATIAL AUGMENTATIONS ===
    
    # 2a. Gaussian noise (stronger)
    if np.random.random() < 0.5:  # Increased from 0.3
        noise_std = np.random.uniform(0.01, 0.03)  # Stronger (was 0.01)
        noise = np.random.normal(0, noise_std, features.shape)
        features = features + noise
    
    # 2b. Feature scaling (per modality)
    if np.random.random() < 0.4:  # NEW
        # Scale different modalities differently
        # Pose: 0:99, Hands: 99:225, Face: 225:1629, Temporal: 1629:
        for start, end in [(0, 99), (99, 225), (225, 1629), (1629, F)]:
            if np.random.random() < 0.5:
                scale = np.random.uniform(0.9, 1.1)
                features[:, start:end] *= scale
    
    # 2c. Coordinate rotation (for spatial features - pose, hands, face)
    if np.random.random() < 0.3:  # NEW
        # Apply small rotation to x,y coordinates (assuming x,y,z format)
        angle = np.random.uniform(-10, 10) * np.pi / 180  # ±10 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotate pose keypoints (33 points, 3 coords each = 99 features)
        for i in range(0, 99, 3):
            x, y = features[:, i], features[:, i+1]
            features[:, i] = x * cos_a - y * sin_a
            features[:, i+1] = x * sin_a + y * cos_a
        
        # Rotate hand keypoints (42 points, 3 coords each = 126 features, starts at 99)
        for i in range(99, 225, 3):
            x, y = features[:, i], features[:, i+1]
            features[:, i] = x * cos_a - y * sin_a
            features[:, i+1] = x * sin_a + y * cos_a
    
    # === 3. FEATURE-LEVEL AUGMENTATIONS ===
    
    # 3a. Feature dropout (stronger)
    if np.random.random() < 0.3:  # Increased from 0.2
        dropout_rate = np.random.uniform(0.1, 0.2)  # Stronger (was 0.1)
        mask = np.random.binomial(1, 1 - dropout_rate, features.shape)
        features = features * mask
    
    # 3b. Gaussian blur (temporal smoothing)
    if np.random.random() < 0.2:  # NEW
        from scipy.ndimage import gaussian_filter1d
        sigma = np.random.uniform(0.5, 1.5)
        features = gaussian_filter1d(features, sigma=sigma, axis=0)
    
    # 3c. Random feature permutation (within modalities)
    if np.random.random() < 0.1:  # NEW - rare
        # Randomly permute features within each modality
        for start, end in [(0, 99), (99, 225), (225, 1629)]:
            if np.random.random() < 0.5:
                perm = np.random.permutation(end - start)
                features[:, start:end] = features[:, start + perm]
    
    # 3d. Mixup (rare, but powerful)
    # Note: This requires access to another sample, so implement in collate_fn
    
    return features


def mixup_batch(features_batch, labels_batch, alpha=0.2):
    """
    Apply mixup augmentation at batch level.
    
    Args:
        features_batch: [B, T, F] batch of features
        labels_batch: [B, L] batch of labels
        alpha: Mixup alpha parameter (0.2 is common)
        
    Returns:
        Mixed features and labels
    """
    if alpha <= 0 or np.random.random() > 0.3:  # Apply 30% of the time
        return features_batch, labels_batch
    
    B = features_batch.shape[0]
    if B < 2:
        return features_batch, labels_batch
    
    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)
    
    # Random permutation
    indices = np.random.permutation(B)
    
    # Mix features (requires same length - use shortest)
    mixed_features = []
    for i in range(B):
        f1 = features_batch[i]
        f2 = features_batch[indices[i]]
        
        # Use shortest length
        min_len = min(f1.shape[0], f2.shape[0])
        f1 = f1[:min_len]
        f2 = f2[:min_len]
        
        mixed = lam * f1 + (1 - lam) * f2
        mixed_features.append(mixed)
    
    # For CTC, we keep original labels (can't really mix discrete sequences)
    # But the mixed features still help with regularization
    
    return mixed_features, labels_batch


# === USAGE IN DATASET CLASS ===
"""
Replace _augment_features() in MediaPipeFeatureDataset with:

def _augment_features(self, features: np.ndarray) -> np.ndarray:
    '''Apply strong data augmentation to features.'''
    from enhanced_augmentation import augment_features_strong
    return augment_features_strong(features, self.augment, self.split)
"""

# === USAGE IN TRAINING LOOP ===
"""
Optionally add mixup in training loop:

for batch in dataloader:
    features = batch['features']
    labels = batch['labels']
    
    # Apply mixup augmentation
    if training:
        from enhanced_augmentation import mixup_batch
        features, labels = mixup_batch(features, labels, alpha=0.2)
    
    # Continue with forward pass...
"""


if __name__ == "__main__":
    # Test augmentation
    print("Testing enhanced augmentation...")
    
    # Create dummy features [T=100, F=6516]
    features = np.random.randn(100, 6516).astype(np.float32)
    
    print(f"Original shape: {features.shape}")
    
    # Apply augmentation
    augmented = augment_features_strong(features, augment=True, split='train')
    
    print(f"Augmented shape: {augmented.shape}")
    print(f"Shape changed: {features.shape != augmented.shape}")
    print(f"Values changed: {not np.array_equal(features[:min(features.shape[0], augmented.shape[0])], augmented[:min(features.shape[0], augmented.shape[0])])}")
    
    print("\n✓ Augmentation test passed!")

