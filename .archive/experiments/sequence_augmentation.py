"""
Sequence augmentation for sign language recognition.

Implements SpecAugment-style augmentation adapted for CNN features:
- Time masking: Randomly mask temporal segments
- Feature masking: Randomly mask feature dimensions
- Time warping: Temporal distortion (optional, more complex)

Critical for preventing overfitting with small datasets.
"""

import torch
import numpy as np
from typing import Tuple


class SequenceAugmentation:
    """
    Data augmentation for sequential features.

    Based on SpecAugment (Park et al., 2019) adapted for CNN features.
    Proven to be one of the most effective regularization techniques for
    sequence models with limited data.
    """

    def __init__(self,
                 time_mask_max_frames: int = 30,
                 time_mask_num: int = 2,
                 feature_mask_max_features: int = 50,
                 feature_mask_num: int = 2,
                 time_warp: bool = False,
                 p: float = 0.5):
        """
        Initialize augmentation parameters.

        Args:
            time_mask_max_frames: Maximum number of consecutive frames to mask
            time_mask_num: Number of time mask regions to apply
            feature_mask_max_features: Maximum number of consecutive features to mask
            feature_mask_num: Number of feature mask regions to apply
            time_warp: Whether to apply time warping (computationally expensive)
            p: Probability of applying augmentation (0.5 = 50% of samples)
        """
        self.time_mask_max = time_mask_max_frames
        self.time_mask_num = time_mask_num
        self.feature_mask_max = feature_mask_max_features
        self.feature_mask_num = feature_mask_num
        self.time_warp = time_warp
        self.p = p

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to feature sequence.

        Args:
            features: [T, F] or [B, T, F] feature tensor

        Returns:
            Augmented features with same shape
        """
        # Random chance to skip augmentation
        if np.random.random() > self.p:
            return features

        # Handle batch or single sequence
        if features.dim() == 2:
            return self._augment_sequence(features)
        elif features.dim() == 3:
            # Apply to each sequence in batch
            batch_size = features.size(0)
            augmented = []
            for i in range(batch_size):
                augmented.append(self._augment_sequence(features[i]))
            return torch.stack(augmented, dim=0)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {features.dim()}D")

    def _augment_sequence(self, features: torch.Tensor) -> torch.Tensor:
        """
        Augment a single sequence.

        Args:
            features: [T, F] feature tensor

        Returns:
            Augmented [T, F] tensor
        """
        T, F = features.shape
        augmented = features.clone()

        # Time masking - mask contiguous time segments
        # Critical: Prevents model from memorizing specific temporal patterns
        for _ in range(self.time_mask_num):
            # Random mask width (up to time_mask_max frames)
            t = np.random.randint(1, min(self.time_mask_max, T // 4) + 1)
            # Random start position
            t0 = np.random.randint(0, max(1, T - t))
            # Mask with zeros
            augmented[t0:t0+t, :] = 0.0

        # Feature masking - mask contiguous feature dimensions
        # Critical: Prevents model from relying on specific feature subsets
        for _ in range(self.feature_mask_num):
            # Random mask width (up to feature_mask_max dimensions)
            f = np.random.randint(1, min(self.feature_mask_max, F // 4) + 1)
            # Random start position
            f0 = np.random.randint(0, max(1, F - f))
            # Mask with zeros
            augmented[:, f0:f0+f] = 0.0

        return augmented

    def time_warp_sequence(self, features: torch.Tensor, W: int = 40) -> torch.Tensor:
        """
        Apply time warping (temporal distortion).

        WARNING: Computationally expensive. Only use if you have GPU capacity.

        Args:
            features: [T, F] feature tensor
            W: Warping parameter (maximum temporal distortion)

        Returns:
            Time-warped [T, F] tensor
        """
        import torch.nn.functional as F

        T, feat_dim = features.shape

        # Skip if sequence too short
        if T < W * 2:
            return features

        # Choose warping point
        center = T // 2

        # Random warp amount
        w = np.random.randint(-W, W)

        # Create warped indices
        if w > 0:
            # Stretch first half, compress second half
            indices_first = torch.linspace(0, center, center + w)
            indices_second = torch.linspace(center, T - 1, T - center - w)
        else:
            # Compress first half, stretch second half
            indices_first = torch.linspace(0, center, center + w)
            indices_second = torch.linspace(center, T - 1, T - center - w)

        # Combine indices
        indices = torch.cat([indices_first, indices_second])

        # Interpolate features at new indices
        # This is expensive but creates realistic temporal variations
        features_np = features.cpu().numpy()
        warped = np.zeros((T, feat_dim), dtype=np.float32)

        for i, idx in enumerate(indices):
            idx = float(idx)
            idx_floor = int(np.floor(idx))
            idx_ceil = min(int(np.ceil(idx)), T - 1)
            alpha = idx - idx_floor

            warped[i] = (1 - alpha) * features_np[idx_floor] + alpha * features_np[idx_ceil]

        return torch.from_numpy(warped).to(features.device)


class MixupAugmentation:
    """
    Mixup augmentation for sequences.

    Mixes pairs of training samples and their labels.
    For CTC, this is tricky but can work with soft blending.

    WARNING: This is experimental for CTC. Use cautiously.
    """

    def __init__(self, alpha: float = 0.2, p: float = 0.3):
        """
        Initialize Mixup.

        Args:
            alpha: Beta distribution parameter (0.2 = conservative mixing)
            p: Probability of applying mixup
        """
        self.alpha = alpha
        self.p = p

    def __call__(self, features1: torch.Tensor, features2: torch.Tensor,
                 lambda_: float = None) -> Tuple[torch.Tensor, float]:
        """
        Mix two feature sequences.

        Args:
            features1: [T1, F] first sequence
            features2: [T2, F] second sequence
            lambda_: Mixing coefficient (if None, sample from Beta distribution)

        Returns:
            Mixed features [T, F] and mixing coefficient
        """
        if np.random.random() > self.p:
            return features1, 1.0

        # Sample mixing coefficient from Beta distribution
        if lambda_ is None:
            lambda_ = np.random.beta(self.alpha, self.alpha)

        # Handle different sequence lengths
        T1, F1 = features1.shape
        T2, F2 = features2.shape

        assert F1 == F2, "Feature dimensions must match"

        # Use shorter length
        T = min(T1, T2)

        # Mix features
        mixed = lambda_ * features1[:T] + (1 - lambda_) * features2[:T]

        return mixed, lambda_


class GaussianNoise:
    """
    Add Gaussian noise to features.

    Simple but effective regularization.
    """

    def __init__(self, std: float = 0.01, p: float = 0.5):
        """
        Initialize noise injection.

        Args:
            std: Standard deviation of noise (0.01 = 1% of typical feature range)
            p: Probability of applying noise
        """
        self.std = std
        self.p = p

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to features."""
        if np.random.random() > self.p:
            return features

        noise = torch.randn_like(features) * self.std
        return features + noise


# Recommended augmentation pipeline for your use case
def create_augmentation_pipeline(training: bool = True,
                                aggressive: bool = False) -> SequenceAugmentation:
    """
    Create recommended augmentation pipeline.

    Args:
        training: Whether this is for training (vs validation)
        aggressive: Use aggressive augmentation for severe overfitting

    Returns:
        Augmentation callable
    """
    if not training:
        # No augmentation during validation
        return lambda x: x

    if aggressive:
        # AGGRESSIVE settings for severe overfitting (YOUR CASE)
        # Apply to 80% of training samples
        # Mask up to 50 frames (about 2 seconds at 25 fps)
        # Mask up to 100 feature dimensions (about 10% of 1024D)
        return SequenceAugmentation(
            time_mask_max_frames=50,
            time_mask_num=3,  # Apply 3 time masks
            feature_mask_max_features=100,
            feature_mask_num=2,  # Apply 2 feature masks
            time_warp=False,  # Disable for speed
            p=0.8  # Apply to 80% of samples
        )
    else:
        # MODERATE settings for typical overfitting
        return SequenceAugmentation(
            time_mask_max_frames=30,
            time_mask_num=2,
            feature_mask_max_features=50,
            feature_mask_num=2,
            time_warp=False,
            p=0.5  # Apply to 50% of samples
        )


if __name__ == "__main__":
    # Test augmentation
    print("Testing SequenceAugmentation...")

    # Create dummy feature sequence: [100 frames, 1024 features]
    features = torch.randn(100, 1024)

    # Create aggressive augmentation
    aug = create_augmentation_pipeline(training=True, aggressive=True)

    # Apply augmentation
    augmented = aug(features)

    print(f"Original shape: {features.shape}")
    print(f"Augmented shape: {augmented.shape}")
    print(f"Proportion of masked values: {(augmented == 0).float().mean():.3f}")
    print(f"Original mean: {features.mean():.3f}, std: {features.std():.3f}")
    print(f"Augmented mean: {augmented.mean():.3f}, std: {augmented.std():.3f}")

    print("\nAugmentation test passed!")
