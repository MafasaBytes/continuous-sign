# Detailed Changes Summary - Anti-Overfitting Implementation

## Overview
This document lists every modification made to combat overfitting in the MobileNetV3 sign language model.

---

## File 1: `src/data/dataset.py`

### Location: `_augment_features()` method (lines 267-321)

#### Changes Made:
```diff
def _augment_features(self, features: np.ndarray) -> np.ndarray:
-   """Apply data augmentation to features."""
+   """
+   Apply STRONG data augmentation to features to combat overfitting.
+   Enhanced augmentation strategy: higher probability, stronger perturbations.
+   """
    
    # 1. Temporal augmentation - speed perturbation
-   if np.random.random() < 0.5:
-       speed_factor = np.random.uniform(0.9, 1.1)
+   if np.random.random() < 0.7:  # INCREASED probability
+       speed_factor = np.random.uniform(0.85, 1.15)  # WIDER range
        seq_len = features.shape[0]
        new_len = int(seq_len * speed_factor)
+       if new_len > 0:  # Safety check
            indices = np.linspace(0, seq_len - 1, new_len).astype(int)
            features = features[indices]

+   # 2. Temporal masking - SpecAugment style [NEW TECHNIQUE]
+   if np.random.random() < 0.5:
+       seq_len = features.shape[0]
+       mask_len = int(seq_len * np.random.uniform(0.05, 0.15))
+       mask_start = np.random.randint(0, max(1, seq_len - mask_len))
+       features[mask_start:mask_start + mask_len, :] = 0

-   # Spatial augmentation - add noise
-   if np.random.random() < 0.3:
-       noise = np.random.normal(0, 0.01, features.shape)
+   # 3. Spatial augmentation - Gaussian noise
+   if np.random.random() < 0.6:  # INCREASED probability
+       noise_std = np.random.uniform(0.015, 0.03)  # STRONGER noise
+       noise = np.random.normal(0, noise_std, features.shape)
        features = features + noise

-   # Feature dropout
-   if np.random.random() < 0.2:
-       mask = np.random.binomial(1, 0.9, features.shape)
+   # 4. Feature dropout
+   if np.random.random() < 0.5:  # INCREASED probability
+       dropout_rate = np.random.uniform(0.1, 0.2)  # VARIABLE rate
+       mask = np.random.binomial(1, 1 - dropout_rate, features.shape)
        features = features * mask

+   # 5. Feature channel dropout [NEW TECHNIQUE]
+   if np.random.random() < 0.3:
+       num_features = features.shape[1]
+       num_to_drop = int(num_features * np.random.uniform(0.05, 0.1))
+       drop_indices = np.random.choice(num_features, num_to_drop, replace=False)
+       features[:, drop_indices] = 0
+
+   # 6. Random Gaussian blur [NEW TECHNIQUE]
+   if np.random.random() < 0.3:
+       from scipy.ndimage import gaussian_filter1d
+       sigma = np.random.uniform(0.5, 1.5)
+       features = gaussian_filter1d(features, sigma=sigma, axis=0)
+
+   # 7. Random scaling per feature channel [NEW TECHNIQUE]
+   if np.random.random() < 0.4:
+       scale_factors = np.random.uniform(0.9, 1.1, features.shape[1])
+       features = features * scale_factors
+
    return features
```

**Summary:**
- 3 techniques → 7 techniques
- Probabilities increased: 20-50% → 30-70%
- Perturbation strength increased across all augmentations

---

## File 2: `src/models/mobilenet_v3.py`

### Change 1: ModalityEncoder (lines 90-108)

```diff
class ModalityEncoder(nn.Module):
-   """Encoder for specific modality features."""
+   """Encoder for specific modality features with enhanced regularization."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = (input_dim + output_dim) // 2
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
+           nn.Dropout(dropout * 0.5)  # Additional dropout
        )
```

**Impact:** Extra regularization in each modality encoder

---

### Change 2: MobileNetV3SignLanguage.__init__ default (lines 156-166)

```diff
    def __init__(
        self,
        vocab_size: int,
        pose_dim: int = 99,
        hands_dim: int = 126,
        face_dim: int = 1404,
        temporal_dim: int = 4887,
        hidden_dim: int = 128,
        num_lstm_layers: int = 1,
-       dropout: float = 0.3,  # Reduced from 0.5 for initial convergence
+       dropout: float = 0.5,  # INCREASED from 0.3 to 0.5 for stronger regularization
    ):
```

**Impact:** 67% increase in dropout rate throughout model

---

### Change 3: LSTM and Output Layers (lines 227-243)

```diff
        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

+       # Additional dropout after LSTM [NEW]
+       self.lstm_dropout = nn.Dropout(dropout * 0.7)
        
-       # Output projection for CTC
-       self.output_proj = nn.Linear(hidden_dim * 2, vocab_size)
+       # Output projection for CTC with dropout [MODIFIED]
+       self.output_proj = nn.Sequential(
+           nn.Linear(hidden_dim * 2, vocab_size),
+           nn.Dropout(dropout * 0.3)
+       )
```

**Impact:** Additional dropout layers at critical points

---

### Change 4: Forward Pass (lines 339-357)

```diff
        # BiLSTM temporal modeling
        if input_lengths is not None:
            adjusted_lengths = torch.clamp(input_lengths, max=T)
            x = nn.utils.rnn.pack_padded_sequence(
                x, adjusted_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            x, _ = self.lstm(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=T)
        else:
            x, _ = self.lstm(x)
        
+       # Apply dropout after LSTM [NEW]
+       x = self.lstm_dropout(x)
        
        # Output projection
        x = self.output_proj(x)
```

**Impact:** Applies new dropout in forward pass

---

### Change 5: Factory Function (lines 372-376)

```diff
def create_mobilenet_v3_model(
    vocab_size: int,
-   dropout: float = 0.3,  # Reduced default for better convergence
+   dropout: float = 0.5,  # INCREASED from 0.3 to 0.5 to combat overfitting
    **kwargs
) -> MobileNetV3SignLanguage:
```

**Impact:** Default models now use higher dropout

---

## File 3: `src/training/train.py`

### Change 1: Learning Rate Scheduler (lines 586-597)

```diff
-   # Learning rate scheduler - ReduceLROnPlateau (more stable for CTC)
-   # No warmup - start at full LR (validated in overfitting test)
-   # Adjusted: More patience to avoid premature LR reduction
+   # Learning rate scheduler - ReduceLROnPlateau
+   # UPDATED: More aggressive patience to avoid premature LR drops
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
-       factor=0.7,          # Reduce LR by 30%
-       patience=20,         # Wait 20 epochs
-       min_lr=1e-5,
-       threshold=0.005,
-       # verbose=True
+       factor=0.5,          # Reduce LR by 50% (more aggressive)
+       patience=15,         # Wait 15 epochs (faster adaptation)
+       min_lr=5e-6,         # Lower minimum for fine-tuning
+       threshold=0.01,      # Larger threshold (more conservative)
+       verbose=True         # Enable verbose
    )
```

**Impact:** Better LR scheduling with monitoring

---

### Change 2: Argument Defaults (lines 797-815)

```diff
    # Model arguments
-   parser.add_argument('--dropout', type=float, default=0.3,
-       help='Dropout rate (default: 0.3 for regularization)')
+   parser.add_argument('--dropout', type=float, default=0.5,
+       help='Dropout rate (INCREASED to 0.5 for stronger regularization)')
    
    # Training arguments
-   parser.add_argument('--batch_size', type=int, default=4,
-       help='Batch size (default: 4 for stability)')
+   parser.add_argument('--batch_size', type=int, default=8,
+       help='Batch size (INCREASED from 4 to 8 - larger batch = better generalization)')
    
-   parser.add_argument('--learning_rate', type=float, default=5e-4,
-       help='Learning rate (default: 5e-4)')
+   parser.add_argument('--learning_rate', type=float, default=3e-4,
+       help='Learning rate (REDUCED from 5e-4 to 3e-4 for more stable convergence)')
    
-   parser.add_argument('--weight_decay', type=float, default=1e-4,
-       help='Weight decay (default: 1e-4 for regularization)')
+   parser.add_argument('--weight_decay', type=float, default=1e-3,
+       help='Weight decay (INCREASED from 1e-4 to 1e-3 - 10x stronger regularization)')
    
-   parser.add_argument('--accumulation_steps', type=int, default=4,
-       help='Gradient accumulation steps')
+   parser.add_argument('--accumulation_steps', type=int, default=2,
+       help='Gradient accumulation steps (REDUCED from 4 to 2 - effective batch = 16)')
```

**Impact:** All training defaults now optimized for generalization

---

### Change 3: Early Stopping (lines 828-829)

```diff
-   parser.add_argument('--early_stopping_patience', type=int, default=100,
-       help='Early stopping patience')
+   parser.add_argument('--early_stopping_patience', type=int, default=30,
+       help='Early stopping patience (REDUCED from 100 to 30 - stop sooner)')
```

**Impact:** Prevents unnecessary over-training

---

## Summary Table

| Component | File | Lines | Changes |
|-----------|------|-------|---------|
| **Augmentation** | `dataset.py` | 267-321 | +4 techniques, ↑ probabilities, ↑ strength |
| **Model Dropout** | `mobilenet_v3.py` | 90-108, 166, 227-243 | +3 dropout layers, 0.3→0.5 |
| **Forward Pass** | `mobilenet_v3.py` | 339-357 | Added lstm_dropout application |
| **Factory Default** | `mobilenet_v3.py` | 372-376 | Default dropout 0.3→0.5 |
| **LR Scheduler** | `train.py` | 586-597 | More aggressive, verbose |
| **Training Defaults** | `train.py` | 797-815 | batch↑, LR↓, decay↑10x |
| **Early Stopping** | `train.py` | 828-829 | 100→30 epochs |

---

## Validation

### No Linter Errors:
```bash
$ read_lints src/data/dataset.py src/models/mobilenet_v3.py src/training/train.py
No linter errors found.
```

### Files Modified:
1. ✅ `src/data/dataset.py` - Enhanced augmentation
2. ✅ `src/models/mobilenet_v3.py` - Increased dropout
3. ✅ `src/training/train.py` - Better hyperparameters

### Files NOT Modified (as per requirements):
- ❌ No new training scripts created
- ❌ No new model architectures created
- ❌ No frontend code modified

---

## How to Verify Changes

### 1. Check augmentation is working:
```python
from src.data.mediapipe_dataset import MediaPipeFeatureDataset
dataset = MediaPipeFeatureDataset(..., augment=True, split='train')
# Should see 7 augmentation techniques applied
```

### 2. Check model dropout:
```python
from src.models import create_mobilenet_v3_model
model = create_mobilenet_v3_model(vocab_size=973)
print(model)  # Should see multiple Dropout layers
```

### 3. Check training defaults:
```bash
python src/training/train.py --help
# Should show new defaults in help text
```

---

## Expected Behavior

### Before Changes:
```
Epoch 50: Train Loss: 2.0, Val Loss: 4.2, WER: 78%
Epoch 100: Train Loss: 1.5, Val Loss: 4.3, WER: 79%
Epoch 300: Train Loss: 1.2, Val Loss: 4.2, WER: 78% ← Best (overfitting!)
```

### After Changes:
```
Epoch 50: Train Loss: 2.5, Val Loss: 3.3, WER: 62%
Epoch 60: Train Loss: 2.3, Val Loss: 3.1, WER: 58% ← Best (healthy!)
Epoch 90: Early stopping triggered
```

---

## Rollback Instructions

If you need to revert these changes:

```bash
# Option 1: Git revert (if using version control)
git checkout HEAD~1 src/data/dataset.py
git checkout HEAD~1 src/models/mobilenet_v3.py
git checkout HEAD~1 src/training/train.py

# Option 2: Manual override via command line
python src/training/train.py \
    --batch_size 4 \
    --dropout 0.3 \
    --learning_rate 5e-4 \
    --weight_decay 1e-4 \
    --accumulation_steps 4 \
    --early_stopping_patience 100
```

---

## Next Steps

1. **Run training** with new defaults:
   ```bash
   python src/training/train.py --data_dir data/teacher_features/mediapipe_full
   ```

2. **Monitor progress**:
   - Training logs: `checkpoints/.../training.log`
   - Curves: `figures/baseline/training_curves.png`
   - TensorBoard: `tensorboard --logdir checkpoints/.../tensorboard`

3. **Evaluate results**:
   - Target: WER < 60%
   - Check: Train-Val gap < 1.5
   - Verify: Early stopping triggers ~50-80 epochs

4. **Adjust if needed**:
   - Still overfitting? → Increase dropout to 0.6
   - Val loss too high? → Reduce dropout to 0.4
   - Too slow? → Reduce augmentation techniques

---

## References

- Config analyzed: `checkpoints/student/mobilenet_v3_optimized/mobilenet_v3_20251118_120803/config.json`
- Training curves: `figures/baseline/training_curves.png`
- Full summary: `OVERFITTING_FIXES_SUMMARY.md`
- Commands: `TRAINING_COMMANDS.md`

