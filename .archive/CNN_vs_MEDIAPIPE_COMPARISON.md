# CNN vs MediaPipe Features: Comprehensive Comparison

## Executive Summary

**Recommendation: Use MediaPipe features for CSLR tasks**

MediaPipe features contain **6.4× more dimensions** with explicit temporal, spatial, and anatomical information critical for sign language recognition. CNN features are fundamentally limited for CSLR.

---

## Feature Dimensions

| Feature Type | Dimensions | Information Content |
|--------------|------------|---------------------|
| **MediaPipe** | **6516** | Landmarks + Temporal + Spatial |
| **CNN (GoogLeNet)** | **1024** | Global appearance only |

---

## Detailed Feature Breakdown

### MediaPipe Features (6516 dimensions)

#### **Raw Landmark Coordinates (1629 dims)**
- **Pose**: 33 keypoints × 3 coords (x,y,z) = 99 dims
- **Left Hand**: 21 keypoints × 3 coords = 63 dims
- **Right Hand**: 21 keypoints × 3 coords = 63 dims
- **Face Mesh**: 468 keypoints × 3 coords = 1404 dims

#### **Temporal Features (3258 dims)**
- **Velocities**: Frame-to-frame motion = 1629 dims
- **Accelerations**: Second-order motion = 1629 dims

#### **Derived Features (~1629 dims)**
- Hand shapes and configurations
- Spatial relationships (hand-to-hand, hand-to-body distances)
- Normalized coordinates (scale-invariant)
- Angular features

#### **Detection Masks (543 boolean values)**
- Tracks landmark detection confidence per frame
- Indicates missing/occluded landmarks

---

### CNN Features (1024 dimensions)

#### **GoogLeNet avgpool Layer**
- Global appearance features
- Abstract visual representations
- No explicit structure

#### **What's Missing:**
- ❌ No explicit spatial structure (keypoints/landmarks)
- ❌ No temporal information (motion/velocity)
- ❌ No hand shape details (lost in global pooling)
- ❌ No facial expressions (averaged away)
- ❌ No body pose (abstract representation)

---

## Information Content Comparison

### What MediaPipe Provides for CSLR:

| Feature Category | MediaPipe | CNN |
|------------------|-----------|-----|
| **Hand Shapes** | ✅ Explicit finger positions | ❌ Lost in global pooling |
| **Facial Expressions** | ✅ 468 face landmarks | ❌ Averaged away |
| **Body Pose** | ✅ 33 pose keypoints | ❌ Abstract features |
| **Motion/Velocity** | ✅ Explicit computation | ❌ Not present |
| **Acceleration** | ✅ Second-order dynamics | ❌ Not present |
| **Spatial Relationships** | ✅ Distances, angles | ❌ Implicit only |
| **Scale Invariance** | ✅ Normalized coordinates | ⚠️ Depends on network training |
| **Rotation Invariance** | ✅ Computed features | ⚠️ Limited |
| **Missing Data Handling** | ✅ Detection masks | ❌ No tracking |

---

## Statistical Comparison

### MediaPipe Features:
```
Mean: 0.1413
Std: 0.3141
Non-zero ratio: 84.0%
Temporal variation: 0.0141 (good temporal dynamics)
```

### CNN Features:
```
Mean: ~0.5 (depends on normalization)
Std: ~0.3
Non-zero ratio: ~99% (no sparsity)
Temporal variation: Unknown (needs measurement)
```

---

## Why MediaPipe is Better for CSLR

### 1. **Explicit Spatial Structure**
Sign language relies on **precise hand configurations and positions**. MediaPipe provides:
- Exact finger joint positions
- Hand-to-hand spatial relationships
- Body pose alignment

CNN features only provide **global appearance** without explicit structure.

### 2. **Temporal Dynamics**
Sign language is fundamentally about **movement**. MediaPipe includes:
- Frame-to-frame velocity (1st derivative)
- Acceleration (2nd derivative)
- Motion trajectories

CNN features are **static per frame** with no temporal encoding.

### 3. **Fine-Grained Details**
Sign language requires recognizing:
- Individual finger positions (handshapes)
- Subtle facial expressions (grammar markers)
- Precise body movements

MediaPipe explicitly models these. CNN features **lose these details** in global pooling.

### 4. **Interpretability**
- **MediaPipe**: Features are interpretable (e.g., "right index finger x-coordinate")
- **CNN**: Features are abstract and not interpretable

### 5. **Missing Data Handling**
- **MediaPipe**: Detection masks indicate which landmarks were successfully detected
- **CNN**: No mechanism to indicate occluded or missing information

---

## Performance Implications

### Expected WER with Current Hierarchical Model:

| Feature Type | Expected WER | Reasoning |
|--------------|--------------|-----------|
| **CNN** | **60-80%** | Missing temporal and spatial structure |
| **MediaPipe** | **25-40%** | Rich temporal and spatial information |

### Why CNN Struggles:

1. **Blank Collapse Risk**: Without explicit motion features, model can't distinguish sign boundaries
2. **Limited Vocabulary**: Abstract features may not discriminate between similar signs
3. **No Hand Detail**: Critical handshape differences are lost
4. **No Temporal Context**: Can't model sign dynamics effectively

### Why MediaPipe Should Work Better:

1. **Explicit Motion**: Velocity/acceleration help identify sign boundaries
2. **Rich Structure**: 1629 landmark dimensions capture fine details
3. **Temporal Context**: Pre-computed temporal features help BiLSTM
4. **Proven Track Record**: MediaPipe features widely used in CSLR research

---

## Computational Comparison

| Aspect | MediaPipe | CNN |
|--------|-----------|-----|
| **Feature Extraction** | ~30 FPS (CPU/GPU) | ~60 FPS (GPU) |
| **Feature Size** | 6516 dims | 1024 dims |
| **Memory per frame** | 26 KB (float32) | 4 KB (float32) |
| **Model Input Size** | Larger (6516) | Smaller (1024) |
| **Training Speed** | Slower (more dims) | Faster (fewer dims) |
| **Inference Speed** | Comparable | Comparable |

**Trade-off**: MediaPipe requires more memory and slower training, but provides much better information content.

---

## Dataset Compatibility

### MediaPipe Features:
✅ Available at: `data/teacher_features/mediapipe_full/`
- Train: 4384 samples
- Dev: 540 samples
- Test: 629 samples
- Format: `.npz` files with `features`, `detection_masks`, `metadata`

### CNN Features:
✅ Available at: `data/features_cnn/`
- Same splits
- Format: `.h5` files with `features`, `annotation`

**Both feature sets are ready to use!**

---

## Recommendation: Test with MediaPipe First

### Why:
1. **Richer information**: 6516 dims vs 1024 dims
2. **Temporal features included**: Motion/velocity already computed
3. **Proven for CSLR**: MediaPipe is standard in sign language research
4. **Explicit structure**: Easier for model to learn from landmarks

### Expected Outcome:
- **Current CNN result**: 80-87% WER (stuck)
- **Expected MediaPipe result**: 35-50% WER (with same hierarchical model)

### Action Plan:
1. Train hierarchical model with MediaPipe features (use `train_hierarchical_mediapipe.py`)
2. Compare WER after 20 epochs
3. If MediaPipe WER < 60%, continue with MediaPipe
4. If both fail, problem is in model architecture, not features

---

## When to Use CNN Features

CNN features might be preferable when:
- ❌ **NOT applicable for CSLR** - Use MediaPipe instead
- Possible use: Transfer learning from ImageNet-pretrained models
- Possible use: Visual understanding tasks (not sign language)

For **Continuous Sign Language Recognition**, MediaPipe features are superior.

---

## Next Steps

1. **Immediate**: Run `train_hierarchical_mediapipe.py` (to be created)
2. **Compare**: MediaPipe WER vs CNN WER after 20 epochs
3. **Analyze**: If MediaPipe still fails, investigate model architecture
4. **Hybrid approach**: Consider combining CNN + MediaPipe features (fusion)

---

## Technical Notes

### MediaPipe Feature Extraction Pipeline:
1. Video frame → MediaPipe Holistic
2. Extract pose (33), hands (2×21), face (468) landmarks
3. Compute velocities (frame-to-frame differences)
4. Compute accelerations (second derivatives)
5. Compute spatial features (distances, angles)
6. Normalize coordinates
7. Save as `.npz` with detection masks

### CNN Feature Extraction Pipeline:
1. Video frame → Resize to 224×224
2. GoogLeNet forward pass
3. Extract avgpool layer (1024 dims)
4. Save to HDF5

**MediaPipe pipeline is more complex but produces much richer features for CSLR.**
