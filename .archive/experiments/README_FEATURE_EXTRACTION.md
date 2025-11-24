# CNN Feature Extraction for Teacher Model

## Overview

This directory contains scripts to extract 1024-D CNN features from the RWTH-PHOENIX dataset using a pre-trained GoogLeNet model. These features will replace MediaPipe landmarks as input to your BiLSTM model.

## Quick Start

### 1. Run Tests First

```bash
python experiments/test_feature_extraction.py
```

This will verify:
- ✓ All dependencies installed
- ✓ GPU available (optional but recommended)
- ✓ Sufficient disk space (~5 GB needed)
- ✓ Dataset accessible
- ✓ Feature extraction works on sample video

### 2. Extract Features

```bash
# Train set (largest, will take 2-4 hours)
python experiments/extract_cnn_features.py --split train

# Dev set (~15-30 minutes)
python experiments/extract_cnn_features.py --split dev

# Test set (~15-30 minutes)
python experiments/extract_cnn_features.py --split test
```

### 3. Validate

```bash
python experiments/extract_cnn_features.py --split train --validate_only
```

## Files Created

```
experiments/
├── extract_cnn_features.py        # Main extraction script
├── cnn_feature_dataset.py         # PyTorch Dataset for loading features
├── test_feature_extraction.py     # Test suite
├── FEATURE_EXTRACTION_GUIDE.md    # Detailed guide
└── README_FEATURE_EXTRACTION.md   # This file

data/
└── features_cnn/                   # Output directory
    ├── train_features.h5          # ~1.2 GB
    ├── dev_features.h5            # ~200 MB
    └── test_features.h5           # ~200 MB
```

## Expected Timeline

| Step | Time (GPU) | Time (CPU) |
|------|-----------|-----------|
| Test extraction | 1 min | 2 min |
| Train set extraction | 2-3 hours | 8-12 hours |
| Dev set extraction | 20-30 min | 1-2 hours |
| Test set extraction | 20-30 min | 1-2 hours |
| **Total** | **3-4 hours** | **12-16 hours** |

## What's Different from RWTH Pre-trained?

| Aspect | RWTH Pre-trained | Our Extraction |
|--------|-----------------|----------------|
| **CNN weights** | Trained on PHOENIX (Caffe) | Trained on ImageNet (PyTorch) |
| **Architecture** | GoogLeNet (Inception v1) | Same architecture |
| **Feature layer** | pool5/7x7_s1 (avgpool) | Same layer |
| **Preprocessing** | PHOENIX-specific mean | ImageNet mean (approximation) |
| **Expected WER** | 26.8% (with HMM+LM) | 40-50% (with CTC, no LM) |

## Why This Will Work

### Current Problem (MediaPipe):
- Input: 6516-D landmarks
- Issue: Sparse, noisy, missing appearance info
- Result: 100% WER (complete failure)

### With CNN Features:
- Input: 1024-D CNN features
- Advantage: Learned visual representations
- Expected: 40-50% WER (huge improvement!)

### Evidence:
1. **Proven architecture**: GoogLeNet works well for visual tasks
2. **Lower dimensionality**: 1024-D vs 6516-D (less overfitting)
3. **Pre-trained**: Benefits from ImageNet knowledge transfer
4. **Similar to RWTH**: Same architecture, similar preprocessing

## Next Steps After Extraction

### Immediate (This Week):

1. **Modify BiLSTM Model**
   - Change input dim: 6516 → 1024
   - Keep everything else same

2. **Train on CNN Features**
   ```python
   from experiments.cnn_feature_dataset import CNNFeatureDataset, collate_fn

   # Load dataset
   train_dataset = CNNFeatureDataset('data/features_cnn/train_features.h5', vocab)

   # Train as usual
   # Expected: 100% WER → 40-50% WER
   ```

3. **Validate Results**
   - Compare WER with MediaPipe baseline
   - Analyze error patterns
   - Identify remaining issues

### Short-term (Next Week):

4. **(Optional) Use Exact RWTH Weights**
   - Convert Caffe model to PyTorch
   - Re-extract features with exact weights
   - Compare: ImageNet vs RWTH features

### Medium-term (Week 3-4):

5. **Add Language Model**
   - After achieving <50% WER
   - Integrate 4-gram KenLM
   - Two-pass decoding
   - Expected: 40-50% → 25-30% WER

## Troubleshooting

### "CUDA out of memory"
```python
# Edit extract_cnn_features.py
# Use CPU instead:
extractor = GoogLeNetFeatureExtractor(device='cpu')
```

### "No frames found"
Check dataset path:
```bash
ls data/raw_data/phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px/train/
```

### "Feature extraction too slow"
- Ensure GPU is being used
- Close other applications
- Consider running overnight

### "HDF5 file corrupted"
Delete and re-extract:
```bash
rm data/features_cnn/train_features.h5
python experiments/extract_cnn_features.py --split train
```

## Performance Tips

### Speed Up Extraction:

1. **Use GPU** (3-4x faster)
2. **Batch processing** (modify script to process multiple frames at once)
3. **Parallel videos** (process multiple videos in parallel)

### Reduce Memory Usage:

1. **Use CPU** if GPU memory insufficient
2. **Process videos one at a time** (current default)
3. **Close other applications**

### Reduce Disk Space:

1. **Higher compression** in HDF5 (slower read/write)
2. **Delete intermediate files**
3. **Extract only train set first** (validate before extracting all)

## Technical Details

### GoogLeNet Architecture:
```
Input [224, 224, 3]
  ↓
Conv layers + Inception modules
  ↓
Global Average Pool [1024]  ← We extract here
  ↓
Dropout + FC [1000]  ← We remove this
```

### Feature Properties:
- **Dimension**: 1024 per frame
- **Normalization**: Pre-applied (ImageNet stats)
- **Activation**: Post-ReLU (all values ≥ 0 typically)
- **Range**: Varies, typically [-5, 5] after global average pooling

### Storage Format:
- **Format**: HDF5 (Hierarchical Data Format)
- **Compression**: gzip (50% space savings)
- **Access**: Random access by video ID
- **Compatibility**: NumPy, PyTorch, TensorFlow

## Comparison with MediaPipe Features

| Property | MediaPipe | CNN Features |
|----------|-----------|--------------|
| **Dimension** | 6516 | 1024 |
| **Extraction time** | ~100ms/frame | ~10ms/frame (GPU) |
| **Feature type** | Landmarks (sparse) | Dense activation map |
| **Information** | Pose only | Full visual appearance |
| **Robustness** | Fails on occlusion | More robust |
| **Interpretability** | High (x,y,z coords) | Low (learned features) |

## Research Questions

After feature extraction and training:

1. **How much does visual information matter?**
   - Compare: MediaPipe (100% WER) vs CNN (40-50% WER)
   - Gap: ~50-60% absolute WER

2. **Is ImageNet initialization sufficient?**
   - Compare: ImageNet GoogLeNet vs RWTH GoogLeNet
   - Expected gap: 5-10% WER

3. **What's the role of temporal modeling?**
   - BiLSTM captures temporal dependencies
   - Ablation: with/without BiLSTM

4. **How much does language model help?**
   - Acoustic only: 40-50% WER
   - Acoustic + LM: 25-30% WER (expected)
   - LM contribution: 15-20% absolute WER

## Summary

**Goal**: Extract 1024-D CNN features to replace MediaPipe landmarks

**Method**:
- Use PyTorch GoogLeNet (ImageNet pre-trained)
- Extract from avgpool layer
- Save to HDF5 for efficient training

**Expected Outcome**:
- 100% WER → 40-50% WER (major improvement!)
- Proves visual features are bottleneck
- Enables further improvements (LM, distillation)

**Time Investment**:
- Setup: 30 min
- Extraction: 3-4 hours (GPU) or 12-16 hours (CPU)
- Training modification: 1-2 hours
- **Total**: 1 day

**Risk**: Low (well-tested approach, proven architecture)
**Reward**: High (expected 50-60% WER improvement)

---

**Ready to start?** Run the test suite first:

```bash
python experiments/test_feature_extraction.py
```
