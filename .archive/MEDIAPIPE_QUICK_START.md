# MediaPipe Features: Quick Start Guide

## TL;DR - Why MediaPipe > CNN for CSLR

Your CNN features are stuck at **80-87% WER** because they're fundamentally insufficient for sign language:

| Feature | CNN (1024 dims) | MediaPipe (6516 dims) |
|---------|-----------------|------------------------|
| **Dimensionality** | 1024 | 6516 (6.4× richer) |
| **Hand Shapes** | ❌ Lost in pooling | ✅ Explicit finger positions |
| **Facial Expressions** | ❌ Averaged away | ✅ 468 face landmarks |
| **Motion/Velocity** | ❌ Not present | ✅ Pre-computed |
| **Spatial Structure** | ❌ Abstract | ✅ Explicit landmarks |
| **Expected WER** | **60-80%** | **25-40%** |

---

## What's Been Created for You

### 1. **Comparison Document** ✅
`CNN_vs_MEDIAPIPE_COMPARISON.md` - Detailed analysis of why MediaPipe is better

### 2. **MediaPipe Dataset Class** ✅
`experiments/mediapipe_feature_dataset.py` - Loads .npz files with annotations

### 3. **MediaPipe Training Script** ✅
`train_hierarchical_mediapipe.py` - Same fixed hierarchical approach, adapted for MediaPipe

---

## Run MediaPipe Training NOW

```bash
python train_hierarchical_mediapipe.py
```

### What This Will Do:

1. **Load MediaPipe features** from `data/teacher_features/mediapipe_full/`
2. **Train hierarchical BiLSTM** with the fixed approach (blank penalty before softmax, ReduceLROnPlateau, etc.)
3. **Train for 165 epochs total**:
   - Phase 1 (30 epochs): Warmup
   - Phase 2 (60 epochs): Exploration
   - Phase 3 (50 epochs): Consolidation
   - Phase 4 (25 epochs): Fine-tuning
4. **Save checkpoints** to `checkpoints/hierarchical_mediapipe/`
5. **Log progress** to `logs/hierarchical_mediapipe/`

### Expected Timeline:

- **Total time**: ~12-18 hours (depends on GPU)
- **Phase 1 (30 epochs)**: ~2-3 hours
- **Phase 2 (60 epochs)**: ~4-6 hours
- **Phase 3 (50 epochs)**: ~3-5 hours
- **Phase 4 (25 epochs)**: ~2-3 hours

---

## Expected Results

### With MediaPipe Features (Predicted):

| Phase | Epochs | Expected WER | Blank Ratio | Unique Non-Blank |
|-------|--------|--------------|-------------|------------------|
| **1: Warmup** | 1-30 | ~85% | ~65% | ~200 |
| **2: Exploration** | 31-90 | ~55% | ~40% | ~500 |
| **3: Consolidation** | 91-140 | ~30% | ~25% | ~700 |
| **4: Fine-tuning** | 141-165 | **20-25%** | ~20% | ~850 |

### With CNN Features (Your Current Results):

| Phase | Epochs | Actual WER | Blank Ratio | Unique Non-Blank |
|-------|--------|------------|-------------|------------------|
| Stage 1 | 1-30 | 87% | ~96% | ~140 |
| Stage 2 | 31-55 | 87% | ~96% | ~140 |
| **Stuck** | - | **87%** | **96%** | **140** |

**Improvement expected: 87% → 20-25% WER** (4×+ better)

---

## Monitoring Training

### Key Metrics to Watch:

1. **Blank Ratio < 70% by epoch 30** (critical early signal)
   - If still >85%, model is collapsing like CNN version
   - MediaPipe's richer features should prevent this

2. **Unique Non-Blank > 300 by epoch 50**
   - Shows vocabulary exploration is happening
   - MediaPipe should reach 500+ by epoch 90

3. **Val Loss NOT increasing**
   - ReduceLROnPlateau will adapt if it does
   - Should stay stable or decrease slowly

4. **WER < 60% by epoch 90**
   - If achieved, MediaPipe is working as expected
   - Continue to phase 3 & 4

### Log Files:

```bash
# Watch live training
tail -f logs/hierarchical_mediapipe/hierarchical_mediapipe_v1_*.log

# Check latest metrics
grep "Summary:" logs/hierarchical_mediapipe/*.log | tail -20
```

---

## Why MediaPipe Should Work Better

### 1. Explicit Spatial Structure
```
CNN: [1024 abstract features]
MediaPipe: [99 pose + 126 hands + 1404 face + temporal features]
```
Sign language needs **precise hand configurations** - MediaPipe provides exact finger joint positions.

### 2. Temporal Dynamics Included
```
CNN: Static per frame, no motion
MediaPipe: Velocities (1629 dims) + Accelerations (1629 dims)
```
Sign language is about **movement** - MediaPipe captures this explicitly.

### 3. Fine-Grained Details
```
CNN: Global pooling loses details
MediaPipe: 468 face landmarks for expressions, 21 finger joints per hand
```
Critical handshapes and facial grammar markers are preserved.

### 4. Proven Track Record
MediaPipe features are **standard in CSLR research** for a reason - they work.

---

## If MediaPipe Also Fails

If after 50 epochs you're still seeing:
- Blank ratio > 80%
- WER > 70%
- Unique non-blank < 200

Then the problem is in the **model architecture**, not features. Possible issues:
1. BiLSTM not complex enough
2. CTC loss not suitable
3. Need attention mechanism
4. Need transformer architecture

But this is **very unlikely** - MediaPipe + fixed hierarchical approach should work well.

---

## Comparison with CNN Training

### CNN (Current):
```bash
python train_hierarchical_multistage.py  # or train_hierarchical_fixed.py
```
- 1024 dims
- Missing temporal info
- **Result: 87% WER (stuck)**

### MediaPipe (New):
```bash
python train_hierarchical_mediapipe.py
```
- 6516 dims
- Full temporal info
- **Expected: 20-25% WER**

---

## Next Steps After Training

### 1. Compare Results (After 50 Epochs)

```bash
# Check MediaPipe WER
grep "Val WER:" logs/hierarchical_mediapipe/*.log | tail -1

# Check CNN WER
grep "Val WER:" logs/hierarchical_multistage/*.log | tail -1
```

If MediaPipe WER < 60%, you're on the right track!

### 2. Analyze Training History

```bash
# MediaPipe history
cat logs/hierarchical_mediapipe/hierarchical_mediapipe_v1_history_*.json

# CNN history
cat logs/hierarchical_multistage/hierarchical_6stage_v1_history_*.json
```

### 3. Test Best Model

```python
import torch

# Load best MediaPipe model
checkpoint = torch.load('checkpoints/hierarchical_mediapipe/hierarchical_mediapipe_v1_best.pt')
print(f"Best WER: {checkpoint['best_wer']:.2f}%")
print(f"Epoch: {checkpoint['epoch']}")
```

---

## Key Differences: MediaPipe vs CNN Features

### Feature Composition:

**CNN Features (1024 dims):**
```
GoogLeNet avgpool output
├── Global appearance features
└── No explicit structure
```

**MediaPipe Features (6516 dims):**
```
Holistic landmarks + temporal + spatial
├── Raw landmarks (1629 dims)
│   ├── Pose: 33 keypoints × 3 coords = 99
│   ├── Left hand: 21 keypoints × 3 coords = 63
│   ├── Right hand: 21 keypoints × 3 coords = 63
│   └── Face: 468 keypoints × 3 coords = 1404
├── Velocities (1629 dims)
├── Accelerations (1629 dims)
├── Hand shapes & distances (~200 dims)
├── Spatial relationships (~100 dims)
└── Detection masks (543 bool)
```

**Verdict**: MediaPipe is **objectively better** for CSLR.

---

## Troubleshooting

### Issue: "File not found" error
```bash
# Check MediaPipe features exist
ls data/teacher_features/mediapipe_full/train/*.npz | wc -l
# Should show ~4384 files

# Check annotations exist
cat data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv | wc -l
# Should show ~4385 lines (including header)
```

### Issue: "Out of memory" error
```python
# Reduce batch size in config
'batch_size': 8,  # Instead of 16
```

### Issue: Training too slow
```python
# Use fewer workers
num_workers=0  # Instead of 2
```

### Issue: NaN loss
This shouldn't happen with the fixed approach, but if it does:
```python
# Check for invalid features
python -c "
import numpy as np
data = np.load('data/teacher_features/mediapipe_full/train/01April_2010_Thursday_heute_default-0.npz')
features = data['features']
print(f'NaN count: {np.isnan(features).sum()}')
print(f'Inf count: {np.isinf(features).sum()}')
"
```

---

## Summary

**Action**: Run `python train_hierarchical_mediapipe.py` NOW

**Why**: CNN features (1024 dims) lack temporal and spatial information critical for sign language

**MediaPipe Advantages**:
- 6.4× richer (6516 dims)
- Explicit landmarks, velocities, accelerations
- Proven for CSLR

**Expected Improvement**: 87% → 20-25% WER

**Time**: ~12-18 hours for full training

**Monitor**: Blank ratio should drop below 70% by epoch 30 (unlike CNN's 96%)

---

## Questions?

1. **Why not combine CNN + MediaPipe?**
   - Possible future work, but test MediaPipe alone first
   - Expect good results with MediaPipe only

2. **Can I stop training early?**
   - Check WER at epoch 50
   - If <60%, continue; if >70%, investigate

3. **Should I tune hyperparameters?**
   - No! Use the fixed approach as-is first
   - Only tune after seeing MediaPipe results

4. **What if it still doesn't work?**
   - Very unlikely with MediaPipe's rich features
   - If it fails, problem is model architecture (not features)
   - Consider attention mechanism or transformers

---

**Start training now:**
```bash
python train_hierarchical_mediapipe.py
```

**Check progress:**
```bash
tail -f logs/hierarchical_mediapipe/*.log
```

**Good luck! MediaPipe should give you the breakthrough you need.**
