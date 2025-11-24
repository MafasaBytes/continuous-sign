# Overfitting Mitigation Strategy - Implementation Summary

## Problem Analysis

Based on the training curves analysis:
- **Training loss**: Continuously decreasing
- **Validation loss**: Plateaus around epoch 50
- **Validation WER**: Stuck at 78-80% (Target: < 25%)
- **Gap**: Severe divergence between train and validation performance

### Root Causes Identified:
1. **Batch size = 4** ← CRITICAL - Extremely small, causing high variance
2. **Dropout = 0.3** ← Insufficient for 14.4M parameters
3. **Weight decay = 0.0001** ← Too weak regularization
4. **Weak augmentation** ← Low probability (20-50%) with minor perturbations
5. **No early stopping** ← Training continues unnecessarily
6. **Model complexity** ← 14.4M parameters for this dataset size

---

## Solutions Implemented

### 1. **Enhanced Data Augmentation** (`src/data/dataset.py`)

#### Previous Augmentation (WEAK):
- Speed perturbation: 50% probability, 0.9-1.1x range
- Gaussian noise: 30% probability, σ=0.01
- Feature dropout: 20% probability, 10% drop rate

#### **NEW Augmentation (STRONG):**
```python
# 1. Temporal speed perturbation
- Probability: 70% (↑ from 50%)
- Speed range: 0.85-1.15x (wider, was 0.9-1.1x)

# 2. Temporal masking (SpecAugment-style) [NEW]
- Probability: 50%
- Masks 5-15% of consecutive frames

# 3. Gaussian noise
- Probability: 60% (↑ from 30%)
- Noise std: 0.015-0.03 (stronger, was 0.01)

# 4. Feature dropout
- Probability: 50% (↑ from 20%)
- Drop rate: 10-20% (variable, was fixed 10%)

# 5. Feature channel dropout [NEW]
- Probability: 30%
- Drops 5-10% of entire feature dimensions

# 6. Gaussian blur [NEW]
- Probability: 30%
- Temporal smoothing with σ=0.5-1.5

# 7. Random channel scaling [NEW]
- Probability: 40%
- Per-channel scale: 0.9-1.1x
```

**Impact**: Much stronger regularization through aggressive augmentation

---

### 2. **Increased Model Dropout** (`src/models/mobilenet_v3.py`)

#### Changes:
```python
# Default dropout increased
- Before: dropout = 0.3
- After:  dropout = 0.5 (↑ 67%)

# Additional dropout layers added:
1. ModalityEncoder: Added 0.5*dropout after output (0.25 dropout)
2. LSTM output: Added 0.7*dropout layer (0.35 dropout)
3. Output projection: Added 0.3*dropout before final layer (0.15 dropout)
```

**Architecture modifications:**
- `ModalityEncoder`: Extra dropout after LayerNorm
- `MobileNetV3SignLanguage.__init__`: New `lstm_dropout` layer
- `MobileNetV3SignLanguage.forward`: Applied `lstm_dropout` after LSTM
- `output_proj`: Changed from `nn.Linear` to `nn.Sequential` with dropout

**Impact**: ~3x more dropout throughout the network

---

### 3. **Stronger Weight Regularization** (`src/training/train.py`)

```python
# Weight decay (L2 regularization)
- Before: 1e-4 (0.0001)
- After:  1e-3 (0.001)  [10x STRONGER]
```

**Impact**: Penalizes large weights much more aggressively

---

### 4. **Improved Training Hyperparameters** (`src/training/train.py`)

#### Batch Size:
```python
- Before: batch_size = 4
- After:  batch_size = 8 (2x increase)
- Effective batch: 8 * 2 (accumulation) = 16
```
**Impact**: More stable gradients, better generalization

#### Learning Rate:
```python
- Before: 5e-4 (0.0005)
- After:  3e-4 (0.0003)  [40% reduction]
```
**Impact**: More conservative updates, smoother convergence

#### Gradient Accumulation:
```python
- Before: accumulation_steps = 4
- After:  accumulation_steps = 2
```
**Impact**: Effective batch size = 8×2 = 16 (vs previous 4×4 = 16, but with larger base batch)

---

### 5. **Adaptive Learning Rate Scheduling** (`src/training/train.py`)

```python
# ReduceLROnPlateau updates:
- factor: 0.5 (was 0.7) - more aggressive reduction
- patience: 15 epochs (was 20) - faster adaptation
- min_lr: 5e-6 (was 1e-5) - lower floor for fine-tuning
- threshold: 0.01 (was 0.005) - more conservative improvement detection
- verbose: True (was commented out) - monitor LR changes
```

**Impact**: Better adaptation when validation plateaus

---

### 6. **Early Stopping** (`src/training/train.py`)

```python
# Early stopping patience:
- Before: 100 epochs
- After:  30 epochs
```

**Impact**: Stops training 70 epochs sooner when no improvement detected

---

## Expected Outcomes

### Immediate Effects:
1. **Training will be slower** - dropout + augmentation add computational overhead
2. **Training loss will be higher** - regularization prevents perfect fitting
3. **Validation loss should improve** - better generalization
4. **WER should decrease** - target < 60% initially, goal < 25%

### Training Behavior:
- **First 20 epochs**: Model learns basic patterns despite strong regularization
- **Epochs 20-50**: Validation WER should steadily decrease
- **After epoch 50**: LR reductions kick in, fine-tuning begins
- **Early stopping**: Triggered around epoch 60-80 if WER plateaus

### Performance Targets:
| Metric | Previous | Target | Stretch Goal |
|--------|----------|--------|--------------|
| Val WER | 78-80% | < 60% | < 40% |
| Train-Val Gap | ~4.0 | < 1.5 | < 1.0 |
| Best Epoch | 300+ | 50-80 | 40-60 |

---

## How to Train with New Settings

### Option 1: Use all new defaults (recommended)
```bash
python src/training/train.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/student/mobilenet_v3_optimized
```

### Option 2: Further increase batch size (if GPU memory allows)
```bash
python src/training/train.py \
    --batch_size 16 \
    --accumulation_steps 1 \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/student/mobilenet_v3_optimized
```

### Option 3: Even more aggressive regularization
```bash
python src/training/train.py \
    --batch_size 8 \
    --dropout 0.6 \
    --weight_decay 0.005 \
    --learning_rate 2e-4 \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/student/mobilenet_v3_optimized
```

---

## Monitoring Training

### Key Metrics to Watch:

1. **Train-Val Loss Gap**
   - Current: ~2.0 (train) vs ~4.0 (val) = 2.0 gap
   - Target: < 1.0 gap
   - Watch: If gap > 2.0 after epoch 50, may need even stronger regularization

2. **Validation WER Trend**
   - Current: Plateaus immediately
   - Target: Steady decrease until epoch 50-60
   - Watch: If plateaus before epoch 30, increase dropout further

3. **Learning Rate Changes**
   - Now visible with `verbose=True`
   - Should see 2-3 reductions before early stopping
   - If no reductions: validation is improving steadily ✓

4. **Early Stopping**
   - Should trigger around epoch 60-80
   - If triggers < 40: may be too aggressive, reduce dropout slightly
   - If triggers > 100: increase patience or improve augmentation

---

## Files Modified

### 1. `src/data/dataset.py`
- Enhanced `_augment_features()` with 7 augmentation techniques
- Increased augmentation probability and strength

### 2. `src/models/mobilenet_v3.py`
- Increased default dropout: 0.3 → 0.5
- Added dropout to `ModalityEncoder` output
- Added `lstm_dropout` layer after BiLSTM
- Changed `output_proj` to Sequential with dropout
- Updated `create_mobilenet_v3_model()` default

### 3. `src/training/train.py`
- Batch size: 4 → 8
- Learning rate: 5e-4 → 3e-4
- Weight decay: 1e-4 → 1e-3
- Accumulation steps: 4 → 2
- Dropout default: 0.3 → 0.5
- Early stopping patience: 100 → 30
- Scheduler: Updated factor, patience, min_lr, threshold
- Scheduler: Enabled verbose logging

---

## Troubleshooting

### If validation loss is too high (> 5.0):
- Reduce dropout to 0.4
- Increase learning rate to 4e-4
- Reduce augmentation probability by 20%

### If still overfitting (val WER > 70%):
- Increase batch size to 16
- Increase dropout to 0.6
- Add more training data or reduce model size

### If training is too slow:
- Reduce some augmentation techniques (keep top 4)
- Increase batch size, reduce accumulation steps
- Use fewer workers or smaller sequences

### If WER doesn't improve below 60%:
- Model capacity may be insufficient despite overfitting
- Consider architecture changes (outside scope)
- Check data quality and preprocessing

---

## Success Criteria

The changes are working if you observe:
1. ✅ Training loss > 2.0 (not too low)
2. ✅ Validation loss < 4.5 (improved from ~4.2)
3. ✅ Train-Val gap < 1.5 (reduced from ~2.0)
4. ✅ Validation WER steadily decreasing
5. ✅ Early stopping triggers 50-80 epochs
6. ✅ Best WER < 60% (ideally < 40%)

---

## Summary of Changes

| Component | Metric | Before | After | Change |
|-----------|--------|--------|-------|--------|
| **Data Aug** | Probability | 20-50% | 30-70% | +50% |
| **Data Aug** | Techniques | 3 | 7 | +133% |
| **Dropout** | Rate | 0.3 | 0.5 | +67% |
| **Dropout** | Layers | 3 | 6 | +100% |
| **Batch Size** | Size | 4 | 8 | +100% |
| **Weight Decay** | λ | 1e-4 | 1e-3 | +900% |
| **Learning Rate** | LR | 5e-4 | 3e-4 | -40% |
| **Early Stop** | Patience | 100 | 30 | -70% |
| **Scheduler** | Patience | 20 | 15 | -25% |

**Total Regularization Increase: ~3-5x stronger**

---

## Conclusion

These comprehensive changes implement a multi-pronged attack on overfitting:
1. **Data-level**: Aggressive augmentation
2. **Model-level**: Much higher dropout
3. **Optimization-level**: Stronger weight decay, smaller LR
4. **Training-level**: Better batching, early stopping

The model should now generalize significantly better. Expect:
- Higher training loss (good!)
- Lower validation loss (good!)
- Much better validation WER (goal!)
- Faster convergence to best model
- Automatic stopping before severe overfitting

**Next Steps**: Run training and monitor the metrics above. If overfitting persists, consider:
- Even larger batch sizes (16-32)
- Dropout up to 0.7
- Reducing model capacity (fewer hidden dims)
- Collecting more training data

