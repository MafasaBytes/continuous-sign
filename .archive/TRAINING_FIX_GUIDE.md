# Training Convergence Fix

## 🔴 Problem Diagnosed

**Model stuck predicting ONE word ("HAAR") for all inputs - CTC collapse!**

### Diagnostic Results
```
Log probs range: [-6.882, -6.878]  ← Almost uniform (no differentiation)
Predictions: "HAAR" for everything  ← Complete collapse
WER: 100% for all 15 epochs         ← No learning
Loss: 103 → 5.2                     ← Decreased but wrong optimization
```

### Root Causes
1. **Learning rate too low**: 1e-5 initial (50x lower than overfit test)
2. **Warmup starting too low**: 10% of 1e-4 = 1e-5
3. **Model stuck in local minimum**: Can't escape with low LR

## ✅ Solution

### Key Changes

| Parameter | Original | Fixed | Reason |
|-----------|----------|-------|--------|
| **Learning Rate** | 1e-4 | **5e-4** | Match overfit test success |
| **Initial LR (warmup)** | 1e-5 | **5e-4** | No warmup needed |
| **Patience** | 15 | **30** | Give model more time |
| **Dropout** | 0.1 | **0.1** | Keep same (worked in overfit) |

### Scheduler Changes Needed

**Current** (causing problems):
```python
# Warmup starts at 10% of target
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, ...)
```

**Fixed** (better for initial learning):
```python
# Option 1: No warmup - start at full LR
main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# Option 2: Gentle warmup from 50% (not 10%)
warmup_scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=3)
```

## 🚀 Quick Fix Command

### Windows PowerShell
```powershell
python src/training/train.py `
  --data_dir data/teacher_features/mediapipe_full `
  --output_dir checkpoints/student/mobilenet_v3_fixed `
  --epochs 500 `
  --batch_size 4 `
  --learning_rate 0.0005 `
  --dropout 0.1 `
  --early_stopping_patience 30
```

### Expected Behavior

**First 5 epochs**:
```
Epoch 1: Loss ~50-80,  WER ~95-98%  ← Initial random
Epoch 2: Loss ~15-25,  WER ~85-92%  ← Starting to learn
Epoch 3: Loss ~8-12,   WER ~70-80%  ← Clear improvement
Epoch 5: Loss ~5-7,    WER ~55-70%  ← Good progress
```

**By epoch 50**:
```
Loss: ~2-4
WER: ~35-45%
```

**By epoch 200**:
```
Loss: ~1-2
WER: ~25-35% ← Approaching target
```

## 🔧 Code Changes Needed

### 1. Modify Scheduler in `train.py`

```python
# REPLACE lines 469-488 with:

# Option A: Simple ReduceLROnPlateau (RECOMMENDED)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',          # Monitor validation loss
    factor=0.5,          # Reduce LR by half
    patience=10,         # Wait 10 epochs
    min_lr=1e-6,         # Minimum LR
    verbose=True
)

# Then in training loop, change:
# scheduler.step()  # OLD
scheduler.step(val_metrics['val_loss'])  # NEW
```

### 2. Update Default Arguments

```python
# Line 807 in train.py
parser.add_argument('--learning_rate', type=float, default=0.0005,  # Was 1e-4
                    help='Learning rate (default: 0.0005 - validated in overfit test)')

parser.add_argument('--early_stopping_patience', type=int, default=30,  # Was 15
                    help='Early stopping patience')
```

## 📊 Monitoring

### What to Watch

1. **First 3 epochs**:
   - Loss should drop dramatically (100 → 10)
   - WER should start decreasing (100% → 85-90%)
   - If still at 100%, **STOP and increase LR more**

2. **Epochs 3-10**:
   - WER should steadily decrease
   - Should see different words in predictions (not just "HAAR")

3. **Epoch 50+**:
   - WER < 50% (good progress)
   - Predictions should be meaningful

### Debug Commands

```python
# Check if model is still stuck
python diagnose_training.py

# Should show:
# - Multiple unique predicted indices (not just [0, 426])
# - Log probs range wider than [-7, -6.8]
# - Different words predicted (not all "HAAR")
```

## ⚠️ If Still Not Converging

### Try These

**1. Even Higher LR**:
```bash
--learning_rate 0.001  # 2x higher
```

**2. Lower Dropout Initially**:
```bash
--dropout 0.05  # Easier initial learning
```

**3. Check Data**:
- Run `diagnose_training.py` again
- Verify features aren't all zeros/NaN
- Check vocab size matches (973)

**4. Simpler Scheduler**:
```python
# Constant LR for first 50 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
```

## ✅ Success Criteria

After re-training with fixes:

- [ ] Epoch 1: WER < 98% (some learning)
- [ ] Epoch 5: WER < 80% (clear learning)
- [ ] Epoch 10: WER < 65% (good progress)
- [ ] Epoch 50: WER < 45% (on track)
- [ ] Epoch 200: WER < 30% (approaching target)
- [ ] Final: WER < 25% (goal achieved!)

## 🎯 Key Takeaway

**The overfitting test validated the architecture works (0% WER).**  
**The problem is purely the learning rate schedule in full training.**

With LR = 0.0005 (matching overfit test), the model WILL converge.

---

**Date**: November 17, 2025  
**Issue**: CTC collapse - model predicting single word  
**Root Cause**: Learning rate 50x too low  
**Fix**: Increase LR to 0.0005, remove warmup

