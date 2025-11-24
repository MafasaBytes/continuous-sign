# 🐛 Teacher Training - Scaler Bug Fix

## Error That Occurred

```
RuntimeError: unscale_() has already been called on this optimizer since the last update().
```

**Line 283**: `scaler.unscale_(optimizer)`

## 🔍 Root Cause

When we detected infinite gradients and tried to skip the batch:

```python
# Old broken code:
scaler.unscale_(optimizer)  # ← Called
grad_norm = clip_grad_norm_(...)

if math.isinf(grad_norm):
    continue  # ← Skipped without calling scaler.update()

# Next iteration:
scaler.unscale_(optimizer)  # ← ERROR! Already unscaled
```

**Problem**: PyTorch's `GradScaler` requires you to call `update()` after every `unscale_()`, even if you skip the optimizer step.

## ✅ Fix Applied

```python
# New fixed code:
scaler.unscale_(optimizer)
grad_norm = clip_grad_norm_(...)

if math.isinf(grad_norm):
    # Must call update() to reset scaler state
    scaler.update()
    optimizer.zero_grad()
    nan_count += 1
    continue  # ← Now safe to skip

# Normal path:
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

**Key changes**:
1. ✅ Call `scaler.update()` even when skipping bad batches
2. ✅ Call `optimizer.zero_grad()` to clear gradients
3. ✅ Use `math.isinf()` instead of `torch.isinf(torch.tensor())`
4. ✅ Track nan_count when skipping bad input

## 🚀 Ready to Train Now

The crash is fixed. Run training again:

**Windows**:
```cmd
python src/training/train_teacher.py ^
    --data_dir data/teacher_features/mediapipe_full ^
    --output_dir checkpoints/teacher ^
    --batch_size 2 ^
    --epochs 50
```

**Or use the ultra-stable script**:
```cmd
train_teacher_stable.bat
```

## 📊 What to Expect Now

### ✅ Good (Stable Training)
```
Epoch 1: loss=12.345, grad_norm=0.48
Epoch 1: loss=11.234, grad_norm=0.45
Epoch 1: loss=10.456, grad_norm=0.42
```
- Loss decreasing
- Gradient norms < 1.0
- No infinite gradients
- No crashes

### 🟡 May Still Skip Some Batches (OK)
```
WARNING - Infinite gradient norm at batch 5, skipping optimizer step
Epoch 1: loss=12.345, grad_norm=0.48
Epoch 1: loss=11.234, grad_norm=0.45 (batch 5 skipped)
Epoch 1: loss=10.456, grad_norm=0.42
```
- A few skipped batches is OK (< 10 per epoch)
- Training continues without crashing
- Loss still decreases overall

### ❌ Bad (Still Unstable)
```
WARNING - Infinite gradient norm at batch 0, skipping
WARNING - Infinite gradient norm at batch 1, skipping
WARNING - Infinite gradient norm at batch 2, skipping
...
ERROR: Too many NaN/Inf outputs, stopping epoch
```
- Too many skipped batches (> 50% of epoch)
- Model fundamentally unstable
- **Need**: Even lower LR or disable mixed precision

## 🔧 If Still Unstable

### Option 1: Disable Mixed Precision

Edit `src/training/train_teacher.py` line 252:

```python
# Comment out autocast:
# with autocast():
#     log_probs = model(features, input_lengths)
#     loss = criterion(...)

# Use FP32 instead:
log_probs = model(features, input_lengths)
loss = criterion(...)
```

### Option 2: Even Lower Learning Rate

```bash
python src/training/train_teacher.py --learning_rate 1e-6
```

### Option 3: Batch Size = 1

```bash
python src/training/train_teacher.py --batch_size 1
```

## 📝 Summary of All Fixes

| Issue | Status |
|-------|--------|
| ✅ NaN/Inf detection | Fixed |
| ✅ Gradient clipping (0.5) | Fixed |
| ✅ Lower LR (1e-5) | Fixed |
| ✅ Smaller weight init | Fixed |
| ✅ Skip infinite grad batches | Fixed |
| ✅ **Scaler state bug** | **Fixed Now** |

## 🎯 Expected Training Time

With all fixes:
- **Epochs**: 50
- **Time per epoch**: ~1-2 hours (depends on GPU)
- **Total time**: ~50-100 hours
- **Target WER**: < 30%

**Note**: Training is SLOW due to:
- Very low LR (1e-5)
- Model size (40M params)
- Batch size (2)
- Some skipped batches

But it should be **STABLE** now!

## 🚀 Next Steps

1. ✅ Crash fixed - safe to run
2. 🔄 Start training (will take days)
3. 📊 Monitor first epoch carefully
4. ✅ If stable (no crash, loss decreasing) → let it run
5. ❌ If still many inf grads → try Options 1-3 above

---

**The training should work now!** 🎉

Monitor the first 10 batches to confirm stability, then let it run for 50 epochs.

