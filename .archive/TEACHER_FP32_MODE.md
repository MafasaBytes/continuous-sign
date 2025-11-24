# 🔧 Teacher Training - Switched to FP32 Mode

## ❌ The Problem You Saw

```
WARNING - Infinite gradient norm at batch 0
WARNING - Infinite gradient norm at batch 1  
WARNING - Infinite gradient norm at batch 2
WARNING - Infinite gradient norm at batch 3
WARNING - Large gradient norm: 81.88 at batch 4
WARNING - Large gradient norm: 66.17 at batch 5
```

**Every single batch** was being skipped due to unstable gradients. Even with:
- LR = 1e-5 (very low)
- Grad clip = 0.5 (very strict)
- Smaller weight initialization

The model was still fundamentally unstable.

## 🎯 The Root Cause

**Mixed Precision (FP16)** was causing numerical instability with the teacher model:

| Component | FP32 Range | FP16 Range | Problem |
|-----------|------------|------------|---------|
| Normal values | ±3.4×10³⁸ | ±6.5×10⁴ | FP16 overflows easily |
| Gradients | Stable | Unstable | Attention + deep network |
| Attention weights | Stable | Explosive | Softmax in FP16 is risky |

### Why Baseline Works but Teacher Doesn't

| Factor | Baseline | Teacher |
|--------|----------|---------|
| Model depth | 20 layers | 50+ layers |
| Attention | None | MultiheadAttention |
| Parameters | 15.7M | 40-50M |
| Mixed precision | ✅ Stable | ❌ Unstable |

## ✅ The Fix: Disabled Mixed Precision

### What Changed

```python
# BEFORE (Mixed Precision - Unstable)
with autocast():  # FP16
    log_probs = model(features, input_lengths)
    loss = criterion(...)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
grad_norm = clip_grad_norm_(...)
scaler.step(optimizer)
scaler.update()

# AFTER (FP32 - Stable)
log_probs = model(features, input_lengths)  # FP32
loss = criterion(...)

loss.backward()  # No scaling
grad_norm = clip_grad_norm_(...)
optimizer.step()  # Direct step
```

### Files Modified

1. **Removed `autocast()`** - Forward pass in FP32
2. **Removed `scaler.scale()`** - No gradient scaling
3. **Removed `scaler.unscale_()`** - No unscaling needed
4. **Removed `scaler.step()`** - Direct optimizer step
5. **Removed `scaler.update()`** - Not needed
6. **Removed `scaler_state_dict`** - From checkpoints

## 🚀 Try Training Again

Now that we're in FP32 mode, try running:

```bash
python src/training/train_teacher.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/teacher \
    --batch_size 2 \
    --epochs 50
```

## 📊 What to Expect Now

### ✅ Success (FP32 Stability)
```
Epoch 1: loss=12.345, grad_norm=0.48
Epoch 1: loss=11.234, grad_norm=0.45
Epoch 1: loss=10.456, grad_norm=0.42
```
- No infinite gradients ✅
- Loss decreasing ✅
- Gradient norms < 1.0 ✅
- Training progresses normally ✅

### 🟡 May Still Skip A FEW Batches (OK)
```
Epoch 1: loss=12.345, grad_norm=0.48
WARNING - Large gradient norm: 11.23 at batch 15, skipping
Epoch 1: loss=11.234, grad_norm=0.45
```
- < 5% of batches skipped: OK
- Loss still decreases overall: OK

### ❌ Still Many Skips (Architecture Problem)
```
WARNING - Infinite gradient at batch 0
WARNING - Large gradient: 95.23 at batch 1
WARNING - Infinite gradient at batch 2
... (many warnings)
```
- If this happens, the **architecture itself is the problem**
- Need to simplify the model or test overfit first

## ⚡ Performance Impact

### Speed

| Mode | Speed | Memory |
|------|-------|--------|
| **FP16 (Mixed)** | Fast (2x) | Low | 
| **FP32 (Current)** | Slower (1x) | High |

**Tradeoff**: Training will be **~2x slower** but **STABLE**.

For a 40M parameter model:
- FP16: ~1 hour/epoch
- FP32: ~2 hours/epoch

Total training time: ~100 hours (4 days) for 50 epochs

### Memory

FP32 uses 2x more memory than FP16:
- Batch size may need to stay at 2
- Cannot increase to 4 without OOM

## 🔍 Still Unstable?

If you still see many infinite gradients in FP32 mode:

### Option 1: Run Overfit Test FIRST

```bash
python overfit_test_teacher.py
```

This will tell you if the model CAN learn at all. If it can't memorize 10 samples, the architecture needs changes.

### Option 2: Simplify Architecture

Reduce model complexity:
- Remove attention mechanism
- Use fewer Inception modules
- Smaller hidden dimensions

### Option 3: Use Simpler Teacher

Instead of I3D, use a scaled-up MobileNetV3:
- Same architecture as baseline
- Just more layers/channels
- Much more stable

## 📝 Summary of ALL Current Settings

| Setting | Value | Reason |
|---------|-------|--------|
| **Precision** | FP32 | Stability (no mixed precision) |
| **Learning Rate** | 1e-5 | Very conservative |
| **Gradient Clip** | 0.5 | Very aggressive |
| **Weight Init** | 0.001 | 10x smaller |
| **Conv Scale** | 0.5 | 50% reduced |
| **Batch Size** | 2 | Memory limit |
| **Skip Bad Batches** | Yes | Continue on errors |

## 🎯 Next Steps

1. **Run training now** - Should be stable in FP32
2. **Monitor first 20 batches** - Check for stability
3. **If stable** - Let it run for 50 epochs
4. **If unstable** - Run overfit test to diagnose

---

**The switch to FP32 should fix the instability!** 🎉

Try it now and monitor the first epoch carefully.

