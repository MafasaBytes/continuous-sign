# 🔥 Teacher Training - Gradient Explosion Emergency Fix

## ❌ What You Saw (NOT Normal)

```
WARNING - Large gradient norm: inf at batch 0
WARNING - Large gradient norm: inf at batch 1
WARNING - Large gradient norm: inf at batch 2
...
loss=104.3560, grad_norm=inf
loss=94.4273, grad_norm=inf
```

**This is GRADIENT EXPLOSION** - the model is unstable and will crash soon.

## 🔧 Emergency Fixes Applied

### 1. Much Lower Learning Rate
```diff
- base_lr = 1e-4  # Was still too high
+ base_lr = 1e-5  # 10x lower for stability
```

### 2. More Aggressive Gradient Clipping
```diff
- max_norm = 1.0  # Still too permissive
+ max_norm = 0.5  # Much stricter clipping
```

### 3. Skip Batches with Infinite Gradients
```python
if torch.isinf(grad_norm) or grad_norm > 10.0:
    logger.warning(f"Infinite gradient, skipping batch")
    continue  # Don't update weights with bad gradients
```

### 4. Smaller Initial Weights
```diff
- nn.init.normal_(m.weight, 0, 0.01)  # Too large
+ nn.init.normal_(m.weight, 0, 0.001)  # 10x smaller

# For Conv layers
m.weight.data *= 0.5  # Scale down by 50%
```

## 🚀 What to Do Now

### Step 1: Stop Current Training
Press `Ctrl+C` in your terminal

### Step 2: Run with Ultra-Stable Config

**Windows**:
```cmd
train_teacher_stable.bat
```

**Linux/Mac**:
```bash
bash train_teacher_stable.sh
```

### Step 3: Watch for Healthy Signs

**Good** ✅:
```
Epoch 1: loss=15.234, grad_norm=0.48
Epoch 1: loss=14.567, grad_norm=0.45
Epoch 1: loss=13.891, grad_norm=0.43
```
- Loss decreasing
- Gradient norms < 1.0
- NO "inf" warnings

**Bad** ❌:
```
WARNING - Infinite gradient norm at batch X
loss=95.XXX, grad_norm=inf
```
- Still seeing inf
- Need even more aggressive fixes

## 📊 What Changed

| Parameter | Before | After Emergency Fix |
|-----------|--------|-------------------|
| **Learning Rate** | 1e-4 | 1e-5 (10x lower) |
| **Grad Clip** | 1.0 | 0.5 (2x stricter) |
| **Skip Inf Grads** | No | Yes ✅ |
| **Weight Init** | 0.01 | 0.001 (10x smaller) |
| **Conv Init Scale** | 1.0 | 0.5 (50% smaller) |
| **Beta2** | 0.999 | 0.98 (more stable) |

## 🎯 Expected Behavior Now

### Epoch 1 (First 10 batches)

**Before (Bad)**:
```
Batch 0: loss=104.36, grad_norm=inf ❌
Batch 1: loss=94.43, grad_norm=inf ❌
Batch 2: loss=94.38, grad_norm=inf ❌
```

**After (Good)**:
```
Batch 0: loss=18.45, grad_norm=0.48 ✅
Batch 1: loss=17.23, grad_norm=0.45 ✅
Batch 2: loss=16.89, grad_norm=0.43 ✅
```

### Full Epoch Progression

| Epoch | Loss | WER | Grad Norm | Status |
|-------|------|-----|-----------|--------|
| 1 | 12-15 | 95-98% | 0.3-0.8 | Stable start ✅ |
| 5 | 10-12 | 90-95% | 0.3-0.7 | Learning ✅ |
| 10 | 8-10 | 80-90% | 0.2-0.6 | Improving ✅ |
| 20 | 6-8 | 60-75% | 0.2-0.5 | Good progress ✅ |
| 50 | 3-5 | 25-35% | 0.1-0.4 | Target range ✅ |

**Warning**: Training will be SLOWER due to very low LR, but it will be STABLE.

## 🔍 Root Cause Analysis

### Why Baseline Works but Teacher Doesn't

| Factor | Baseline | Teacher |
|--------|----------|---------|
| Model depth | 20 layers | 50+ layers |
| Gradient path | Short | Very long |
| Parameters | 15.7M | 40-50M |
| Attention | None | MultiheadAttention |
| Batch size | 4 (stable BN) | 2 (unstable BN) |
| **Stability** | ✅ Inherently stable | ❌ Needs careful tuning |

### The Attention Problem

```python
# MultiheadAttention does this internally:
attention_weights = softmax(Q @ K.T / sqrt(d_k))
output = attention_weights @ V

# If Q or K have large values:
# - Q @ K.T → huge values
# - softmax → extreme (0 or 1)
# - Gradients explode backward
```

**Our fix**: Clamp attention outputs to [-10, 10]

### The Depth Problem

```
Input → 50+ layers → Output

Gradients flow backward:
Output grad → Layer 50 → Layer 49 → ... → Layer 1

At each layer: grad_out * local_grad = grad_in

If local_grad > 1.0 at each layer:
  grad_1 = 1.5^50 = 637,621,500,000 (explosion!)

Our fix: Clip to max_norm=0.5 (keep gradients small)
```

## 🛠️ Alternative: Disable Mixed Precision

If still unstable, try FP32 only (no autocast):

<function_calls>
<invoke name="read_file">
<parameter name="target_file">src/training/train_teacher.py
