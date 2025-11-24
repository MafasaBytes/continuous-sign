# Data Augmentation Strategy for Overfitting

## 📊 Current Situation

**Your model:** 27M parameters  
**Your data:** 4,376 training samples  
**Current augmentation:** Weak (speed 0.9-1.1x, noise σ=0.01, 10% dropout)  
**Current regularization:** Too weak (dropout 0.3, weight_decay 0.0001)  
**Result:** Overfitting (train/val gap 2.93, val WER 90%+)

---

## 🎯 Solution Priority

### Priority 1: Fix Regularization (Do This First!) ⭐⭐⭐

**Why first:**
- Instant effect (no code changes needed)
- No training time overhead
- Most effective for your situation
- Already implemented in updated `train_teacher.py`

**Action:**
```bash
python train_teacher.py \
  --dropout 0.5 \
  --weight_decay 0.001 \
  --output_dir checkpoints/teacher_fixed
```

**Expected impact:**
- Train/val gap: 2.93 → <1.0 ✓
- Val WER: 92% → 70% (epoch 50) → 40% (epoch 100) ✓
- Training time: Same
- Predictions: Multi-word instead of empty ✓

---

### Priority 2: Stronger Augmentation (If Still Overfitting) ⭐⭐

**When to use:**
- After trying Priority 1 for 40-50 epochs
- If train/val gap still >1.5
- If val WER plateaus above 35%

**Action:**
1. Replace `_augment_features()` in `src/data/dataset.py` with code from `enhanced_augmentation.py`
2. Restart training

**Expected impact:**
- Further reduces overfitting
- Val WER: Potentially 35% → 28% ✓
- Training time: +10-20% (augmentation overhead)
- More diverse training data

---

### Priority 3: Advanced Techniques (Optional) ⭐

Only if Priorities 1 and 2 don't achieve < 25% WER:
- Mixup augmentation (batch-level)
- Label smoothing
- Cutout/DropBlock variants
- Multi-scale training

---

## 📈 Comparison: Weak vs Strong Augmentation

| Augmentation | Current (Weak) | Enhanced (Strong) | Impact |
|--------------|----------------|-------------------|--------|
| **Speed perturbation** | 0.9-1.1x (50%) | 0.85-1.15x (60%) | Medium |
| **Gaussian noise** | σ=0.01 (30%) | σ=0.01-0.03 (50%) | Medium |
| **Feature dropout** | 10% (20%) | 10-20% (30%) | Medium |
| **Time masking** | ❌ None | 5-15% masked (30%) | High |
| **Frame sampling** | ❌ None | Drop 5-15% (20%) | Medium |
| **Rotation** | ❌ None | ±10° (30%) | Medium |
| **Feature scaling** | ❌ None | 0.9-1.1x (40%) | Low |
| **Temporal reverse** | ❌ None | Reverse (10%) | Low |
| **Gaussian blur** | ❌ None | σ=0.5-1.5 (20%) | Low |

**Current:** 3 augmentations (weak)  
**Enhanced:** 9 augmentations (strong)

---

## 🔬 Expected Results

### Scenario 1: Regularization Only (Recommended Start)
```
Configuration:
  dropout: 0.5
  weight_decay: 0.001
  augmentation: Current (weak)

Expected Results:
  Epoch 50:  Train 65% | Val 70% | Gap 5%  ✓
  Epoch 100: Train 45% | Val 50% | Gap 5%  ✓
  Epoch 150: Train 30% | Val 35% | Gap 5%  ✓ (may not hit 25%)
```

### Scenario 2: Regularization + Strong Augmentation
```
Configuration:
  dropout: 0.5
  weight_decay: 0.001
  augmentation: Enhanced (strong)

Expected Results:
  Epoch 50:  Train 70% | Val 72% | Gap 2%  ✓✓
  Epoch 100: Train 50% | Val 52% | Gap 2%  ✓✓
  Epoch 150: Train 32% | Val 30% | Gap -2% ✓✓ (val better!)
  Epoch 180: Train 28% | Val 24% | Gap -4% ✓✓✓ TARGET!
```

Note: With strong augmentation, validation can sometimes be better than training (healthy sign!).

---

## 🚀 Implementation Steps

### Step 1: Try Regularization First (Today)

```bash
# 1. Stop current training
Ctrl+C

# 2. Restart with strong regularization
python train_teacher.py \
  --dropout 0.5 \
  --weight_decay 0.001 \
  --patience 40 \
  --output_dir checkpoints/teacher_reg_only

# 3. Monitor for 40-50 epochs (~8-10 hours)
tensorboard --logdir checkpoints/teacher_reg_only/tensorboard
```

**Decision point at epoch 40-50:**
- ✅ If val WER < 60% → Continue, it's working!
- ⚠️ If val WER 60-70% and gap < 1.0 → Continue, might reach 35-40%
- ❌ If val WER > 70% or gap > 1.5 → Move to Step 2

---

### Step 2: Add Strong Augmentation (If Needed)

```bash
# 1. Update dataset with strong augmentation
# Copy augment_features_strong() from enhanced_augmentation.py
# to src/data/dataset.py (replace _augment_features)

# 2. Restart training
python train_teacher.py \
  --dropout 0.5 \
  --weight_decay 0.001 \
  --patience 50 \
  --epochs 200 \
  --output_dir checkpoints/teacher_strong_aug

# 3. Train longer (augmentation helps more over time)
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Adding Augmentation Without Regularization
```
dropout: 0.3, weight_decay: 0.0001, strong augmentation
Result: Still overfits! Augmentation alone isn't enough.
```

### Pitfall 2: Too Strong Augmentation Too Early
```
dropout: 0.5, weight_decay: 0.001, very strong augmentation
Result: Model can't learn anything! Train WER stays >80%.
Solution: Start moderate, increase gradually.
```

### Pitfall 3: Not Training Long Enough
```
Strong regularization + augmentation but only 100 epochs
Result: Val WER 40% (seems stuck)
Solution: Train 150-200 epochs, convergence is slower.
```

---

## 📊 Monitoring Guide

### Healthy Training (What You Want):
```
Epoch 20:  Train Loss 4.5 | Val Loss 4.7 | Gap 0.2
Epoch 40:  Train Loss 3.8 | Val Loss 3.9 | Gap 0.1
Epoch 60:  Train Loss 3.2 | Val Loss 3.3 | Gap 0.1
Epoch 100: Train Loss 2.5 | Val Loss 2.6 | Gap 0.1
```
- Gap stays small (<1.0)
- Both losses decreasing
- Val WER improving consistently

### Still Overfitting (Need More Aug):
```
Epoch 60:  Train Loss 3.0 | Val Loss 4.2 | Gap 1.2
Epoch 100: Train Loss 2.2 | Val Loss 4.5 | Gap 2.3
```
- Gap widening (even with dropout 0.5)
- Val loss plateauing/increasing
- → Add stronger augmentation

### Underfitting (Too Much Reg/Aug):
```
Epoch 60:  Train Loss 5.5 | Val Loss 5.6 | Gap 0.1
Epoch 100: Train Loss 5.2 | Val Loss 5.3 | Gap 0.1
```
- Losses not decreasing much
- Train WER stuck >75%
- → Reduce dropout or augmentation

---

## 🎯 Decision Tree

```
START: Overfitting detected
  ↓
[1] Increase dropout to 0.5, weight_decay to 0.001
  ↓
Train 40-50 epochs
  ↓
Is train/val gap < 1.0?
  ├─ YES → Continue training to 150 epochs ✓
  │         Check if Val WER < 25%
  │         ├─ YES → SUCCESS! ✓✓✓
  │         └─ NO  → Try Step 2 (strong aug)
  │
  └─ NO  → [2] Add strong augmentation
            ↓
          Train 100-150 epochs
            ↓
          Is gap < 1.0 now?
            ├─ YES → Continue to 200 epochs ✓
            │         Should reach < 25% WER
            │
            └─ NO  → [3] Consider:
                      - Reduce model size (hidden_dim)
                      - Train MobileNetV3 first
                      - Use pre-trained weights
```

---

## ✅ TL;DR - What to Do NOW

1. **Stop current training** (it's overfitting)

2. **Restart with strong regularization:**
   ```bash
   python train_teacher.py --dropout 0.5 --weight_decay 0.001
   ```

3. **Wait 40-50 epochs (~8-10 hours)**

4. **Check results:**
   - If gap < 1.0 → ✓ Working, continue
   - If gap > 1.5 → Add strong augmentation (Step 2)

5. **Strong augmentation is OPTIONAL and SECONDARY**
   - Only if regularization isn't enough
   - Adds 10-20% training time
   - Can help get from 35% → 25% WER

**Most important:** Fix regularization first! It's free and will likely solve your problem.

---

## 📝 Files Created

- `enhanced_augmentation.py` - Strong augmentation implementation
- `AUGMENTATION_STRATEGY.md` - This guide
- `train_teacher.py` - Already updated with dropout=0.5, weight_decay=0.001

**Ready to go!** Just restart training with the updated defaults.

