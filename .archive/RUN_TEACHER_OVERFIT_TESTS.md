# Teacher Overfit Test - Post-Stability-Fixes

## 🎯 Purpose

After adding stability fixes to the teacher model (attention clamping, gradient clipping, etc.), we need to verify the model **can still learn**. This overfit test ensures our fixes didn't over-regularize the model.

## ✅ What Changed (Requiring Retest)

| Component | Original | With Stability Fixes |
|-----------|----------|---------------------|
| Attention Init | Random (default) | Xavier gain=0.5 (smaller) |
| Attention Output | No limit | Clamped [-10, 10] |
| Attention Dropout | 0.0 | 0.1 |
| Architecture | Post-norm | Pre-norm |
| Gradient Clip | 5.0 | 1.0 (stricter) |

**Question**: Can the model still memorize 10 samples with these restrictions?

## 🚀 Run Tests

### Test 1: Standard Overfit Test (Run This First)

```bash
python overfit_test_teacher.py
```

**Expected Result**: 0% WER within 1500-2000 epochs

**Configuration**:
- Samples: 10
- Epochs: 3000
- Learning Rate: 0.0005
- Dropout: 0.1 (from model)
- Uses stability fixes: ✅

### Test 2: Relaxed Overfit Test (If Test 1 Struggles)

```bash
# Only run if Test 1 doesn't reach 0% WER
python overfit_test_teacher_relaxed.py
```

**Configuration**:
- Samples: 10
- Epochs: 3000
- Learning Rate: 0.001 (2x higher)
- Dropout: 0.1 (reduced)
- Uses stability fixes: ✅

## 📊 Interpreting Results

### Scenario A: Both Tests Pass ✅✅

```
Standard Test: 0% WER at epoch 1200
Relaxed Test:  0% WER at epoch 800
```

**Meaning**: Perfect! Stability fixes work AND model can learn.

**Action**: Proceed with full training using standard settings.

### Scenario B: Only Relaxed Test Passes 🟡✅

```
Standard Test: 15% WER at epoch 3000 (stuck)
Relaxed Test:  0% WER at epoch 1200
```

**Meaning**: Stability fixes are too conservative for learning.

**Action**: Use relaxed settings for full training:
```bash
python src/training/train_teacher.py \
    --learning_rate 0.001 \
    --dropout 0.1
```

### Scenario C: Both Tests Struggle ❌❌

```
Standard Test: 75% WER at epoch 3000
Relaxed Test:  65% WER at epoch 3000
```

**Meaning**: Something is fundamentally broken.

**Action**: Check the checklist below.

## 🔍 Troubleshooting Checklist

### If Overfit Test Fails

1. **Check Model Output**
   ```python
   # Add to overfit_test_teacher.py after line 334
   with torch.no_grad():
       print(f"Log probs shape: {log_probs.shape}")
       print(f"Log probs min/max: {log_probs.min():.3f} / {log_probs.max():.3f}")
       print(f"Contains NaN: {torch.isnan(log_probs).any()}")
       print(f"Contains Inf: {torch.isinf(log_probs).any()}")
   ```

2. **Check Attention Clamping**
   - Maybe [-10, 10] is too restrictive?
   - Try [-20, 20] in `src/models/i3d_teacher.py` line 142

3. **Check Gradient Flow**
   ```python
   # Add after backward pass
   total_norm = 0
   for p in model.parameters():
       if p.grad is not None:
           param_norm = p.grad.data.norm(2)
           total_norm += param_norm.item() ** 2
   total_norm = total_norm ** 0.5
   print(f"Total gradient norm: {total_norm:.4f}")
   ```
   
   - Should be 0.1-5.0 in healthy training
   - If < 0.001: Vanishing gradients (clamping too aggressive)
   - If > 100: Exploding gradients (need more clipping)

4. **Disable Attention Temporarily**
   ```python
   # In i3d_teacher.py, ModalityFusion.forward()
   # Replace attention with simple average
   output = fused  # Skip attention completely
   ```
   
   If this works → Attention mechanism is the issue

5. **Check Data**
   ```python
   # Print first batch statistics
   print(f"Features shape: {features.shape}")
   print(f"Features min/max: {features.min():.3f} / {features.max():.3f}")
   print(f"Labels: {labels}")
   print(f"Vocab size: {len(vocab)}")
   ```

## 📈 Expected Learning Curves

### Standard Test (Healthy)

```
Epoch    Loss      WER    Status
   50    8.234    95.0%   Starting to learn
  100    6.123    85.0%   Learning
  200    4.567    65.0%   Improving
  500    2.345    35.0%   Getting good
  800    0.856    15.0%   Almost there
 1200    0.234     5.0%   Close
 1500    0.045     0.0%   ✅ SUCCESS
```

### If Stuck (Problem)

```
Epoch    Loss      WER    Status
   50    9.123    98.0%   Barely learning
  500    8.456    95.0%   Not improving
 1500    7.890    92.0%   Stuck
 3000    7.234    89.0%   ❌ FAILED - Can't learn
```

## 🎯 Comparison with Baseline

Your baseline achieved 0% WER, so teacher should too:

| Model | Overfit Test Result | Analysis |
|-------|---------------------|----------|
| Baseline (MobileNetV3) | 0% @ epoch 800 | ✅ No stability issues |
| Teacher (Before Fixes) | NaN/Inf | ❌ Unstable |
| Teacher (After Fixes) | **To be tested** | Should reach 0% |

## 📝 What to Report

After running the test, note:

1. **Final WER**: X.XX%
2. **Epoch when 0% reached**: XXXX (or N/A)
3. **Any NaN/Inf warnings**: Yes/No
4. **Final loss**: X.XXX
5. **Gradient norms**: Min X.X, Max X.X

Example:
```
✅ Standard test passed!
- Final WER: 0.00%
- Reached 0% at epoch: 1234
- No NaN/Inf warnings
- Final loss: 0.0023
- Gradient norms: 0.5 - 2.3
```

## 🚀 Next Steps After Overfit Test

### If Test Passes ✅

```bash
# Run full teacher training
python src/training/train_teacher.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/teacher \
    --batch_size 2 \
    --epochs 50
```

Monitor for:
- No NaN/Inf warnings ✅
- Loss decreasing steadily ✅
- WER improving to < 30% ✅
- Gradient norms stable (< 5) ✅

### If Test Fails ❌

1. **Try relaxed test**
2. **Check troubleshooting checklist**
3. **Adjust attention clamping**
4. **Report findings** for further debugging

## 📊 Files Generated

### Standard Test
- `overfit_test_teacher_results.png` - Training curves
- `overfit_test_teacher_report.txt` - Detailed results

### Relaxed Test (if needed)
- `overfit_test_teacher_relaxed_results.png` - Training curves
- `overfit_test_teacher_relaxed_report.txt` - Detailed results

## 🎯 Success Criteria

✅ **Minimum**: WER < 15% (proves model can learn)  
✅ **Target**: WER < 5% (proves good capacity)  
🎯 **Ideal**: WER = 0% (proves perfect capacity)

**Your baseline achieved 0%, so teacher should too!**

---

**Ready to test?** Run:
```bash
python overfit_test_teacher.py
```

Good luck! 🚀

