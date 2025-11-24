# Overfit Test Guide - Hierarchical Model

## 🎯 Purpose

Before committing to a 4+ hour full training run, verify that:
1. ✅ Hierarchical architecture can learn (not broken)
2. ✅ Multi-scale features work correctly
3. ✅ Hierarchical attention and LSTM function properly
4. ✅ Model can achieve very low loss/WER on tiny dataset

**Time investment**: ~10-30 minutes (vs 4+ hours for full training)

## 🚀 Quick Start

### Step 1: Run Overfit Test
```bash
python overfit_test_hierarchical.py
```

### Step 2: Wait for Results
- Should complete in 10-30 minutes
- Tests on 5 samples
- Runs for up to 2000 epochs (usually converges much faster)

### Step 3: Check Results

**Success Criteria:**
- ✅ Loss < 0.5
- ✅ WER < 15% (standard) or < 5% (strict)
- ✅ Model can memorize the 5 samples

## 📊 Expected Output

### Successful Test:
```
OVERFITTING TEST RESULTS - Hierarchical Teacher
======================================================================
Final Loss:     0.023456
Minimum Loss:   0.023456 ✓ PASS (threshold: < 0.5)
Final WER:      0.00%
Minimum WER:    0.00%
  Standard:     ✓ PASS (threshold: < 15%)
  Strict:       ✓ PASS (threshold: < 5%)
  Improvement:  100.0% reduction from baseline

Overall: ✓ PASSED
```

### What This Means:
- ✅ Architecture is working correctly
- ✅ Model can learn and memorize
- ✅ Safe to proceed with full training

## ⚠️ If Test Fails

### Failure Indicators:
- Loss stays high (> 5.0)
- WER doesn't decrease (< 50% improvement)
- Model predictions are empty or random

### Troubleshooting:

1. **Check Feature Loading**
   ```python
   # Verify features are loaded correctly
   # Check feature dimensions match expected (6516)
   ```

2. **Verify Model Forward Pass**
   ```python
   # Test model with dummy input
   model = create_hierarchical_teacher(vocab_size=973, dropout=0.1)
   dummy_input = torch.randn(1, 100, 6516)  # [B, T, features]
   output = model(dummy_input)
   print(f"Output shape: {output.shape}")  # Should be [T, B, vocab_size]
   ```

3. **Check Gradient Flow**
   - Add gradient logging to see if gradients are flowing
   - Check for vanishing/exploding gradients

4. **Try Different Hyperparameters**
   - Lower learning rate: 0.0001
   - Different optimizer: AdamW
   - Adjust dropout: 0.0 (no dropout for overfit test)

## 📈 Comparison with I3D Teacher

### I3D Teacher Overfit Test Results:
- Typically achieves: Loss < 0.01, WER < 5% in ~1000-2000 epochs
- Architecture validated: ✅

### Hierarchical Teacher Expected:
- Should achieve similar or better results
- May converge faster due to multi-scale features
- Architecture should be validated: ✅

## 🔍 What the Test Checks

1. **Architecture Correctness**
   - Forward pass works without errors
   - Output dimensions are correct
   - CTC loss can be computed

2. **Learning Capability**
   - Model can reduce loss on training data
   - WER decreases over epochs
   - Predictions improve over time

3. **Multi-scale Features**
   - Frame-level, sign-level, sentence-level features work
   - Hierarchical attention functions correctly
   - Hierarchical LSTM processes sequences

## 📝 Files Created

After running the test:
- `overfit_test_hierarchical_results.png` - Training curves plot
- `overfit_test_hierarchical_report.txt` - Detailed text report

## ✅ Next Steps After Success

1. **Proceed with Full Training**
   ```bash
   python train_teacher.py \
     --model_type hierarchical \
     --dropout 0.5 \
     --weight_decay 0.001 \
     --output_dir checkpoints/hierarchical_teacher \
     --epochs 200
   ```

2. **Monitor Training**
   - Watch for overfitting (train/val gap)
   - Check if WER improves
   - Compare with I3D teacher baseline

3. **If Still Overfitting**
   - Add stronger augmentation
   - Increase dropout slightly
   - Consider reducing model capacity

## 🎯 Success Indicators

**Good Signs:**
- ✅ Loss decreases steadily
- ✅ WER improves over epochs
- ✅ Model predictions match targets
- ✅ Test completes successfully

**Warning Signs:**
- ⚠️ Loss plateaus early
- ⚠️ WER doesn't improve
- ⚠️ Predictions are empty/random
- ⚠️ Gradients are zero or NaN

## 💡 Tips

1. **Run on GPU** if available (much faster)
2. **Monitor first 50 epochs** - should see rapid improvement
3. **Check predictions** - should see words appearing, not just empty strings
4. **Compare with I3D** - run both tests to compare architectures

---

**Ready to test!** Run `python overfit_test_hierarchical.py` and wait ~10-30 minutes for results.

