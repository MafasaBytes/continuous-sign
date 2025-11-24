# Overfitting Tests - Baseline vs Teacher

## Overview

This directory contains two overfitting tests to validate that both models can memorize a small dataset (sanity check):

1. **Baseline Model** (`overfit_test.py`) - MobileNetV3 student model (15.7M params)
2. **Teacher Model** (`overfit_test_teacher.py`) - I3D teacher model (~40-50M params)

## Purpose

Overfitting tests ensure that:
- The model architecture has sufficient capacity to learn
- The training pipeline (loss, optimizer, data loading) works correctly
- The model can achieve 0% WER on memorizable data

**Success Criteria:**
- ✅ Loss < 0.5
- ✅ WER < 15% (standard)
- ✅ WER < 5% (stretch goal)
- 🎯 WER = 0% (perfect memorization)

## Results Summary

### Baseline Model (MobileNetV3)
- **Status**: ✅ PASSED (0% WER achieved)
- **Epochs**: ~1200/3000
- **Final Loss**: ~0.002
- **Final WER**: 0.00%
- **Parameters**: 15.7M
- **Output Files**:
  - `overfit_test_results.png`
  - `overfit_test_report.txt`

### Teacher Model (I3D)
- **Status**: 🔄 To be tested
- **Expected Performance**: Should achieve 0% WER (larger capacity)
- **Parameters**: ~40-50M (3x more than baseline)
- **Output Files**:
  - `overfit_test_teacher_results.png`
  - `overfit_test_teacher_report.txt`

## How to Run

### Option 1: Direct Python (Cross-platform)

```bash
# Test baseline model
python overfit_test.py

# Test teacher model
python overfit_test_teacher.py
```

### Option 2: Shell Scripts (Linux/Mac)

```bash
# Test baseline model
bash run_overfit_test.sh

# Test teacher model
bash run_overfit_test_teacher.sh
```

### Option 3: Batch Files (Windows)

```cmd
# Test baseline model
run_overfit_test.bat

# Test teacher model
run_overfit_test_teacher.bat
```

## Configuration

Both tests use the same configuration (adjustable in `main()` function):

```python
num_samples = 10           # Number of samples to overfit on
num_epochs = 3000          # Maximum training epochs
learning_rate = 0.0005     # Learning rate (teacher might converge faster)
```

### Key Differences

| Aspect | Baseline | Teacher |
|--------|----------|---------|
| Model | MobileNetV3 | I3D Teacher |
| Parameters | 15.7M | ~40-50M |
| Architecture | Lightweight CNN + BiLSTM | Inception modules + BiLSTM |
| Training Speed | Faster (smaller model) | Slower (larger model) |
| Expected WER | 0% ✅ | 0% (expected) |
| Use Case | Deployment (student) | Knowledge distillation (teacher) |

## Understanding Results

### Loss Curve
- Should show exponential decay (log scale)
- Target: < 0.5, ideally < 0.01

### WER Curve
- Should decrease to 0% or near 0%
- May plateau temporarily before final convergence

### Sample Predictions
- Printed every 50 epochs
- Shows exact matches between target and prediction
- All should show `Match: True` at convergence

## Troubleshooting

### Teacher Model Fails to Converge
If the teacher model doesn't achieve 0% WER:

1. **Increase epochs**: Try 5000 epochs
2. **Adjust learning rate**: 
   - Too high: Loss oscillates
   - Too low: Very slow convergence
3. **Check data**: Ensure features are properly extracted
4. **Reduce num_samples**: Try with 5 samples first

### Out of Memory
Teacher model requires more GPU memory:

```python
# Reduce batch size in dataloader creation
batch_size=num_samples // 2  # Use half the samples per batch
```

Or reduce `num_samples` to 5.

## Next Steps

Once both overfit tests pass:

1. ✅ **Baseline model validated** - Ready for full training
2. 🔄 **Teacher model validated** - Ready for full training
3. 🎯 **Full training pipeline**:
   - Train teacher on full dataset (target: 20-30% WER)
   - Train baseline student (target: 40% WER)
   - Apply knowledge distillation (target: <25% WER)

## File Structure

```
sign-language-recognition/
├── overfit_test.py                    # Baseline overfit test
├── overfit_test_teacher.py            # Teacher overfit test
├── run_overfit_test.sh               # Linux/Mac script (baseline)
├── run_overfit_test_teacher.sh       # Linux/Mac script (teacher)
├── run_overfit_test.bat              # Windows script (baseline)
├── run_overfit_test_teacher.bat      # Windows script (teacher)
├── OVERFIT_TEST_README.md            # This file
├── overfit_test_results.png          # Baseline results plot
├── overfit_test_report.txt           # Baseline detailed report
├── overfit_test_teacher_results.png  # Teacher results plot (after running)
└── overfit_test_teacher_report.txt   # Teacher detailed report (after running)
```

## Notes

- Both tests use the same data (10 samples from train set, seed=42)
- Tests are reproducible with fixed random seed
- Teacher model should achieve similar or better results than baseline
- Success validates both architecture capacity and training setup

