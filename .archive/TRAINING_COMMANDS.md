# Training Commands - Overfitting Fix

## Quick Start

### Recommended: Use New Defaults (All fixes included)
```bash
python src/training/train.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/student/mobilenet_v3_anti_overfit \
    --epochs 100
```

**This automatically uses:**
- Batch size: 8 (was 4)
- Dropout: 0.5 (was 0.3)
- Weight decay: 0.001 (was 0.0001)
- Learning rate: 0.0003 (was 0.0005)
- Accumulation steps: 2 (effective batch = 16)
- Early stopping: 30 epochs patience
- Enhanced data augmentation (7 techniques)

---

## Configuration Comparison

### Previous Config (config.json - Overfitting)
```json
{
  "model": "MobileNetV3SignLanguage",
  "vocab_size": 973,
  "batch_size": 4,              ← TOO SMALL
  "learning_rate": 0.0005,      ← TOO HIGH
  "dropout": 0.3,               ← TOO LOW
  "weight_decay": 0.0001,       ← TOO WEAK
  "epochs": 500,
  "seed": 42,
  "remove_pca": true,
  "mixed_precision": true,
  "gradient_checkpointing": false,
  "model_params": 14404568
}
```

### New Config (Anti-Overfitting)
```json
{
  "model": "MobileNetV3SignLanguage",
  "vocab_size": 973,
  "batch_size": 8,              ← DOUBLED
  "learning_rate": 0.0003,      ← REDUCED 40%
  "dropout": 0.5,               ← INCREASED 67%
  "weight_decay": 0.001,        ← 10x STRONGER
  "epochs": 100,
  "seed": 42,
  "remove_pca": true,
  "mixed_precision": true,
  "gradient_checkpointing": false,
  "accumulation_steps": 2,      ← NEW (effective batch=16)
  "early_stopping": 30,         ← NEW (was 100)
  "model_params": 14404568,
  "augmentation": "enhanced"    ← 7 techniques
}
```

---

## Advanced Options

### Option 1: Maximum Regularization (If still overfitting)
```bash
python src/training/train.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/student/mobilenet_v3_max_regularization \
    --batch_size 16 \
    --dropout 0.6 \
    --weight_decay 0.005 \
    --learning_rate 2e-4 \
    --accumulation_steps 1 \
    --early_stopping_patience 25 \
    --epochs 100
```

**Use this if:** WER still > 70% after first run

---

### Option 2: Balanced (Moderate regularization)
```bash
python src/training/train.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/student/mobilenet_v3_balanced \
    --batch_size 12 \
    --dropout 0.45 \
    --weight_decay 0.0005 \
    --learning_rate 3e-4 \
    --accumulation_steps 1 \
    --epochs 100
```

**Use this if:** First run was too slow or validation loss > 5.0

---

### Option 3: Fast Training (Lower memory, faster epochs)
```bash
python src/training/train.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/student/mobilenet_v3_fast \
    --batch_size 4 \
    --dropout 0.5 \
    --weight_decay 0.001 \
    --learning_rate 3e-4 \
    --accumulation_steps 4 \
    --num_workers 2 \
    --epochs 100
```

**Use this if:** GPU memory is limited or want faster iteration

---

## Expected Training Time

| Setup | Batch Size | Epochs | Time/Epoch | Total Time |
|-------|------------|--------|------------|------------|
| **Default (Recommended)** | 8 × 2 = 16 | ~60-80 | ~5-8 min | ~6-10 hrs |
| **Max Regularization** | 16 × 1 = 16 | ~50-70 | ~4-6 min | ~4-7 hrs |
| **Fast Training** | 4 × 4 = 16 | ~60-80 | ~8-12 min | ~8-15 hrs |

*Times assume single GPU (V100/A100), ~5000 training samples*

---

## Monitoring During Training

### Terminal Output to Watch:
```
Epoch 1/100
Training:   0%|          | 0/625 [00:00<?, ?it/s]
loss: 12.3456, lr: 3.00e-04

Validation:   0%|          | 0/156 [00:00<?, ?it/s]
Val Loss: 5.4321
WER: 95.12%
SER: 98.45%

# After ~20 epochs, you should see:
WER: 85-90%  ← Improving

# After ~40 epochs:
WER: 70-80%  ← Getting better

# After ~60 epochs:
WER: 60-70%  ← Target range
Early stopping triggered after 30 epochs without improvement
```

### TensorBoard (Real-time visualization):
```bash
# In another terminal:
tensorboard --logdir checkpoints/student/mobilenet_v3_anti_overfit/tensorboard
```

Open browser: `http://localhost:6006`

**Watch these curves:**
1. **Loss/Train** vs **Loss/Val**: Gap should be < 1.5
2. **Metrics/WER**: Should steadily decrease
3. **LR**: Should drop 2-3 times during training

---

## What Success Looks Like

### Training Log (Good):
```
Epoch 1:  Train Loss: 8.234, Val Loss: 8.567, WER: 98.23%
Epoch 10: Train Loss: 4.123, Val Loss: 5.234, WER: 89.45%
Epoch 20: Train Loss: 3.456, Val Loss: 4.123, WER: 82.34%  ← Steady improvement
Epoch 30: Train Loss: 2.987, Val Loss: 3.789, WER: 75.67%
Epoch 40: Train Loss: 2.654, Val Loss: 3.512, WER: 68.23%
Epoch 50: Train Loss: 2.432, Val Loss: 3.289, WER: 62.45%  ← Breaking through!
Epoch 60: Train Loss: 2.287, Val Loss: 3.134, WER: 58.76%  ← Best
Epoch 61-90: No improvement...
Early stopping triggered at epoch 91
Best model: Epoch 60, WER: 58.76%
```

### Training Log (Still Overfitting - needs more regularization):
```
Epoch 1:  Train Loss: 8.234, Val Loss: 8.567, WER: 98.23%
Epoch 10: Train Loss: 4.123, Val Loss: 5.234, WER: 89.45%
Epoch 20: Train Loss: 2.456, Val Loss: 4.523, WER: 85.34%  ← Val loss not improving
Epoch 30: Train Loss: 1.987, Val Loss: 4.489, WER: 84.67%  ← Stuck!
Epoch 40: Train Loss: 1.654, Val Loss: 4.512, WER: 85.23%
Early stopping triggered at epoch 70
Best model: Epoch 25, WER: 84.12%  ← Not good enough

→ Solution: Use "Option 1: Maximum Regularization"
```

---

## Comparing Results

### Before (Original config.json):
- Training Loss: 1.5 (overfitting!)
- Validation Loss: 4.2
- Train-Val Gap: 2.7 (HUGE)
- Best WER: 78.24%
- Epochs to best: 300+
- Clear overfitting ✗

### After (Expected with new settings):
- Training Loss: 2.3-2.5 (healthy)
- Validation Loss: 3.0-3.3
- Train-Val Gap: 0.7-1.0 (good!)
- Best WER: 55-65%
- Epochs to best: 50-70
- Much better generalization ✓

---

## Troubleshooting

### Problem: "Out of memory"
**Solution:**
```bash
python src/training/train.py \
    --batch_size 4 \
    --accumulation_steps 4 \
    --dynamic_truncation \
    [other args...]
```

### Problem: "Training too slow"
**Solutions:**
1. Increase `num_workers`:
   ```bash
   --num_workers 4
   ```
2. Reduce augmentation overhead (edit `src/data/dataset.py`, comment out gaussian_filter1d)

### Problem: "WER not improving below 70%"
**Solutions:**
1. Increase batch size:
   ```bash
   --batch_size 16 --accumulation_steps 1
   ```
2. Stronger dropout:
   ```bash
   --dropout 0.6
   ```
3. More epochs (patience):
   ```bash
   --epochs 150 --early_stopping_patience 50
   ```

### Problem: "Validation loss > 5.0 and increasing"
**Solutions:**
1. Reduce dropout:
   ```bash
   --dropout 0.4
   ```
2. Increase learning rate:
   ```bash
   --learning_rate 4e-4
   ```
3. Check data preprocessing

---

## Summary

**Just run this command to get started:**
```bash
python src/training/train.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/student/mobilenet_v3_anti_overfit
```

All the anti-overfitting fixes are now in the defaults!

**Check progress:**
- Training logs: `checkpoints/student/mobilenet_v3_anti_overfit/training.log`
- Training curves: `figures/baseline/training_curves.png`
- TensorBoard: `tensorboard --logdir checkpoints/student/mobilenet_v3_anti_overfit/tensorboard`

**Expected outcome:** WER drops from 78% → 55-65% with proper generalization.

