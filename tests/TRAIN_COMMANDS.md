# Training Commands Reference

## 🎯 Quick Start - Start Training Now!

### Option 1: Conservative (Recommended First)
```bash
python train_teacher.py \
  --batch_size 8 \
  --learning_rate 0.0003 \
  --dropout 0.3 \
  --epochs 150 \
  --warmup_epochs 5 \
  --output_dir checkpoints/teacher_conservative
```

**Expected:**
- Training time: ~12-15 hours on single GPU
- Target WER: < 25%
- Stable, predictable training

---

### Option 2: Faster Convergence
```bash
python train_teacher.py \
  --batch_size 12 \
  --learning_rate 0.0005 \
  --dropout 0.25 \
  --epochs 100 \
  --warmup_epochs 3 \
  --accumulation_steps 1 \
  --output_dir checkpoints/teacher_fast
```

**Expected:**
- Training time: ~8-10 hours
- Faster but potentially higher final WER (28-32%)
- Good for rapid experimentation

---

### Option 3: Heavy Regularization (If Overfitting)
```bash
python train_teacher.py \
  --batch_size 8 \
  --learning_rate 0.0002 \
  --dropout 0.4 \
  --weight_decay 0.001 \
  --epochs 200 \
  --warmup_epochs 10 \
  --output_dir checkpoints/teacher_regularized
```

**Expected:**
- Best generalization
- Slower convergence
- Lower validation WER

---

## 📊 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir checkpoints/teacher_conservative/tensorboard
# Open browser: http://localhost:6006
```

### Check Progress
```bash
# View training log
tail -f checkpoints/teacher_conservative/training.log

# Check latest results
cat checkpoints/teacher_conservative/results.json

# View training curves
open checkpoints/teacher_conservative/training_curves.png
```

---

## 🔧 Advanced Options

### Memory Optimization (If OOM)
```bash
python train_teacher.py \
  --batch_size 4 \
  --accumulation_steps 4 \
  --use_amp \
  --output_dir checkpoints/teacher_low_mem
```

### Resume Training
```bash
python train_teacher.py \
  --resume checkpoints/teacher_conservative/checkpoint_epoch_50.pth \
  --epochs 200 \
  --output_dir checkpoints/teacher_continued
```

### Quick Test Run (Small Dataset)
```bash
python train_teacher.py \
  --batch_size 4 \
  --epochs 5 \
  --num_train_samples 100 \
  --output_dir checkpoints/teacher_test
```

---

## 📈 What to Expect

### First 10 Epochs
```
Epoch 1  | Train Loss: 15.234 | Val WER: 97.8%
Epoch 5  | Train Loss:  8.456 | Val WER: 78.3%
Epoch 10 | Train Loss:  5.432 | Val WER: 65.2%
```

### Convergence (50-100 epochs)
```
Epoch 50  | Train Loss: 1.234 | Val WER: 32.5%
Epoch 80  | Train Loss: 0.892 | Val WER: 27.8%
Epoch 100 | Train Loss: 0.654 | Val WER: 24.7% ✓
```

### Success Indicators
- ✅ Loss decreasing smoothly
- ✅ Val WER improving consistently
- ✅ No NaN/Inf warnings
- ✅ Gap between train/val loss reasonable (< 2x)

---

## 🚨 Troubleshooting

### Issue: OOM Error
```bash
# Solution: Reduce batch size and increase accumulation
python train_teacher.py --batch_size 4 --accumulation_steps 4
```

### Issue: Val WER Not Improving
```bash
# Solution: Increase regularization
python train_teacher.py --dropout 0.4 --weight_decay 0.001
```

### Issue: Training Too Slow
```bash
# Solution: Increase learning rate and batch size
python train_teacher.py --learning_rate 0.0005 --batch_size 12
```

### Issue: Overfitting (Train WER << Val WER)
```bash
# Solution: More regularization, data augmentation
python train_teacher.py --dropout 0.4 --weight_decay 0.001
```

---

## 🎓 After Training Succeeds

### 1. Evaluate Test Set
The script automatically evaluates on test set at the end.
Results are in `checkpoints/teacher_*/results.json`

### 2. Extract Soft Targets for Distillation
```python
# Use trained teacher to generate soft targets
python extract_soft_targets.py \
  --checkpoint checkpoints/teacher_conservative/best_model.pth \
  --output data/soft_targets/teacher_predictions.pt
```

### 3. Train Student with Distillation
```python
python train_student_distill.py \
  --teacher_checkpoint checkpoints/teacher_conservative/best_model.pth \
  --soft_targets data/soft_targets/teacher_predictions.pt \
  --alpha 0.7 \
  --temperature 3.0
```

---

## 📊 Comparing Runs

```bash
# Compare different configurations
tensorboard --logdir checkpoints/

# Or use the comparison script
python compare_runs.py \
  checkpoints/teacher_conservative \
  checkpoints/teacher_fast \
  checkpoints/teacher_regularized
```

---

## 💾 Checkpoint Management

### Important Checkpoints
- `best_model.pth` - Best validation WER (use this!)
- `checkpoint_epoch_*.pth` - Regular checkpoints (every 10 epochs)

### Checkpoint Size
- Teacher model: ~200-250 MB per checkpoint
- Keep best + last 3 checkpoints to save disk space

---

## 🎯 Success Criteria Checklist

- [ ] Training completes without crashes
- [ ] Validation WER < 25% achieved
- [ ] Test WER documented
- [ ] Training curves saved and look healthy
- [ ] Best checkpoint saved
- [ ] Results JSON saved
- [ ] Ready for knowledge distillation

---

## 📝 Training Configuration Template

Save this as `config_teacher.yaml`:
```yaml
# Data
data_dir: data/teacher_features/mediapipe_full

# Model
dropout: 0.3

# Training
batch_size: 8
epochs: 150
learning_rate: 0.0003
min_lr: 1.0e-6
warmup_epochs: 5

# Optimizer
optimizer: adamw
weight_decay: 0.0001

# Regularization
max_grad_norm: 1.0
accumulation_steps: 2

# Other
use_amp: true
patience: 15
num_workers: 4
seed: 42
```

Then run:
```bash
python train_teacher.py --config config_teacher.yaml
```

---

Good luck with training! Your model is validated and ready to learn on the full dataset! 🚀

