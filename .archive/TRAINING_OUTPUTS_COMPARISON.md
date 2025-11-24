# Training Outputs Comparison: Baseline vs Teacher

## 📁 Directory Structure

```
sign-language-recognition/
│
├── figures/
│   ├── baseline/                    # ✅ Baseline model visualizations
│   │   ├── training_curves.png     # ✅ High-res PNG (300 DPI)
│   │   └── training_curves.pdf     # ✅ Publication PDF
│   │
│   └── teacher/                     # ✅ Teacher model visualizations (NEW)
│       ├── training_curves.png     # ✅ High-res PNG (300 DPI)
│       └── training_curves.pdf     # ✅ Publication PDF
│
├── checkpoints/
│   ├── baseline/
│   │   └── mobilenet_v3_YYYYMMDD_HHMMSS/
│   │       ├── best_model.pth
│   │       ├── training_curves.png
│   │       ├── training_history.json
│   │       ├── config.json
│   │       ├── results.json
│   │       └── tensorboard/
│   │
│   └── teacher/
│       └── i3d_teacher_YYYYMMDD_HHMMSS/
│           ├── best_i3d.pth
│           ├── training_curves.png      # ✅ NEW
│           ├── training_history.json    # ✅ NEW
│           ├── config.json
│           ├── results.json
│           ├── teacher_training.log
│           └── tensorboard/
│
├── overfit_test_results.png          # Baseline overfit test
├── overfit_test_report.txt           # Baseline overfit report
├── overfit_test_teacher_results.png  # Teacher overfit test
└── overfit_test_teacher_report.txt   # Teacher overfit report
```

## 📊 Visualization Comparison

### Baseline Model (`figures/baseline/training_curves.png`)
```
┌─────────────────────────────────────────────────────────────┐
│     MobileNetV3 Sign Language Model - Training Progress     │
├──────────────────────────┬──────────────────────────────────┤
│ Training & Validation    │  Validation Word Error Rate      │
│ Loss                     │                                  │
│                          │  Target: 25%                     │
│ [Blue: Train Loss]       │  [Green line with markers]       │
│ [Red: Val Loss]          │  [Red: Best WER]                 │
│                          │  [Orange: Target 25%]            │
├──────────────────────────┼──────────────────────────────────┤
│ Learning Rate Schedule   │  Training Overview               │
│                          │                                  │
│ [Purple line, log scale] │  [Combined: Loss + WER]          │
│ [Square markers]         │  [Dual y-axis]                   │
└──────────────────────────┴──────────────────────────────────┘
```

### Teacher Model (`figures/teacher/training_curves.png`)
```
┌─────────────────────────────────────────────────────────────┐
│        I3D Teacher Model - Training Progress                 │
├──────────────────────────┬──────────────────────────────────┤
│ Training & Validation    │  Validation Word Error Rate      │
│ Loss                     │                                  │
│                          │  Teacher Target: 30%             │
│ [Blue: Train Loss]       │  [Green line with markers]       │
│ [Red: Val Loss]          │  [Red: Best WER]                 │
│                          │  [Orange: Target 30%]            │
├──────────────────────────┼──────────────────────────────────┤
│ Learning Rate Schedule   │  Training Overview               │
│                          │                                  │
│ [Purple line, log scale] │  [Combined: Loss + WER]          │
│ [Square markers]         │  [Dual y-axis]                   │
└──────────────────────────┴──────────────────────────────────┘
```

## 🎯 Key Differences

| Feature | Baseline | Teacher |
|---------|----------|---------|
| **Title** | "MobileNetV3 Sign Language Model" | "I3D Teacher Model" |
| **Target WER** | 25% | 30% |
| **Parameters** | 15.7M | ~40-50M |
| **Batch Size** | 4 | 2 (larger model) |
| **Learning Rate** | 5e-5 | 5e-5 (adjusted in code to 2e-4) |
| **Output Dir** | `figures/baseline/` | `figures/teacher/` |
| **Checkpoint Name** | `best_model.pth` | `best_i3d.pth` |
| **Purpose** | Student model (deployment) | Teacher model (distillation) |

## 📈 Metrics Tracked

Both models track identical metrics:

### During Training
- **train_loss**: CTC loss on training set
- **val_loss**: CTC loss on validation set
- **val_wer**: Word Error Rate on validation set (%)
- **val_ser**: Sentence Error Rate on validation set (%)
- **learning_rate**: Current learning rate (log scale)

### Saved to `training_history.json`
```json
{
  "train_losses": [10.5, 8.3, 6.2, ...],
  "val_losses": [11.2, 9.1, 7.4, ...],
  "val_wers": [95.3, 87.2, 78.4, ...],
  "val_sers": [98.1, 94.3, 89.7, ...],
  "learning_rates": [0.0002, 0.0002, 0.0001, ...],
  "best_wer": 42.5
}
```

## 🎨 Plot Generation Schedule

Both models generate plots:
- **Every 5 epochs** during training
- **At the final epoch** (even if not divisible by 5)
- **After test evaluation** (final plot with complete history)

## 📝 Training Commands

### Baseline Training
```bash
python src/training/train.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/baseline \
    --batch_size 4 \
    --epochs 100 \
    --learning_rate 5e-5
```

**Expected Output:**
- Best WER: ~25-30% (target < 25%)
- Training time: ~2-4 hours per epoch (depends on GPU)
- Plots: `figures/baseline/training_curves.{png,pdf}`

### Teacher Training
```bash
python src/training/train_teacher.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/teacher \
    --batch_size 2 \
    --epochs 50 \
    --learning_rate 5e-5
```

**Expected Output:**
- Best WER: ~20-30% (target < 30%)
- Training time: ~4-6 hours per epoch (larger model)
- Plots: `figures/teacher/training_curves.{png,pdf}`

## 🔍 Monitoring Training

### Option 1: Watch Plots (Updated every 5 epochs)
```bash
# Linux/Mac
watch -n 10 ls -lh figures/baseline/
watch -n 10 ls -lh figures/teacher/

# Windows (PowerShell)
while($true) { cls; ls figures\baseline\; ls figures\teacher\; sleep 10 }
```

### Option 2: TensorBoard
```bash
# Baseline
tensorboard --logdir checkpoints/baseline --port 6006

# Teacher
tensorboard --logdir checkpoints/teacher --port 6007

# Both
tensorboard --logdir checkpoints/ --port 6008
```

### Option 3: Log Files
```bash
# Baseline
tail -f checkpoints/baseline/*/training.log

# Teacher
tail -f checkpoints/teacher/*/teacher_training.log
```

## 📊 Side-by-Side Comparison

After training, compare the models:

```bash
# View both plots
eog figures/baseline/training_curves.png figures/teacher/training_curves.png

# Or on Windows
start figures\baseline\training_curves.png
start figures\teacher\training_curves.png
```

### Expected Results

**Baseline (MobileNetV3)**
- Faster training per epoch
- May struggle to reach < 25% WER
- Smaller, deployable model

**Teacher (I3D)**
- Slower training per epoch
- Should achieve < 30% WER more easily
- Larger, more accurate model for distillation

## ✅ Update Summary

**BEFORE**: Teacher training only used TensorBoard (real-time but requires separate viewer)

**AFTER**: Teacher training generates:
- ✅ matplotlib/seaborn plots (same as baseline)
- ✅ High-resolution PNG exports
- ✅ Publication-ready PDF exports
- ✅ Training history JSON
- ✅ Periodic updates every 5 epochs
- ✅ Automatic figure directory creation

**Result**: Complete feature parity between baseline and teacher training scripts!

## 🚀 Verification Checklist

Run a short teacher training to verify:

```bash
# Test with 5 epochs
python src/training/train_teacher.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/teacher_test \
    --batch_size 2 \
    --epochs 5
```

Then check:
- [ ] `figures/teacher/` directory created
- [ ] `figures/teacher/training_curves.png` exists
- [ ] `figures/teacher/training_curves.pdf` exists
- [ ] Plot shows "I3D Teacher Model" title
- [ ] Target line shows 30%
- [ ] All 4 subplots visible
- [ ] training_history.json saved
- [ ] Logs mention "Training curves saved to figures/teacher/"

If all checks pass: ✅ Teacher training visualization is working!

