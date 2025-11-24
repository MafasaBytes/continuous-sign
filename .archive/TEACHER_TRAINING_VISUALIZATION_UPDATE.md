# Teacher Training Script - Visualization Update

## Summary

The teacher training script (`src/training/train_teacher.py`) has been **updated to match the baseline visualization capabilities**. It now generates the same comprehensive training plots as the baseline model.

## ✅ Changes Made

### 1. **Added Visualization Libraries** (Line 32-33)
```python
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. **Added `plot_training_curves()` Function** (Lines 111-225)
- Creates 2×2 subplot layout with:
  - **Top Left**: Training vs Validation Loss
  - **Top Right**: Validation WER with target line at 30% (teacher target)
  - **Bottom Left**: Learning Rate Schedule (log scale)
  - **Bottom Right**: Combined metrics overview
- Saves plots to:
  - `figures/teacher/training_curves.png` (high-res PNG)
  - `figures/teacher/training_curves.pdf` (publication quality)
  - `{checkpoint_dir}/training_curves.png` (backup)
- Uses thesis-quality styling with seaborn

### 3. **Added Training History Tracking** (Lines 415-420)
```python
train_losses = []
val_losses = []
val_wers = []
val_sers = []
learning_rates = []
```

### 4. **Metric Recording During Training** (Lines 439-444)
After each validation step:
```python
train_losses.append(train_metrics['train_loss'])
val_losses.append(val_metrics['val_loss'])
val_wers.append(val_metrics['wer'])
val_sers.append(val_metrics['ser'])
learning_rates.append(optimizer.param_groups[0]['lr'])
```

### 5. **Periodic Plot Generation** (Lines 488-497)
Generates plots every 5 epochs and at the end:
```python
if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
    plot_training_curves(
        train_losses, val_losses, val_wers, 
        learning_rates, best_wer, output_dir
    )
```

### 6. **Training History JSON Export** (Lines 499-509)
Saves complete training history for analysis:
```python
history = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'val_wers': val_wers,
    'val_sers': val_sers,
    'learning_rates': learning_rates,
    'best_wer': best_wer
}
```

### 7. **Final Plot Generation** (Lines 538-546)
Creates final visualization after test evaluation

### 8. **Enhanced Logging** (Line 566)
Added training curves path to final summary:
```python
logger.info(f"Training curves: figures/teacher/training_curves.png")
```

## 📊 Output Files

The teacher training now generates:

### Visualization Files
- `figures/teacher/training_curves.png` (300 DPI PNG)
- `figures/teacher/training_curves.pdf` (Vector PDF)
- `{checkpoint_dir}/training_curves.png` (Backup copy)

### Data Files
- `{checkpoint_dir}/training_history.json` (Complete metrics)
- `{checkpoint_dir}/config.json` (Training configuration)
- `{checkpoint_dir}/results.json` (Final results)

### Model Checkpoints
- `{checkpoint_dir}/best_i3d.pth` (Best model)
- `{checkpoint_dir}/checkpoint_epoch_X.pth` (Periodic checkpoints)
- `checkpoints/teacher/best_i3d.pth` (Copy for distillation)

### Logs
- `{checkpoint_dir}/teacher_training.log` (Text log)
- `{checkpoint_dir}/tensorboard/` (TensorBoard logs)

## 🎨 Visualization Features

### Thesis-Quality Styling
- Professional seaborn paper style
- High-resolution (300 DPI) exports
- Publication-ready PDF format
- Consistent color scheme with baseline

### Key Differences from Baseline
1. **Title**: "I3D Teacher Model - Training Progress"
2. **Target Line**: 30% WER (teacher target) vs 25% (baseline target)
3. **Output Directory**: `figures/teacher/` vs `figures/baseline/`
4. **Model Name**: I3D Teacher vs MobileNetV3

### Subplot Details

#### 1. Training and Validation Loss
- Blue line: Training loss
- Red line: Validation loss
- Grid for easy reading
- Automatic x-axis scaling

#### 2. Validation Word Error Rate
- Green line with markers
- Red dashed line: Best WER achieved
- Orange dotted line: Teacher target (30%)
- Legend shows both best and target

#### 3. Learning Rate Schedule
- Purple line with square markers
- Log scale for visibility
- Tracks ReduceLROnPlateau adjustments

#### 4. Training Overview
- Dual y-axis plot
- Left axis: Train/Val loss
- Right axis: WER (%)
- Combined legend

## 🔄 Comparison with Baseline

| Feature | Baseline (`train.py`) | Teacher (`train_teacher.py`) |
|---------|----------------------|------------------------------|
| Visualization | ✅ Yes | ✅ Yes (NOW) |
| 2×2 Subplot Layout | ✅ | ✅ |
| PDF Export | ✅ | ✅ |
| PNG Export | ✅ | ✅ |
| Training History JSON | ✅ | ✅ |
| Periodic Updates | Every 5 epochs | Every 5 epochs |
| Final Plot | ✅ | ✅ |
| TensorBoard | ✅ | ✅ |
| Target WER Line | 25% | 30% |
| Output Directory | `figures/baseline/` | `figures/teacher/` |

## 📝 Usage

### Run Teacher Training
```bash
python src/training/train_teacher.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/teacher \
    --batch_size 2 \
    --epochs 50 \
    --learning_rate 5e-5
```

### Monitor Training
1. **Real-time**: Check TensorBoard
   ```bash
   tensorboard --logdir checkpoints/teacher/*/tensorboard
   ```

2. **Plots**: View updates every 5 epochs
   - `figures/teacher/training_curves.png`

3. **Logs**: Follow text log
   ```bash
   tail -f checkpoints/teacher/*/teacher_training.log
   ```

### After Training
Check final outputs:
```bash
ls figures/teacher/
# training_curves.png
# training_curves.pdf

ls checkpoints/teacher/i3d_teacher_*/
# best_i3d.pth
# training_curves.png
# training_history.json
# results.json
# config.json
```

## 🎯 Benefits

1. **Consistent Visualization**: Teacher and baseline use identical plot styles
2. **Progress Monitoring**: Real-time tracking without TensorBoard
3. **Publication Ready**: High-quality PDF exports for papers
4. **Easy Comparison**: Side-by-side comparison with baseline plots
5. **Reproducibility**: Complete training history saved
6. **No Manual Work**: Automatic plot generation every 5 epochs

## ✅ Verification

To verify the update works correctly:

1. Run teacher training for a few epochs
2. Check that `figures/teacher/` directory is created
3. Verify plots are generated every 5 epochs
4. Confirm plot shows teacher-specific elements:
   - Title: "I3D Teacher Model"
   - Target line at 30%
   - All 4 subplots present

## 🚀 Next Steps

Now that both training scripts have matching visualization:

1. ✅ **Baseline trained** with visualizations
2. ✅ **Teacher script updated** with visualizations  
3. 🔄 **Run teacher training** to generate teacher plots
4. 🎯 **Compare plots** side-by-side (baseline vs teacher)
5. 📊 **Knowledge distillation** using both trained models

The teacher training script is now feature-complete and matches the baseline in terms of monitoring and visualization capabilities!

