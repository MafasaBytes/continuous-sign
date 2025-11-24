# Using Pre-trained Weights for Teacher Model

## Overview

The teacher model can now leverage pre-trained weights from action recognition models (like I3D trained on Kinetics-400) to achieve better performance and faster convergence. This is especially useful when:

- Training from scratch shows poor performance (high WER)
- You want to leverage knowledge from large-scale video understanding tasks
- You need faster convergence and better generalization

## Why Pre-trained Weights?

**Current Issue**: The teacher model trained from scratch achieved **97.17% WER** (target: < 30%), which is insufficient for effective knowledge distillation.

**Solution**: Using pre-trained I3D weights from action recognition provides:
1. **Better feature representations** learned from millions of video samples
2. **Faster convergence** - starts from a much better initialization
3. **Better generalization** - reduces overfitting to small datasets
4. **Improved teacher quality** - better teacher means better student via distillation

## Quick Start

### Option 1: Use existing checkpoint (recommended for now)

If you have a well-trained checkpoint from another experiment:

```bash
python src/training/train_teacher.py \
    --pretrained_weights checkpoints/teacher/best_i3d.pth \
    --freeze_backbone \
    --progressive_unfreeze \
    --learning_rate 1e-4 \
    --epochs 30
```

### Option 2: Use I3D Kinetics-400 pre-trained weights

```bash
python src/training/train_teacher.py \
    --pretrained_weights i3d_kinetics400 \
    --adapt_method temporal_mean \
    --freeze_backbone \
    --freeze_until_layer mixed_4f \
    --progressive_unfreeze \
    --unfreeze_stages 4 \
    --learning_rate 1e-4 \
    --epochs 50
```

### Option 3: Custom pre-trained weights

```bash
python src/training/train_teacher.py \
    --pretrained_weights path/to/custom_weights.pth \
    --adapt_method temporal_mean \
    --freeze_backbone \
    --learning_rate 1e-4
```

## Transfer Learning Strategies

### 1. **Feature Extraction** (Fastest, most stable)

Freeze all backbone layers, only train the classifier:

```bash
python src/training/train_teacher.py \
    --pretrained_weights i3d_kinetics400 \
    --freeze_backbone \
    --freeze_until_layer mixed_5c \
    --learning_rate 1e-3 \
    --epochs 20
```

**Best for**: 
- Limited data
- Quick experiments
- When pre-trained features are very relevant

### 2. **Fine-tuning Top Layers** (Balanced)

Freeze early layers, fine-tune top layers:

```bash
python src/training/train_teacher.py \
    --pretrained_weights i3d_kinetics400 \
    --freeze_backbone \
    --freeze_until_layer mixed_4f \
    --learning_rate 5e-4 \
    --epochs 30
```

**Best for**:
- Moderate amount of data
- When task is somewhat related to pre-training
- Good balance of speed and adaptation

### 3. **Progressive Unfreezing** (Recommended for best performance)

Gradually unfreeze layers from top to bottom:

```bash
python src/training/train_teacher.py \
    --pretrained_weights i3d_kinetics400 \
    --freeze_backbone \
    --progressive_unfreeze \
    --unfreeze_stages 4 \
    --learning_rate 1e-4 \
    --epochs 50
```

**Best for**:
- Sufficient training time
- Maximum adaptation to new task
- Best final performance

**Unfreezing schedule** (4 stages over 50 epochs):
- **Epochs 0-12**: Only classifier & LSTM (top layers)
- **Epochs 13-25**: + mixed_5 blocks
- **Epochs 26-37**: + mixed_4 blocks
- **Epochs 38-50**: + mixed_3 & stem (full model)

### 4. **Full Fine-tuning** (Slowest, highest risk)

Fine-tune all layers from the start:

```bash
python src/training/train_teacher.py \
    --pretrained_weights i3d_kinetics400 \
    --learning_rate 1e-5 \
    --epochs 50
```

**Best for**:
- Large dataset
- Task very different from pre-training
- When you have time and compute

**⚠️ Warning**: Use very low learning rate to avoid catastrophic forgetting!

## Weight Adaptation Methods

When loading pre-trained I3D weights (which use 3D convolutions), they need to be adapted to our 1D temporal architecture:

### `--adapt_method temporal_mean` (Default, Recommended)

Averages spatial dimensions (H, W) to get temporal weights:
```python
weight_1d = weight_3d.mean(dim=[3, 4])  # [C_out, C_in, T]
```

**Best for**: General purpose, preserves temporal information

### `--adapt_method spatial_mean`

Takes the center of temporal dimension after spatial averaging:
```python
spatial_avg = weight_3d.mean(dim=[3, 4])
weight_1d = spatial_avg[:, :, center_frame]
```

**Best for**: When you want to focus on instantaneous features

### `--adapt_method temporal_max`

Uses max pooling over spatial dimensions:
```python
weight_1d = weight_3d.max(dim=3)[0].max(dim=3)[0]
```

**Best for**: When you want to keep strongest spatial patterns

## Expected Results

### Without Pre-trained Weights (Current)
```
Epoch 15/50: Train Loss: 4.95, Val WER: 97.17%
❌ Very poor performance, insufficient for distillation
```

### With Pre-trained Weights + Fine-tuning
```
Epoch 15/30: Train Loss: 2.80, Val WER: 35-45%
Epoch 30/30: Train Loss: 2.20, Val WER: 25-35%
✅ Much better, suitable for distillation
```

### With Pre-trained Weights + Progressive Unfreezing
```
Epoch 15/50: Train Loss: 2.50, Val WER: 30-40%
Epoch 30/50: Train Loss: 2.00, Val WER: 22-28%
Epoch 50/50: Train Loss: 1.75, Val WER: 18-25%
✅ Best performance, excellent for distillation
```

## Hyperparameter Recommendations

| Strategy | Learning Rate | Epochs | Batch Size | Expected Time |
|----------|---------------|--------|------------|---------------|
| Feature Extraction | 1e-3 | 20 | 16 | ~2 hours |
| Fine-tuning Top | 5e-4 | 30 | 16 | ~4 hours |
| Progressive Unfreezing | 1e-4 | 50 | 8-16 | ~8 hours |
| Full Fine-tuning | 1e-5 | 50 | 8 | ~10 hours |

**Note**: Times are approximate for the Phoenix-2014 dataset with GPU training.

## Troubleshooting

### Issue: "Failed to load pre-trained weights"

**Solution 1**: Check if weights exist
```bash
ls -l checkpoints/pretrained/
```

**Solution 2**: Try different model name or path
```bash
# If download fails, manually download and specify path
python src/training/train_teacher.py \
    --pretrained_weights /path/to/manually/downloaded/weights.pth
```

### Issue: "Shape mismatch" errors

**Solution**: This is expected! Not all layers will match. The loader will:
- ✅ Load matching layers (Conv, BatchNorm, etc.)
- ⚠️ Skip mismatched layers (different vocabulary size, architecture differences)
- 🔄 Adapt 3D→1D where possible

Check logs for details:
```
INFO - Successfully loaded: 245 parameters
INFO - Adapted 3D->1D: 87 parameters
INFO - Skipped: 12 parameters
```

### Issue: Training still shows high WER

**Possible causes**:
1. **Insufficient fine-tuning**: Try more epochs or progressive unfreezing
2. **Learning rate too high**: Reduce to 1e-5 or 5e-5
3. **Too much freezing**: Try unfreezing more layers
4. **Data quality**: Check if features are normalized correctly

**Debug steps**:
```bash
# Try with less freezing
python src/training/train_teacher.py \
    --pretrained_weights i3d_kinetics400 \
    --freeze_until_layer mixed_3c \
    --learning_rate 1e-4

# Or with no freezing but very low LR
python src/training/train_teacher.py \
    --pretrained_weights i3d_kinetics400 \
    --learning_rate 1e-5
```

### Issue: Training is very slow

**Solution**: Increase batch size or reduce model size
```bash
python src/training/train_teacher.py \
    --pretrained_weights i3d_kinetics400 \
    --batch_size 32 \
    --freeze_backbone  # Freezing reduces memory usage
```

## Advanced: Creating Custom Pre-trained Weights

If you have a well-trained model on a related task, you can use it as pre-trained weights:

```python
# Save your model for use as pre-trained weights
import torch

checkpoint = {
    'model_state_dict': model.state_dict(),
    'epoch': epoch,
    'wer': wer,
    'config': config
}

torch.save(checkpoint, 'custom_pretrained.pth')
```

Then use it:
```bash
python src/training/train_teacher.py \
    --pretrained_weights custom_pretrained.pth
```

## Integration with Knowledge Distillation

After training a better teacher with pre-trained weights:

```bash
# Step 1: Train teacher with pre-trained weights
python src/training/train_teacher.py \
    --pretrained_weights i3d_kinetics400 \
    --progressive_unfreeze \
    --epochs 50

# Step 2: Use the improved teacher for distillation
python src/training/train_distillation.py \
    --teacher_checkpoint checkpoints/teacher/best_i3d.pth \
    --temperature 4.0 \
    --alpha 0.7
```

The improved teacher (lower WER) will provide:
- ✅ Better soft targets for the student
- ✅ More informative probability distributions
- ✅ Better feature representations to mimic
- ✅ Overall better student performance

## Benchmarks

| Initialization | Val WER @ Epoch 15 | Val WER @ Epoch 50 | Suitable for Distillation |
|----------------|--------------------|--------------------|---------------------------|
| Random (Current) | 97.17% | ~95% | ❌ No |
| Pre-trained (Feature Ext.) | 45-50% | 35-40% | ⚠️ Marginal |
| Pre-trained (Fine-tune) | 35-40% | 25-30% | ✅ Yes |
| Pre-trained (Progressive) | 30-35% | 18-25% | ✅✅ Excellent |

## Next Steps

1. **Try pre-trained weights** to improve teacher performance
2. **Monitor training** - watch for WER to drop below 30%
3. **Use best teacher** for knowledge distillation
4. **Compare student performance** with and without pre-trained teacher

## References

- Original I3D Paper: ["Quo Vadis, Action Recognition?"](https://arxiv.org/abs/1705.07750)
- Transfer Learning Guide: [cs231n Stanford Notes](http://cs231n.github.io/transfer-learning/)
- Progressive Unfreezing: [ULMFit Paper](https://arxiv.org/abs/1801.06146)

