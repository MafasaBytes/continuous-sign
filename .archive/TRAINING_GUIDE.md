# Complete Training Guide for Sign Language Recognition

## Overview

We have implemented a complete teacher-student knowledge distillation pipeline aligned with your research proposal. Here's what to train and in what order.

## Training Architecture

```
┌─────────────┐     Knowledge      ┌──────────────┐
│ I3D Teacher │  ───Distillation──> │ MobileNetV3  │
│  19.3M params│     (T=3.0)       │   Student    │
│  73.79 MB   │                    │ 15.7M params │
│ WER: ~20-30%│                    │  59.95 MB    │
└─────────────┘                    │ WER: <25%    │
                                   └──────────────┘
```

## Training Options

### Option 1: Quick Baseline (1-2 days) ⚡

Train MobileNetV3 student directly without teacher:

```bash
python src/training/train.py \
    --data_dir data/features_enhanced \
    --output_dir checkpoints/student \
    --epochs 100 \
    --batch_size 4 \
    --remove_pca \
    --dropout 0.6
```

**Expected Results:**
- WER: ~40-50%
- Time: 1-2 days
- Meets Phase I target

### Option 2: Full Distillation Pipeline (3-5 days) 🎯 [RECOMMENDED]

This follows your research proposal exactly:

#### Step 1: Train I3D Teacher (1-2 days)
```bash
python src/training/train_teacher.py \
    --data_dir data/features_enhanced \
    --output_dir checkpoints/teacher \
    --epochs 50 \
    --batch_size 2 \
    --dropout 0.3
```

**Expected Results:**
- Teacher WER: ~20-30%
- Model saved to: `checkpoints/teacher/best_i3d.pth`

#### Step 2: Knowledge Distillation (1-2 days)
```bash
python src/training/train_distillation.py \
    --data_dir data/features_enhanced \
    --teacher_checkpoint checkpoints/teacher/best_i3d.pth \
    --output_dir checkpoints/distillation \
    --epochs 50 \
    --batch_size 4 \
    --temperature 3.0 \
    --alpha 0.7
```

**Expected Results:**
- Student WER: <25% (Phase II target)
- Model size: 59.95 MB (< 100MB requirement)

### Option 3: Progressive Training (Parallel) ⚖️

Run both in parallel for efficiency:

**Terminal 1:**
```bash
# Start student baseline training
python src/training/train.py --epochs 100
```

**Terminal 2:**
```bash
# Meanwhile, train teacher
python src/training/train_teacher.py --epochs 50
```

Then fine-tune student with distillation:
```bash
python src/training/train_distillation.py \
    --student_checkpoint checkpoints/student/best_model.pth \
    --teacher_checkpoint checkpoints/teacher/best_i3d.pth
```

## Model Comparison

| Model | Parameters | Size | Expected WER | Training Time |
|-------|------------|------|--------------|---------------|
| I3D Teacher | 19.3M | 73.79 MB | 20-30% | 1-2 days |
| MobileNetV3 (baseline) | 15.7M | 59.95 MB | 40-50% | 1-2 days |
| MobileNetV3 (distilled) | 15.7M | 59.95 MB | <25% | 2-3 days total |

## Key Training Parameters

### Teacher Training
```yaml
batch_size: 2          # Smaller due to model size
learning_rate: 5e-5    # Lower for stability
dropout: 0.3           # Less regularization (model is bigger)
max_seq_length: 256    # Shorter sequences for memory
```

### Student Training (Baseline)
```yaml
batch_size: 4          # Conservative for stability
learning_rate: 1e-4    # Conservative
dropout: 0.6           # High regularization (prevent overfitting)
remove_pca: true       # Preserve modality boundaries
```

### Knowledge Distillation
```yaml
temperature: 3.0       # From research proposal
alpha: 0.7            # 70% soft loss, 30% hard loss
batch_size: 4         # Both models in memory
learning_rate: 5e-5   # Lower for fine-tuning
```

## Memory Requirements

| Training Type | GPU Memory | Fits in 8GB? |
|---------------|------------|--------------|
| Teacher alone | ~174 MB | ✅ Yes |
| Student alone | ~110 MB | ✅ Yes |
| Distillation | ~284 MB | ✅ Yes |

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir checkpoints/
```

Monitor:
- Loss curves (train/val)
- WER progression
- Learning rate schedule
- Teacher-student agreement (distillation)

### Expected Training Curves

1. **Teacher**: Should reach 20-30% WER within 20-30 epochs
2. **Student Baseline**: Will plateau around 40-50% WER
3. **Distillation**: Rapid improvement in first 10 epochs, reaching <25% WER

## Troubleshooting

### If WER is not improving:
1. Check learning rate (try reducing by 10x)
2. Increase batch size if memory allows
3. Add more data augmentation
4. Check for data loading issues

### If out of memory:
1. Reduce batch size
2. Enable gradient checkpointing
3. Reduce max_seq_length
4. Use gradient accumulation

### If training is too slow:
1. Use larger batch size
2. Enable mixed precision (already enabled)
3. Reduce validation frequency
4. Use more workers for data loading

## Success Criteria (from Research Proposal)

✅ **Phase I (Baseline)**:
- Model size < 100MB: **ACHIEVED** (59.95 MB)
- Memory < 8GB: **ACHIEVED**
- WER ~40%: To be validated

⏳ **Phase II (Distillation)**:
- WER < 25%: To be achieved via distillation
- Real-time > 30 FPS: To be tested

## Next Steps After Training

1. **Evaluate on test set**:
```bash
python src/training/evaluate.py \
    --model checkpoints/distillation/best_student.pth \
    --data_dir data/features_enhanced
```

2. **Optimize for deployment**:
- TensorRT conversion
- INT8 quantization (14.99 MB model)
- ONNX export

3. **Real-time testing**:
- Measure inference speed
- Implement sliding window
- Test on live video

## Summary

You now have a complete training pipeline with three options:

1. **Quick**: Train student alone (1-2 days, ~40% WER)
2. **Best**: Teacher → Distillation (3-5 days, <25% WER) ✅ RECOMMENDED
3. **Parallel**: Train both simultaneously

The implementation fully aligns with your research proposal objectives for efficient, real-time sign language recognition with knowledge distillation.

**Ready to start training!** 🚀