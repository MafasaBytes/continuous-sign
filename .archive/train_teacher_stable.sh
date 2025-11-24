#!/bin/bash
# Emergency stable configuration for teacher training
# Use this if standard training shows infinite gradients

echo "=========================================="
echo "Teacher Training - ULTRA STABLE CONFIG"
echo "=========================================="
echo "Changes from standard:"
echo "  - Very low LR: 1e-5"
echo "  - Aggressive grad clip: 0.5"
echo "  - Higher dropout: 0.4"
echo "  - Smaller batch: 1 (if needed)"
echo "=========================================="
echo ""

python src/training/train_teacher.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/teacher_stable \
    --batch_size 2 \
    --epochs 50 \
    --learning_rate 1e-5 \
    --dropout 0.4 \
    --weight_decay 1e-2 \
    --num_workers 0

echo ""
echo "Training complete!"
echo "Check: checkpoints/teacher_stable/"
echo "Plots: figures/teacher/"

