#!/bin/bash
# Fixed training configuration based on overfitting test success

python src/training/train.py \
  --data_dir data/teacher_features/mediapipe_full \
  --output_dir checkpoints/student/mobilenet_v3_fixed \
  --epochs 500 \
  --batch_size 4 \
  --learning_rate 0.0005 \
  --dropout 0.1 \
  --weight_decay 0.0001 \
  --max_grad_norm 1.0 \
  --accumulation_steps 1 \
  --num_workers 0 \
  --early_stopping_patience 30 \
  --device cuda

