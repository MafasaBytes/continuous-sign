#!/bin/bash

################################################################################
# Train Teacher Model with Pre-trained Weights
# 
# This script provides easy-to-use commands for training the teacher model
# with different pre-trained weight configurations.
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default arguments
DATA_DIR="data/teacher_features/mediapipe_full"
OUTPUT_DIR="checkpoints/teacher"
BATCH_SIZE=16
NUM_WORKERS=0
SEED=42

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}Teacher Model Training Script${NC}"
echo -e "${BLUE}With Pre-trained Weights Support${NC}"
echo -e "${BLUE}=================================${NC}\n"

# Function to display usage
usage() {
    echo "Usage: $0 [STRATEGY] [OPTIONS]"
    echo ""
    echo "Strategies:"
    echo "  baseline              - Train from scratch (no pre-trained weights)"
    echo "  feature-extraction    - Freeze backbone, train classifier only"
    echo "  fine-tune             - Fine-tune top layers"
    echo "  progressive           - Progressive unfreezing (recommended)"
    echo "  full-finetune         - Fine-tune all layers"
    echo ""
    echo "Options:"
    echo "  --pretrained PATH     - Path to pre-trained weights"
    echo "  --epochs N            - Number of epochs (default: varies by strategy)"
    echo "  --lr RATE             - Learning rate (default: varies by strategy)"
    echo "  --batch-size N        - Batch size (default: 16)"
    echo "  --gpu ID              - GPU device ID (default: 0)"
    echo ""
    echo "Examples:"
    echo "  $0 progressive                    # Use progressive unfreezing"
    echo "  $0 fine-tune --epochs 40          # Fine-tune for 40 epochs"
    echo "  $0 feature-extraction --lr 1e-3   # Feature extraction with custom LR"
    echo ""
    exit 1
}

# Check if strategy is provided
if [ $# -eq 0 ]; then
    usage
fi

STRATEGY=$1
shift

# Parse additional options
CUSTOM_PRETRAINED=""
CUSTOM_EPOCHS=""
CUSTOM_LR=""
CUSTOM_BATCH_SIZE=""
GPU_ID="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --pretrained)
            CUSTOM_PRETRAINED="$2"
            shift 2
            ;;
        --epochs)
            CUSTOM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            CUSTOM_LR="$2"
            shift 2
            ;;
        --batch-size)
            CUSTOM_BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Strategy-specific configurations
case $STRATEGY in
    baseline)
        echo -e "${YELLOW}Strategy: Baseline (No pre-trained weights)${NC}"
        PRETRAINED=""
        EPOCHS=${CUSTOM_EPOCHS:-50}
        LR=${CUSTOM_LR:-5e-4}
        EXTRA_ARGS=""
        ;;
        
    feature-extraction)
        echo -e "${YELLOW}Strategy: Feature Extraction${NC}"
        echo -e "  - Freeze backbone up to mixed_5c"
        echo -e "  - Train only classifier & LSTM"
        PRETRAINED=${CUSTOM_PRETRAINED:-"checkpoints/teacher/best_i3d.pth"}
        EPOCHS=${CUSTOM_EPOCHS:-20}
        LR=${CUSTOM_LR:-1e-3}
        EXTRA_ARGS="--freeze_backbone --freeze_until_layer mixed_5c"
        ;;
        
    fine-tune)
        echo -e "${YELLOW}Strategy: Fine-tuning Top Layers${NC}"
        echo -e "  - Freeze backbone up to mixed_4f"
        echo -e "  - Fine-tune top layers"
        PRETRAINED=${CUSTOM_PRETRAINED:-"checkpoints/teacher/best_i3d.pth"}
        EPOCHS=${CUSTOM_EPOCHS:-30}
        LR=${CUSTOM_LR:-5e-4}
        EXTRA_ARGS="--freeze_backbone --freeze_until_layer mixed_4f"
        ;;
        
    progressive)
        echo -e "${YELLOW}Strategy: Progressive Unfreezing (Recommended)${NC}"
        echo -e "  - Start with frozen backbone"
        echo -e "  - Gradually unfreeze in 4 stages"
        PRETRAINED=${CUSTOM_PRETRAINED:-"checkpoints/teacher/best_i3d.pth"}
        EPOCHS=${CUSTOM_EPOCHS:-50}
        LR=${CUSTOM_LR:-1e-4}
        EXTRA_ARGS="--freeze_backbone --freeze_until_layer mixed_4f --progressive_unfreeze --unfreeze_stages 4"
        ;;
        
    full-finetune)
        echo -e "${YELLOW}Strategy: Full Fine-tuning${NC}"
        echo -e "  - Fine-tune all layers"
        echo -e "  - ⚠️  Requires very low learning rate"
        PRETRAINED=${CUSTOM_PRETRAINED:-"checkpoints/teacher/best_i3d.pth"}
        EPOCHS=${CUSTOM_EPOCHS:-50}
        LR=${CUSTOM_LR:-1e-5}
        EXTRA_ARGS=""
        ;;
        
    *)
        echo -e "${RED}Unknown strategy: $STRATEGY${NC}"
        usage
        ;;
esac

# Check if pre-trained weights exist (if specified)
if [ ! -z "$PRETRAINED" ] && [ "$PRETRAINED" != "i3d_kinetics400" ]; then
    if [ ! -f "$PRETRAINED" ]; then
        echo -e "${RED}Error: Pre-trained weights not found at: $PRETRAINED${NC}"
        echo -e "${YELLOW}Available checkpoints:${NC}"
        find checkpoints/teacher -name "*.pth" 2>/dev/null || echo "  None found"
        echo ""
        echo -e "${YELLOW}Options:${NC}"
        echo "  1. Train baseline model first: $0 baseline"
        echo "  2. Specify a different path: $0 $STRATEGY --pretrained /path/to/weights.pth"
        echo "  3. Use Kinetics pre-trained: $0 $STRATEGY --pretrained i3d_kinetics400"
        exit 1
    fi
fi

# Override batch size if specified
if [ ! -z "$CUSTOM_BATCH_SIZE" ]; then
    BATCH_SIZE=$CUSTOM_BATCH_SIZE
fi

# Display configuration
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Data directory:  $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Pre-trained weights: ${PRETRAINED:-"None (training from scratch)"}"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Batch size: $BATCH_SIZE"
echo "  GPU: $GPU_ID"
echo ""

# Confirm before starting
echo -e "${BLUE}Starting training in 3 seconds... (Ctrl+C to cancel)${NC}"
sleep 3

# Build the command
CMD="python src/training/train_teacher.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LR \
    --num_workers $NUM_WORKERS \
    --seed $SEED"

if [ ! -z "$PRETRAINED" ]; then
    CMD="$CMD --pretrained_weights $PRETRAINED"
fi

if [ ! -z "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

echo -e "${GREEN}Executing:${NC}"
echo "$CMD"
echo ""

# Run the training
eval $CMD

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================${NC}"
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}=================================${NC}"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Check training curves: figures/teacher/training_curves.png"
    echo "  2. Review best model: checkpoints/teacher/best_i3d.pth"
    echo "  3. Use for distillation: python src/training/train_distillation.py"
else
    echo ""
    echo -e "${RED}=================================${NC}"
    echo -e "${RED}Training failed!${NC}"
    echo -e "${RED}=================================${NC}"
    echo ""
    echo "Check the logs for errors."
    exit 1
fi

