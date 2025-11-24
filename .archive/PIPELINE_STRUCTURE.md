# Sign Language Recognition Pipeline Structure

## Overview
This document describes the clean, organized pipeline for sign language recognition aligned with the research proposal objectives.

## Directory Structure

```
sign-language-recognition/
├── src/                      # Core implementation (clean pipeline)
│   ├── data/                # Data loading and preprocessing
│   │   ├── mediapipe_dataset.py    # Dataset class for MediaPipe features
│   │   └── loaders.py              # Data loading utilities
│   │
│   ├── models/              # Model architectures
│   │   ├── mobilenet_v3.py        # MobileNetV3 backbone (TO BE IMPLEMENTED)
│   │   └── bilstm_ctc.py          # BiLSTM + CTC head (TO BE IMPLEMENTED)
│   │
│   ├── training/            # Training scripts
│   │   ├── train.py               # Main training script (TO BE IMPLEMENTED)
│   │   └── distillation.py        # Knowledge distillation (TO BE IMPLEMENTED)
│   │
│   └── utils/               # Utility functions
│       ├── vocabulary.py          # Vocabulary management
│       ├── metrics.py             # WER/SER metrics
│       ├── ctc.py                # CTC utilities
│       └── forced_alignment.py    # Alignment utilities
│
├── data/                    # Raw and processed data
│   ├── raw_data/           # Original RWTH-PHOENIX dataset
│   ├── features_enhanced/  # Extracted MediaPipe features (6,516 dims)
│   └── vocabulary/         # Vocabulary files
│
├── configs/                # Configuration files
│   └── mobilenet_v3_config.yaml  # Model configuration (TO BE CREATED)
│
├── checkpoints/           # Model checkpoints
│   ├── teacher/          # Pre-trained teacher models (I3D/SlowFast)
│   └── student/          # MobileNetV3 student models
│
├── experiments/          # Experimental code and analysis
│
├── .archive/            # Archived old implementations
│   └── phase1_efficient_hybrid/  # Old EfficientHybridModel implementation
│
└── research-proposal.md  # Research objectives and methodology

```

## Pipeline Flow

### 1. Data Processing Pipeline
```
Raw Videos (RWTH-PHOENIX-Weather 2014)
    ↓
MediaPipe Holistic Extraction
    ↓
6,516 Dimensional Features
- Pose: 33 keypoints × 3 = 99 dims
- Hands: 21 × 2 × 3 = 126 dims
- Face: 468 × 3 = 1,404 dims
- Velocity: 1,629 dims
- Acceleration: 1,629 dims
- Spatial features: 1,629 dims
    ↓
NO PCA REDUCTION (preserve modality boundaries)
    ↓
Modality-Specific Encoding
```

### 2. Model Architecture Pipeline (Aligned with Proposal)
```
Input Features (6,516 dims)
    ↓
Modality-Specific Encoders
- Pose Encoder: 99 → 64 dims
- Hand Encoder: 126 → 128 dims
- Face Encoder: 1,404 → 64 dims
- Temporal Encoder: 4,887 → 128 dims
    ↓
MobileNetV3-Small Backbone (3.2M params)
    ↓
Cross-Modal Attention (4 heads, 256 dims)
    ↓
BiLSTM Layers (2 layers, 128 units each)
    ↓
CTC Output Layer
    ↓
Predictions
```

### 3. Training Pipeline (Three Phases)

#### Phase I: Baseline (Current)
- Train MobileNetV3 + BiLSTM baseline
- Target: 40% WER
- Memory: < 8GB VRAM

#### Phase II: Knowledge Distillation
- Teacher: Pre-trained I3D/SlowFast
- Student: MobileNetV3 architecture
- Loss: 0.7 × L_soft + 0.3 × L_hard
- Temperature: 3.0
- Target: < 25% WER

#### Phase III: Deployment
- TensorRT optimization
- Real-time inference (> 30 FPS)
- Model size: < 100MB
- Web deployment via TensorFlow.js

## Key Improvements from Old Pipeline

### Removed/Archived:
- ❌ EfficientHybridModel (36M params - too large)
- ❌ PCA reduction (destroyed modality information)
- ❌ Complex hierarchical training
- ❌ Overfitted models

### Added/Aligned:
- ✅ MobileNetV3 backbone (as per proposal)
- ✅ Modality-specific processing
- ✅ Knowledge distillation framework
- ✅ Memory-efficient design (< 100MB)
- ✅ Clear three-phase training

## Implementation Status

### Completed:
- [x] MediaPipe feature extraction
- [x] Dataset preparation
- [x] Basic utilities

### To Implement (Priority Order):
1. [ ] MobileNetV3 backbone architecture
2. [ ] Modality-specific encoders
3. [ ] BiLSTM + CTC integration
4. [ ] Training script with mixed-precision
5. [ ] Knowledge distillation from I3D
6. [ ] TensorRT optimization
7. [ ] Web deployment

## Usage

### Training
```bash
python src/training/train.py \
    --config configs/mobilenet_v3_config.yaml \
    --data data/features_enhanced \
    --checkpoint checkpoints/student
```

### Evaluation
```bash
python src/training/evaluate.py \
    --model checkpoints/student/best_model.pth \
    --data data/features_enhanced/test
```

### Inference
```bash
python src/inference/predict.py \
    --model checkpoints/student/best_model.pth \
    --input video.mp4
```

## Performance Targets (from Research Proposal)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| WER | < 25% | > 50% | ❌ Need improvement |
| Model Size | < 100MB | 144MB | ❌ Need reduction |
| FPS | > 30 | TBD | ⏳ Not tested |
| Memory | < 8GB | OK | ✅ |
| Accuracy | > 85% | TBD | ⏳ Not tested |

## Next Steps

1. **Immediate**: Implement MobileNetV3 backbone
2. **Week 1**: Complete Phase I baseline training
3. **Week 2**: Implement knowledge distillation
4. **Week 3**: Optimize for deployment
5. **Week 4**: User study and evaluation

---

This structure ensures a clean, maintainable pipeline that directly aligns with the research proposal objectives.