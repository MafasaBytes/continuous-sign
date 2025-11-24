# Feature Extraction from Pre-trained Teacher Model

## Goal
Extract 1024D CNN features from the pre-trained GoogLeNet model to replace MediaPipe landmarks as input to your BiLSTM model.

## Expected Improvement
100% WER → 40-50% WER (based on using proven visual features)

---

## Challenge: Caffe Framework

The pre-trained model is in **Caffe format**, which is deprecated and challenging to install on modern systems.

### Options for Feature Extraction:

#### Option 1: Caffe Installation (Traditional, Complex)
**Pros**:
- Direct use of original model
- Guaranteed compatibility

**Cons**:
- Caffe is deprecated (last update 2018)
- Difficult to compile on modern systems
- Requires custom Reverse layer
- C++ compilation issues on Windows

**Estimated effort**: 2-3 days of troubleshooting

---

#### Option 2: Caffe Model Conversion (Recommended)
**Pros**:
- Use modern frameworks (PyTorch/TensorFlow)
- Easier deployment
- Better tooling and debugging

**Cons**:
- Conversion may have slight numerical differences
- Need to validate outputs match

**Tools**:
- `caffemodel2pytorch`: Converts Caffe → PyTorch
- `MMdnn`: Multi-framework conversion tool
- `caffe2tensorflow`: Caffe → TensorFlow

**Estimated effort**: 1 day

---

#### Option 3: Re-implement GoogLeNet in PyTorch (Most Reliable)
**Pros**:
- Full control over architecture
- Load pre-trained weights manually
- No dependency on Caffe

**Cons**:
- Need to manually map layer names
- Requires careful weight loading

**Estimated effort**: 1-2 days

---

#### Option 4: Use PyTorch Pre-trained GoogLeNet + Fine-tune
**Pros**:
- Fastest to implement
- PyTorch has GoogLeNet in torchvision
- Can initialize from ImageNet weights

**Cons**:
- Not the exact RWTH pre-trained model
- May need fine-tuning on PHOENIX dataset
- Different weight initialization

**Estimated effort**: 1 hour (but may not match RWTH performance)

---

## Recommended Approach: Hybrid Strategy

### Phase 1: Quick Start (This Week)
Use **Option 4** (PyTorch GoogLeNet) as a baseline:
- Extract features using torchvision.models.googlenet
- Train BiLSTM on these features
- Validate pipeline works

### Phase 2: Exact Replication (Next Week)
Use **Option 2 or 3** to load exact RWTH weights:
- Convert Caffe model to PyTorch
- Extract features using exact pre-trained weights
- Compare with Phase 1 results

---

## Implementation Plan

### Step 1: Video Frame Extraction
Extract raw video frames from RWTH-PHOENIX dataset.

**Input**: Video files in `data/raw_data/phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px/`
**Output**: Preprocessed frames (224×224, center-cropped, mean-subtracted)

### Step 2: Feature Extraction
Run GoogLeNet (up to pool5 layer) on each frame.

**Input**: 224×224×3 RGB frames
**Output**: 1024-D feature vectors per frame

### Step 3: Feature Caching
Save extracted features to disk for efficient training.

**Format**: HDF5 or NPY
**Structure**:
```
features_train.h5
├── video_001: [T1, 1024]
├── video_002: [T2, 1024]
└── ...

features_dev.h5
features_test.h5
```

### Step 4: BiLSTM Training
Train your BiLSTM + CTC model on extracted features.

**Input**: 1024-D features (instead of 6516-D MediaPipe)
**Architecture**: Reduce first LSTM layer input from 6516 → 1024

---

## Detailed Implementation

### 1. Check Dataset Structure

First, let's understand the video file organization.

