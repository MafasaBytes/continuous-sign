# CNN Feature Extraction Guide

## Quick Start

### Step 1: Install Dependencies

```bash
pip install torch torchvision h5py tqdm pillow
```

### Step 2: Extract Features from Training Set

```bash
# Extract features for train split (this will take some time)
python experiments/extract_cnn_features.py --split train

# Expected output: data/features_cnn/train_features.h5
```

**Estimated time**:
- ~3000 videos in train set
- ~100 frames per video on average
- GPU: ~2-3 hours
- CPU: ~8-12 hours

### Step 3: Extract Features for Dev and Test

```bash
python experiments/extract_cnn_features.py --split dev
python experiments/extract_cnn_features.py --split test
```

### Step 4: Validate Extracted Features

```bash
python experiments/extract_cnn_features.py --split train --validate_only
```

Expected output:
```
Total videos: ~3000
Feature shape: [T, 1024]  # T varies per video
Feature range: typically [-5, 5] after normalization
```

---

## Understanding the Output

### HDF5 File Structure

```
train_features.h5
├── video_id_001
│   ├── features [T1, 1024]  # CNN features
│   └── attributes
│       ├── annotation: "SIGN1 SIGN2 SIGN3"
│       ├── folder: "path/to/frames/*.png"
│       └── num_frames: T1
├── video_id_002
│   ├── features [T2, 1024]
│   └── ...
└── ...
```

### Feature Properties

- **Dimension**: 1024 per frame (from GoogLeNet avgpool layer)
- **Type**: float32
- **Compression**: gzip (saves ~50% space)
- **Temporal resolution**: 1 feature per frame (25 FPS)

---

## Using Features for Training

### Option 1: Direct Usage (Recommended)

Use the provided `CNNFeatureDataset`:

```python
from experiments.cnn_feature_dataset import CNNFeatureDataset, collate_fn, build_vocabulary
from torch.utils.data import DataLoader

# Build vocabulary
vocab, idx2sign = build_vocabulary([
    'data/features_cnn/train_features.h5',
    'data/features_cnn/dev_features.h5'
])

# Create datasets
train_dataset = CNNFeatureDataset(
    h5_file='data/features_cnn/train_features.h5',
    vocabulary=vocab
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4
)

# Training loop
for batch in train_loader:
    features = batch['features']  # [B, T, 1024]
    labels = batch['labels']  # [B, L]
    input_lengths = batch['input_lengths']  # [B]
    target_lengths = batch['target_lengths']  # [B]

    # Feed to your BiLSTM model
    outputs = model(features, input_lengths)
    loss = ctc_loss(outputs, labels, input_lengths, target_lengths)
    # ...
```

### Option 2: Modify Your Existing Pipeline

If you have an existing training script, modify it to:

1. **Change input dimension**: 6516 → 1024
2. **Load features from HDF5** instead of computing MediaPipe
3. **Keep everything else the same** (BiLSTM, CTC loss, etc.)

---

## Model Architecture Changes

### Before (MediaPipe):
```python
# Input: [B, T, 6516] MediaPipe landmarks

class SignLanguageModel(nn.Module):
    def __init__(self):
        self.lstm1 = nn.LSTM(6516, 256, bidirectional=True)  # First layer
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True)
        self.lstm3 = nn.LSTM(512, 256, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)
```

### After (CNN Features):
```python
# Input: [B, T, 1024] CNN features

class SignLanguageModel(nn.Module):
    def __init__(self):
        self.lstm1 = nn.LSTM(1024, 256, bidirectional=True)  # Changed: 6516 → 1024
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True)  # Rest unchanged
        self.lstm3 = nn.LSTM(512, 256, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)
```

**Only change**: First LSTM layer input dimension 6516 → 1024

---

## Expected Results

### Baseline Comparison

| Model | Input Features | Expected WER |
|-------|---------------|--------------|
| Your current model | MediaPipe (6516D) | 100% (failing) |
| With CNN features | GoogLeNet (1024D) | **40-50%** |
| Pre-trained RWTH | GoogLeNet + HMM/LM | 26.8% |

### Why CNN Features Should Work Better:

1. **Proven features**: GoogLeNet is trained on visual tasks
2. **Reduced dimensionality**: 1024D vs 6516D (less overfitting)
3. **Learned representations**: CNN adapts to sign-specific patterns
4. **Lower noise**: CNN filters out irrelevant information

---

## Troubleshooting

### Issue: "Out of memory during feature extraction"

**Solution**: Process in smaller batches
```python
# Modify extract_cnn_features.py
# Add batch processing in extract_video_features()
```

### Issue: "Features are all zeros"

**Check**:
1. Are frame paths correct?
2. Are images loading properly?
3. Is GPU available?

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: "HDF5 file is too large"

**Current size estimation**:
- Train: ~3000 videos × 100 frames × 1024 features × 4 bytes ≈ 1.2 GB
- Total (all splits): ~2-3 GB

**If too large**: Increase HDF5 compression level in script

### Issue: "Different number of frames than expected"

**This is normal**:
- Videos have variable lengths
- Some frames may be missing
- Dataset may have preprocessing artifacts

**Validation**:
```python
python experiments/extract_cnn_features.py --split train --validate_only
```

---

## Next Steps After Feature Extraction

### 1. Train BiLSTM on CNN Features (This Week)

```bash
# Modify your existing training script to use CNN features
# Expected: 100% WER → 40-50% WER
```

### 2. Compare with MediaPipe Baseline

Create ablation study:
- Model A: MediaPipe features (6516D)
- Model B: CNN features (1024D)
- Compare WER on same test set

### 3. (Optional) Use Exact RWTH Weights (Next Week)

Convert Caffe model to PyTorch:
- Install conversion tools
- Load exact pre-trained weights
- Re-extract features
- Compare: ImageNet GoogLeNet vs RWTH GoogLeNet

### 4. Add Language Model (Week 4)

After achieving <50% WER with CNN features:
- Integrate 4-gram KenLM
- Two-pass decoding (CTC + LM)
- Tune hyperparameters
- Expected: 40-50% → 25-30% WER

---

## Advanced: Converting RWTH Caffe Model

If you want to use the exact pre-trained RWTH weights:

### Method 1: Use caffemodel2pytorch

```bash
pip install caffemodel2pytorch

# Convert model
python -c "
from caffemodel2pytorch import Net
net = Net('data/raw_data/.../models/CNN-LSTM/net.prototxt',
          'data/raw_data/.../models/CNN-LSTM/googlenet_iter_76500.caffemodel')
# Extract weights layer by layer
"
```

### Method 2: Manual Weight Loading

1. Read Caffe .caffemodel (protobuf format)
2. Map layer names: Caffe → PyTorch
3. Load weights into PyTorch GoogLeNet
4. Validate outputs match

**Warning**: This is time-consuming and error-prone. Only attempt if:
- You have experience with model conversion
- ImageNet GoogLeNet features give poor results
- You need exact replication

---

## Performance Optimization

### GPU Utilization

```python
# Enable mixed precision for faster extraction
from torch.cuda.amp import autocast

with autocast():
    features = self.feature_extractor(img_tensor)
```

### Parallel Processing

```python
# Process multiple videos in parallel
from multiprocessing import Pool

def extract_single_video(args):
    video_id, frame_pattern = args
    # Extract features
    return features

with Pool(4) as pool:
    results = pool.map(extract_single_video, video_args)
```

### Batch Processing

Instead of 1 frame at a time, process frames in batches:

```python
# Load 32 frames
frame_batch = torch.stack([transform(img) for img in frames[:32]])
# Extract in batch
features_batch = feature_extractor(frame_batch)
```

---

## Summary

**What we're doing**:
1. Extract 1024D CNN features using GoogLeNet
2. Cache features to HDF5 for fast loading
3. Train BiLSTM + CTC on extracted features
4. Bypass MediaPipe → Use proven visual features

**Expected outcome**:
- 100% WER → 40-50% WER (major improvement!)
- Validates that visual features are the bottleneck
- Provides path to further improvements (LM, distillation)

**Time investment**:
- Feature extraction: 3-5 hours (one-time)
- Training modification: 1-2 hours
- Total: ~1 day

**Next document**: See `experiments/train_with_cnn_features.md` for training guide
