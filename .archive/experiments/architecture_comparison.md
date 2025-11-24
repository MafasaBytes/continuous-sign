# Architecture Comparison: Pre-trained vs. Current Model

## Executive Summary

This document compares the RWTH-PHOENIX pre-trained CNN-BLSTM model (26.8% WER) with your current MediaPipe + BiLSTM + CTC approach (100% WER).

**Key Finding**: The pre-trained model uses **sub-sign temporal states** (3 states per sign), explaining why it has 3694 classes instead of the expected ~1200 signs.

---

## 1. Class Vocabulary Analysis

### 1.1 Pre-trained Model Classes (3694 total)

**Pattern Discovery:**
```
AACHEN0-0 → class 3
AACHEN0-1 → class 4
AACHEN0-2 → class 5
```

Each sign is decomposed into **3 temporal states** (beginning, middle, end):
- `SIGNNAME0-0`: Beginning state
- `SIGNNAME0-1`: Middle state
- `SIGNNAME0-2`: End state

**Total classes**: 1231 signs × 3 states = 3693 classes + 1 blank = 3694 classes

**Why This Design?**
- **Temporal sub-structure**: Models within-sign dynamics
- **HMM integration**: 3-state left-to-right HMM per sign
- **Better alignment**: Frame-level loss can learn sign progression
- **Inspired by speech recognition**: Tri-phone models in ASR

---

### 1.2 Your Current Model Classes (1295)

Based on typical RWTH-PHOENIX vocabulary:
- ~1200 unique signs
- +1 blank token (CTC)
- Possibly some special tokens

**Hypothesis**: Your model uses **full-sign units** without temporal decomposition.

---

### 1.3 Comparison

| Aspect | Pre-trained Model | Your Model |
|--------|------------------|------------|
| **Vocabulary size** | 1231 signs | ~1200-1294 signs |
| **Temporal states** | 3 per sign (begin/mid/end) | 1 per sign (full) |
| **Total classes** | 3694 | 1295 |
| **Blank token** | Included | CTC blank (class 0) |
| **Granularity** | Sub-sign level | Sign level |

**Implication**: Pre-trained model has finer temporal resolution, which may help with alignment.

---

## 2. Complete Architecture Comparison

### 2.1 Input Processing

| Component | Pre-trained Model | Your Model |
|-----------|------------------|------------|
| **Input type** | Raw RGB video frames | MediaPipe Holistic landmarks |
| **Resolution** | 224×224 pixels | 543 landmarks × 4 coords (x,y,z,visibility) |
| **Preprocessing** | Center crop, mean subtraction | Normalization, flattening |
| **Feature dimension** | 224×224×3 = 150,528D | 543×4 + derived = 6516D |
| **Advantages** | Complete visual information | Lightweight, interpretable |
| **Disadvantages** | Computationally expensive | Loses appearance/texture info |

---

### 2.2 Visual Feature Extraction

#### Pre-trained Model: GoogLeNet (Inception v1)
```
Input: 224×224×3 RGB
  ↓
Conv 7×7, stride 2 → 64 filters
  ↓
MaxPool, LRN
  ↓
Conv Block (1×1 + 3×3) → 192 filters
  ↓
9× Inception Modules
  ├─ inception_3a, 3b
  ├─ inception_4a, 4b, 4c, 4d, 4e
  └─ inception_5a, 5b
  ↓
Global Average Pool 7×7
  ↓
Output: 1024D feature vector per frame
```

**Parameters**: ~6 million (GoogLeNet portion)
**Computational cost**: ~1.5 GFLOPs per frame
**Training**: Pre-trained on ImageNet, fine-tuned on PHOENIX

---

#### Your Model: Direct MediaPipe Features
```
Input: 543 landmarks × 4 coords
  ↓
Normalization (pose centering, hand alignment)
  ↓
Feature engineering (velocities, angles, etc.)
  ↓
Output: 6516D feature vector per frame
```

**Parameters**: 0 (no learned visual features)
**Computational cost**: Negligible
**Training**: MediaPipe is frozen, pre-trained on COCO/other datasets

---

### 2.3 Temporal Modeling

#### Pre-trained Model: 2-Layer Bidirectional LSTM
```
Input: 1024D per frame
  ↓
BiLSTM Layer 1
  ├─ Forward LSTM: 1024 units
  └─ Backward LSTM: 1024 units
  → Concat: 2048D
  ↓
BiLSTM Layer 2
  ├─ Forward LSTM: 1024 units
  └─ Backward LSTM: 1024 units
  → Concat: 2048D
  ↓
Fully Connected: 2048 → 3694 classes
  ↓
Softmax Loss (frame-level)
```

**Parameters**:
- Layer 1: ~33M params (1024 input → 1024×2 bidir)
- Layer 2: ~33M params (2048 input → 1024×2 bidir)
- FC: ~7.5M params (2048 → 3694)
- **Total**: ~73M parameters

**Training**:
- Loss: Frame-level softmax (with auxiliary losses)
- Alignment: Handled by HMM during decoding
- Batch size: 32 sequences

---

#### Your Model: 3-Layer Bidirectional LSTM
```
Input: 6516D per frame
  ↓
BiLSTM Layer 1
  ├─ Forward LSTM: 256 units
  └─ Backward LSTM: 256 units
  → Concat: 512D
  ↓
BiLSTM Layer 2
  ├─ Forward LSTM: 256 units
  └─ Backward LSTM: 256 units
  → Concat: 512D
  ↓
BiLSTM Layer 3
  ├─ Forward LSTM: 256 units
  └─ Backward LSTM: 256 units
  → Concat: 512D
  ↓
Fully Connected: 512 → 1295 classes
  ↓
CTC Loss (sequence-level)
```

**Parameters** (estimated):
- Layer 1: ~27M params (6516 input → 256×2 bidir)
- Layer 2: ~2M params (512 input → 256×2 bidir)
- Layer 3: ~2M params (512 input → 256×2 bidir)
- FC: ~0.66M params (512 → 1295)
- **Total**: ~32M parameters

**Training**:
- Loss: CTC (end-to-end alignment)
- Batch processing: Handle variable-length sequences
- Blank token: Included for CTC alignment

---

### 2.4 Loss Function & Training Objective

| Aspect | Pre-trained Model | Your Model |
|--------|------------------|------------|
| **Main loss** | Frame-level Softmax | CTC Loss |
| **Auxiliary losses** | 2 auxiliary classifiers (weight 0.3 each) | None |
| **Alignment** | Post-hoc (HMM Viterbi) | Integrated (CTC) |
| **Training signal** | Every frame labeled | Sequence-level supervision |
| **Label requirements** | Frame-to-class alignment needed | Only sequence transcription needed |

**Key Difference**:
- **Pre-trained**: Requires frame-level labels (HMM forced alignment)
- **Your model**: Only needs sequence-level transcriptions (CTC handles alignment)

---

### 2.5 Decoding Strategy

#### Pre-trained Model: HMM + Language Model
```
1. CNN-BLSTM Forward Pass
   → Frame posteriors: P(class | frame)

2. HMM Viterbi Decoding
   → Find best state sequence through sign HMMs
   → Each sign = 3-state left-to-right HMM

3. Language Model Rescoring
   → 4-gram SRILM (perplexity 58.56)
   → Rescore top-K hypotheses
   → Combined score: α × acoustic + β × LM

4. Output: Best word sequence
```

**Hyperparameters**:
- Beam width: 10-20
- Acoustic pruning: 2000
- LM pruning: 4000
- LM scale: 9 (strong LM influence)

---

#### Your Model: CTC Beam Search (No LM yet)
```
1. BiLSTM Forward Pass
   → Frame posteriors: P(class | frame)

2. CTC Beam Search
   → Beam width: 10 (typical)
   → Collapse repeated tokens
   → Remove blank tokens

3. Output: Best word sequence
```

**Missing Component**: Language model rescoring (not yet implemented)

---

## 3. Parameter & Complexity Comparison

| Metric | Pre-trained Model | Your Model | Ratio |
|--------|------------------|------------|-------|
| **Total parameters** | ~79M | ~32M | 2.5× |
| **Visual backbone params** | ~6M (GoogLeNet) | 0 (MediaPipe frozen) | ∞ |
| **LSTM params** | ~73M | ~31M | 2.4× |
| **Input dimension** | 150,528D (pixels) | 6516D (landmarks) | 23× |
| **LSTM hidden units** | 1024 per direction | 256 per direction | 4× |
| **LSTM layers** | 2 | 3 | 0.67× |
| **Output classes** | 3694 | 1295 | 2.9× |
| **Inference FLOPs** | ~1.5 GFLOPs (CNN) + LSTM | ~0.01 GFLOPs (LSTM only) | 150× |

**Conclusion**: Your model is **much lighter** due to MediaPipe preprocessing, but may lack representational capacity.

---

## 4. Why Pre-trained Model Achieves 26.8% WER

### 4.1 Advantages:
1. **Learned visual features**: CNN adapts to sign-specific patterns
2. **Deep supervision**: Auxiliary losses prevent vanishing gradients
3. **Fine temporal granularity**: 3-state HMM per sign
4. **Strong language model**: Perplexity 58.56, only 0.5% OOV
5. **Large capacity**: 79M parameters
6. **Proven architecture**: GoogLeNet pre-trained on ImageNet

---

### 4.2 Disadvantages:
1. **Computationally expensive**: ~1.5 GFLOPs per frame (CNN)
2. **Real-time infeasible**: Cannot achieve 30+ FPS on edge devices
3. **Large model size**: ~300MB model file
4. **Caffe dependency**: Outdated framework
5. **Frame-level labels**: Requires HMM forced alignment for training

---

## 5. Why Your Model Achieves 100% WER

### 5.1 Potential Issues (Hypotheses):

#### Hypothesis 1: Feature Quality
- MediaPipe landmarks may be too noisy/sparse
- Missing appearance information (colors, textures)
- 6516D may have redundant/irrelevant features

**Diagnostic**: Train a simple classifier on isolated signs. If accuracy is low, features are bad.

---

#### Hypothesis 2: CTC Alignment Failure
- CTC requires `T ≥ 2×L + 1` (T=timesteps, L=target_length)
- If sequences are too short, CTC cannot align
- Check if model outputs only blanks

**Diagnostic**: Print CTC loss and inspect predictions. If all blanks, alignment failed.

---

#### Hypothesis 3: Model Capacity
- 256 LSTM units may be too small
- 3 layers may be too deep (vanishing gradients)
- 32M params may be insufficient for 6516D input

**Diagnostic**: Reduce complexity (1 layer, 512 units). If improves, capacity issue.

---

#### Hypothesis 4: Training Issues
- Learning rate too high/low
- Batch size too small/large
- Insufficient training epochs
- Bad weight initialization

**Diagnostic**: Check training curves. If loss doesn't decrease, training issue.

---

#### Hypothesis 5: Vocabulary Mismatch
- 1295 classes vs. 3694 in pre-trained
- May be using wrong vocabulary file
- Class indices may be misaligned

**Diagnostic**: Verify vocabulary matches annotation files.

---

## 6. Recommendations

### 6.1 Immediate Diagnostics (This Week)

**Step 1: Verify CTC Constraints**
```python
# Check if sequences are long enough
for batch in dataloader:
    input_lengths = batch['input_lengths']  # T
    target_lengths = batch['target_lengths']  # L
    assert (input_lengths >= 2 * target_lengths + 1).all()
```

**Step 2: Check for Blank Collapse**
```python
# Inspect predictions
predictions = model(features)
pred_classes = predictions.argmax(dim=-1)
print(f"Unique predictions: {torch.unique(pred_classes)}")
# If only blank (class 0), model collapsed
```

**Step 3: Simplify Model**
```python
# Reduce to 1 BiLSTM layer, 512 units
# Train on 10% of data
# Target: <80% WER (shows pipeline works)
```

---

### 6.2 Transfer Learning from Pre-trained Model (Weeks 2-3)

**Option A: Extract GoogLeNet Features**
1. Install Caffe + custom Reverse layer
2. Load `googlenet_iter_76500.caffemodel`
3. Extract 1024D features from `pool5/7x7_s1` layer
4. Save as HDF5/NPY files
5. Train BiLSTM + CTC on extracted features

**Expected Improvement**: 100% WER → 40-50% WER

---

**Option B: Knowledge Distillation**
1. Run pre-trained model on your training data
2. Extract soft targets from `my_2blstm_loss3/classifier`
3. Train student model with distillation loss:
   ```python
   loss = alpha * ctc_loss + (1-alpha) * distillation_loss
   ```

**Expected Improvement**: Better generalization, smoother training

---

### 6.3 Language Model Integration (Week 4)

**After achieving <50% WER with visual model:**

1. **Install KenLM**:
   ```bash
   pip install https://github.com/kpu/kenlm/archive/master.zip
   ```

2. **Decompress Language Model**:
   ```bash
   gunzip SI5-train-4gram.sri.lm.gz
   ```

3. **Implement Two-Pass Decoding**:
   ```python
   import kenlm
   lm = kenlm.Model('SI5-train-4gram.sri.lm')

   # Pass 1: CTC beam search
   hypotheses = ctc_beam_search(logits, beam_width=10)

   # Pass 2: LM rescoring
   for hyp in hypotheses:
       lm_score = lm.score(hyp.text)
       hyp.final_score = 0.7 * hyp.acoustic + 0.3 * lm_score

   best = max(hypotheses, key=lambda h: h.final_score)
   ```

4. **Tune Weights on Dev Set**:
   - Grid search: α ∈ {0.5, 0.6, 0.7, 0.8}, β ∈ {0.2, 0.3, 0.4, 0.5}
   - Optimize for WER

**Expected Improvement**: 40-50% WER → 25-30% WER

---

### 6.4 Architecture Experiments (Weeks 5-6)

**Experiment 1: Sub-sign States** (like pre-trained)
- Decompose each sign into 3 temporal states
- Expand vocabulary: 1295 → 3885 classes
- May improve fine-grained alignment

**Experiment 2: Deep Supervision**
- Add auxiliary classifiers at intermediate layers
- Weight: 0.3 for auxiliary, 1.0 for final
- Helps with deep networks

**Experiment 3: Attention Mechanisms**
- Add self-attention layer after BiLSTM
- May capture long-range dependencies better

**Experiment 4: Hybrid CNN-Landmark Features**
- Extract lightweight CNN features (MobileNet)
- Concatenate with MediaPipe landmarks
- Best of both worlds?

---

## 7. Critical Path to Success

### Phase 1: Fix Base Model (Week 1)
- Goal: 100% WER → <80% WER
- Actions:
  1. Diagnose CTC alignment
  2. Simplify architecture (1 layer)
  3. Verify data pipeline
  4. Train on subset (10% data)

### Phase 2: Transfer Learning (Weeks 2-3)
- Goal: <80% WER → 40-50% WER
- Actions:
  1. Extract CNN features OR
  2. Use knowledge distillation
  3. Train full model on full data
  4. Validate on dev set

### Phase 3: Language Model Integration (Week 4)
- Goal: 40-50% WER → 25-30% WER
- Actions:
  1. Implement CTC + LM beam search
  2. Tune hyperparameters (α, β, beam width)
  3. Evaluate on test set

### Phase 4: Optimization (Weeks 5-6)
- Goal: Optimize for edge deployment
- Actions:
  1. Quantization (FP32 → INT8)
  2. Model pruning
  3. TensorRT conversion
  4. Measure FPS and latency

---

## 8. Key Research Questions

1. **Can MediaPipe landmarks alone achieve competitive WER?**
   - Pre-trained uses pixels → 26.8% WER
   - Your model uses landmarks → ??? WER
   - Gap attributable to visual information loss?

2. **Is CTC better than frame-level + HMM for this task?**
   - CTC: End-to-end, no frame labels needed
   - HMM: Traditional, requires forced alignment
   - Which is more suitable for sign language?

3. **How much does the language model contribute?**
   - Ablation needed: with LM vs. without LM
   - Expected: 15-20% relative WER reduction

4. **What is the optimal trade-off: accuracy vs. efficiency?**
   - CNN features: High accuracy, slow inference
   - Landmarks: Lower accuracy, fast inference
   - Hybrid approach?

---

## 9. Next Steps

1. **Create diagnostic notebook** (experiments/diagnostics.ipynb)
   - Check CTC constraints
   - Visualize predictions
   - Analyze failure modes

2. **Simplify model for debugging** (experiments/simple_model.py)
   - 1 BiLSTM layer
   - 512 hidden units
   - Train on 10% data

3. **Plan transfer learning** (experiments/transfer_learning.md)
   - Caffe installation guide
   - Feature extraction script
   - Training pipeline

4. **Study language model integration** (experiments/lm_integration.md)
   - KenLM tutorial
   - Beam search implementation
   - Hyperparameter tuning

---

**Next Action**: Would you like me to:
1. Create the diagnostic notebook?
2. Set up Caffe for feature extraction?
3. Implement a simplified model for debugging?
4. Explore your current model architecture to identify specific issues?

---

**Author**: Claude Code
**Date**: 2025-11-08
**Purpose**: Comprehensive comparison to guide debugging and improvement strategy
