# Pre-trained CNN-LSTM Architecture Analysis

## Overview
Analysis of the RWTH-PHOENIX pre-trained models for sign language recognition (26.8% WER on test set).

**Source**: Koller, Zargaran, Ney. "Re-Sign: Re-Aligned End-to-End Sequence Modeling with Deep Recurrent CNN-HMMs" CVPR 2017

---

## 1. Pre-trained Model Components

### Files Available:
- `googlenet_iter_76500.caffemodel` - Pre-trained weights (Caffe format)
- `net.prototxt` - Network architecture definition (Caffe)
- `trainingClasses.txt` - 1231 output classes
- `solver.prototxt` - Training hyperparameters

### Language Model:
- `SI5-train-4gram.sri.lm.gz` - 4-gram SRILM language model
- Perplexity: 58.56 on dev set
- Only 6 OOV (out-of-vocabulary) words on 1167 word dev set

---

## 2. Architecture Breakdown

### Overall Pipeline:
```
Raw Video Frames (224×224, RGB)
    ↓
GoogLeNet CNN (Visual Feature Extraction)
    ↓
Bidirectional LSTM (Temporal Modeling - 2 layers)
    ↓
Fully Connected Classifier (3694 outputs)
    ↓
Softmax Loss (frame-level classification)
```

### 2.1 Visual Backbone: GoogLeNet (Inception v1)

**Input Processing:**
- Input size: 224×224 pixels (center crop from 256×256)
- Mean subtraction from training data
- Batch size: 32

**Architecture Layers:**
1. **Initial Convolution Block**
   - Conv 7×7, stride 2, 64 filters → ReLU
   - MaxPool 3×3, stride 2
   - LRN (Local Response Normalization)

2. **Second Convolution Block**
   - Conv 1×1, 64 filters (dimensionality reduction)
   - Conv 3×3, 192 filters
   - LRN, MaxPool 3×3, stride 2

3. **Inception Modules** (9 total)
   - **inception_3a, 3b** (after pool2)
   - **inception_4a, 4b, 4c, 4d, 4e** (after pool3)
   - **inception_5a, 5b** (after pool4)

4. **Auxiliary Classifiers** (for deep supervision)
   - **loss1** at inception_4a output (weight: 0.3)
   - **loss2** at inception_4d output (weight: 0.3)
   - Both output 3694 classes

5. **Final Pooling**
   - Average Pool 7×7, stride 1
   - Dropout 0.4
   - Output: 1024-dimensional feature vector per frame

**Key Design Choices:**
- Uses **deep supervision** (auxiliary losses) to combat vanishing gradients
- Multi-scale feature extraction via inception modules
- Dropout ratio: 0.7 for auxiliary classifiers, 0.4 for main path

---

### 2.2 Temporal Modeling: Bidirectional LSTM (2 layers)

**Architecture:**

```
CNN Features (1024D per frame)
    ↓
Reshape to (batch=32, time=T, features=1024)
    ↓
┌─────────────────────────────────────┐
│ Layer 1: Bidirectional LSTM         │
│   - Forward LSTM: 1024 units        │
│   - Backward LSTM: 1024 units       │
│   - Concatenate → 2048D             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Layer 2: Bidirectional LSTM         │
│   - Forward LSTM: 1024 units        │
│   - Backward LSTM: 1024 units       │
│   - Concatenate → 2048D             │
└─────────────────────────────────────┘
    ↓
Fully Connected: 2048 → 3694 classes
    ↓
Softmax Loss (frame-level)
```

**LSTM Configuration:**
- Number of units per direction: 1024
- Total output per layer: 2048 (bidirectional concat)
- Weight initialization: Uniform(-0.08, 0.08)
- Bias initialization: Constant(0)
- No dropout between LSTM layers (!)

**Sequence Handling:**
- Uses `cont` (continuation) flags for sequence boundaries
- Forward continuation: `cont/fwd`
- Backward continuation: `cont/bck/rev` (reversed)
- Custom `Reverse` layer for backward LSTM inputs

---

### 2.3 Output Layer

**Classifier:**
- Input: 2048D (from BiLSTM layer 2)
- Output: 3694 classes
- Includes blank token for alignment
- Activation: Softmax with cross-entropy loss

**Training Strategy:**
- Multi-task loss: auxiliary losses + main loss
  - `loss1` (at inception_4a): weight 0.3
  - `loss2` (at inception_4d): weight 0.3
  - `loss3` (final output): weight 1.0
- Top-1 and Top-5 accuracy metrics

---

## 3. Key Differences from Your Current Architecture

| Component | Pre-trained Model | Your Current Model |
|-----------|-------------------|-------------------|
| **Input** | Raw pixels (224×224 RGB) | MediaPipe landmarks (6516D) |
| **CNN Backbone** | GoogLeNet (Inception v1) | None (direct landmarks) |
| **Feature Dim** | 1024D (from CNN pool) | 6516D (raw landmarks) |
| **BiLSTM Layers** | 2 layers, 1024 units each | 3 layers, 256 units each |
| **BiLSTM Output** | 2048D per timestep | 512D per timestep |
| **Sequence Loss** | Frame-level Softmax | CTC Loss |
| **Deep Supervision** | Yes (2 auxiliary losses) | No |
| **Dropout** | 0.4 (main), 0.7 (auxiliary) | Unknown (check your model) |
| **Output Classes** | 3694 | 1295 |
| **Decoding** | HMM + Language Model | CTC Beam Search (no LM yet) |

---

## 4. Critical Architectural Insights

### 4.1 Why GoogLeNet Works Well:
1. **Hierarchical feature extraction**: Multi-scale temporal patterns
2. **Deep supervision**: Auxiliary losses prevent vanishing gradients in deep network
3. **Inception modules**: Capture signs at multiple spatial scales
4. **Learned features**: CNN learns sign-specific visual patterns from pixels

### 4.2 Why Frame-level Loss Works:
- **Alignment-free training**: No need for frame-to-sign alignment
- **Combined with HMM decoding**: HMM handles temporal alignment during inference
- **Language model integration**: Strong 4-gram LM (perplexity 58.56) rescores hypotheses

### 4.3 The Two-Stage Decoding Strategy:
```
1. CNN-BLSTM → Frame-level posteriors (3694 classes per frame)
2. HMM Viterbi Decoding → Forced alignment to signs
3. Language Model Rescoring → Re-rank hypotheses
```

**This is fundamentally different from CTC!**
- CTC: End-to-end alignment during training
- HMM: Post-hoc alignment during decoding

---

## 5. Potential Uses for Your Project

### 5.1 Transfer Learning (Recommended)
**Option A: Feature Extractor**
- Extract 1024D features from `pool5/7x7_s1` layer
- Replace your MediaPipe features with CNN features
- Train only the BiLSTM + CTC layers

**Pros:**
- Proven visual features (achieved 26.8% WER)
- Reduced feature dimensionality (1024D vs 6516D)
- Pre-trained on same dataset

**Cons:**
- Requires Caffe installation
- Can't process videos in real-time (heavy CNN)
- Need to convert Caffe model to PyTorch/TensorFlow

---

### 5.2 Knowledge Distillation (Teacher Model)
- Use pre-trained CNN-BLSTM as teacher
- Distill knowledge into your lightweight student model
- Soft targets from `my_2blstm_loss3/classifier` layer

**Distillation Strategy:**
```python
# Pseudo-code
teacher_logits = pretrained_model(frames)  # 3694 classes
student_logits = your_model(mediapipe_features)  # 1295 classes

# Map vocabularies (teacher has more classes)
loss = distillation_loss(student_logits, teacher_logits, temperature=3.0)
```

---

### 5.3 Baseline Comparison
- Run pre-trained model on your data splits
- Verify evaluation pipeline correctness
- If pre-trained gets 26.8% WER but yours gets 100%, problem is in your training

**How to use:**
1. Install Caffe + custom Reverse layer
2. Load `googlenet_iter_76500.caffemodel`
3. Run inference on test set
4. Compare outputs with your model

---

### 5.4 Vocabulary Alignment Analysis
- Pre-trained uses 3694 classes (trainingClasses.txt)
- Your model uses 1295 classes
- **Action**: Read trainingClasses.txt to understand class definition
- Check if pre-trained uses sub-word units vs. full signs

---

## 6. Language Model Integration Insights

### 6.1 Language Model Statistics:
- **Type**: 4-gram SRILM
- **Perplexity**: 58.56 (relatively low = good predictions)
- **OOV Rate**: 6/1167 = 0.5% (excellent vocabulary coverage)

### 6.2 Why This Matters:
1. **Strong linguistic prior**: Weather domain has predictable word sequences
2. **Low perplexity**: LM can disambiguate visual confusions
3. **Minimal OOV**: All signs well-represented in training data

### 6.3 Integration Strategy (when visual model works):
```python
# Two-pass decoding
# Pass 1: Acoustic model (your BiLSTM + CTC)
hypotheses = ctc_beam_search(acoustic_scores, beam_width=10)

# Pass 2: Language model rescoring
for hyp in hypotheses:
    lm_score = kenlm_model.score(hyp.text)
    final_score = alpha * hyp.acoustic_score + beta * lm_score

# Best hypothesis
best_hyp = max(hypotheses, key=lambda h: h.final_score)
```

**Hyperparameters** (from RWTH paper):
- `alpha` (acoustic weight): ~0.7
- `beta` (LM weight): ~0.3
- Beam width: 10-20
- LM pruning threshold: 4000

---

## 7. Recommendations for Your Project

### Immediate Actions:
1. **Read trainingClasses.txt** (next step)
   - Understand class definitions
   - Compare with your vocabulary (1295 classes)

2. **Fix your base model first**
   - 100% WER → <50% WER before adding complexity
   - Diagnose CTC alignment issues
   - Verify MediaPipe features are meaningful

3. **Simplify for debugging**
   - Reduce BiLSTM from 3 layers to 1 layer
   - Reduce hidden units from 256 to 128
   - Train on 10% of data to verify pipeline

### Medium-term (After Visual Model Works):
4. **Extract CNN features** (if Caffe installation feasible)
   - Replace MediaPipe with GoogLeNet features
   - Compare: CNN features vs. MediaPipe landmarks

5. **Implement Language Model Rescoring**
   - Use provided 4-gram SRILM model
   - Integrate with CTC beam search
   - Tune alpha/beta weights on dev set

### Long-term (Phase II/III):
6. **Knowledge Distillation**
   - Distill CNN-BLSTM → lightweight student
   - Combine visual features + soft targets

7. **Hybrid Decoding**
   - CTC for alignment
   - HMM for temporal modeling
   - LM for linguistic constraints

---

## 8. Technical Notes

### Caffe Installation (if needed):
```bash
# Install standard Caffe
git clone https://github.com/BVLC/caffe.git
cd caffe
git checkout Oct-19-2016  # Specific version

# Add custom Reverse layer
wget https://raw.githubusercontent.com/ChWick/caffe/.../reverse_layer.hpp
wget https://raw.githubusercontent.com/ChWick/caffe/.../reverse_layer.cpp
wget https://raw.githubusercontent.com/ChWick/caffe/.../reverse_layer.cu

# Compile (follow Caffe installation guide)
```

### Model Conversion (Caffe → PyTorch):
- Use `caffemodel2pytorch` or `MMdnn` libraries
- Extract layer-by-layer if needed
- Validate outputs match original Caffe model

---

## 9. Key Takeaways

1. **Architecture is well-designed**:
   - Proven to work (26.8% WER)
   - Deep supervision prevents gradient issues
   - Bidirectional context crucial for sequence modeling

2. **Your approach differs fundamentally**:
   - Landmarks vs. pixels
   - CTC vs. frame-level loss + HMM
   - No language model (yet)

3. **Language model is powerful**:
   - Low perplexity (58.56)
   - Minimal OOV (0.5%)
   - Critical for final performance

4. **Transfer learning is viable**:
   - Extract CNN features as drop-in replacement
   - Use as teacher for distillation
   - Validate your evaluation pipeline

5. **Don't use LM until visual model works**:
   - Fix 100% WER first
   - Then add LM rescoring
   - Research principle: debug sequentially

---

## Next Steps

1. **Read trainingClasses.txt** - Understand class vocabulary
2. **Create diagnostic script** - Analyze why current model fails
3. **Compare architectures side-by-side** - Your model vs. pre-trained
4. **Plan transfer learning experiment** - Extract CNN features

---

**Author**: Claude Code
**Date**: 2025-11-08
**Purpose**: Understanding RWTH pre-trained models for sign language recognition
