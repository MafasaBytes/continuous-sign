# Experiments Directory

This directory contains analysis and experimental work for the sign language recognition project.

## Documents Created

### 1. `pretrained_architecture_analysis.md`
**Purpose**: Deep dive into the RWTH-PHOENIX pre-trained CNN-BLSTM model

**Key Findings**:
- Architecture: GoogLeNet (Inception v1) + 2-layer BiLSTM
- Performance: 26.8% WER on test set
- Uses 3-state temporal decomposition (3694 classes = 1231 signs × 3 states)
- Strong 4-gram language model (perplexity 58.56)
- Frame-level softmax + HMM decoding (not CTC)

**Contents**:
- Complete architecture breakdown
- Layer-by-layer analysis
- Language model statistics
- Transfer learning recommendations

---

### 2. `architecture_comparison.md`
**Purpose**: Side-by-side comparison of pre-trained model vs. your current implementation

**Key Findings**:
- Pre-trained: 79M params, CNN + 2-layer BiLSTM, frame-level loss
- Your model: 32M params, MediaPipe + 3-layer BiLSTM, CTC loss
- Feature difference: Pixels (150K-D) vs. Landmarks (6.5K-D)
- Missing component: Language model integration

**Contents**:
- Detailed parameter comparison
- Hypothesis for 100% WER failure
- Step-by-step debugging recommendations
- Critical path to success (4-phase plan)

---

## Critical Insights

### Why Pre-trained Model Works (26.8% WER):
1. Learned visual features from GoogLeNet CNN
2. Fine temporal granularity (3 states per sign)
3. Deep supervision (auxiliary losses)
4. Strong language model (perplexity 58.56)
5. Large model capacity (79M parameters)

### Why Your Model Fails (100% WER):
**Primary Hypotheses** (in order of likelihood):
1. **CTC alignment failure** - Model outputs only blanks
2. **Feature quality issues** - MediaPipe landmarks insufficient
3. **Insufficient model capacity** - 256 LSTM units too small
4. **Training issues** - Bad hyperparameters or initialization
5. **Vocabulary mismatch** - Wrong class definitions

---

## Recommended Next Steps

### Immediate (This Week):
1. **Diagnose CTC alignment**
   - Check if T ≥ 2L + 1 constraint holds
   - Inspect predictions (blank collapse?)
   - Simplify model (1 layer, 512 units)

2. **Verify data pipeline**
   - Check vocabulary consistency
   - Validate input/target shapes
   - Test on small dataset (10% data)

### Short-term (Weeks 2-3):
3. **Transfer learning**
   - Extract GoogLeNet features (1024D)
   - Replace MediaPipe with CNN features
   - Expected: 100% → 40-50% WER

4. **Knowledge distillation**
   - Use pre-trained as teacher
   - Distill into lightweight student
   - Soft targets + hard CTC loss

### Medium-term (Week 4):
5. **Language model integration** (ONLY after visual model works)
   - Decompress 4-gram SRILM model
   - Implement two-pass decoding (CTC + LM rescoring)
   - Tune α/β weights on dev set
   - Expected: 40-50% → 25-30% WER

### Long-term (Weeks 5-6):
6. **Architecture experiments**
   - Sub-sign temporal states (like pre-trained)
   - Deep supervision (auxiliary losses)
   - Attention mechanisms
   - Hybrid CNN-landmark features

---

## How to Use KenLM Language Model

### Files Available:
- `data/raw_data/.../models/LanguageModel/SI5-train-4gram.sri.lm.gz`
- Perplexity: 58.56
- OOV rate: 0.5% (6/1167 words)

### Integration Steps:

1. **Install KenLM**:
```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
```

2. **Decompress Model**:
```bash
cd data/raw_data/phoenix-2014-signerindependent-SI5/models/LanguageModel/
gunzip SI5-train-4gram.sri.lm.gz
```

3. **Two-Pass Decoding**:
```python
import kenlm

# Load language model
lm = kenlm.Model('path/to/SI5-train-4gram.sri.lm')

# Pass 1: CTC beam search (acoustic model only)
hypotheses = ctc_beam_search(logits, beam_width=10)

# Pass 2: Language model rescoring
for hyp in hypotheses:
    lm_score = lm.score(hyp.text)  # Log probability
    hyp.final_score = alpha * hyp.acoustic_score + beta * lm_score

# Best hypothesis
best_hyp = max(hypotheses, key=lambda h: h.final_score)
```

4. **Hyperparameter Tuning** (on dev set):
- Alpha (acoustic weight): 0.5 - 0.8 (try 0.7)
- Beta (LM weight): 0.2 - 0.5 (try 0.3)
- Beam width: 5, 10, 20

**IMPORTANT**: Only integrate LM after achieving <50% WER with visual model alone!

---

## How to Use Pre-trained CNN-LSTM

### Option A: Feature Extraction (Recommended)

**Pros**:
- No Caffe training needed
- Proven features (26.8% WER)
- Lightweight (extract once, cache)

**Cons**:
- Requires Caffe installation
- One-time processing overhead

**Steps**:
1. Install Caffe + custom Reverse layer (see pretrained_architecture_analysis.md)
2. Load model: `googlenet_iter_76500.caffemodel`
3. Extract features from `pool5/7x7_s1` layer (1024D per frame)
4. Save to HDF5/NPY: `features_train.h5`, `features_dev.h5`, `features_test.h5`
5. Train BiLSTM + CTC on extracted features

**Expected Result**: 100% WER → 40-50% WER

---

### Option B: Knowledge Distillation

**Pros**:
- Improve student model training
- Learn from teacher's soft targets
- No feature extraction overhead at inference

**Cons**:
- Still requires Caffe for teacher inference
- More complex training pipeline

**Steps**:
1. Run teacher model on training data
2. Extract soft targets: `softmax(logits / temperature)`
3. Train student with combined loss:
   ```python
   loss = 0.7 * kl_divergence(student, teacher) + 0.3 * ctc_loss(student, labels)
   ```
4. Temperature: 3.0 (typical for distillation)

**Expected Result**: Better generalization, smoother training curves

---

### Option C: Baseline Comparison

**Pros**:
- Validates your evaluation pipeline
- Quick sanity check

**Steps**:
1. Run pre-trained model on YOUR test set
2. If gets 26.8% WER → your eval is correct
3. If gets 100% WER → data mismatch issue
4. Compare predictions with your model

---

## Research Questions to Answer

1. **Feature representation**:
   - Can landmarks alone match pixel-based features?
   - Gap: CNN features vs. MediaPipe landmarks

2. **Loss function**:
   - CTC vs. frame-level + HMM
   - Which is better for sign language?

3. **Language model contribution**:
   - How much does 4-gram LM help?
   - Ablation: acoustic only vs. acoustic + LM

4. **Temporal granularity**:
   - Full signs vs. 3-state sub-signs
   - Impact on alignment quality

5. **Model capacity vs. efficiency**:
   - Trade-off: accuracy vs. inference speed
   - Optimal architecture for edge devices (8GB VRAM, 30 FPS)

---

## File Organization

```
experiments/
├── README.md (this file)
├── pretrained_architecture_analysis.md
├── architecture_comparison.md
└── (future experiments)
    ├── diagnostics.ipynb
    ├── simple_model.py
    ├── feature_extraction.py
    ├── lm_integration.py
    └── results/
```

---

## Key Takeaways

1. **Don't use KenLM yet** - Fix base model first (100% → <50% WER)
2. **Pre-trained model is valuable** - Use for transfer learning or distillation
3. **Root cause likely**: CTC alignment failure or feature quality
4. **Critical path**: Diagnose → Transfer learning → LM integration → Optimization
5. **Research rigor**: Debug sequentially, not simultaneously

---

**Last Updated**: 2025-11-08
**Next Action**: Create diagnostic notebook to identify failure mode
