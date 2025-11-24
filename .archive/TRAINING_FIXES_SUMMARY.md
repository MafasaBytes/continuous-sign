# Training Fixes Applied - MobileNetV3 Baseline Model

## Date: November 16, 2025
## Status: ✅ CRITICAL FIXES APPLIED

---

## Problems Identified

### 🚨 Critical Issue #1: Architecture Bug - Sequence Collapse
**Problem**: The MobileNetV3 model used `AdaptiveAvgPool1d(1)` which collapsed the entire temporal dimension to a single timestep, then artificially expanded it back. This meant every timestep had **identical features**, making CTC training impossible.

**Impact**: 
- Model could not learn temporal alignments
- WER stuck at 100%
- CTC loss plateaued at ~4.2

**Fix Applied**:
- Replaced global pooling with proper temporal upsampling using `ConvTranspose1d`
- Reduced stride configuration to preserve more temporal information (4x downsampling instead of 32x)
- Added proper temporal projection layers
- Ensured sequence length is restored to match input for CTC loss

### 🚨 Critical Issue #2: Vocabulary Mapping Bug
**Problem**: The `Vocabulary` class had both `<pad>` and `<blank>` mapped to index 0, but `idx2word` dictionary was not updated when adding new words.

**Impact**:
- All decoded predictions returned `"<blank>"` for every token
- WER always 100% because predictions were empty

**Fix Applied**:
- Removed duplicate `<pad>` mapping (CTC only needs `<blank>`)
- Added proper `idx2word[idx] = word` assignment in `add_word()` method
- Now vocabulary properly maps both directions

### 🚨 Issue #3: Training Instability
**Problem**: 
- Learning rate parameter handling had fallback logic that didn't work correctly
- No learning rate warmup leading to gradient spikes
- Dropout too high (0.5) preventing initial convergence
- Weight decay too high (5e-3) over-regularizing

**Impact**:
- NaN/Inf losses appearing after epoch 23
- Model couldn't converge

**Fix Applied**:
- Fixed learning rate to use `args.learning_rate` directly
- Added 5-epoch warmup starting at 10% of target LR
- Reduced dropout from 0.5 → 0.3
- Reduced weight decay from 5e-3 → 1e-4
- Increased default learning rate from 1e-4 → 3e-4

### Issue #4: Model Complexity
**Problem**: Model was too complex for baseline convergence:
- 2 LSTM layers
- 11 MobileNetV3 blocks
- Cross-modal attention

**Fix Applied**:
- Reduced to 1 LSTM layer for stability
- Simplified to 6 MobileNetV3 blocks
- Removed cross-modal attention (will add back after baseline works)

---

## Changes Summary

### File: `src/data/dataset.py`
```python
class Vocabulary:
    def __init__(self):
        # FIXED: Removed duplicate <pad> mapping
        self.word2idx = {"<blank>": 0}
        self.idx2word = {0: "<blank>"}
        self.blank_id = 0
        self.pad_id = 0

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word  # CRITICAL FIX: Added this line
        return self.word2idx[word]
```

### File: `src/models/mobilenet_v3.py`
**Key Changes**:
1. Simplified MobileNetV3 blocks (11 → 6 blocks)
2. Reduced strides to preserve temporal dimension (32x → 4x downsampling)
3. Replaced `AdaptiveAvgPool1d(1) + expand` with proper upsampling:
   ```python
   self.temporal_upsample = nn.Sequential(
       nn.ConvTranspose1d(48, hidden_dim, kernel_size=4, stride=4),
       nn.BatchNorm1d(hidden_dim),
       nn.ReLU()
   )
   ```
4. Added temporal projection layer for feature refinement
5. Proper sequence length handling with trim/pad logic
6. Reduced default LSTM layers: 2 → 1
7. Reduced default dropout: 0.5 → 0.3

### File: `src/training/train.py`
**Key Changes**:
1. Fixed learning rate handling - now uses `args.learning_rate` directly
2. Added learning rate warmup:
   ```python
   warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
   scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[5])
   ```
3. Updated default hyperparameters:
   - Learning rate: 1e-4 → 3e-4
   - Dropout: 0.5 → 0.3
   - Weight decay: 5e-3 → 1e-4
   - Max grad norm: 5.0 → 1.0

---

## Expected Results

### Before Fixes:
- ❌ Loss: plateaus at ~4.2
- ❌ WER: stuck at 100%
- ❌ NaN/Inf losses after epoch 23
- ❌ No learning progress

### After Fixes:
- ✅ Loss should drop below 3.0 by epoch 10
- ✅ Loss should reach 1.5-2.0 by epoch 50
- ✅ WER should drop below 80% by epoch 5
- ✅ WER should reach 40-60% for initial baseline (target: <25% with fine-tuning)
- ✅ No NaN/Inf losses
- ✅ Smooth training progression

---

## Training Command

Run training with the fixed implementation:

```bash
python src/training/train.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/student \
    --batch_size 4 \
    --learning_rate 3e-4 \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --epochs 100 \
    --early_stopping_patience 15
```

---

## Monitoring Training

Watch for these positive signs:
1. **Epoch 1-5**: Loss should drop from ~10 to ~5 (warmup phase)
2. **Epoch 5-10**: Loss should drop from ~5 to ~3
3. **Epoch 10-20**: WER should drop below 80%
4. **Epoch 20-50**: WER should reach 50-60%
5. **No NaN/Inf**: Should see smooth training throughout

If you still see issues:
- Check feature files are loading correctly
- Verify vocabulary size matches model output
- Monitor GPU memory usage
- Check for data loading errors in logs

---

## Next Steps (After Baseline Converges)

1. **Add back cross-modal attention** once baseline WER < 60%
2. **Increase LSTM layers** to 2 if needed
3. **Fine-tune with stronger augmentation**
4. **Implement knowledge distillation** from teacher model
5. **Add beam search decoding** for inference

---

## Architecture Diagram (After Fixes)

```
Input [B, T, 6516]
    ↓
Modality Encoders
    → Pose [B, T, 64]
    → Hands [B, T, 128]
    → Face [B, T, 64]
    → Temporal [B, T, 128]
    ↓
Concatenate [B, T, 384]
    ↓
Transpose [B, 384, T]
    ↓
MobileNetV3 Blocks (6 blocks, 4x downsampling)
    → [B, 48, T/4]
    ↓
Temporal Upsample (TransposedConv1d 4x)
    → [B, 128, T]
    ↓
Temporal Projection
    → [B, 128, T]
    ↓
Transpose [B, T, 128]
    ↓
BiLSTM (1 layer, bidirectional)
    → [B, T, 256]
    ↓
Output Projection
    → [B, T, vocab_size]
    ↓
LogSoftmax + Transpose
    → [T, B, vocab_size]
    ↓
CTC Loss
```

**Key Difference**: Temporal dimension is now properly preserved throughout the network!

---

## Validation

All changes have been:
- ✅ Applied to source files
- ✅ Syntax checked (no linter errors)
- ✅ Architecturally sound
- ✅ Ready for training

**Status**: Ready to train! 🚀

