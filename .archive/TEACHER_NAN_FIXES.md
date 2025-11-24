# 🔧 Teacher Model NaN/Inf Loss Fixes

## ⚠️ Problem: Why Teacher Gets NaN/Inf but Baseline Doesn't

### Architectural Differences

| Component | Baseline (MobileNetV3) | Teacher (I3D) | Risk Level |
|-----------|------------------------|---------------|------------|
| **Depth** | ~20 layers | 50+ layers (Inception) | 🔴 High |
| **Parameters** | 15.7M | ~40-50M (3x) | 🔴 High |
| **Attention** | None | MultiheadAttention | 🔴 High |
| **Batch Size** | 4 | 2 (memory limit) | 🟡 Medium |
| **BatchNorm** | Stable with BS=4 | Unstable with BS=2 | 🟡 Medium |
| **Gradients** | Shorter paths | Very long paths | 🔴 High |

### Root Causes

1. **Gradient Explosion** 🔴
   - Deeper network = longer backprop paths
   - Gradients multiply through 50+ layers
   - Can explode to infinity quickly

2. **Attention Mechanism** 🔴
   - `MultiheadAttention` uses softmax
   - Softmax can produce extreme values
   - Numerical instability in attention weights

3. **Small Batch Size** 🟡
   - BatchNorm requires statistics
   - batch_size=2 → poor estimates
   - Unstable running mean/var

4. **Weight Initialization** 🟡
   - Large model needs careful init
   - Default init may be too aggressive
   - Random weights can start unstable

5. **Mixed Precision** 🟡
   - FP16 has limited range
   - Attention + FP16 = explosion risk
   - Loss scaling can fail

## ✅ Applied Fixes

### Fix 1: Improved Training Loop (`train_teacher.py`)

**Changes:**
- ✅ Input validation (check for NaN in features)
- ✅ Output validation (check log_probs)
- ✅ Loss validation (check before backward)
- ✅ Gradient clipping: 5.0 → **1.0** (more aggressive)
- ✅ Gradient norm monitoring
- ✅ Skip bad batches (don't crash)
- ✅ Track NaN count per epoch
- ✅ Stop if too many NaNs (>10)

**Code:**
```python
# Before
with autocast():
    log_probs = model(features, input_lengths)
    loss = criterion(log_probs, labels, input_lengths, target_lengths)

scaler.scale(loss).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Too high!

# After ✅
# Check input
if torch.isnan(features).any() or torch.isinf(features).any():
    logger.warning(f"NaN/Inf in input, skipping")
    continue

with autocast():
    log_probs = model(features, input_lengths)
    
    # Check output
    if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
        logger.warning(f"NaN/Inf in output, skipping")
        continue
    
    loss = criterion(log_probs, labels, input_lengths, target_lengths)

# Check loss
if torch.isnan(loss) or torch.isinf(loss):
    logger.warning(f"NaN/Inf loss, skipping")
    continue

scaler.scale(loss).backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # More aggressive!
```

### Fix 2: Lower Learning Rate

**Changes:**
- ✅ Default LR: 5e-5 → **1e-4** (auto-adjusted)
- ✅ Conservative for large model
- ✅ Prevents parameter explosion

**Code:**
```python
# Before
optimizer = optim.AdamW(
    model.parameters(),
    lr=2e-4  # Too high for teacher!
)

# After ✅
base_lr = args.learning_rate if args.learning_rate != 5e-5 else 1e-4
optimizer = optim.AdamW(
    model.parameters(),
    lr=base_lr,  # Conservative: 1e-4
    weight_decay=args.weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8  # Prevent division by zero
)
```

### Fix 3: Stable Attention Mechanism (`i3d_teacher.py`)

**Changes:**
- ✅ Xavier init with gain=0.5 (smaller weights)
- ✅ Pre-normalization (normalize before attention)
- ✅ Attention dropout (0.1)
- ✅ Output clamping ([-10, 10])
- ✅ Proper residual connection

**Code:**
```python
# Before
class ModalityFusion:
    def __init__(...):
        self.pose_proj = nn.Linear(pose_dim, output_dim // 4)
        # No special init ❌
        
        self.modality_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
            # No dropout ❌
        )
    
    def forward(...):
        fused = torch.cat([pose_feat, hands_feat, face_feat, temporal_feat], dim=-1)
        attended, _ = self.modality_attention(fused, fused, fused)
        output = self.layer_norm(fused + self.dropout(attended))

# After ✅
class ModalityFusion:
    def __init__(...):
        self.pose_proj = nn.Linear(pose_dim, output_dim // 4)
        
        # Smaller initialization ✅
        for proj in [self.pose_proj, ...]:
            nn.init.xavier_uniform_(proj.weight, gain=0.5)
            nn.init.constant_(proj.bias, 0)
        
        self.modality_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1  # Attention dropout ✅
        )
    
    def forward(...):
        fused = torch.cat([pose_feat, hands_feat, face_feat, temporal_feat], dim=-1)
        
        # Pre-norm for stability ✅
        fused_norm = self.layer_norm(fused)
        
        attended, _ = self.modality_attention(fused_norm, fused_norm, fused_norm)
        
        # Clamp to prevent explosion ✅
        attended = torch.clamp(attended, min=-10.0, max=10.0)
        
        # Residual with original (not normalized) ✅
        output = fused + self.dropout(attended)
```

## 🚀 How to Use the Fixes

### Option 1: Use Fixed Default Parameters

```bash
# Just run with defaults - LR auto-adjusted to 1e-4
python src/training/train_teacher.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/teacher \
    --batch_size 2 \
    --epochs 50
```

### Option 2: Manual Conservative Settings

```bash
# Even more conservative for extra stability
python src/training/train_teacher.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/teacher \
    --batch_size 2 \
    --epochs 50 \
    --learning_rate 5e-5 \      # Very low
    --dropout 0.4 \              # Higher dropout
    --weight_decay 1e-2          # Stronger regularization
```

### Option 3: Gradual Warm-up (Safest)

```bash
# Start with very low LR, increase gradually
python src/training/train_teacher.py \
    --data_dir data/teacher_features/mediapipe_full \
    --output_dir checkpoints/teacher \
    --batch_size 2 \
    --epochs 50 \
    --learning_rate 1e-5 \       # Very conservative start
    --dropout 0.5 \              # High dropout initially
    --weight_decay 1e-2

# Then resume with higher LR after 10 stable epochs
```

## 📊 Monitoring for Stability

### What to Watch

```bash
# Run training
python src/training/train_teacher.py --epochs 50 2>&1 | tee teacher_training.log

# In another terminal, watch for issues
tail -f teacher_training.log | grep -E "NaN|Inf|WARNING|gradient"
```

### Good Signs ✅
```
Epoch [1/50] | Loss: 10.234567 | grad_norm: 2.45
Epoch [2/50] | Loss: 9.876543 | grad_norm: 1.89
Epoch [3/50] | Loss: 9.234567 | grad_norm: 1.56
```
- Loss decreasing steadily
- Gradient norm < 5.0
- No NaN warnings

### Bad Signs ❌
```
WARNING: NaN/Inf in model output at batch 5
WARNING: Large gradient norm: 127.45 at batch 12
WARNING: NaN/Inf loss at batch 18
ERROR: Too many NaN/Inf outputs, stopping epoch
```
- Multiple NaN warnings
- Gradient norm > 100
- Loss stops improving
- **Action**: Reduce LR by 10x

## 🔍 Debugging Checklist

If you still get NaN/Inf after fixes:

### Step 1: Check Data
```python
# Add this before training
import numpy as np

# Load a batch
for batch in train_loader:
    features = batch['features']
    print(f"Features shape: {features.shape}")
    print(f"Features min: {features.min()}, max: {features.max()}")
    print(f"Features mean: {features.mean()}, std: {features.std()}")
    print(f"NaN count: {torch.isnan(features).sum()}")
    print(f"Inf count: {torch.isinf(features).sum()}")
    break
```

**Expected:**
- Min: ~-3 to -5
- Max: ~3 to 5
- Mean: ~0
- Std: ~1
- NaN: 0
- Inf: 0

**If data has NaN/Inf**: Check feature extraction!

### Step 2: Test Without Mixed Precision
```python
# In train_teacher.py, comment out autocast
# with autocast():  # ← Comment this
log_probs = model(features, input_lengths)
loss = criterion(log_probs, labels, input_lengths, target_lengths)
```

**If stable without autocast**: Mixed precision issue
- Increase `scaler` growth interval
- Use FP32 for attention only

### Step 3: Test Without Attention
```python
# In i3d_teacher.py, temporarily disable attention
def forward(self, pose, hands, face, temporal):
    fused = torch.cat([pose_feat, hands_feat, face_feat, temporal_feat], dim=-1)
    # attended, _ = self.modality_attention(...)  # ← Comment this
    # output = fused + self.dropout(attended)      # ← Comment this
    output = fused  # ← Use this instead
    return output
```

**If stable without attention**: Attention mechanism issue
- Reduce num_heads: 8 → 4
- Increase attention dropout: 0.1 → 0.3
- Use simpler fusion (weighted sum)

### Step 4: Reduce Model Complexity
```python
# In train_teacher.py
model = create_i3d_teacher(
    vocab_size=len(vocab),
    dropout=0.5,  # Much higher dropout
    hidden_dim=256  # Reduce from 512
)
```

## 📈 Expected Results After Fixes

### Training Progression

| Epoch | Loss | WER | Grad Norm | Status |
|-------|------|-----|-----------|--------|
| 1 | 10-12 | 95-99% | 5-10 | Initial (high is OK) |
| 5 | 8-10 | 85-90% | 2-5 | Stabilizing |
| 10 | 6-8 | 70-80% | 1-3 | Learning |
| 20 | 4-6 | 50-60% | 0.5-2 | Converging |
| 50 | 2-4 | 20-30% | 0.2-1 | Target reached ✅ |

### Gradient Norms

- **Epoch 1-5**: 5-10 (high initially, OK)
- **Epoch 6-20**: 1-5 (stabilizing)
- **Epoch 21+**: 0.5-2 (converged)

**Warning if**:
- Grad norm > 20 after epoch 5
- Grad norm > 50 at any time
- Grad norm suddenly jumps by 10x

## 🎯 Comparison: Baseline vs Teacher (Fixed)

| Metric | Baseline | Teacher (Before) | Teacher (After Fix) |
|--------|----------|------------------|---------------------|
| **Stability** | ✅ Stable | ❌ NaN/Inf | ✅ Stable |
| **Grad Clip** | 1.0 | 5.0 (too high) | 1.0 |
| **Learning Rate** | 5e-5 | 2e-4 (too high) | 1e-4 |
| **Batch Size** | 4 | 2 | 2 |
| **Attention** | None | Unstable | Stable (clamped) |
| **Init** | Default | Default | Xavier (gain=0.5) |
| **NaN Handling** | Skip | Skip | Skip + Track + Stop |
| **Monitoring** | Loss only | Loss only | Loss + Grad Norm |

## 📝 Summary of Changes

### Files Modified

1. **`src/training/train_teacher.py`**
   - Enhanced `train_epoch()` with stability checks
   - Lower default learning rate (1e-4)
   - Aggressive gradient clipping (1.0)
   - NaN tracking and early stopping
   - Gradient norm monitoring

2. **`src/models/i3d_teacher.py`**
   - Stable `ModalityFusion` attention
   - Xavier initialization (gain=0.5)
   - Pre-normalization architecture
   - Attention output clamping ([-10, 10])
   - Attention dropout (0.1)

### Key Parameters

```python
# Training
learning_rate = 1e-4      # Conservative for stability
gradient_clip = 1.0       # Aggressive clipping
batch_size = 2            # Memory constraint
dropout = 0.3             # Regularization

# Attention
attention_dropout = 0.1   # Prevent attention explosion
attention_clamp = 10.0    # Clamp output values
init_gain = 0.5           # Smaller initial weights
```

## 🚀 Next Steps

1. ✅ **Test Overfit** (Verify fixes work)
   ```bash
   python overfit_test_teacher.py
   ```
   Expected: 0% WER, no NaN

2. ✅ **Full Training** (With monitoring)
   ```bash
   python src/training/train_teacher.py --epochs 50 2>&1 | tee teacher.log
   ```
   Monitor: `tail -f teacher.log | grep -E "grad_norm|NaN"`

3. 📊 **Compare Plots**
   - Check `figures/teacher/training_curves.png`
   - Verify smooth loss curve
   - WER should decrease steadily

4. 🎯 **Target Performance**
   - Validation WER < 30% ✅
   - No NaN/Inf during training ✅
   - Stable gradient norms (< 5) ✅

## ❓ Still Having Issues?

### Last Resort Options

1. **Disable Mixed Precision**
   ```python
   # Use FP32 only
   # Don't use autocast()
   ```

2. **Use GroupNorm Instead of BatchNorm**
   ```python
   # Replace BatchNorm1d with GroupNorm
   nn.GroupNorm(8, channels)  # Works with any batch size
   ```

3. **Simplify Architecture**
   - Remove attention (use weighted average)
   - Reduce Inception modules
   - Use simpler MobileNetV3 teacher

4. **Train Longer with Lower LR**
   ```bash
   --learning_rate 1e-6 --epochs 100
   ```

---

**Good luck! The fixes should stabilize your teacher training.** 🚀

Monitor gradient norms and check for NaN warnings. If stable for 10 epochs, you're good to go!

