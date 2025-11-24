# Systematic Overfitting Test Strategy

## Executive Summary

After a month stuck at 93% WER with uniform probability outputs, we need to take a systematic approach to identify where the model learning breaks down. This document provides a complete strategy for testing and fixing the overfitting capability of the ORIGINAL MobileNetV3 model.

## Current Situation

### What We Know
- ✅ Model CAN overfit on 2 samples (loss: 124 → -0.24)
- ❌ Model FAILS with 20+ samples (WER stays at 100%)
- ⚠️ Model outputs uniform probabilities: -6.88 ≈ log(1/973)
- ✅ Feature dimensions are preserved correctly (176 temporal frames maintained)
- ✅ MediaPipe features are available in `data/teacher_features/mediapipe_full/`

### Core Problem
The model outputs nearly uniform probabilities across all 973 vocabulary tokens, indicating the signal is dying somewhere in the network. This happens even with aggressive overfitting configurations.

## Test Strategy

### Phase 1: Progressive Overfitting Test (START HERE)

Run the systematic test to find the exact breaking point:

```bash
# Run progressive test (1, 2, 5, 10, 20, 50 samples)
python test_overfitting_clean.py --mode progressive
```

This will:
1. Start with 1 sample and progressively increase
2. Use optimized configs for each scale
3. Stop when overfitting fails
4. Generate a detailed report

Expected outcomes:
- **Success at 1-2 samples**: Architecture is fundamentally sound
- **Failure at 5-10 samples**: Capacity or optimization issue
- **Failure at 1 sample**: Critical architecture problem

### Phase 2: Diagnostic Analysis

If overfitting fails early, run diagnostics:

```bash
# Analyze where signal dies in the network
python diagnose_uniform_output.py
```

This will identify:
- Which layers have dead neurons
- Where gradients vanish
- If initialization is problematic
- Whether specific components fail

### Phase 3: Configuration Testing

Test specific configurations at the breaking point:

```bash
# Test with specific configuration
python test_overfitting_clean.py --mode specific

# Or test single sample size with custom config
python test_overfitting_clean.py --mode single --n_samples 10
```

### Phase 4: Systematic Grid Search

If basic tests fail, run comprehensive grid search:

```bash
# Full systematic test with multiple configs
python overfit_test_systematic.py
```

## Recommended Configurations by Scale

### 1 Sample (Memorization)
```python
{
    "optimizer": "sgd",
    "lr": 1.0,  # Very high for single sample
    "momentum": 0.9,
    "gradient_clip": 10.0,
    "batch_size": 1,
    "dropout": 0.0,
    "hidden_dim": 32,  # Small model
    "max_epochs": 500
}
```

### 2-5 Samples (Basic Overfitting)
```python
{
    "optimizer": "adam",
    "lr": 0.1,  # High learning rate
    "gradient_clip": 5.0,
    "batch_size": 2-5,
    "dropout": 0.0,
    "hidden_dim": 64,
    "max_epochs": 400
}
```

### 10-20 Samples (Challenging)
```python
{
    "optimizer": "adam",
    "lr": 0.01,
    "gradient_clip": 5.0,
    "batch_size": 5,
    "dropout": 0.0,
    "hidden_dim": 128,
    "use_scheduler": True,
    "max_epochs": 300
}
```

### 50+ Samples (Full Capacity Test)
```python
{
    "optimizer": "adamw",
    "lr": 0.001,
    "gradient_clip": 1.0,
    "batch_size": 8,
    "dropout": 0.0,
    "hidden_dim": 128,
    "use_scheduler": True,
    "warmup_epochs": 10,
    "max_epochs": 500
}
```

## Potential Issues and Fixes

### Issue 1: Uniform Outputs (Most Likely)

**Symptoms**:
- Output entropy ≈ maximum entropy
- All logits nearly identical
- Loss plateaus immediately

**Potential Causes**:
1. Signal dying in MobileNet blocks
2. Temporal downsampling losing information
3. Poor weight initialization
4. BatchNorm killing gradients

**Fixes to Try**:
```python
# Fix 1: Bypass MobileNet blocks temporarily
model.blocks = nn.Identity()

# Fix 2: Remove temporal downsampling
# Comment out temporal_upsample and adjust accordingly

# Fix 3: Different initialization
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

# Fix 4: Remove BatchNorm
# Replace BatchNorm1d with LayerNorm or Identity
```

### Issue 2: Gradient Vanishing

**Symptoms**:
- Gradients < 1e-6
- No weight updates
- Loss doesn't decrease

**Fixes**:
```python
# Higher learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

# Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Layer-wise learning rates
optimizer = torch.optim.Adam([
    {'params': model.pose_encoder.parameters(), 'lr': 0.01},
    {'params': model.lstm.parameters(), 'lr': 0.1},
    {'params': model.output_proj.parameters(), 'lr': 1.0}
])
```

### Issue 3: CTC Loss Problems

**Symptoms**:
- Loss becomes NaN or Inf
- Loss doesn't decrease below certain threshold
- Blank token dominates predictions

**Fixes**:
```python
# Ensure proper length handling
assert all(input_lengths > label_lengths)
assert all(label_lengths > 0)

# Use different CTC reduction
criterion = nn.CTCLoss(blank=0, reduction='sum')  # Try sum vs mean

# Add label smoothing
def smooth_ctc_loss(output, target, smoothing=0.1):
    # Implementation of label-smoothed CTC
    pass
```

## Success Criteria

### Stage 1: Single Sample Memorization
- ✅ Loss < 0.01
- ✅ WER = 0%
- ✅ Perfect reconstruction

### Stage 2: Small Batch (2-5 samples)
- ✅ Loss < 1.0
- ✅ WER < 20%
- ✅ Can distinguish between samples

### Stage 3: Medium Batch (10-20 samples)
- ✅ Loss < 5.0
- ✅ WER < 50%
- ✅ Learning curve shows improvement

### Stage 4: Large Batch (50+ samples)
- ✅ Loss decreasing consistently
- ✅ WER < 70%
- ✅ No uniform outputs

## Next Steps After Successful Overfitting

Once overfitting works at reasonable scale (20+ samples):

### 1. Gradual Scale-Up
```python
# Start with working configuration
config = get_working_config(20)  # From successful test

# Gradually increase dataset size
for n_samples in [50, 100, 200, 500, 1000]:
    train_with_curriculum(n_samples, config)
```

### 2. Add Regularization Carefully
```python
# Add one at a time, monitoring impact
regularization_schedule = [
    {'dropout': 0.1},
    {'dropout': 0.2},
    {'weight_decay': 1e-5},
    {'weight_decay': 1e-4},
    {'data_augmentation': True}
]
```

### 3. Optimize Learning Schedule
```python
# Implement warm-up + cosine annealing
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Restart every 10 epochs
    T_mult=2,  # Double period after restart
    eta_min=1e-6
)
```

### 4. Architecture Refinements
Only after basic learning works:
- Re-enable MobileNet blocks
- Add attention mechanisms
- Increase model capacity
- Implement skip connections

## Command Reference

```bash
# Quick test commands
python test_overfitting_clean.py --mode progressive  # Full test
python test_overfitting_clean.py --mode single --n_samples 2  # Quick check
python diagnose_uniform_output.py  # Diagnose issues
python overfit_test_systematic.py  # Comprehensive analysis

# Monitor training
tensorboard --logdir=logs/overfitting_tests

# Check results
cat overfitting_test_report.txt
```

## Expected Timeline

1. **Day 1**: Run progressive overfitting test (2-3 hours)
2. **Day 1**: Analyze results, identify breaking point (1 hour)
3. **Day 2**: Test fixes for identified issues (3-4 hours)
4. **Day 2**: Validate working configuration (2 hours)
5. **Day 3**: Scale up gradually to full dataset (4-6 hours)
6. **Day 4+**: Full training with optimized configuration

## Critical Insights

Based on the symptoms (uniform outputs, WER stuck at 93-100%):

1. **The model is NOT learning features** - it's outputting random guesses
2. **The architecture may be too complex** for the CTC objective
3. **Temporal downsampling (4x) might be losing critical information**
4. **The initialization might be inappropriate** for this architecture

## Files Created

1. `test_overfitting_clean.py` - Main overfitting test script
2. `overfit_test_systematic.py` - Comprehensive grid search
3. `diagnose_uniform_output.py` - Diagnostic tool for uniform outputs
4. `overfit_config.py` - Configuration library for different scales
5. `OVERFITTING_STRATEGY.md` - This strategy document

## Conclusion

The key is to start simple and gradually increase complexity. If the model can't overfit 5-10 samples with aggressive settings, there's a fundamental issue that needs to be fixed before attempting full training.

**Start with:** `python test_overfitting_clean.py --mode progressive`

This will quickly identify where the model breaks and guide the next steps.