# Adaptive Decoding Configuration Guide

## Problem Solved

**Issue**: Length normalization was:
- ✅ Working well on **short sequences** (removing extra "MORGEN")
- ❌ Too aggressive on **long sequences** (clipping valid words)

**Solution**: Adaptive thresholds that scale with sequence length

## How Adaptive Thresholds Work

The decoder now uses **length-aware filtering**:

```python
Short sequences (<= 5 words):   std_multiplier = 0.5  # Aggressive filtering
Medium sequences (5-15 words):  std_multiplier = 0.3  # Moderate filtering  
Long sequences (> 15 words):    std_multiplier = 0.2  # Gentle filtering
```

### Why This Works

**Short sequences** (e.g., "HAUPTSAECHLICH SONNE"):
- Few words = easier to identify outliers
- Aggressive filtering safely removes "MORGEN"
- Low risk of removing valid words

**Long sequences** (e.g., 15+ word weather reports):
- More variance in confidence scores
- Gentle filtering preserves valid low-confidence words
- Prevents premature truncation

## Configuration Parameters

Located in `decode_predictions()` function:

```python
config = {
    'length_penalty': 0.6,           # Overall length preference
    'confidence_threshold': -8.0,    # Absolute minimum confidence
    'min_sequence_length': 2,        # Protected words (never filter)
    'max_sequence_length': 50,       # Hard cap (prevent runaway)
    'adaptive_threshold': True       # Enable length-aware filtering
}
```

### Parameter Details

#### 1. `min_sequence_length` (default: 2)
**What it does**: Always keeps first N words, no matter what

| Value | Effect | Use When |
|-------|--------|----------|
| 1 | Minimal protection | Very confident in filtering |
| 2 | Standard | **Start here** (balanced) |
| 3 | Conservative | Short sequences being over-filtered |
| 5 | Very safe | Want to ensure core words survive |

**Example**:
```python
Sequence: "HEUTE MORGEN REGEN WIRD KOMMEN"
min_sequence_length: 2
→ "HEUTE MORGEN" are guaranteed, others evaluated
```

#### 2. `max_sequence_length` (default: 50)
**What it does**: Hard cap on output length (safety measure)

| Value | Effect | Use When |
|-------|--------|----------|
| 30 | Strict cap | Prevent runaway predictions |
| 50 | Standard | **Start here** (typical weather reports) |
| 100 | Permissive | Dataset has very long sequences |
| None | No limit | Trust model completely (risky) |

**Note**: Phoenix dataset typically has 5-20 words per sequence

#### 3. `confidence_threshold` (default: -8.0)
**What it does**: Absolute minimum log probability to accept a word

| Value | Effect | Impact |
|-------|--------|--------|
| -10.0 | Very permissive | Keep almost everything |
| -8.0 | Standard | **Start here** (balanced) |
| -6.0 | Strict | Filter uncertain words |
| -5.0 | Very strict | Only high-confidence words |

**Guidance**:
- Model outputting nonsense? → Lower to -6.0
- Clipping valid words? → Raise to -9.0 or -10.0

#### 4. `adaptive_threshold` (default: True)
**What it does**: Enables length-aware filtering multipliers

| Value | Behavior |
|-------|----------|
| True | Different thresholds for short/medium/long sequences |
| False | Same threshold for all (0.5 * std) |

**When to disable**:
- You want consistent behavior across all lengths
- Debugging (simpler to reason about)
- Very uniform sequence lengths in dataset

#### 5. `length_penalty` (default: 0.6)
**What it does**: Global bias toward shorter/longer sequences

| Value | Effect | Use When |
|-------|--------|----------|
| 0.3 | Weak penalty | Model under-predicts |
| 0.6 | Standard | **Start here** (balanced) |
| 0.8 | Strong penalty | Model over-predicts consistently |
| 1.0 | Very strong | Aggressive length control needed |

**Note**: With adaptive thresholds, this has less impact but still useful

## Tuning Strategy

### Step 1: Identify Your Issue

**Problem A: Short sequences getting extra words**
```
Target: "HAUPTSAECHLICH SONNE"
Pred:   "HAUPTSAECHLICH SONNE MORGEN"  ❌
```

**Fix**:
```python
# Already handled by adaptive thresholds!
# But if still happening:
min_sequence_length: 1  # Less protection
confidence_threshold: -6.0  # Stricter filtering
```

**Problem B: Long sequences being clipped**
```
Target: "WOCHENENDE WETTER WECHSELHAFT ABER NICHT-GEWITTER NICHT-MEHR KUEHL BLEIBEN AUCH MONTAG AB MEHR WOLKE SONNE WENIG"
Pred:   "WOCHENENDE WETTER WECHSELHAFT ABER NICHT-GEWITTER NICHT-MEHR KUEHL BLEIBEN"  ❌ (truncated)
```

**Fix**:
```python
min_sequence_length: 3  # More protection
confidence_threshold: -9.0  # More permissive
max_sequence_length: 100  # Higher cap (if needed)
```

**Problem C: Inconsistent - some good, some bad**
```
Sample 1: ✅ Perfect
Sample 2: ❌ Extra word
Sample 3: ❌ Missing words
```

**Fix**: Use debug mode to see patterns
```python
# Enable debug in training loop
predictions = decode_predictions(log_probs, vocab, debug=True)
```

### Step 2: Adjust Adaptive Thresholds (Advanced)

If default ranges don't work, modify in the decoder:

```python
# In decode_predictions_with_length_norm(), lines 119-130

# Current (default):
if len(words) <= 5:
    std_multiplier = 0.5  # Aggressive
elif len(words) <= 15:
    std_multiplier = 0.3  # Moderate
else:
    std_multiplier = 0.2  # Gentle

# If short sequences still have insertions:
if len(words) <= 5:
    std_multiplier = 0.7  # More aggressive
elif len(words) <= 15:
    std_multiplier = 0.4  # Moderate
else:
    std_multiplier = 0.2  # Gentle

# If long sequences being clipped:
if len(words) <= 5:
    std_multiplier = 0.5  # Aggressive
elif len(words) <= 15:
    std_multiplier = 0.2  # More lenient
else:
    std_multiplier = 0.1  # Very gentle
```

### Step 3: Test and Iterate

```bash
# Run with current config
python overfit_test.py

# Check predictions in output
# Look for patterns:
# - Are short sequences correct?
# - Are long sequences complete?
# - What's the WER breakdown by length?
```

## Common Scenarios

### Scenario 1: "Working perfectly on 4/5 samples, but one long sequence is clipped"

**Solution**: Increase protection for longer sequences
```python
config = {
    'min_sequence_length': 3,      # Up from 2
    'confidence_threshold': -9.0,  # Up from -8.0
    'adaptive_threshold': True,     # Keep enabled
}
```

### Scenario 2: "Short sequences perfect, all long sequences have extra words at end"

**Solution**: Make long-sequence filtering stricter
```python
# Modify std_multiplier for long sequences
else:  # len(words) > 15
    std_multiplier = 0.3  # Up from 0.2 (more aggressive)
```

### Scenario 3: "Completely random - can't find pattern"

**Solution**: Disable adaptive and use fixed threshold
```python
config = {
    'adaptive_threshold': False,    # Disable
    'confidence_threshold': -7.0,   # Tune this single parameter
}
```

### Scenario 4: "All predictions are perfect length but WER still high"

**Issue**: Not a length problem - might be:
- Word confusion (e.g., SECHZEHN vs SIEBZEHN)
- Model capacity issue
- Need more training epochs

**Action**: Keep current decoder, focus on model training

## Quick Reference: Config by Sequence Length

### Your Dataset has SHORT sequences (2-8 words)

```python
config = {
    'min_sequence_length': 1,
    'max_sequence_length': 20,
    'confidence_threshold': -7.0,
    'adaptive_threshold': True,  # Will use aggressive filtering
}
```

### Your Dataset has MEDIUM sequences (5-15 words)

```python
config = {
    'min_sequence_length': 2,
    'max_sequence_length': 30,
    'confidence_threshold': -8.0,
    'adaptive_threshold': True,  # Balanced
}
# ← This is the default!
```

### Your Dataset has LONG sequences (15+ words)

```python
config = {
    'min_sequence_length': 3,
    'max_sequence_length': 100,
    'confidence_threshold': -9.0,
    'adaptive_threshold': True,  # Will use gentle filtering
}
```

### Your Dataset has MIXED lengths (2-20 words)

```python
config = {
    'min_sequence_length': 2,
    'max_sequence_length': 50,
    'confidence_threshold': -8.0,
    'adaptive_threshold': True,  # Handles all lengths
}
# ← This is the default! (Phoenix dataset)
```

## Debug Mode Output

Enable debug to see what's happening:

```python
# In train_overfit_test(), find the decode call (around line 300)
predictions = decode_predictions(log_probs, vocab, debug=True)
```

**Output example**:
```
=== Decoding Comparison ===
Sample 0:
  Greedy (16 words): 'WOCHENENDE WETTER ... WENIG MORGEN'
  Adaptive (15 words): 'WOCHENENDE WETTER ... WENIG'
  Removed: 'MORGEN'

Sample 2:
  Greedy (7 words): 'MONTAG KOMMEN KUEHL REGEN UND GEWITTER KOENNEN'
  Adaptive (7 words): 'MONTAG KOMMEN KUEHL REGEN UND GEWITTER KOENNEN'
  (no change - good!)
```

**What to look for**:
- ✅ Short sequences: Should show "Removed: MORGEN" or similar
- ✅ Long sequences: Should show "(no change)" or minimal removals
- ❌ Long sequences being clipped: Shows "Removed: valid words" → increase `min_sequence_length`

## Performance Impact

**Computational cost**: Negligible
- Adaptive thresholds: Simple if-else logic
- Confidence filtering: Already doing stats calculation
- ~0.1ms per sequence added

**Memory**: None (no additional tensors)

**Accuracy improvement expected**:
```
Before: WER = 10.81% (insertion errors)
After:  WER = 3-7%   (insertions removed, long sequences preserved)
```

## Integration with Full Training

Once tuned on overfitting test, copy config to main training:

```python
# In src/training/train.py
from overfit_test import decode_predictions_with_length_norm

# Replace greedy decoder with:
config = {
    'min_sequence_length': 2,      # From tuning
    'max_sequence_length': 50,
    'confidence_threshold': -8.0,
    'adaptive_threshold': True,
}

predictions = decode_predictions_with_length_norm(
    log_probs, vocabulary, **config
)
```

---

**Last Updated**: November 17, 2025  
**Issue**: Length normalization clipping long sequences  
**Solution**: Adaptive thresholds based on sequence length  
**Status**: Ready for testing

