# Length Normalization for CTC Decoding

## Problem Identified

Your model was consistently adding extra words (especially "MORGEN") at the end of predictions:

```
Target: 'MONTAG KOMMEN KUEHL REGEN UND GEWITTER KOENNEN'
Pred:   'MONTAG KOMMEN KUEHL REGEN UND GEWITTER KOENNEN MORGEN'  ❌

Target: 'HAUPTSAECHLICH SONNE'
Pred:   'HAUPTSAECHLICH SONNE MORGEN'  ❌
```

This is a classic CTC issue where the model learns to pad sequences with common words.

## Solution Implemented

Added **length-normalized decoding with confidence filtering** that:

1. ✅ **Tracks confidence scores** for each predicted word
2. ✅ **Applies length penalty** to discourage overly long sequences
3. ✅ **Filters low-confidence trailing words** (statistical outliers)
4. ✅ **Removes junk tokens** typically added at sequence end

## How It Works

### Length Normalization Formula
```
normalized_score = avg_confidence / (sequence_length ** length_penalty)
```

- **Higher penalty** (e.g., 0.8) → prefers shorter sequences
- **Lower penalty** (e.g., 0.3) → allows longer sequences
- **Standard value**: 0.6 (balanced)

### Confidence Filtering

1. **Absolute threshold**: Rejects tokens with log_prob < -8.0
2. **Statistical filtering**: 
   - Calculates mean and std of word confidence scores
   - Removes trailing words with score < (mean - 0.5 * std)
   - Stops at first low-confidence word

## Configuration

In `overfit_test.py`, adjust these parameters in the `decode_predictions()` function:

```python
def decode_predictions(log_probs: torch.Tensor, vocab_obj, debug: bool = False):
    return decode_predictions_with_length_norm(
        log_probs, 
        vocab_obj,
        length_penalty=0.6,        # ← Adjust this (0.3-1.0)
        confidence_threshold=-8.0  # ← Adjust this (-10.0 to -5.0)
    )
```

### Parameter Tuning Guide

#### `length_penalty` (default: 0.6)

| Value | Effect | Use When |
|-------|--------|----------|
| 0.3 | Very mild penalty | Model under-predicts (missing words) |
| 0.6 | Standard penalty | **Start here** (balanced) |
| 0.8 | Strong penalty | Model over-predicts (extra words) |
| 1.0 | Very strong penalty | Aggressive filtering needed |

#### `confidence_threshold` (default: -8.0)

| Value | Effect | Use When |
|-------|--------|----------|
| -10.0 | Very permissive | Keep most predictions |
| -8.0 | Standard | **Start here** (balanced) |
| -6.0 | Strict | Filter uncertain predictions |
| -5.0 | Very strict | Aggressive filtering |

## Expected Results

### Before Length Normalization
```
Sample 1: 'WOCHENENDE WETTER ... WENIG MORGEN'  ❌ Extra word
Sample 2: 'MONTAG KOMMEN ... KOENNEN MORGEN'    ❌ Extra word
Sample 3: 'HAUPTSAECHLICH SONNE MORGEN'         ❌ Extra word
WER: 10.81% (due to insertions)
```

### After Length Normalization
```
Sample 1: 'WOCHENENDE WETTER ... WENIG'         ✅ Exact match
Sample 2: 'MONTAG KOMMEN ... KOENNEN'           ✅ Exact match
Sample 3: 'HAUPTSAECHLICH SONNE'                ✅ Exact match
WER: 0-5% (only hard cases remain)
```

## Debug Mode

To see what's being filtered, enable debug mode:

```python
# In train_overfit_test(), add debug parameter:
predictions = decode_predictions(log_probs, vocab, debug=True)
```

This will print:
```
=== Decoding Comparison ===
Sample 0:
  Greedy:         'MONTAG KOMMEN KUEHL REGEN UND GEWITTER KOENNEN MORGEN'
  Length-normed:  'MONTAG KOMMEN KUEHL REGEN UND GEWITTER KOENNEN'
  Removed:        ' MORGEN'
```

## Advanced: Fine-Tuning Strategy

If you still see issues after applying length normalization:

### Issue: Still adding extra words
```python
# Increase length penalty
length_penalty=0.8  # or 1.0

# Or stricter confidence threshold
confidence_threshold=-6.0  # or -5.0
```

### Issue: Missing real words
```python
# Decrease length penalty
length_penalty=0.4  # or 0.3

# Or more permissive confidence threshold
confidence_threshold=-9.0  # or -10.0
```

### Issue: Inconsistent filtering
```python
# Adjust statistical filtering in decode_predictions_with_length_norm()
# Line 124: Change multiplier
threshold_score = mean_score - 0.3 * std_score  # Less aggressive (was 0.5)
threshold_score = mean_score - 0.7 * std_score  # More aggressive
```

## Alternative Decoders

The implementation includes three decoding strategies:

1. **`decode_predictions_greedy()`** - Simple, no filtering
   - Use for: Baseline comparison
   - Pros: Fast, no tuning needed
   - Cons: Can produce extra words

2. **`decode_predictions_with_length_norm()`** - Length normalized (default)
   - Use for: Production overfitting test
   - Pros: Filters junk tokens
   - Cons: Requires parameter tuning

3. **Custom decoder** - You can implement your own
   - Example: Beam search, prefix search, etc.

## Integration with Full Training

Once overfitting test passes, apply same decoding to full training:

```python
# In src/training/train.py, replace greedy decoder with:
from overfit_test import decode_predictions_with_length_norm

# In validation/test loops:
predictions = decode_predictions_with_length_norm(
    log_probs,
    vocabulary,
    length_penalty=0.6,
    confidence_threshold=-8.0
)
```

## Theoretical Background

### Why CTC Produces Extra Tokens

CTC loss doesn't explicitly penalize sequence length, so models can:
1. Learn to repeat common words ("MORGEN")
2. Pad sequences to match training patterns
3. Insert high-frequency tokens as "filler"

### Length Normalization Theory

Based on:
- **Google NMT** (Wu et al., 2016): `score / length^α` where α=0.6
- **CTC beam search**: Common in speech recognition
- **Statistical filtering**: Novel approach for word-level CTC

### Trade-offs

| Metric | No Normalization | With Normalization |
|--------|-----------------|-------------------|
| Precision | Lower (false positives) | Higher (fewer insertions) |
| Recall | Higher (catches all) | Slightly lower (may miss some) |
| F1 Score | Moderate | **Higher** (better balance) |
| WER | 10-15% | **5-10%** (target) |

## Testing Your Changes

After updating decoding:

```bash
# Run overfitting test with new decoder
python overfit_test.py
```

Look for:
- ✅ Reduced insertions (extra "MORGEN" gone)
- ✅ Lower WER (target: < 5%)
- ✅ Exact matches on simple sequences
- ⚠️ Check for deletions (missing words)

## Benchmark

Expected improvement:
```
Before: WER = 10.81% (mostly insertions)
After:  WER = 3-5%   (close to perfect)
```

---

**Implementation Date**: November 17, 2025  
**Issue**: CTC inserting extra words ("MORGEN")  
**Solution**: Length-normalized decoding with confidence filtering  
**Status**: Ready for testing

