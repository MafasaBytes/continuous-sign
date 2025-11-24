# WER Argument Order Fix

## Issue Found

The `compute_wer()` and `compute_ser()` functions were being called with **swapped arguments** in the teacher and distillation training scripts.

## Correct Order

According to `src/utils/metrics.py`:

```python
def compute_wer(references, hypotheses):
    """
    Args:
        references: Ground truth / targets
        hypotheses: Model predictions
    """
```

**Correct order:** `compute_wer(targets, predictions)`

## Files Fixed

### 1. `src/training/train_teacher.py` (Line 394-395)

**Before (WRONG):**
```python
wer = compute_wer(all_predictions, all_targets)  # ❌ Swapped!
ser = compute_ser(all_predictions, all_targets)  # ❌ Swapped!
```

**After (CORRECT):**
```python
wer = compute_wer(all_targets, all_predictions)  # ✅ Fixed
ser = compute_ser(all_targets, all_predictions)  # ✅ Fixed
```

### 2. `src/training/train_distillation.py` (Lines 297-300)

**Before (WRONG):**
```python
student_wer = compute_wer(student_predictions, all_targets)  # ❌ Swapped!
teacher_wer = compute_wer(teacher_predictions, all_targets)  # ❌ Swapped!
student_ser = compute_ser(student_predictions, all_targets)  # ❌ Swapped!
teacher_ser = compute_ser(teacher_predictions, all_targets)  # ❌ Swapped!
```

**After (CORRECT):**
```python
student_wer = compute_wer(all_targets, student_predictions)  # ✅ Fixed
teacher_wer = compute_wer(all_targets, teacher_predictions)  # ✅ Fixed
student_ser = compute_ser(all_targets, student_predictions)  # ✅ Fixed
teacher_ser = compute_ser(all_targets, teacher_predictions)  # ✅ Fixed
```

### 3. `src/training/train.py` ✅

Already correct (no changes needed):
```python
wer = compute_wer(all_targets, all_predictions)  # ✅ Correct
```

### 4. `overfit_test_teacher.py` ✅

Already correct (no changes needed):
```python
wer = compute_wer(epoch_targets, epoch_predictions)  # ✅ Correct
```

## Impact

### Before Fix
The WER/SER values would be **incorrect** because the function was comparing:
- Predictions as references (wrong baseline)
- Targets as hypotheses (wrong comparison)

This could lead to:
- Misleading WER/SER scores
- Incorrect model evaluation
- Wrong decisions about which checkpoint is best

### After Fix
The WER/SER values will now be **correct**:
- Targets as references (correct baseline)
- Predictions as hypotheses (correct comparison)

## Verification

To verify the fix is working correctly, compare WER values before and after:

**Expected behavior:**
- WER should be meaningful relative to model performance
- WER should decrease as training progresses
- WER should match expectations (e.g., 20-30% for teacher after 100 epochs)

**If you see WER > 100%:**
- This was likely happening with the swapped arguments
- Now it should be fixed

## Summary

| Script | Status | Lines Changed |
|--------|--------|---------------|
| `train_teacher.py` | ✅ Fixed | 394-395 |
| `train_distillation.py` | ✅ Fixed | 297-300 |
| `train.py` | ✅ Already correct | - |
| `overfit_test_teacher.py` | ✅ Already correct | - |

---

**Status:** ✅ **FIXED**  
**All WER/SER computations now use correct argument order**

