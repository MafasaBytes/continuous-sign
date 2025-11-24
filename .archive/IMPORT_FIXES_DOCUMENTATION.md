# Module Import Fixes Documentation

## Issue Summary
The sign language recognition codebase encountered `ModuleNotFoundError` when trying to import modules from the `utils` directory while running scripts from the `teacher` subdirectory.

## Root Cause
Python's module resolution system couldn't find the `utils` module when scripts were executed directly from the `teacher` directory using `python teacher/train.py`. This occurred because:
1. Python adds the script's directory to `sys.path`, not the project root
2. Import statements were executed before `sys.path` modifications took effect
3. Nested imports (e.g., `teacher.loaders` importing `utils.vocabulary`) failed before path fixes could be applied

## Fixes Applied

### 1. Fixed Import Order in teacher/train.py
**File:** `C:\Users\Masia\OneDrive\Desktop\sign-language-recognition\teacher\train.py`
- Moved `sys.path.append()` to the top of the file, BEFORE any local imports
- This ensures the parent directory is in the path before any module imports are attempted

### 2. Added Path Fix to teacher/loaders.py
**File:** `C:\Users\Masia\OneDrive\Desktop\sign-language-recognition\teacher\loaders.py`
- Added `sys.path.append()` to include the parent directory
- Allows the module to import from `utils` when loaded by other modules

### 3. Fixed teacher/trainers/ctc_trainer.py
**File:** `C:\Users\Masia\OneDrive\Desktop\sign-language-recognition\teacher\trainers\ctc_trainer.py`
- Added proper path configuration with two parent directory levels
- Ensures utils can be imported from this nested module

### 4. Updated teacher/overfit_test.py
**File:** `C:\Users\Masia\OneDrive\Desktop\sign-language-recognition\teacher\overfit_test.py`
- Fixed import order (sys.path modification before imports)
- Updated model import from `teacher.model` to `teacher.models.efficient_hybrid`
- Updated function call from `create_teacher_model` to `create_efficient_model`

## Recommended Usage

### Option 1: Run as Module (RECOMMENDED)
Run scripts as Python modules from the project root directory:
```bash
# From project root directory:
python -m teacher.train --config configs/teacher/config.yaml
python -m teacher.overfit_test --config configs/teacher/config.yaml
```

### Option 2: Direct Execution
If you must run scripts directly, ensure you're in the project root:
```bash
# From project root directory:
python teacher/train.py --config configs/teacher/config.yaml
```

## Project Structure
```
sign-language-recognition/
├── configs/
│   └── teacher/
│       └── config.yaml
├── data/
│   ├── teacher_features/
│   │   └── mediapipe_pca1024/
│   ├── raw_data/
│   │   └── phoenix-2014-multisigner/
│   └── clean_vocabulary/
│       └── vocabulary.txt
├── teacher/
│   ├── models/
│   │   └── efficient_hybrid.py
│   ├── trainers/
│   │   └── ctc_trainer.py
│   ├── utils/
│   │   ├── decoding.py
│   │   ├── metrics.py
│   │   └── ...
│   ├── loaders.py
│   ├── train.py
│   └── overfit_test.py
└── utils/
    ├── vocabulary.py
    ├── ctc.py
    ├── metrics.py
    └── ...
```

## Import Pattern
All Python files that need to import from the root-level `utils` directory now follow this pattern:

```python
# Fix path BEFORE any local imports
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now safe to import from utils and teacher modules
from utils.vocabulary import Vocabulary
from teacher.models.efficient_hybrid import create_efficient_model
```

## Testing the Fix
To verify the fixes work correctly:

1. Test imports directly:
```bash
python -c "from teacher.loaders import create_dataloaders; print('Success')"
```

2. Test the training script:
```bash
python -m teacher.train --help
```

3. Run a quick overfit test:
```bash
python -m teacher.overfit_test --config configs/teacher/config.yaml --num-samples 5 --num-epochs 10
```

## Common Issues and Solutions

### Issue: ModuleNotFoundError when running from wrong directory
**Solution:** Always run scripts from the project root directory

### Issue: Import errors after moving files
**Solution:** Update the sys.path.append() call to reflect the new file location relative to project root

### Issue: Circular imports
**Solution:** Ensure sys.path modifications happen before any local imports

## Future Recommendations

1. Consider creating a proper package structure with `setup.py` or `pyproject.toml`
2. Use absolute imports consistently throughout the project
3. Create a unified entry point script at the project root
4. Consider using environment variables or configuration files for path management