# Overfitting Test for MobileNetV3 Sign Language Model

## Purpose

This test validates that the MobileNetV3 architecture has the capacity to learn by attempting to memorize a tiny dataset (3 samples). If a model cannot overfit to a small dataset, it indicates fundamental issues with the architecture, training setup, or data pipeline.

## What It Tests

✅ **Model Capacity**: Can the architecture learn patterns?  
✅ **Training Setup**: Are the loss function, optimizer, and gradients working correctly?  
✅ **Data Pipeline**: Are features and labels properly formatted?  
✅ **WER Reduction**: Can the model reduce Word Error Rate to near zero?  

## Success Criteria

- **Loss**: Should drop below 0.5 (ideally close to 0)
- **WER**: Should drop below 5% (ideally close to 0%)
- **Convergence**: Should show consistent improvement over 500 epochs

## How to Run

### Prerequisites

```bash
# Ensure you have processed data
python data_preprocessing.py

# Ensure all dependencies are installed
pip install torch numpy matplotlib tqdm
```

### Run the Test

```bash
python overfit_test.py
```

### Configuration

You can modify these parameters in the `main()` function:

```python
num_samples = 3          # Number of samples to overfit on (default: 3)
num_epochs = 500         # Training epochs (default: 500)
learning_rate = 0.001    # Learning rate (default: 0.001)
```

## Output Files

After running, you'll get:

1. **`overfit_test_results.png`**: Plots showing Loss and WER over training
2. **`overfit_test_report.txt`**: Detailed text report with:
   - Configuration parameters
   - Final metrics
   - Pass/Fail status
   - Sample predictions at key epochs

## Interpreting Results

### ✅ PASSED (Good Signs)

```
Minimum Loss: < 0.5
Minimum WER: < 5%
Predictions match targets exactly
```

**Meaning**: The architecture is capable of learning. Any issues with full training are likely due to:
- Need for more training time
- Hyperparameter tuning
- Regularization settings
- Data augmentation

### ⚠️ FAILED (Troubleshooting Needed)

```
Loss not decreasing or stuck at high value
WER not improving or stuck at 100%
```

**Possible Issues**:
1. **Architecture Problems**
   - Check model forward pass
   - Verify dimensions match throughout
   - Check for vanishing/exploding gradients

2. **Loss Function Issues**
   - CTC loss requirements not met
   - Input/target length mismatch
   - Blank token index incorrect

3. **Data Pipeline**
   - Features not normalized correctly
   - Labels not encoded properly
   - Sequence lengths incorrect

4. **Optimization Issues**
   - Learning rate too high/low
   - Gradient clipping too aggressive
   - Weight initialization problems

## Example Output

### Successful Run

```
Epoch [500/500] | Loss: 0.024123 (Best: 0.024123) | WER: 0.00% (Best: 0.00%)

Sample Predictions (Epoch 500):
  Sample train_00123:
    Target: 'HELLO'
    Pred:   'HELLO'
    Match:  True

✓ PASSED - The model can learn and memorize the training data.
```

### Failed Run

```
Epoch [500/500] | Loss: 8.456789 (Best: 8.234567) | WER: 100.00% (Best: 95.23%)

Sample Predictions (Epoch 500):
  Sample train_00123:
    Target: 'HELLO'
    Pred:   'HXXXH'
    Match:  False

✗ FAILED - The model failed to overfit to the training data.
```

## Next Steps

### If Test Passes ✅
- Proceed with full training on the complete dataset
- Tune hyperparameters for generalization
- Add regularization and data augmentation
- Monitor validation WER

### If Test Fails ❌
1. Check model architecture dimensions
2. Verify CTC loss setup
3. Inspect data preprocessing
4. Test with even simpler data (1 sample)
5. Add debugging prints for gradient flow
6. Verify feature extraction is working

## Troubleshooting Commands

```bash
# Check data exists
ls data/processed/train/*_features.npy | head -5

# Verify vocab file
cat data/processed/train/vocab.json

# Test model creation only
python -c "from src.models.mobilenet_v3 import create_mobilenet_v3_model; model = create_mobilenet_v3_model(1232); print('Model created successfully')"
```

## Related Files

- **Model**: `src/models/mobilenet_v3.py`
- **Metrics**: `src/utils/metrics.py`
- **Training**: `src/training/train.py`
- **Data**: `data/processed/train/`

## Tips

1. **Start Small**: Begin with 1-2 samples if 3 samples don't converge
2. **Increase Epochs**: Try 1000 epochs if 500 isn't enough
3. **Lower Dropout**: The script uses 0.1 dropout (lower than production)
4. **Monitor Gradients**: Add gradient norm logging if suspecting vanishing gradients
5. **Check Predictions**: Even if WER is high, check if predictions show any structure

## Contact

If the test consistently fails after troubleshooting, review:
- Feature dimensionality in model vs. data
- CTC blank token configuration
- Sequence length handling in forward pass

