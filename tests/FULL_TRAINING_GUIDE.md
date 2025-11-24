# Full Training Guide - I3D Teacher Model

## 🎉 Overfitting Test Results

**SUCCESS!** Achieved 0% WER at epoch 1712
- Final Loss: 0.009003
- Perfect memorization on 5 samples
- Model architecture validated ✅

---

## 🚀 Transition from Overfitting to Full Training

### Key Changes Needed

| Parameter | Overfitting Test | Full Training | Reasoning |
|-----------|-----------------|---------------|-----------|
| **Dropout** | 0.1 | 0.3-0.4 | Need regularization for generalization |
| **Learning Rate** | 0.001 | 0.0003-0.0005 | More conservative for large dataset |
| **Batch Size** | 5 (all samples) | 8-16 | Balance memory and gradient quality |
| **Epochs** | 2000 | 100-200 | More data = faster convergence per epoch |
| **Warmup** | 50 epochs | 5-10 epochs | Proportional to total epochs |
| **Weight Decay** | 0.0 | 0.0001-0.001 | L2 regularization for generalization |
| **Data Augmentation** | False | True | Improve robustness |
| **Gradient Accumulation** | 1 | 2-4 | Simulate larger batch sizes |

---

## 📋 Recommended Training Configurations

### Configuration 1: Conservative (Recommended Start)
```python
# Training hyperparameters
batch_size = 8
num_epochs = 150
learning_rate = 0.0003
warmup_epochs = 5
dropout = 0.3

# Optimizer
optimizer = 'adamw'
weight_decay = 0.0001
betas = (0.9, 0.999)

# Regularization
gradient_clip = 1.0
label_smoothing = 0.0
gradient_accumulation_steps = 2  # Effective batch size = 16

# Data
augment = True
num_workers = 4

# Learning rate schedule
scheduler = 'cosine'  # Cosine annealing after warmup
min_lr = 1e-6
```

**Expected Results:**
- WER: 25-35% (first run, baseline)
- Training time: ~10-15 hours on single GPU
- Model convergence: ~80-100 epochs

---

### Configuration 2: Aggressive (If baseline is too slow)
```python
# Training hyperparameters
batch_size = 12
num_epochs = 100
learning_rate = 0.0005
warmup_epochs = 3
dropout = 0.25

# Optimizer
optimizer = 'adam'
weight_decay = 0.0001

# Regularization
gradient_clip = 0.5
gradient_accumulation_steps = 1

# Learning rate schedule
scheduler = 'cosine'
min_lr = 1e-6
```

**Expected Results:**
- Faster convergence
- Potentially higher WER (30-40%) due to less regularization
- Good for initial experiments

---

### Configuration 3: Heavy Regularization (If overfitting)
```python
# Training hyperparameters
batch_size = 8
num_epochs = 200
learning_rate = 0.0002
warmup_epochs = 10
dropout = 0.4

# Optimizer
optimizer = 'adamw'
weight_decay = 0.001

# Regularization
gradient_clip = 1.0
label_smoothing = 0.1
gradient_accumulation_steps = 2

# Data
augment = True
mixup_alpha = 0.2  # Optional: temporal mixup

# Learning rate schedule
scheduler = 'cosine_with_restarts'
min_lr = 1e-6
```

**Expected Results:**
- Best generalization
- Lower training WER, better validation WER
- Slower convergence but more stable

---

## 🔧 Training Script Modifications

### 1. Add Validation Loop
```python
def validate(model, val_loader, vocab, device):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            feature_lengths = batch['input_lengths'].to(device)
            label_lengths = batch['target_lengths'].to(device)
            
            log_probs = model(features, feature_lengths)
            loss = ctc_loss(log_probs, labels, feature_lengths, label_lengths)
            
            total_loss += loss.item()
            predictions = decode_predictions(log_probs, vocab)
            targets = [' '.join(words) for words in batch['words']]
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    avg_loss = total_loss / len(val_loader)
    wer = compute_wer(all_targets, all_predictions)
    
    return avg_loss, wer, all_predictions, all_targets
```

### 2. Add Learning Rate Scheduler
```python
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# After warmup, use cosine annealing
scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs - warmup_epochs,
    eta_min=1e-6
)
```

### 3. Add Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

### 4. Add Gradient Accumulation
```python
optimizer.zero_grad()
for i, batch in enumerate(train_loader):
    # Forward pass
    log_probs = model(features, feature_lengths)
    loss = ctc_loss(...) / gradient_accumulation_steps
    
    # Backward pass
    loss.backward()
    
    # Update weights every N steps
    if (i + 1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()
        optimizer.zero_grad()
```

### 5. Add Checkpoint Saving
```python
def save_checkpoint(model, optimizer, epoch, best_wer, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_wer': best_wer,
        'vocab_size': len(vocab),
    }
    torch.save(checkpoint, save_path)
    
# Save best model
if val_wer < best_wer:
    best_wer = val_wer
    save_checkpoint(
        model, optimizer, epoch, best_wer,
        'checkpoints/teacher_best.pt'
    )
```

---

## 📊 Monitoring Strategy

### Metrics to Track

1. **Training Loss** - Should decrease smoothly
2. **Validation Loss** - Monitor for overfitting (gap with training)
3. **Training WER** - Can be lower due to overfitting
4. **Validation WER** - Primary metric for model selection
5. **Learning Rate** - Track schedule progression
6. **Gradient Norm** - Monitor for instability

### Logging Strategy
```python
# Log every N batches
if batch_idx % 50 == 0:
    print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
          f"Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

# Validate every epoch
val_loss, val_wer, _, _ = validate(model, val_loader, vocab, device)
print(f"Epoch {epoch}: Val Loss: {val_loss:.4f} | Val WER: {val_wer:.2f}%")

# Save metrics
wandb.log({
    'train/loss': train_loss,
    'train/wer': train_wer,
    'val/loss': val_loss,
    'val/wer': val_wer,
    'lr': current_lr
})
```

---

## 🎯 Target Performance Metrics

Based on your research proposal and PHOENIX-2014 dataset:

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| **Validation WER** | < 25% | < 20% |
| **Model Size** | ~50-100 MB | N/A (teacher can be large) |
| **Training Time** | < 24 hours | < 12 hours |
| **Convergence** | < 100 epochs | < 80 epochs |

### Comparison with Literature
- Koller et al. (2015) CNN-HMM: ~40% WER
- Camgöz et al. (2018) Neural Translation: ~30% WER
- Your target: < 25% WER

---

## 🔄 Training Workflow

### Phase 1: Initial Training (Days 1-2)
```bash
# Start with conservative config
python src/training/train.py \
  --config configs/teacher_conservative.yaml \
  --batch_size 8 \
  --learning_rate 0.0003 \
  --epochs 150
```

**Goals:**
- Establish baseline WER
- Verify training stability
- Monitor for any issues

---

### Phase 2: Hyperparameter Tuning (Days 3-5)
```bash
# Try different configurations
python src/training/train.py --config configs/teacher_aggressive.yaml
python src/training/train.py --config configs/teacher_regularized.yaml

# Experiment with:
# - Learning rates: [0.0001, 0.0003, 0.0005]
# - Dropout: [0.2, 0.3, 0.4]
# - Batch sizes: [8, 12, 16]
```

**Goals:**
- Find optimal hyperparameters
- Achieve < 25% validation WER
- Select best checkpoint

---

### Phase 3: Fine-tuning (Days 6-7)
```bash
# Load best checkpoint and fine-tune
python src/training/train.py \
  --resume checkpoints/teacher_best.pt \
  --learning_rate 0.00005 \
  --epochs 50
```

**Goals:**
- Squeeze out final performance gains
- Prepare for knowledge distillation

---

## 🎓 Knowledge Distillation Preparation

Once teacher achieves < 25% WER:

### 1. Extract Soft Targets
```python
# Save teacher predictions for training set
def extract_soft_targets(teacher, dataloader, save_path):
    teacher.eval()
    soft_targets = {}
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            feature_lengths = batch['input_lengths'].to(device)
            
            # Get soft predictions (with temperature)
            logits = teacher(features, feature_lengths)
            soft_probs = F.softmax(logits / temperature, dim=-1)
            
            for i, video_id in enumerate(batch['video_ids']):
                soft_targets[video_id] = soft_probs[:, i, :].cpu()
    
    torch.save(soft_targets, save_path)
```

### 2. Distillation Loss
```python
def distillation_loss(student_logits, teacher_logits, targets, 
                     alpha=0.7, temperature=3.0):
    # Soft target loss (KL divergence)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard target loss (CTC)
    hard_loss = ctc_loss(student_logits, targets, ...)
    
    # Combined loss
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### 3. Student Training Strategy
```python
# Configuration for student (MobileNetV3)
student_config = {
    'learning_rate': 0.001,  # Higher than teacher
    'dropout': 0.3,
    'alpha': 0.7,  # Distillation weight
    'temperature': 3.0,
    'epochs': 200,
    'target_wer': teacher_wer * 1.1  # Allow 10% degradation
}
```

---

## 🐛 Common Issues and Solutions

### Issue 1: Validation WER not improving
**Solutions:**
- Lower learning rate
- Increase regularization (dropout, weight_decay)
- More data augmentation
- Train longer (patience)

### Issue 2: Training loss plateaus
**Solutions:**
- Increase learning rate
- Reduce regularization
- Check for gradient clipping too aggressive
- Verify data quality

### Issue 3: Overfitting (train WER << val WER)
**Solutions:**
- Increase dropout to 0.4-0.5
- Add label smoothing
- More aggressive data augmentation
- Early stopping

### Issue 4: OOM (Out of Memory)
**Solutions:**
- Reduce batch size
- Increase gradient accumulation steps
- Use mixed precision training (FP16)
- Reduce sequence length (truncate)

---

## 📈 Expected Training Curves

### Healthy Training
```
Epoch    Train Loss    Val Loss    Train WER    Val WER
  1        15.234       16.123       95.3%       97.8%
 10         5.432        6.234       65.2%       72.1%
 20         2.345        3.456       42.1%       48.5%
 40         1.234        2.123       28.5%       35.2%
 60         0.892        1.789       22.3%       29.8%
 80         0.654        1.567       18.9%       26.4%
100         0.543        1.456       16.2%       24.7%  ← Best
120         0.478        1.523       14.8%       25.1%  ← Overfitting
```

### Red Flags
- Val loss increasing while train loss decreasing → Overfitting
- Both losses plateauing early → Learning rate too low
- Loss becomes NaN → Learning rate too high or gradient explosion

---

## ✅ Checklist Before Full Training

- [x] Overfitting test passed (0% WER achieved)
- [ ] Training script updated with validation loop
- [ ] Data loaders configured (train/val/test splits)
- [ ] Learning rate scheduler implemented
- [ ] Checkpoint saving configured
- [ ] Logging system set up (TensorBoard/Wandb)
- [ ] GPU memory profiled
- [ ] Backup strategy for checkpoints
- [ ] Monitoring alerts configured

---

## 🎯 Success Criteria

### Minimum Viable Performance
- ✅ Validation WER < 30%
- ✅ Training completes without crashes
- ✅ Model checkpoints saved

### Target Performance (Research Proposal)
- 🎯 Validation WER < 25%
- 🎯 Stable training curves
- 🎯 Ready for knowledge distillation

### Stretch Goals
- 🌟 Validation WER < 20%
- 🌟 Beats published baselines
- 🌟 Student model within 5% of teacher

---

## 🚀 Quick Start Commands

```bash
# 1. Verify data is ready
ls data/teacher_features/mediapipe_full/train/*.npz | wc -l
# Should show ~6000+ files

# 2. Start training with recommended config
python src/training/train.py \
  --model teacher \
  --batch_size 8 \
  --learning_rate 0.0003 \
  --dropout 0.3 \
  --epochs 150 \
  --warmup_epochs 5

# 3. Monitor training
tensorboard --logdir runs/

# 4. Evaluate best checkpoint
python src/training/evaluate.py \
  --checkpoint checkpoints/teacher_best.pt \
  --split test
```

---

Good luck with full training! Your model is ready to learn! 🎉

