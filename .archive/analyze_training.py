"""Quick analysis of current training."""
import json
from pathlib import Path

history_file = Path("checkpoints/student/mobilenet_v3_20251117_221723/training_history.json")
with open(history_file) as f:
    h = json.load(f)

print("="*70)
print("TRAINING ANALYSIS")
print("="*70)

epochs = len(h['train_losses'])
val_wers = h['val_wers']
train_losses = h['train_losses']
val_losses = h['val_losses']
lrs = h['learning_rates']

print(f"\n📊 Overall Progress:")
print(f"   Total epochs: {epochs}")
print(f"   Best WER: {min(val_wers):.2f}% (epoch {val_wers.index(min(val_wers))+1})")
print(f"   Current WER: {val_wers[-1]:.2f}%")
print(f"   Current LR: {lrs[-1]:.6f}")

print(f"\n📈 WER Progression:")
print(f"   Epoch 10:  {val_wers[9]:.1f}%")
print(f"   Epoch 50:  {val_wers[49]:.1f}%")
print(f"   Epoch 100: {val_wers[99]:.1f}%")
print(f"   Epoch 150: {val_wers[149]:.1f}%")
print(f"   Epoch 200: {val_wers[199]:.1f}%")
print(f"   Current:   {val_wers[-1]:.1f}%")

print(f"\n📉 Loss Progression:")
print(f"   Epoch 10:  Train={train_losses[9]:.2f}, Val={val_losses[9]:.2f}")
print(f"   Epoch 50:  Train={train_losses[49]:.2f}, Val={val_losses[49]:.2f}")
print(f"   Epoch 100: Train={train_losses[99]:.2f}, Val={val_losses[99]:.2f}")
print(f"   Current:   Train={train_losses[-1]:.2f}, Val={val_losses[-1]:.2f}")

print(f"\n🎓 Learning Rate Schedule:")
print(f"   Initial: {lrs[0]:.6f}")
print(f"   Epoch 50: {lrs[49]:.6f}")
print(f"   Epoch 100: {lrs[99]:.6f}")
print(f"   Current: {lrs[-1]:.6f}")
print(f"   Min LR hit at epoch: {next((i for i, lr in enumerate(lrs) if lr <= 1.1e-6), 'not yet')}")

print(f"\n⚠️  Train-Val Gap:")
gap = val_losses[-1] - train_losses[-1]
print(f"   Current: {gap:.2f} (Val - Train)")
print(f"   Status: {'SEVERE OVERFITTING' if gap > 1.5 else 'Moderate overfitting' if gap > 0.5 else 'Good'}")

print(f"\n🎯 Target Analysis:")
target_wer = 25.0
print(f"   Target WER: {target_wer}%")
print(f"   Current WER: {val_wers[-1]:.2f}%")
print(f"   Gap: {val_wers[-1] - target_wer:.2f}%")
print(f"   Status: {'FAR FROM TARGET' if val_wers[-1] > 50 else 'APPROACHING' if val_wers[-1] > 30 else 'CLOSE'}")

# Find when WER stopped improving significantly
improvements = [val_wers[i-1] - val_wers[i] for i in range(1, len(val_wers))]
last_significant = next((i for i in range(len(improvements)-1, 0, -1) 
                        if improvements[i] > 1.0), 0)
print(f"\n📌 Last significant WER improvement (>1%): Epoch {last_significant+1}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

print("\n🔴 Critical Issues:")
print("   1. LR hit minimum (1e-6) - can't improve further")
print("   2. Severe overfitting (val loss 2x train loss)")
print("   3. WER plateaued at ~81-82% (far from 25% target)")
print("   4. Model capacity issue or wrong optimization strategy")

print("\n💡 Root Cause:")
print("   The model learned to predict SOME patterns but:")
print("   - LR scheduler reduced too aggressively")
print("   - Dropout too low (0.1) → overfitting")
print("   - Decoder may still be too aggressive")
print("   - Or: Architecture can't reach <25% WER with current setup")

print("\n" + "="*70)

