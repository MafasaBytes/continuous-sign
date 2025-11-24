"""
Minimal training test to verify model can overfit on small batch.
This will help identify if the issue is with the model or training config.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.models.mobilenet_v3 import MobileNetV3SignLanguage
from src.data.dataset import MediaPipeFeatureDataset, build_vocabulary, collate_fn
from torch.utils.data import DataLoader

def test_overfit_single_batch():
    """Test if model can overfit on a single batch."""

    print("="*60)
    print("MINIMAL OVERFITTING TEST")
    print("="*60)

    # Build vocabulary
    annotation_file = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    vocab = build_vocabulary(annotation_file)
    print(f"Vocabulary size: {len(vocab)}")

    # Create dataset (only 1 sample for overfitting test)
    dataset = MediaPipeFeatureDataset(
        data_dir=Path("data/teacher_features/mediapipe_full"),
        annotation_file=annotation_file,
        vocabulary=vocab,
        split='train',
        augment=False,
        normalize=False
    )

    # Use only first 2 samples for quick overfitting test
    subset_indices = list(range(min(2, len(dataset))))
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

    # Create dataloader
    dataloader = DataLoader(
        subset_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False
    )

    # Get one batch
    batch = next(iter(dataloader))
    features = batch['features']
    labels = batch['labels']
    input_lengths = batch['input_lengths']
    label_lengths = batch['target_lengths']  # Fixed: was 'label_lengths'

    print(f"\nBatch info:")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Input lengths: {input_lengths}")
    print(f"  Label lengths: {label_lengths}")

    # Create model
    model = MobileNetV3SignLanguage(vocab_size=len(vocab), dropout=0.0)  # No dropout for overfitting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nDevice: {device}")

    # Move batch to device
    features = features.to(device)
    labels = labels.to(device)
    input_lengths = input_lengths.to(device)
    label_lengths = label_lengths.to(device)

    # Setup training
    criterion = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("\nTraining on single batch (overfitting test):")
    print("-" * 40)

    losses = []
    for epoch in range(50):  # 50 epochs should be enough to overfit 2 samples
        # Forward pass
        output = model(features, input_lengths)  # [T, B, vocab_size]

        # Check output
        if epoch == 0:
            print(f"\nFirst epoch output shape: {output.shape}")
            print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")

        # Compute CTC loss
        loss = criterion(output, labels, input_lengths, label_lengths)

        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[ERROR] Invalid loss at epoch {epoch}: {loss.item()}")
            print(f"  Input lengths: {input_lengths}")
            print(f"  Label lengths: {label_lengths}")
            print(f"  Output shape: {output.shape}")
            break

        losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        optimizer.step()

        # Print progress
        if epoch % 10 == 0 or epoch == 49:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}, Grad Norm = {total_grad_norm:.4f}")

            # Decode predictions for first sample
            with torch.no_grad():
                # Get predictions
                probs = output[:input_lengths[0], 0, :]  # First sample
                predicted = torch.argmax(probs, dim=-1).cpu().numpy()

                # Simple greedy decoding (remove consecutive duplicates and blanks)
                decoded = []
                prev = -1
                for p in predicted:
                    if p != prev and p != vocab.blank_id:
                        decoded.append(p)
                    prev = p

                # Convert to words
                predicted_words = vocab.indices_to_words(decoded)[:5]  # First 5 words
                actual_words = batch['words'][0][:5]

                print(f"  Predicted (first 5): {predicted_words}")
                print(f"  Actual (first 5): {actual_words}")

    # Check if model learned anything
    print("\n" + "="*60)
    if len(losses) >= 2:
        initial_loss = losses[0]
        final_loss = losses[-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100

        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Improvement: {improvement:.2f}%")

        if improvement < 10:
            print("\n[WARNING] Model is NOT learning!")
            print("Possible issues:")
            print("  1. Learning rate too high/low")
            print("  2. Model initialization problems")
            print("  3. Feature preprocessing issues")
            print("  4. CTC loss computation problems")
        elif final_loss > 1.0:
            print("\n[WARNING] Model is learning slowly")
            print("Consider:")
            print("  1. Increasing learning rate")
            print("  2. Checking feature normalization")
            print("  3. Reducing model complexity")
        else:
            print("\n[OK] Model CAN overfit on small batch!")
            print("The model architecture is working correctly.")
            print("Training issues might be due to:")
            print("  1. Dataset size/quality")
            print("  2. Hyperparameter tuning")
            print("  3. Regularization (dropout, weight decay)")

if __name__ == "__main__":
    test_overfit_single_batch()