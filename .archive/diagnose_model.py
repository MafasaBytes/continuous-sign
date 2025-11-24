"""
Diagnostic script to understand why the model has 93% WER.
"""

import torch
import numpy as np
from pathlib import Path
from src.models.mobilenet_v3 import create_mobilenet_v3_model
from src.data.dataset import build_vocabulary
from src.models.bilstm_ctc import CTCDecoder

# Load vocabulary
print("Loading vocabulary...")
vocab = build_vocabulary(
    Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
)
print(f"Vocabulary size: {len(vocab)}")

# Load best checkpoint
checkpoint_path = Path("checkpoints/student/mobilenet_v3_20251116_193021/best_model.pth")
if not checkpoint_path.exists():
    print(f"Checkpoint not found at {checkpoint_path}")
    exit()

print(f"\nLoading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(f"Best WER from checkpoint: {checkpoint['best_wer']:.2f}%")

# Create model and load weights
model = create_mobilenet_v3_model(vocab_size=len(vocab), dropout=0.6)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create synthetic input to test model output
batch_size = 2
seq_length = 50
feature_dim = 6516

# Random features
test_features = torch.randn(batch_size, seq_length, feature_dim)
test_lengths = torch.tensor([seq_length, seq_length])

print("\n" + "="*60)
print("Testing model predictions on random input:")
print("="*60)

with torch.no_grad():
    # Get model predictions
    log_probs = model(test_features, test_lengths)  # [T, B, V]

    # Check the distribution of predictions
    probs = torch.exp(log_probs)  # Convert log probs to probabilities

    # Get predicted classes
    predicted_classes = log_probs.argmax(dim=-1)  # [T, B]

    print(f"Output shape: {log_probs.shape}")
    print(f"Min log prob: {log_probs.min():.3f}, Max log prob: {log_probs.max():.3f}")

    # Check if model is predicting mostly blanks
    blank_id = vocab.blank_id
    blank_predictions = (predicted_classes == blank_id).sum().item()
    total_predictions = predicted_classes.numel()
    blank_ratio = blank_predictions / total_predictions

    print(f"\nBlank predictions: {blank_predictions}/{total_predictions} ({blank_ratio*100:.1f}%)")

    # Get unique predicted classes
    unique_predictions = torch.unique(predicted_classes)
    print(f"Unique predicted classes: {len(unique_predictions)} out of {len(vocab)} possible")
    print(f"First 10 unique predictions: {unique_predictions[:10].tolist()}")

    # Use CTC decoder
    decoder = CTCDecoder()
    decoded_sequences = decoder.greedy_decode(log_probs, test_lengths, blank_id=blank_id)

    print("\n" + "="*60)
    print("CTC Decoded sequences:")
    print("="*60)
    for i, seq in enumerate(decoded_sequences):
        words = vocab.indices_to_words(seq)
        print(f"Sequence {i+1}: {' '.join(words) if words else '[EMPTY]'}")

    # Check probability distribution
    print("\n" + "="*60)
    print("Probability distribution analysis:")
    print("="*60)

    # Average probability per class across all timesteps
    avg_probs = probs.mean(dim=[0, 1])  # Average over T and B dimensions
    top_k = 10
    top_probs, top_indices = torch.topk(avg_probs, top_k)

    print(f"\nTop {top_k} most probable classes (averaged):")
    for prob, idx in zip(top_probs, top_indices):
        word = vocab.idx2word.get(idx.item(), f"UNK_{idx.item()}")
        print(f"  Class {idx:4d} ({word:15s}): {prob:.4f}")

    # Check if model outputs are collapsing
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
    max_entropy = np.log(len(vocab))
    print(f"\nOutput entropy: {entropy:.3f} (max possible: {max_entropy:.3f})")
    print(f"Entropy ratio: {(entropy/max_entropy)*100:.1f}% (higher = more diverse predictions)")

print("\n" + "="*60)
print("Diagnosis complete!")
print("="*60)

print("\nPossible issues to investigate:")
print("1. If blank ratio is >90%, CTC decoder is removing most predictions")
print("2. If unique predictions < 50, model is collapsing to few outputs")
print("3. If entropy ratio < 20%, model outputs are not diverse enough")
print("4. Check if top predicted words make sense for sign language")