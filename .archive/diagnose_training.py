"""
Diagnostic script to check why training isn't converging.
"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.mobilenet_v3 import create_mobilenet_v3_model
from src.data.mediapipe_dataset import MediaPipeFeatureDataset, build_vocabulary, collate_fn
from torch.utils.data import DataLoader

def diagnose():
    print("\n" + "="*70)
    print("TRAINING CONVERGENCE DIAGNOSIS")
    print("="*70)
    
    # Load vocabulary
    print("\n1. Checking Vocabulary...")
    annotation_file = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    vocab = build_vocabulary(annotation_file)
    print(f"   Vocabulary size: {len(vocab)}")
    print(f"   Blank ID: {vocab.blank_id}")
    print(f"   Sample words: {list(vocab.idx2word.values())[:10]}")
    
    # Load model
    print("\n2. Checking Model...")
    model = create_mobilenet_v3_model(vocab_size=len(vocab), dropout=0.1)
    print(f"   Model parameters: {model.count_parameters():,}")
    print(f"   Model size: {model.count_parameters() * 4 / 1024 / 1024:.2f} MB")
    
    # Load one batch
    print("\n3. Checking Data Loading...")
    feature_dir = Path("data/teacher_features/mediapipe_full")
    dataset = MediaPipeFeatureDataset(
        data_dir=feature_dir,
        annotation_file=annotation_file,
        vocabulary=vocab,
        split='train',
        augment=False
    )
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(loader))
    
    print(f"   Batch features shape: {batch['features'].shape}")
    print(f"   Batch labels shape: {batch['labels'].shape}")
    print(f"   Input lengths: {batch['input_lengths']}")
    print(f"   Target lengths: {batch['target_lengths']}")
    print(f"   Sample label text: {batch['words'][0]}")
    
    # Test forward pass
    print("\n4. Testing Forward Pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        features = batch['features'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        
        log_probs = model(features, input_lengths)
        print(f"   Log probs shape: {log_probs.shape}")  # [T, B, V]
        print(f"   Log probs range: [{log_probs.min():.3f}, {log_probs.max():.3f}]")
        
        # Check if all predictions are blank
        predictions = torch.argmax(log_probs, dim=-1)  # [T, B]
        print(f"   Unique predicted indices: {torch.unique(predictions).tolist()[:20]}")
        print(f"   Blank predictions: {(predictions == vocab.blank_id).sum().item()} / {predictions.numel()}")
        
    # Test decoder
    print("\n5. Testing Adaptive Decoder...")
    from src.training.train import adaptive_greedy_decode
    
    decoded = adaptive_greedy_decode(
        log_probs.cpu(),
        input_lengths.cpu(),
        blank_id=vocab.blank_id,
        confidence_threshold=-8.0,
        min_sequence_length=2,
        max_sequence_length=50,
        adaptive_threshold=True
    )
    
    print(f"   Decoded sequences: {decoded}")
    for i, (pred_indices, target_words) in enumerate(zip(decoded, batch['words'])):
        pred_words = vocab.indices_to_words(pred_indices)
        pred_text = ' '.join(pred_words)
        target_text = ' '.join(target_words)
        print(f"   Sample {i}:")
        print(f"     Target ({len(target_words)} words): {target_text}")
        print(f"     Predicted ({len(pred_indices)} words): {pred_text}")
        print(f"     Match: {pred_text == target_text}")
    
    # Test with less aggressive decoder
    print("\n6. Testing Less Aggressive Decoder...")
    decoded_lenient = adaptive_greedy_decode(
        log_probs.cpu(),
        input_lengths.cpu(),
        blank_id=vocab.blank_id,
        confidence_threshold=-12.0,  # More permissive
        min_sequence_length=1,
        max_sequence_length=100,
        adaptive_threshold=False  # Disable adaptive
    )
    
    print(f"   Lenient decoded sequences: {decoded_lenient}")
    for i, pred_indices in enumerate(decoded_lenient):
        pred_words = vocab.indices_to_words(pred_indices)
        pred_text = ' '.join(pred_words)
        print(f"   Sample {i}: {pred_text} ({len(pred_indices)} words)")
    
    # Check learning rate schedule
    print("\n7. Analyzing Learning Rate Schedule...")
    import json
    history_file = Path("checkpoints/student/mobilenet_v3_20251117_210149/training_history.json")
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        
        print(f"   Initial LR: {history['learning_rates'][0]:.6f}")
        print(f"   Epoch 5 LR: {history['learning_rates'][4]:.6f}")
        print(f"   Max LR: {max(history['learning_rates']):.6f}")
        print(f"   Final LR: {history['learning_rates'][-1]:.6f}")
        
        print(f"\n   Loss progress:")
        print(f"     Epoch 1: {history['train_losses'][0]:.2f} -> {history['val_losses'][0]:.2f}")
        print(f"     Epoch 5: {history['train_losses'][4]:.2f} -> {history['val_losses'][4]:.2f}")
        print(f"     Epoch 15: {history['train_losses'][-1]:.2f} -> {history['val_losses'][-1]:.2f}")
        
        print(f"\n   WER stuck at: {set(history['val_wers'])}")
    
    # Recommendations
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    print("\n⚠️  Issues Found:")
    print("  1. Learning rate too low (starts at 1e-5)")
    print("  2. WER stuck at 100% despite loss decreasing")
    print("  3. Model might be predicting mostly blanks")
    
    print("\n✅ Recommended Fixes:")
    print("  1. Increase learning rate: 0.0001 → 0.0005 (match overfit test)")
    print("  2. Remove/reduce warmup: Start at target LR immediately")
    print("  3. Check decoder: May need more permissive thresholds initially")
    print("  4. Reduce dropout: 0.1 → 0.05 for initial learning")
    print("  5. Check gradient flow: Add gradient norm logging")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    diagnose()

