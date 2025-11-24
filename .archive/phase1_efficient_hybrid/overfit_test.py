"""Preflight overfit test for teacher model.

This script tests the training pipeline on a tiny dataset to ensure:
1. Model can overfit (loss → 0)
2. Forward/backward passes work correctly
3. Data loading functions properly
4. All components are integrated correctly
"""

# Fix path BEFORE imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml
import argparse
from tqdm import tqdm
import numpy as np

from teacher.models.efficient_hybrid import create_efficient_model
from teacher.loaders import create_dataloaders, collate_fn
from utils.vocabulary import Vocabulary, load_vocabulary_from_file
from utils.ctc import CTCLoss, ctc_decode, prepare_ctc_targets
from utils.metrics import compute_wer


def overfit_test(config_path: str, num_samples: int = 5, num_epochs: int = 10,
                 multi_stage: bool = False, 
                 zero_regularization: bool = False, curriculum: bool = False):
    """
    Run overfit test on a small subset of data.
    
    Args:
        config_path: Path to config YAML file
        num_samples: Number of samples to use for overfitting
        num_epochs: Number of epochs to train
    """
    print("="*80)
    print("TEACHER MODEL OVERFIT TEST")
    print("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Set random seeds
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)
    
    # Paths - resolve relative to project root (where script is run from)
    # Get project root (parent of teacher directory)
    project_root = Path(__file__).parent.parent
    data_dir = (project_root / config['data']['data_dir']).resolve()
    features_dir = data_dir / config['data']['features_dir']
    annotations_dir = data_dir / config['data']['annotations_dir']
    vocab_file = data_dir / config['data']['vocab_file']
    
    print(f"\nProject root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Features directory: {features_dir}")
    print(f"Annotations directory: {annotations_dir}")
    print(f"Vocabulary file: {vocab_file}")
    
    # Verify paths exist
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    if not features_dir.exists():
        raise ValueError(f"Features directory not found: {features_dir}")
    if not annotations_dir.exists():
        raise ValueError(f"Annotations directory not found: {annotations_dir}")
    if not vocab_file.exists():
        raise ValueError(f"Vocabulary file not found: {vocab_file}")
    
    # Load vocabulary
    print(f"\nLoading vocabulary from {vocab_file}")
    vocabulary = load_vocabulary_from_file(vocab_file)
    print(f"Vocabulary size: {len(vocabulary)}")
    
    # Create full data loaders
    print("\nCreating data loaders...")
    loaders = create_dataloaders(
        features_dir=features_dir,
        annotations_dir=annotations_dir,
        vocabulary=vocabulary,
        batch_size=min(num_samples, config['training']['batch_size']),
        num_workers=0,  # Use 0 workers for debugging
        max_length=config['training'].get('max_length'),
        normalize=config['training'].get('normalize', True)
    )
    
    if 'train' not in loaders:
        raise ValueError("Training data loader not found!")
    
    # Create small subset for overfitting
    train_dataset = loaders['train'].dataset
    num_samples = min(num_samples, len(train_dataset))
    subset_indices = list(range(num_samples))
    subset_dataset = Subset(train_dataset, subset_indices)
    
    # CURRICULUM LEARNING: Start with 1 sample, gradually increase
    if curriculum:
        print("\nCURRICULUM LEARNING MODE: Starting with 1 sample, increasing gradually")
        curriculum_schedule = [
            (1, num_epochs // 3),      # First third: 1 sample
            (min(2, num_samples), num_epochs // 3),  # Second third: 2 samples
            (num_samples, num_epochs - 2 * (num_epochs // 3))  # Final: all samples
        ]
    else:
        # Standard: use all samples from start
        curriculum_schedule = [(num_samples, num_epochs)]
    
    print(f"\nUsing {num_samples} samples for overfit test")
    if not curriculum:
        print(f"Batch size: {min(num_samples, config['training']['batch_size'])}")
    
    # Create model (using EfficientHybridModel)
    print("\nCreating model...")
    model = create_efficient_model(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_classes=len(vocabulary),
        dropout=config['model']['dropout'],
        blank_idx=vocabulary.blank_idx
    )
    model = model.to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    
    # Loss function with blank and repetition penalties
    # MULTI-STAGE TRAINING: Gradually introduce penalties
    if multi_stage:
        print("\n" + "="*80)
        print("MULTI-STAGE TRAINING MODE")
        print("="*80)
        print("Stage 1 (0-33%):  Zero penalties - learn vocabulary")
        print("Stage 2 (33-66%): Low penalties - start enforcing alignment")
        print("Stage 3 (66-100%): Full penalties - enforce generalization")
        print("="*80)
        # Will be updated dynamically during training
        blank_penalty_val = 0.0
        repetition_penalty_val = 0.0
    elif zero_regularization:
        blank_penalty_val = 0.0
        repetition_penalty_val = 0.0
    else:
        blank_penalty_val = 1.5
        repetition_penalty_val = 0.5

    criterion = CTCLoss(
        blank_idx=vocabulary.blank_idx,
        reduction='mean',
        blank_penalty=blank_penalty_val,
        repetition_penalty=repetition_penalty_val
    )

    if multi_stage:
        print(f"\nInitial penalties: blank={blank_penalty_val}, repetition={repetition_penalty_val}")
    elif zero_regularization:
        print("\nDIAGNOSTIC MODE: Zero regularization (blank_penalty=0.0, repetition_penalty=0.0)")
    else:
        print(f"\nUsing blank penalty: {blank_penalty_val} (to prevent blank collapse)")
        print(f"Using repetition penalty: {repetition_penalty_val} (to prevent single-token collapse)")
    
    # Optimizer - use lower LR to avoid instability
    weight_decay = config['training'].get('weight_decay', 1e-5)
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    # Use moderate learning rate for stable CTC training
    # Lower LR provides more stable gradient updates with proper batching
    # DIAGNOSTIC EXPERIMENT B: Zero weight decay if zero_regularization
    weight_decay_val = 0.0 if zero_regularization else weight_decay

    # DIAGNOSTIC: Test higher LR to escape mode collapse
    # Standard: 0.0003, Higher: 0.001-0.003 to help model explore better
    lr_val = config['training'].get('learning_rate', 0.0003)
    print(f"\nUsing learning rate: {lr_val}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_val,
        weight_decay=weight_decay_val,
        betas=(0.9, 0.995)  # Standard Adam betas
    )
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING OVERFIT TEST")
    print("="*80)
    
    initial_loss = None
    final_loss = None
    
    # Create standard loader (for non-curriculum mode)
    # Use batch_size=4 for proper gradient estimation
    # CTC loss correctly handles batches with different targets by averaging per-sample losses
    # Batch size of 1 causes poor gradient estimates and slow convergence
    small_loader = DataLoader(
        subset_dataset,
        batch_size=4,  # Use small batches for better gradient estimates
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    for epoch in range(num_epochs):
        # Determine which curriculum phase we're in
        if curriculum:
            epoch_accum = 0
            current_samples = num_samples
            for cs, ce in curriculum_schedule:
                if epoch < epoch_accum + ce:
                    current_samples = cs
                    break
                epoch_accum += ce
            
            # Create loader for current curriculum phase
            # Use batch_size=4 for proper gradient estimation
            # CTC loss correctly handles batches with different targets by averaging per-sample losses
            phase_subset = Subset(train_dataset, subset_indices[:current_samples])
            current_loader = DataLoader(
                phase_subset,
                batch_size=4,  # Use small batches for better gradient estimates
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available()
            )
        else:
            current_loader = small_loader
            current_samples = num_samples
        
        # MULTI-STAGE: Update penalties based on training progress
        if multi_stage:
            progress = epoch / num_epochs

            if progress < 0.33:  # Stage 1: Learn vocabulary (0-33%)
                stage = 1
                blank_penalty_curr = 0.0
                repetition_penalty_curr = 0.0
            elif progress < 0.66:  # Stage 2: Start alignment (33-66%)
                stage = 2
                # Gradually increase from 0.0 to 0.75 (half of final)
                stage_progress = (progress - 0.33) / 0.33
                blank_penalty_curr = 0.75 * stage_progress
                repetition_penalty_curr = 0.25 * stage_progress
            else:  # Stage 3: Full regularization (66-100%)
                stage = 3
                # Gradually increase from 0.75 to 1.5 (final)
                stage_progress = (progress - 0.66) / 0.34
                blank_penalty_curr = 0.75 + 0.75 * stage_progress
                repetition_penalty_curr = 0.25 + 0.25 * stage_progress

            # Update criterion with new penalties
            criterion.blank_penalty = blank_penalty_curr
            criterion.repetition_penalty = repetition_penalty_curr

            if epoch % 5 == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1}: Stage {stage}, blank_penalty={blank_penalty_curr:.3f}, repetition_penalty={repetition_penalty_curr:.3f}")

        model.train()

        total_loss = 0.0
        num_batches = 0

        # Standard training loop: update weights after each batch
        # CTC loss correctly handles batches with different targets
        pbar = tqdm(current_loader, desc=f'Epoch {epoch+1}/{num_epochs} (samples={current_samples})')
        for batch_idx, batch in enumerate(pbar):
            # Zero gradients for this batch
            optimizer.zero_grad()
            features = batch['features'].to(device)
            sequence_lengths = batch['sequence_lengths'].to(device)
            targets = batch['targets']
            target_lengths = batch['target_lengths']
            
            # Prepare CTC targets
            targets_list = [targets[i, :target_lengths[i]].tolist() 
                          for i in range(len(targets))]
            
            # DIAGNOSTIC EXPERIMENT A: Check input diversity
            if epoch == 0 and num_batches == 0:
                print("\n" + "="*80)
                print("DIAGNOSTIC EXPERIMENT A: Checking Input Diversity")
                print("="*80)
                
                # Check if features are different across batch
                print("\n--- Feature Diversity Check ---")
                for i in range(min(3, features.shape[0])):
                    feat_norm = torch.norm(features[i]).item()
                    feat_mean = features[i].mean().item()
                    feat_std = features[i].std().item()
                    feat_min = features[i].min().item()
                    feat_max = features[i].max().item()
                    print(f"Sample {i}:")
                    print(f"  Norm: {feat_norm:.4f}")
                    print(f"  Mean: {feat_mean:.6f}, Std: {feat_std:.6f}")
                    print(f"  Range: [{feat_min:.4f}, {feat_max:.4f}]")
                    print(f"  First 10 values: {features[i, 0, :10].cpu().tolist()}")
                    print(f"  Sequence length: {sequence_lengths[i].item()}")
                
                # Check if features are identical
                if features.shape[0] > 1:
                    feat_diff = torch.abs(features[0] - features[1]).mean().item()
                    print(f"\nFeature difference (sample 0 vs 1): {feat_diff:.6f}")
                    if feat_diff < 1e-6:
                        print("  WARNING: Features are IDENTICAL! Data loading bug!")
                    else:
                        print("  Features are different")
                
                # Check if targets are different
                print("\n--- Target Diversity Check ---")
                for i in range(min(3, len(targets_list))):
                    print(f"Sample {i} target ({len(targets_list[i])} tokens):")
                    print(f"  Token indices: {targets_list[i][:15]}")
                    decoded = vocabulary.decode(targets_list[i][:15])
                    print(f"  Decoded: {decoded}")
                    annotation = batch['annotations'][i] if i < len(batch['annotations']) else "N/A"
                    print(f"  Original annotation: {annotation}")
                
                # Check if targets are identical
                if len(targets_list) > 1:
                    if targets_list[0] == targets_list[1]:
                        print("\n  WARNING: Targets are IDENTICAL! Annotation loading bug!")
                    else:
                        print("\n  Targets are different")
                
                print("="*80 + "\n")
            
            targets_tensor, target_lengths_tensor = prepare_ctc_targets(
                targets_list, device
            )
            
            # Forward pass (gradients already zeroed at epoch start)
            log_probs = model(features, sequence_lengths)
            
            # Verify output shape
            T, N, C = log_probs.shape
            assert N == len(sequence_lengths), f"Batch size mismatch: {N} != {len(sequence_lengths)}"
            assert C == len(vocabulary), f"Vocab size mismatch: {C} != {len(vocabulary)}"
            
            # Model preserves full temporal resolution (no downsampling)
            # CTC loss expects input_lengths to match log_probs length T
            # Sequence lengths stay the same since there's no downsampling
            sequence_lengths_adjusted = torch.clamp(sequence_lengths, max=T)
            
            # Compute loss
            loss = criterion(
                log_probs,
                targets_tensor,
                sequence_lengths_adjusted,
                target_lengths_tensor
            )
            
            # Debug: print penalty breakdown periodically
            if batch_idx == 0 and epoch % 2 == 0:
                # Manually compute penalties to show breakdown
                with torch.no_grad():
                    probs = torch.exp(log_probs)
                    batch_size = probs.shape[1]
                    
                    # Blank penalty
                    blank_probs = probs[:, :, vocabulary.blank_idx]
                    blank_avg = blank_probs.mean()
                    
                    # UNK penalty
                    unk_probs = probs[:, :, vocabulary.unk_idx]
                    unk_avg = unk_probs.mean()
                    
                    # Repetition penalty (simplified)
                    total_unique = 0
                    for n in range(batch_size):
                        seq_len = sequence_lengths_adjusted[n].item()
                        if seq_len > 0:
                            argmax = torch.argmax(probs[:seq_len, n, :], dim=1)
                            non_blank = argmax[argmax != vocabulary.blank_idx]
                            if len(non_blank) > 0:
                                total_unique += len(torch.unique(non_blank))
                    
                    avg_unique = total_unique / batch_size if batch_size > 0 else 0
                    
                    print(f"\n  Loss breakdown (epoch {epoch+1}):")
                    print(f"    CTC loss: {loss.item():.4f}")
                    print(f"    Blank prob: {blank_avg.item():.4f}")
                    print(f"    UNK prob: {unk_avg.item():.4f}")
                    print(f"    Avg unique tokens: {avg_unique:.1f}")
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nERROR: Loss is NaN or Inf at epoch {epoch+1}, batch {num_batches+1}")
                return False

            # Backward pass
            loss.backward()

            # Clip gradients to prevent explosion
            grad_clip = 5.0  # Standard gradient clipping value
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Update weights for this batch
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Show average loss so far
            avg_loss_so_far = total_loss / num_batches if num_batches > 0 else 0
            pbar.set_postfix({
                'loss': f'{avg_loss_so_far:.2f}',
                'batches': num_batches
            })

            # Clear CUDA cache periodically to prevent OOM
            if num_batches % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches
        if epoch == 0:
            initial_loss = avg_loss
        final_loss = avg_loss

        # Clear CUDA cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Test decoding
        model.eval()
        with torch.no_grad():
            test_batch = next(iter(current_loader))
            test_features = test_batch['features'].to(device)
            test_sequence_lengths = test_batch['sequence_lengths'].to(device)
            
            log_probs = model(test_features, test_sequence_lengths)
            # log_probs is already [T, N, C] format from model
            # Use beam search for better decoding (more robust than greedy)
            # Beam search can find longer sequences if they exist in probability space
            decode_method = 'beam_search'
            predictions = ctc_decode(log_probs, 
                                   blank_idx=vocabulary.blank_idx,
                                   method=decode_method,
                                   beam_width=15)
            
            # Print sample predictions
            print(f"\nSample predictions (epoch {epoch+1}):")
            annotations = test_batch['annotations']
            for i in range(min(2, len(predictions), len(annotations))):
                pred_text = vocabulary.decode(predictions[i])
                ref_text = annotations[i]
                print(f"  Reference: {ref_text}")
                print(f"  Prediction: {pred_text}")
                # Debug: show raw prediction tokens and top predictions
                if len(predictions[i]) > 0:
                    print(f"  Raw tokens (first 10): {predictions[i][:10]}")
                    # Also check what the model probabilities look like across timesteps
                    probs = torch.exp(log_probs[:, i, :])
                    seq_len = test_sequence_lengths[i].item()
                    # Check non-blank predictions across timesteps
                    argmax_preds = torch.argmax(probs[:seq_len], dim=1).cpu()
                    non_blank_preds = argmax_preds[argmax_preds != vocabulary.blank_idx]
                    unique_tokens = torch.unique(non_blank_preds) if len(non_blank_preds) > 0 else torch.tensor([])
                    print(f"  Model output: {len(non_blank_preds)} non-blank timesteps, {len(unique_tokens)} unique tokens")
                    if len(non_blank_preds) > 0:
                        top_tokens = unique_tokens[:5].tolist()
                        print(f"  Top predicted tokens: {top_tokens}")
                        # Check probability distribution across timesteps
                        # Sample at beginning, middle, and end of sequence
                        sample_indices = [0, seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 1]
                        sample_indices = [idx for idx in sample_indices if idx < seq_len]
                        print(f"  Probability distribution across timesteps:")
                        for t_idx in sample_indices[:3]:  # Show first 3 samples
                            top_probs_t, top_tokens_t = torch.topk(probs[t_idx], k=5)
                            print(f"    Timestep {t_idx}: top tokens {top_tokens_t.cpu().tolist()}, probs {top_probs_t.cpu().tolist()}")
                        # Check if there's diversity in non-argmax predictions
                        # Look at top-2 probability at each timestep
                        top2_probs = torch.topk(probs[:seq_len], k=2, dim=1)[0]
                        top2_entropy = (top2_probs[:, 1] / (top2_probs[:, 0] + top2_probs[:, 1] + 1e-10)).mean().item()
                        print(f"  Top-2 diversity: {top2_entropy:.4f} (higher = more uncertainty between top tokens)")
                else:
                    # Check what the model is predicting
                    probs = torch.exp(log_probs[:, i, :])
                    top_probs, top_indices = torch.topk(probs.mean(dim=0), k=10)
                    print(f"  Top 10 avg predictions: {top_indices.cpu().tolist()}")
                    print(f"  Top 10 avg probs: {top_probs.cpu().tolist()}")
                    # Check if it's predicting mostly blank or UNK
                    blank_prob = probs[:, vocabulary.blank_idx].mean().item()
                    unk_prob = probs[:, vocabulary.unk_idx].mean().item()
                    print(f"  Avg blank prob: {blank_prob:.4f}")
                    print(f"  Avg UNK prob: {unk_prob:.4f}")
                    # Check entropy (diversity of predictions)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean().item()
                    print(f"  Avg entropy: {entropy:.4f} (higher = more diverse, max={np.log(len(vocabulary)):.4f})")
                    # Check what the raw argmax predictions are
                    argmax_preds = torch.argmax(probs, dim=1).cpu()
                    unique_tokens, counts = argmax_preds.unique(return_counts=True)
                    unique_tokens_list = unique_tokens.tolist()
                    token_counts_list = counts.tolist()
                    seq_len = len(argmax_preds)
                    diversity_ratio = len(unique_tokens) / seq_len if seq_len > 0 else 0.0
                    print(f"  Unique argmax tokens: {unique_tokens_list[:10]}")
                    print(f"  Token counts: {token_counts_list[:10]}")
                    print(f"  Diversity ratio: {diversity_ratio:.4f} (1.0 = all unique, 0.0 = all same)")
                    # Check sequence length
                    target_len = len(test_batch['targets'][i])
                    predicted_non_blank = (unique_tokens != vocabulary.blank_idx).sum().item()
                    print(f"  Target length: {target_len}, Predicted unique tokens: {predicted_non_blank}")
    
    # Results
    print("\n" + "="*80)
    print("OVERFIT TEST RESULTS")
    print("="*80)
    
    print(f"\nInitial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Loss Reduction: {((initial_loss - final_loss) / initial_loss * 100):.1f}%")
    
    # Check if overfitting occurred
    loss_reduction_ratio = (initial_loss - final_loss) / initial_loss
    
    if loss_reduction_ratio > 0.5:
        print(f"\nSUCCESS: Model is learning (loss reduced by {loss_reduction_ratio*100:.1f}%)")
        print("   The training pipeline is working correctly!")
        print("   You can proceed with full training.")
    elif loss_reduction_ratio > 0.2:
        print(f"\nWARNING: Model is learning but slowly (loss reduced by {loss_reduction_ratio*100:.1f}%)")
        print("   Consider increasing learning rate or training for more epochs.")
    else:
        print(f"\nFAILURE: Model is not learning (loss reduced by only {loss_reduction_ratio*100:.1f}%)")
        print("   Check:")
        print("   - Model architecture")
        print("   - Loss function")
        print("   - Data loading")
        print("   - Learning rate")
        return False
    
    # Compute WER on overfit set
    print("\nComputing WER on overfit set...")
    model.eval()
    all_references = []
    all_hypotheses = []

    with torch.no_grad():
        for batch in small_loader:
            features = batch['features'].to(device)
            sequence_lengths = batch['sequence_lengths'].to(device)

            log_probs = model(features, sequence_lengths)
            # log_probs is already [T, N, C] format, no transpose needed

            # DIAGNOSTIC: Check model output distribution
            probs = torch.exp(log_probs)  # [T, N, C]
            blank_probs = probs[:, :, vocabulary.blank_idx]  # [T, N]
            mean_blank_prob = blank_probs.mean().item()

            # Get top-k predictions (excluding blank)
            T, N, C = log_probs.shape
            top_k_indices = torch.topk(log_probs, k=5, dim=2)  # Top 5 predictions

            print(f"\n  DIAGNOSTIC - Sample 0:")
            print(f"    Mean blank probability: {mean_blank_prob*100:.2f}%")
            print(f"    Blank idx: {vocabulary.blank_idx}")
            print(f"    Vocab size: {len(vocabulary)}")
            print(f"    First timestep top-5 predictions (token_idx: log_prob):")
            for k in range(min(5, C)):
                idx = top_k_indices.indices[0, 0, k].item()
                log_p = top_k_indices.values[0, 0, k].item()
                prob = torch.exp(torch.tensor(log_p)).item()
                token_name = vocabulary.idx_to_word.get(idx, f"IDX_{idx}")
                print(f"      {idx}: {token_name} (prob={prob*100:.2f}%, log_prob={log_p:.4f})")

            # Use greedy decoding for fast evaluation
            # Beam search is more accurate but much slower
            predictions = ctc_decode(log_probs,
                                   blank_idx=vocabulary.blank_idx,
                                   method='greedy',
                                   beam_width=10)

            # Get annotations from batch
            annotations = batch['annotations']

            for i, pred in enumerate(predictions):
                if i < len(annotations):
                    hyp_text = vocabulary.decode(pred)
                    ref_text = annotations[i]
                    all_references.append(ref_text)
                    all_hypotheses.append(hyp_text)
                    # Debug: print first few
                    if len(all_references) <= 3:
                        print(f"\n  Sample {len(all_references)-1}:")
                        print(f"    Ref: {ref_text}")
                        print(f"    Hyp: {hyp_text}")
                        print(f"    Pred tokens: {pred[:10] if len(pred) > 0 else 'empty'}")
                        print(f"    Pred token names: {[vocabulary.idx_to_word.get(idx, f'IDX_{idx}') for idx in pred[:10]] if len(pred) > 0 else 'empty'}")
    
    wer, wer_stats = compute_wer(all_references, all_hypotheses)
    print(f"WER on overfit set: {wer*100:.2f}%")

    if wer < 0.1:  # Less than 10% WER on overfit set
        print("Excellent overfitting! Model can memorize the small dataset.")
    elif wer < 0.5:
        print("Partial overfitting. Model is learning but may need more training.")
    else:
        print("Poor overfitting. Check model architecture and training setup.")

    # Final CUDA cache cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("OVERFIT TEST COMPLETE")
    print("="*80)
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Overfit test for teacher model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to use for overfitting (default: 5)')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of epochs to train (default: 10)')
    parser.add_argument('--zero_reg', action='store_true',
                       help='Use zero regularization (diagnostic experiment B)')
    parser.add_argument('--curriculum', action='store_true',
                       help='Use curriculum learning (start with 1 sample, gradually increase)')
    parser.add_argument('--multi_stage', action='store_true',
                       help='Use multi-stage training (gradually introduce penalties)')
    args = parser.parse_args()

    success = overfit_test(
        config_path=args.config,
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        multi_stage=args.multi_stage,
        zero_regularization=args.zero_reg,
        curriculum=args.curriculum
    )
    
    if not success:
        exit(1)


if __name__ == '__main__':
    main()

