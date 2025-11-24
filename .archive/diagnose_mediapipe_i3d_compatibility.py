"""
Diagnostic Script: MediaPipe Features ↔ I3D Teacher Architecture Compatibility
===============================================================================

This script performs comprehensive analysis to identify why I3D teacher:
✓ Works during overfit test (0% WER)
✗ Produces NaN/Inf during full training

Tests:
1. Feature statistics and distribution
2. Normalization methods comparison
3. Forward pass stability at different scales
4. Gradient flow analysis
5. Batch size effects
6. Modality-specific issues
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.i3d_teacher import create_i3d_teacher
from src.data.dataset import MediaPipeFeatureDataset, collate_fn, build_vocabulary
from torch.utils.data import DataLoader, Subset


class FeatureCompatibilityDiagnostics:
    """Comprehensive diagnostics for MediaPipe + I3D compatibility."""
    
    def __init__(self, feature_dir: Path, annotation_file: Path, device: str = 'cuda'):
        self.feature_dir = feature_dir
        self.annotation_file = annotation_file
        self.device = device
        
        # Build vocabulary
        print("=" * 80)
        print("INITIALIZING DIAGNOSTICS")
        print("=" * 80)
        self.vocab = build_vocabulary(annotation_file)
        print(f"[OK] Vocabulary: {len(self.vocab)} words")
        
        # Load dataset
        self.dataset = MediaPipeFeatureDataset(
            data_dir=feature_dir,
            annotation_file=annotation_file,
            vocabulary=self.vocab,
            split='train',
            augment=False,
            normalize=False  # We'll test normalization ourselves
        )
        print(f"[OK] Dataset: {len(self.dataset)} samples")
        
        # Create model
        self.model = create_i3d_teacher(vocab_size=len(self.vocab), dropout=0.3)
        self.model = self.model.to(device)
        self.model.eval()
        print(f"[OK] Model: {self.model.count_parameters():,} parameters")
        print()
    
    def test_1_feature_statistics(self, num_samples: int = 100) -> Dict:
        """Test 1: Analyze raw feature statistics."""
        print("=" * 80)
        print("TEST 1: RAW FEATURE STATISTICS")
        print("=" * 80)
        
        # Sample features
        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        all_features = []
        
        for idx in indices:
            sample = self.dataset[idx]
            all_features.append(sample['features'].numpy())
        
        # Stack all features
        stacked = np.vstack(all_features)  # [total_frames, 6516]
        
        # Compute statistics
        stats = {
            'mean': np.mean(stacked, axis=0),
            'std': np.std(stacked, axis=0),
            'min': np.min(stacked, axis=0),
            'max': np.max(stacked, axis=0),
            'median': np.median(stacked, axis=0),
            'has_nan': np.isnan(stacked).any(),
            'has_inf': np.isinf(stacked).any(),
            'num_zeros': np.sum(stacked == 0, axis=0),
        }
        
        # Overall statistics
        print(f"\nOverall Statistics (across all {stacked.shape[1]} features):")
        print(f"  Shape: {stacked.shape}")
        print(f"  Mean: {stats['mean'].mean():.6f} ± {stats['mean'].std():.6f}")
        print(f"  Std:  {stats['std'].mean():.6f} ± {stats['std'].std():.6f}")
        print(f"  Min:  {stats['min'].min():.6f}")
        print(f"  Max:  {stats['max'].max():.6f}")
        print(f"  Has NaN: {stats['has_nan']}")
        print(f"  Has Inf: {stats['has_inf']}")
        
        # Check for problematic features
        zero_std_features = np.sum(stats['std'] < 1e-6)
        extreme_mean_features = np.sum(np.abs(stats['mean']) > 100)
        extreme_std_features = np.sum(stats['std'] > 100)
        
        print(f"\nPotential Issues:")
        print(f"  Features with near-zero std: {zero_std_features} / {stacked.shape[1]}")
        print(f"  Features with extreme mean (|mean| > 100): {extreme_mean_features}")
        print(f"  Features with extreme std (std > 100): {extreme_std_features}")
        
        # Modality breakdown
        print(f"\nModality Breakdown:")
        modalities = {
            'Pose': (0, 99),
            'Hands': (99, 225),
            'Face': (225, 1629),
            'Temporal': (1629, 6516)
        }
        
        for name, (start, end) in modalities.items():
            mod_mean = stats['mean'][start:end].mean()
            mod_std = stats['std'][start:end].mean()
            mod_min = stats['min'][start:end].min()
            mod_max = stats['max'][start:end].max()
            print(f"  {name:10s}: mean={mod_mean:8.4f}, std={mod_std:8.4f}, "
                  f"range=[{mod_min:8.4f}, {mod_max:8.4f}]")
        
        return stats
    
    def test_2_normalization_comparison(self, sample_idx: int = 0) -> Dict:
        """Test 2: Compare different normalization methods."""
        print("\n" + "=" * 80)
        print("TEST 2: NORMALIZATION METHODS COMPARISON")
        print("=" * 80)
        
        # Get a sample
        sample = self.dataset[sample_idx]
        features = sample['features'].unsqueeze(0)  # [1, T, 6516]
        
        print(f"\nSample: {sample['video_id']}")
        print(f"Shape: {features.shape}")
        print(f"Original range: [{features.min():.4f}, {features.max():.4f}]")
        
        # Method 1: No normalization
        feat_none = features.clone()
        
        # Method 2: Global z-score (dataset statistics)
        if hasattr(self.dataset, 'feature_mean'):
            mean = torch.from_numpy(self.dataset.feature_mean).float()
            std = torch.from_numpy(self.dataset.feature_std).float()
            feat_global = (features - mean) / std
        else:
            print("  WARNING: Dataset has no precomputed statistics")
            feat_global = features.clone()
        
        # Method 3: Per-sample z-score (what overfit test uses)
        feat_mean = features.mean(dim=(1, 2), keepdim=True)
        feat_std = features.std(dim=(1, 2), keepdim=True) + 1e-6
        feat_per_sample = (features - feat_mean) / feat_std
        
        # Method 4: Per-sample min-max
        feat_min = features.min()
        feat_max = features.max()
        feat_minmax = (features - feat_min) / (feat_max - feat_min + 1e-6)
        
        # Method 5: Clipping + per-sample z-score
        feat_clipped = torch.clamp(features, min=-100, max=100)
        feat_mean_clip = feat_clipped.mean(dim=(1, 2), keepdim=True)
        feat_std_clip = feat_clipped.std(dim=(1, 2), keepdim=True) + 1e-6
        feat_clip_norm = (feat_clipped - feat_mean_clip) / feat_std_clip
        
        results = {}
        methods = {
            'None (raw)': feat_none,
            'Global z-score': feat_global,
            'Per-sample z-score (overfit)': feat_per_sample,
            'Per-sample min-max': feat_minmax,
            'Clipped + z-score': feat_clip_norm
        }
        
        print(f"\nNormalization Results:")
        for name, feat in methods.items():
            has_nan = torch.isnan(feat).any().item()
            has_inf = torch.isinf(feat).any().item()
            mean = feat.mean().item()
            std = feat.std().item()
            min_val = feat.min().item()
            max_val = feat.max().item()
            
            print(f"\n  {name}:")
            print(f"    Mean: {mean:8.4f}, Std: {std:8.4f}")
            print(f"    Range: [{min_val:8.4f}, {max_val:8.4f}]")
            print(f"    NaN: {has_nan}, Inf: {has_inf}")
            
            results[name] = {
                'mean': mean, 'std': std, 'min': min_val, 'max': max_val,
                'has_nan': has_nan, 'has_inf': has_inf
            }
        
        return results
    
    def test_3_forward_pass_stability(self, num_samples: int = 5) -> Dict:
        """Test 3: Test forward pass with different normalization methods."""
        print("\n" + "=" * 80)
        print("TEST 3: FORWARD PASS STABILITY")
        print("=" * 80)
        
        # Sample data
        subset = Subset(self.dataset, range(min(num_samples, len(self.dataset))))
        loader = DataLoader(subset, batch_size=num_samples, collate_fn=collate_fn)
        batch = next(iter(loader))
        
        features = batch['features'].to(self.device)
        input_lengths = batch['input_lengths'].to(self.device)
        
        print(f"\nBatch shape: {features.shape}")
        print(f"Input lengths: {input_lengths.tolist()}")
        
        results = {}
        
        # Test different normalizations
        normalizations = {
            'Raw (no norm)': lambda x: x,
            'Per-sample z-score': lambda x: (x - x.mean(dim=(1,2), keepdim=True)) / (x.std(dim=(1,2), keepdim=True) + 1e-6),
            'Clipped [-100, 100]': lambda x: torch.clamp(x, -100, 100),
            'Clipped + per-sample': lambda x: (
                (torch.clamp(x, -100, 100) - torch.clamp(x, -100, 100).mean(dim=(1,2), keepdim=True)) / 
                (torch.clamp(x, -100, 100).std(dim=(1,2), keepdim=True) + 1e-6)
            ),
        }
        
        for name, norm_fn in normalizations.items():
            print(f"\n  Testing: {name}")
            
            try:
                # Apply normalization
                feat_norm = norm_fn(features)
                
                # Check input
                input_nan = torch.isnan(feat_norm).any().item()
                input_inf = torch.isinf(feat_norm).any().item()
                input_mean = feat_norm.mean().item()
                input_std = feat_norm.std().item()
                
                print(f"    Input: mean={input_mean:.4f}, std={input_std:.4f}, "
                      f"NaN={input_nan}, Inf={input_inf}")
                
                if input_nan or input_inf:
                    print(f"    [FAIL] Input has NaN/Inf")
                    results[name] = {'status': 'input_invalid'}
                    continue
                
                # Forward pass
                with torch.no_grad():
                    log_probs = self.model(feat_norm, input_lengths)
                
                # Check output
                output_nan = torch.isnan(log_probs).any().item()
                output_inf = torch.isinf(log_probs).any().item()
                output_mean = log_probs.mean().item()
                output_std = log_probs.std().item()
                output_min = log_probs.min().item()
                output_max = log_probs.max().item()
                
                print(f"    Output: mean={output_mean:.4f}, std={output_std:.4f}")
                print(f"            range=[{output_min:.4f}, {output_max:.4f}]")
                print(f"            NaN={output_nan}, Inf={output_inf}")
                
                if output_nan or output_inf:
                    print(f"    [FAIL] Output has NaN/Inf")
                    results[name] = {'status': 'output_invalid'}
                else:
                    print(f"    [PASS] Clean forward pass")
                    results[name] = {'status': 'passed', 'output_mean': output_mean}
                
            except Exception as e:
                print(f"    [ERROR] EXCEPTION: {str(e)}")
                results[name] = {'status': 'exception', 'error': str(e)}
        
        return results
    
    def test_4_gradient_flow(self, num_samples: int = 3) -> Dict:
        """Test 4: Analyze gradient flow during backward pass."""
        print("\n" + "=" * 80)
        print("TEST 4: GRADIENT FLOW ANALYSIS")
        print("=" * 80)
        
        # Enable training mode
        self.model.train()
        
        # Sample data
        subset = Subset(self.dataset, range(min(num_samples, len(self.dataset))))
        loader = DataLoader(subset, batch_size=num_samples, collate_fn=collate_fn)
        batch = next(iter(loader))
        
        features = batch['features'].to(self.device)
        labels = batch['labels'].to(self.device)
        input_lengths = batch['input_lengths'].to(self.device)
        target_lengths = batch['target_lengths'].to(self.device)
        
        print(f"\nBatch: {num_samples} samples")
        
        # Test with per-sample normalization (what works in overfit)
        feat_mean = features.mean(dim=(1, 2), keepdim=True)
        feat_std = features.std(dim=(1, 2), keepdim=True) + 1e-6
        features = (features - feat_mean) / feat_std
        
        print(f"Features after normalization:")
        print(f"  Mean: {features.mean():.4f}, Std: {features.std():.4f}")
        print(f"  Range: [{features.min():.4f}, {features.max():.4f}]")
        
        # Forward pass
        log_probs = self.model(features, input_lengths)
        
        print(f"\nLog probs:")
        print(f"  Shape: {log_probs.shape}")
        print(f"  Mean: {log_probs.mean():.4f}, Std: {log_probs.std():.4f}")
        print(f"  Range: [{log_probs.min():.4f}, {log_probs.max():.4f}]")
        
        # CTC Loss
        ctc_loss = nn.CTCLoss(blank=self.vocab.blank_id, zero_infinity=True)
        loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)
        
        print(f"\nCTC Loss: {loss.item():.6f}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("  [FAIL] Loss is NaN/Inf")
            return {'status': 'loss_invalid'}
        
        # Backward pass
        loss.backward()
        
        # Analyze gradients
        print(f"\nGradient Analysis:")
        
        gradient_stats = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.abs().max().item()
                has_nan = torch.isnan(param.grad).any().item()
                has_inf = torch.isinf(param.grad).any().item()
                
                gradient_stats.append({
                    'name': name,
                    'shape': list(param.grad.shape),
                    'mean': grad_mean,
                    'std': grad_std,
                    'norm': grad_norm,
                    'max': grad_max,
                    'has_nan': has_nan,
                    'has_inf': has_inf
                })
        
        # Find problematic layers
        problematic = [g for g in gradient_stats if g['has_nan'] or g['has_inf'] or g['max'] > 100]
        
        if problematic:
            print(f"\n  [FAIL] FOUND {len(problematic)} PROBLEMATIC LAYERS:")
            for g in problematic[:10]:  # Show first 10
                print(f"    {g['name']}: max={g['max']:.2e}, NaN={g['has_nan']}, Inf={g['has_inf']}")
        else:
            print(f"  [PASS] All gradients are healthy")
        
        # Summary statistics
        all_norms = [g['norm'] for g in gradient_stats]
        all_maxes = [g['max'] for g in gradient_stats]
        
        print(f"\n  Gradient norms: min={min(all_norms):.2e}, max={max(all_norms):.2e}, "
              f"mean={np.mean(all_norms):.2e}")
        print(f"  Gradient maxes: min={min(all_maxes):.2e}, max={max(all_maxes):.2e}, "
              f"mean={np.mean(all_maxes):.2e}")
        
        self.model.eval()
        
        return {
            'status': 'completed',
            'num_problematic': len(problematic),
            'gradient_stats': gradient_stats
        }
    
    def test_5_batch_size_effect(self) -> Dict:
        """Test 5: Test how batch size affects stability."""
        print("\n" + "=" * 80)
        print("TEST 5: BATCH SIZE EFFECT")
        print("=" * 80)
        
        batch_sizes = [1, 2, 4]
        results = {}
        
        for bs in batch_sizes:
            if bs > len(self.dataset):
                continue
            
            print(f"\n  Testing batch_size={bs}")
            
            subset = Subset(self.dataset, range(bs))
            loader = DataLoader(subset, batch_size=bs, collate_fn=collate_fn)
            batch = next(iter(loader))
            
            features = batch['features'].to(self.device)
            input_lengths = batch['input_lengths'].to(self.device)
            
            # Use per-sample normalization
            feat_mean = features.mean(dim=(1, 2), keepdim=True)
            feat_std = features.std(dim=(1, 2), keepdim=True) + 1e-6
            features_norm = (features - feat_mean) / feat_std
            
            try:
                with torch.no_grad():
                    log_probs = self.model(features_norm, input_lengths)
                
                has_nan = torch.isnan(log_probs).any().item()
                has_inf = torch.isinf(log_probs).any().item()
                mean = log_probs.mean().item()
                
                print(f"    Output: mean={mean:.4f}, NaN={has_nan}, Inf={has_inf}")
                
                status = 'passed' if not (has_nan or has_inf) else 'failed'
                results[bs] = {'status': status}
                
            except Exception as e:
                print(f"    ✗ Exception: {str(e)}")
                results[bs] = {'status': 'exception'}
        
        return results
    
    def test_6_modality_specific_issues(self) -> Dict:
        """Test 6: Check if specific modalities cause issues."""
        print("\n" + "=" * 80)
        print("TEST 6: MODALITY-SPECIFIC ISSUES")
        print("=" * 80)
        
        # Get a sample
        sample = self.dataset[0]
        features = sample['features'].unsqueeze(0).to(self.device)  # [1, T, 6516]
        input_lengths = torch.tensor([features.shape[1]], device=self.device)
        
        # Extract modalities
        pose = features[:, :, :99]
        hands = features[:, :, 99:225]
        face = features[:, :, 225:1629]
        temporal = features[:, :, 1629:6516]
        
        modalities = {
            'Pose': pose,
            'Hands': hands,
            'Face': face,
            'Temporal': temporal,
        }
        
        print("\nModality Statistics (raw):")
        for name, feat in modalities.items():
            print(f"\n  {name}: shape={feat.shape}")
            print(f"    Mean: {feat.mean():.6f}, Std: {feat.std():.6f}")
            print(f"    Range: [{feat.min():.6f}, {feat.max():.6f}]")
            print(f"    Zeros: {(feat == 0).sum().item()} / {feat.numel()} "
                  f"({100 * (feat == 0).sum().item() / feat.numel():.1f}%)")
            print(f"    NaN: {torch.isnan(feat).any().item()}")
            print(f"    Inf: {torch.isinf(feat).any().item()}")
        
        # Test zeroing out each modality
        print(f"\n\nForward Pass with Modality Ablation:")
        
        results = {}
        for ablate_name in ['None'] + list(modalities.keys()):
            # Create ablated features
            feat_ablated = features.clone()
            
            if ablate_name == 'Pose':
                feat_ablated[:, :, :99] = 0
            elif ablate_name == 'Hands':
                feat_ablated[:, :, 99:225] = 0
            elif ablate_name == 'Face':
                feat_ablated[:, :, 225:1629] = 0
            elif ablate_name == 'Temporal':
                feat_ablated[:, :, 1629:6516] = 0
            
            # Normalize
            feat_mean = feat_ablated.mean(dim=(1, 2), keepdim=True)
            feat_std = feat_ablated.std(dim=(1, 2), keepdim=True) + 1e-6
            feat_norm = (feat_ablated - feat_mean) / feat_std
            
            try:
                with torch.no_grad():
                    log_probs = self.model(feat_norm, input_lengths)
                
                has_nan = torch.isnan(log_probs).any().item()
                has_inf = torch.isinf(log_probs).any().item()
                
                status = '[PASS]' if not (has_nan or has_inf) else '[FAIL]'
                print(f"  Ablate {ablate_name:10s}: {status}")
                
                results[ablate_name] = {'status': 'passed' if not (has_nan or has_inf) else 'failed'}
                
            except Exception as e:
                print(f"  Ablate {ablate_name:10s}: [ERROR] - {str(e)}")
                results[ablate_name] = {'status': 'exception'}
        
        return results
    
    def run_all_tests(self):
        """Run all diagnostic tests."""
        print("\n" + "=" * 80)
        print(" " * 15 + "MEDIAPIPE <-> I3D COMPATIBILITY DIAGNOSTICS")
        print("=" * 80)
        
        all_results = {}
        
        try:
            all_results['test_1'] = self.test_1_feature_statistics()
        except Exception as e:
            print(f"\n[ERROR] Test 1 failed with exception: {e}")
            all_results['test_1'] = {'error': str(e)}
        
        try:
            all_results['test_2'] = self.test_2_normalization_comparison()
        except Exception as e:
            print(f"\n[ERROR] Test 2 failed with exception: {e}")
            all_results['test_2'] = {'error': str(e)}
        
        try:
            all_results['test_3'] = self.test_3_forward_pass_stability()
        except Exception as e:
            print(f"\n[ERROR] Test 3 failed with exception: {e}")
            all_results['test_3'] = {'error': str(e)}
        
        try:
            all_results['test_4'] = self.test_4_gradient_flow()
        except Exception as e:
            print(f"\n[ERROR] Test 4 failed with exception: {e}")
            all_results['test_4'] = {'error': str(e)}
        
        try:
            all_results['test_5'] = self.test_5_batch_size_effect()
        except Exception as e:
            print(f"\n[ERROR] Test 5 failed with exception: {e}")
            all_results['test_5'] = {'error': str(e)}
        
        try:
            all_results['test_6'] = self.test_6_modality_specific_issues()
        except Exception as e:
            print(f"\n[ERROR] Test 6 failed with exception: {e}")
            all_results['test_6'] = {'error': str(e)}
        
        # Final summary
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict):
        """Print comprehensive summary of all tests."""
        print("\n" + "=" * 80)
        print("FINAL DIAGNOSIS SUMMARY")
        print("=" * 80)
        
        print("\nKEY FINDINGS:\n")
        
        # Test 1: Feature statistics
        if 'test_1' in results and 'has_nan' in results['test_1']:
            t1 = results['test_1']
            if t1['has_nan'] or t1['has_inf']:
                print("[FAIL] Test 1: CRITICAL - Raw features contain NaN/Inf")
            else:
                print("[PASS] Test 1: Raw features are clean (no NaN/Inf)")
        
        # Test 3: Forward pass
        if 'test_3' in results:
            t3 = results['test_3']
            passed = sum(1 for v in t3.values() if isinstance(v, dict) and v.get('status') == 'passed')
            total = len(t3)
            print(f"[PASS] Test 3: {passed}/{total} normalization methods pass forward pass")
        
        # Test 4: Gradients
        if 'test_4' in results and 'num_problematic' in results['test_4']:
            t4 = results['test_4']
            if t4['num_problematic'] > 0:
                print(f"[FAIL] Test 4: CRITICAL - {t4['num_problematic']} layers have problematic gradients")
            else:
                print("[PASS] Test 4: All gradients are healthy")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS:")
        print("=" * 80)
        
        # Determine root cause
        if 'test_1' in results:
            t1 = results['test_1']
            if t1.get('has_nan') or t1.get('has_inf'):
                print("\n[CRITICAL] ROOT CAUSE: MediaPipe features contain NaN/Inf")
                print("\n   SOLUTIONS:")
                print("   1. Re-extract MediaPipe features with NaN handling")
                print("   2. Add NaN/Inf filtering in dataset preprocessing")
                print("   3. Use torch.nan_to_num() in data loading")
            else:
                print("\n[WARNING] Features are clean, but normalization may be problematic")
                print("\n   RECOMMENDATIONS:")
                print("   1. Use per-sample z-score normalization (matches overfit test)")
                print("   2. Add input clamping: torch.clamp(features, -100, 100)")
                print("   3. Consider using GroupNorm instead of BatchNorm")
                print("   4. Reduce learning rate for stability")
        
        print("\n" + "=" * 80)


def main():
    """Main diagnostic function."""
    # Configuration
    feature_dir = Path("data/teacher_features/mediapipe_full")
    annotation_file = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    
    # Check if data exists
    if not feature_dir.exists():
        print(f"ERROR: Feature directory not found: {feature_dir}")
        print("\nPlease ensure MediaPipe features have been extracted.")
        return
    
    if not annotation_file.exists():
        print(f"ERROR: Annotation file not found: {annotation_file}")
        print("\nPlease ensure the raw data is available.")
        return
    
    # Run diagnostics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    diagnostics = FeatureCompatibilityDiagnostics(
        feature_dir=feature_dir,
        annotation_file=annotation_file,
        device=device
    )
    
    results = diagnostics.run_all_tests()
    
    # Save results
    output_file = Path("mediapipe_i3d_diagnostics.txt")
    print(f"\n\nResults will be saved to: {output_file}")


if __name__ == "__main__":
    main()

