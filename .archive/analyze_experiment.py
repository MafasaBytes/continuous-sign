"""
Analyze experimental results and compare with baseline.

Helps make data-driven decisions for next iterations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def load_history(history_file: str) -> List[Dict]:
    """Load training history JSON."""
    with open(history_file, 'r') as f:
        return json.load(f)


def analyze_experiment(history_file: str, baseline_wer: float = 95.0):
    """
    Analyze experiment results.
    
    Args:
        history_file: Path to history JSON
        baseline_wer: Baseline WER for comparison
    """
    history = load_history(history_file)
    
    if not history:
        print("No history data found!")
        return
    
    print("="*80)
    print("EXPERIMENT ANALYSIS")
    print("="*80)
    
    # Separate by stage (supports 2-6 stages)
    stages = {}
    for h in history:
        stage_num = h.get('stage', 0)
        if stage_num not in stages:
            stages[stage_num] = []
        stages[stage_num].append(h)
    
    # Sort stages
    stage_numbers = sorted(stages.keys())
    stage1_history = stages.get(1, [])
    stage2_history = stages.get(2, [])
    
    # Stage 1 Analysis
    if stage1_history:
        print("\nSTAGE 1 (Exploration) Analysis:")
        print("-" * 80)
        stage1_wer_start = stage1_history[0]['val_wer']
        stage1_wer_end = stage1_history[-1]['val_wer']
        stage1_blank_start = stage1_history[0]['blank_ratio']
        stage1_blank_end = stage1_history[-1]['blank_ratio']
        stage1_unique_start = stage1_history[0]['unique_nonblank_predictions']
        stage1_unique_end = stage1_history[-1]['unique_nonblank_predictions']
        
        print(f"  WER: {stage1_wer_start:.2f}% → {stage1_wer_end:.2f}% "
              f"({stage1_wer_end - stage1_wer_start:+.2f}%)")
        print(f"  Blank Ratio: {stage1_blank_start:.2f}% → {stage1_blank_end:.2f}% "
              f"({stage1_blank_end - stage1_blank_start:+.2f}%)")
        print(f"  Unique Non-Blank: {stage1_unique_start} → {stage1_unique_end} "
              f"({stage1_unique_end - stage1_unique_start:+d})")
        print(f"  Val Loss: {stage1_history[0]['val_loss']:.4f} → "
              f"{stage1_history[-1]['val_loss']:.4f}")
        
        # Check if blank ratio decreased
        if stage1_blank_end < stage1_blank_start:
            print(f" Blank ratio decreased (good!)")
        else:
            print(f" Blank ratio increased (may need stronger blank bias)")
        
        # Check if unique predictions increased
        if stage1_unique_end > stage1_unique_start:
            print(f"  Vocabulary exploration increased (good!)")
        else:
            print(f"  Vocabulary exploration stagnated (may need stronger blank bias)")
    
    # Additional stages analysis (if 6-stage)
    for stage_num in stage_numbers[2:]:  # Stages 3-6
        stage_history = stages.get(stage_num, [])
        if stage_history:
            stage_name = stage_history[0].get('stage_name', f'Stage {stage_num}')
            print(f"\n{stage_name.upper()} Analysis:")
            print("-" * 80)
            stage_wer_start = stage_history[0]['val_wer']
            stage_wer_end = stage_history[-1]['val_wer']
            stage_blank_start = stage_history[0]['blank_ratio']
            stage_blank_end = stage_history[-1]['blank_ratio']
            stage_unique_start = stage_history[0]['unique_nonblank_predictions']
            stage_unique_end = stage_history[-1]['unique_nonblank_predictions']
            
            print(f"  WER: {stage_wer_start:.2f}% → {stage_wer_end:.2f}% "
                  f"({stage_wer_end - stage_wer_start:+.2f}%)")
            print(f"  Blank Ratio: {stage_blank_start:.2f}% → {stage_blank_end:.2f}%")
            print(f"  Unique Non-Blank: {stage_unique_start} → {stage_unique_end}")
            print(f"  Val Loss: {stage_history[0]['val_loss']:.4f} → "
                  f"{stage_history[-1]['val_loss']:.4f}")
            
            # Check overfitting
            if stage_history[-1]['val_loss'] > stage_history[0]['val_loss']:
                print(f"  WARNING: Val loss increased (overfitting)")
            else:
                print(f"  OK: Val loss decreased (good)")
    
    # Stage 2 Analysis
    if stage2_history:
        print("\nSTAGE 2 (Refinement) Analysis:")
        print("-" * 80)
        stage2_wer_start = stage2_history[0]['val_wer']
        stage2_wer_end = stage2_history[-1]['val_wer']
        stage2_blank_start = stage2_history[0]['blank_ratio']
        stage2_blank_end = stage2_history[-1]['blank_ratio']
        stage2_unique_start = stage2_history[0]['unique_nonblank_predictions']
        stage2_unique_end = stage2_history[-1]['unique_nonblank_predictions']
        
        print(f"  WER: {stage2_wer_start:.2f}% → {stage2_wer_end:.2f}% "
              f"({stage2_wer_end - stage2_wer_start:+.2f}%)")
        print(f"  Blank Ratio: {stage2_blank_start:.2f}% → {stage2_blank_end:.2f}%")
        print(f"  Unique Non-Blank: {stage2_unique_start} → {stage2_unique_end}")
        print(f"  Val Loss: {stage2_history[0]['val_loss']:.4f} → "
              f"{stage2_history[-1]['val_loss']:.4f}")
        
        # Check if WER improved
        if stage2_wer_end < stage2_wer_start:
            print(f"  WER improved (good!)")
        else:
            print(f"  WER did not improve (may need more epochs or different LR)")
        
        # Check if val loss decreased
        if stage2_history[-1]['val_loss'] < stage2_history[0]['val_loss']:
            print(f"  Val loss decreased (good - overfitting addressed)")
        else:
            print(f"  Val loss increased (overfitting still present)")
    
    # Overall Analysis
    print("\nOVERALL Analysis:")
    print("-" * 80)
    best_wer = min(h['val_wer'] for h in history)
    best_epoch = min(h['epoch'] for h in history if h['val_wer'] == best_wer)
    final_wer = history[-1]['val_wer']
    final_unique = history[-1]['unique_nonblank_predictions']
    final_blank = history[-1]['blank_ratio']
    
    print(f"  Best WER: {best_wer:.2f}% (epoch {best_epoch})")
    print(f"  Final WER: {final_wer:.2f}%")
    print(f"  Final Unique Non-Blank: {final_unique}")
    print(f"  Final Blank Ratio: {final_blank:.2f}%")
    
    # Comparison with baseline
    print(f"\n  Baseline WER: {baseline_wer:.2f}%")
    improvement = baseline_wer - best_wer
    if improvement > 0:
        print(f"  Improvement: {improvement:.2f}% (better than baseline)")
    else:
        print(f"  Worse by: {abs(improvement):.2f}% (worse than baseline)")
    
    # Target comparison
    target_wer = 85.0
    if best_wer < target_wer:
        print(f"  Achieved target (< {target_wer}% WER)")
        print(f"  Next: Refine hyperparameters with statistical approach")
    else:
        print(f"  Did not achieve target (< {target_wer}% WER)")
        print(f"  Next: Try different blank bias or architecture")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 80)
    
    if final_blank > 90:
        print("  1. Blank ratio still very high (>90%)")
        print("     Try stronger blank bias (e.g., -4.0 or -5.0)")
    
    if final_unique < 100:
        print("  2. Vocabulary exploration still low (<100 unique tokens)")
        print("     Try stronger blank bias or longer Stage 1")
    
    if stage2_history and stage2_history[-1]['val_loss'] > stage2_history[0]['val_loss']:
        print("  3. Val loss increasing in Stage 2 (overfitting)")
        print("     Increase dropout or reduce learning rate")
    
    if best_wer < target_wer:
        print("  4. Target achieved! Next steps:")
        print("     Run hyperparameter search on blank bias")
        print("     Test different dropout schedules")
        print("     Consider 3-stage if needed")
    
    print("="*80)


def compare_experiments(history_files: List[str], labels: List[str]):
    """Compare multiple experiments."""
    print("="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    
    results = []
    for hist_file, label in zip(history_files, labels):
        history = load_history(hist_file)
        if history:
            best_wer = min(h['val_wer'] for h in history)
            final_unique = history[-1]['unique_nonblank_predictions']
            final_blank = history[-1]['blank_ratio']
            results.append({
                'label': label,
                'best_wer': best_wer,
                'final_unique': final_unique,
                'final_blank': final_blank
            })
    
    # Sort by best WER
    results.sort(key=lambda x: x['best_wer'])
    
    print("\nRanked by Best WER:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['label']}:")
        print(f"     Best WER: {r['best_wer']:.2f}%")
        print(f"     Unique Non-Blank: {r['final_unique']}")
        print(f"     Blank Ratio: {r['final_blank']:.2f}%")
    
    print("="*80)


if __name__ == "__main__":
    import sys
    import glob
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_experiment.py <history_file.json> [baseline_wer]")
        print("\nExample:")
        print("  python analyze_experiment.py logs/hierarchical_experimental/hierarchical_2stage_v1_history_*.json 95.0")
        print("  python analyze_experiment.py logs/hierarchical_experimental/hierarchical_2stage_v1_history_20251110_*.json 95.0")
        sys.exit(1)
    
    history_file_pattern = sys.argv[1]
    baseline_wer = float(sys.argv[2]) if len(sys.argv) > 2 else 95.0
    
    # Handle glob patterns
    if '*' in history_file_pattern or '?' in history_file_pattern:
        matching_files = glob.glob(history_file_pattern)
        if not matching_files:
            print(f"No files found matching pattern: {history_file_pattern}")
            sys.exit(1)
        # Use most recent file
        history_file = max(matching_files, key=os.path.getmtime)
        print(f"Found {len(matching_files)} matching file(s), using most recent: {history_file}")
    else:
        history_file = history_file_pattern
    
    if not os.path.exists(history_file):
        print(f"History file not found: {history_file}")
        sys.exit(1)
    
    analyze_experiment(history_file, baseline_wer)

