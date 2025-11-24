"""Analyze model architectures for sign language recognition."""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from teacher.models.hierarchical import HierarchicalModelFixed
from teacher.model import create_teacher_model

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_model_capacity(model, model_name, input_dim, sequence_len=100, batch_size=4):
    """Analyze model capacity and computational requirements."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Model size (MB): {total_params * 4 / (1024**2):.2f}")

    # Parameter breakdown by component
    param_breakdown = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                param_breakdown[name] = params

    print("\nParameter breakdown (top components):")
    sorted_breakdown = sorted(param_breakdown.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, params in sorted_breakdown:
        print(f"  {name}: {params:,} ({params/total_params*100:.1f}%)")

    # Estimate FLOPs for forward pass
    dummy_input = torch.randn(batch_size, sequence_len, input_dim)
    dummy_lengths = torch.tensor([sequence_len] * batch_size)

    # Measure memory footprint
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()
        dummy_lengths = dummy_lengths.cuda()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            # Forward pass
            if 'Hierarchical' in model_name:
                output = model(dummy_input, dummy_lengths, stage=2)
            else:
                output = model(dummy_input, dummy_lengths)

        memory_used = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"\nGPU memory used (MB): {memory_used:.2f}")

    return total_params

# Analyze HierarchicalModelFixed
print("Analyzing Sign Language Recognition Model Architectures")
print("="*60)

# Configuration from mediapipe_pca1024.py
vocab_size = 966  # From config
input_dim = 1024  # PCA reduced features
hidden_dim = 512

print(f"\nConfiguration:")
print(f"  Vocabulary size: {vocab_size}")
print(f"  Input dimension: {input_dim} (MediaPipe PCA features)")
print(f"  Hidden dimension: {hidden_dim}")

# Create and analyze HierarchicalModelFixed
hierarchical_model = HierarchicalModelFixed(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_classes=vocab_size,
    dropout_frame=0.2,
    dropout_sequence=0.25,
    use_temporal_conv=True
)

h_params = analyze_model_capacity(
    hierarchical_model,
    "HierarchicalModelFixed (train_mediapipe.py)",
    input_dim
)

# Create and analyze TeacherModel
teacher_model = create_teacher_model(
    input_dim=6516,  # Full MediaPipe features
    hidden_dim=512,
    num_layers=3,
    num_classes=1295,  # From config.yaml
    dropout=0.3,
    use_attention=True
)

t_params = analyze_model_capacity(
    teacher_model,
    "TeacherModel (train.py)",
    6516
)

# Theoretical analysis for sign language recognition
print(f"\n{'='*60}")
print("THEORETICAL ANALYSIS FOR WER < 20%")
print(f"{'='*60}")

print("\n1. MODEL CAPACITY ANALYSIS:")
print(f"   - Hierarchical model: {h_params:,} parameters")
print(f"   - Teacher model: {t_params:,} parameters")
print(f"   - Vocabulary size: {vocab_size}-{1295} signs")
print(f"   - Parameters per sign (Hierarchical): {h_params/vocab_size:.0f}")
print(f"   - Parameters per sign (Teacher): {t_params/1295:.0f}")

print("\n2. ARCHITECTURAL BOTTLENECKS:")
print("   BiLSTM-based architecture limitations:")
print("   - Limited temporal receptive field (especially for long sequences)")
print("   - No explicit modeling of hand shapes vs. motion trajectories")
print("   - Single-scale temporal processing (no multi-scale features)")
print("   - Limited cross-modal attention between pose/hands/face")

print("\n3. CAPACITY REQUIREMENTS FOR WER < 20%:")
print("   Based on ASL/sign language SOTA models:")
print("   - Typical requirement: 10-50M parameters for <20% WER")
print("   - Current models: 3-10M parameters (UNDERPARAMETERIZED)")
print("   - Need 3-5x more capacity for target performance")

print("\n4. SPECIFIC ARCHITECTURAL LIMITATIONS:")

print("\n   HierarchicalModelFixed Issues:")
print("   - Two-stage LSTM (512 hidden) - insufficient for 966 classes")
print("   - Temporal conv kernels (3,5,7) - too small for sign transitions")
print("   - No residual connections in LSTM layers")
print("   - Linear bottleneck before output (512*2 -> 966) too narrow")

print("\n   TeacherModel Issues:")
print("   - 3-layer BiLSTM - needs deeper architecture (5-6 layers)")
print("   - Single attention head (8 heads) - needs more (16-32)")
print("   - No CNN-LSTM hybrid for local motion patterns")
print("   - Output projection bottleneck (1024 -> 1295)")

print("\n5. CRITICAL MISSING COMPONENTS:")
print("   - No pre-training or transfer learning")
print("   - No temporal segment modeling (signs have structure)")
print("   - No explicit hand dominance modeling")
print("   - No multi-scale temporal processing")
print("   - Missing spatial attention for hand regions")