"""
MobileNetV3-based Sign Language Recognition Model - FIXED VERSION
Addresses gradient vanishing and CTC blank collapse issues
Target: < 100MB model size, < 25% WER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for MobileNetV3."""

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        squeeze_channels = max(1, in_channels // reduction)
        self.fc1 = nn.Conv1d(in_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv1d(squeeze_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool1d(x, 1)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale)
        return x * scale


class MobileNetV3Block(nn.Module):
    """MobileNetV3 building block with depthwise separable convolution."""

    def __init__(
        self,
        in_channels: int,
        exp_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = False,
        activation: str = 'relu'
    ):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []

        # Expansion layer
        if exp_channels != in_channels:
            layers.append(nn.Conv1d(in_channels, exp_channels, 1, bias=False))
            layers.append(nn.BatchNorm1d(exp_channels))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            else:  # Hard swish
                layers.append(nn.Hardswish(inplace=True))

        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        layers.append(nn.Conv1d(
            exp_channels, exp_channels, kernel_size,
            stride=stride, padding=padding, groups=exp_channels, bias=False
        ))
        layers.append(nn.BatchNorm1d(exp_channels))

        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Hardswish(inplace=True))

        # Squeeze-and-Excitation (disable initially for better gradient flow)
        if use_se and False:  # Disabled for now to improve gradient flow
            layers.append(SqueezeExcitation(exp_channels))

        # Pointwise linear projection
        layers.append(nn.Conv1d(exp_channels, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm1d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class ModalityEncoder(nn.Module):
    """Encoder for specific modality features."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.05):  # Reduced dropout
        super().__init__()
        hidden_dim = (input_dim + output_dim) // 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # Much lower dropout
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MobileNetV3SignLanguageFixed(nn.Module):
    """
    Fixed MobileNetV3-based architecture for sign language recognition.
    Addresses gradient vanishing and temporal alignment issues.
    """

    def __init__(
        self,
        vocab_size: int,
        pose_dim: int = 99,      # 33 keypoints × 3
        hands_dim: int = 126,    # 21 × 2 × 3
        face_dim: int = 1404,    # 468 × 3
        temporal_dim: int = 4887, # velocity + acceleration + spatial
        hidden_dim: int = 128,    # Reduced from 768 for efficiency
        num_lstm_layers: int = 1, # Reduced to 1 for baseline stability
        dropout: float = 0.05,    # MUCH LOWER dropout for gradient flow
    ):
        super().__init__()
        self.vocab_size = vocab_size

        # Modality-specific encoders with reduced dropout
        self.pose_encoder = ModalityEncoder(pose_dim, 64, dropout)
        self.hands_encoder = ModalityEncoder(hands_dim, 128, dropout)
        self.face_encoder = ModalityEncoder(face_dim, 64, dropout)
        self.temporal_encoder = ModalityEncoder(temporal_dim, 128, dropout)

        # Total encoded dimension
        encoded_dim = 64 + 128 + 64 + 128  # 384

        # MobileNetV3-Small backbone (adapted for 1D temporal data)
        self.stem = nn.Sequential(
            nn.Conv1d(encoded_dim, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.Hardswish(inplace=True)
        )

        # MobileNetV3-Small configuration - SIMPLIFIED
        mobilenet_config = [
            # Stage 1 - keep some downsampling for efficiency
            [16, 16, 3, 1, False, 'relu'],      # 1/2 (from stem)
            # Stage 2
            [72, 24, 3, 2, False, 'relu'],      # 1/4
            [88, 24, 3, 1, False, 'relu'],
            # Stage 3 - NO stride to preserve temporal info
            [96, 40, 5, 1, False, 'hswish'],    # SE disabled
            [240, 40, 5, 1, False, 'hswish'],
            # Stage 4 - reduced complexity
            [120, 48, 5, 1, False, 'hswish'],
        ]

        # Build MobileNetV3 layers
        in_channels = 16
        self.blocks = nn.ModuleList()
        for exp, out, k, s, se, act in mobilenet_config:
            self.blocks.append(MobileNetV3Block(
                in_channels, exp, out, k, s, se, act
            ))
            in_channels = out

        # Project to hidden dimension
        self.feature_proj = nn.Sequential(
            nn.Conv1d(48, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0  # No dropout in single-layer LSTM
        )

        # Output projection for CTC
        self.output_proj = nn.Linear(hidden_dim * 2, vocab_size)  # *2 for bidirectional

        # Initialize weights with better strategy
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights with improved strategy."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Special initialization for output layer
                if m.out_features == self.vocab_size:
                    # Use Xavier/Glorot initialization for better gradient flow
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    # Larger variance for other layers
                    nn.init.normal_(m.weight, 0, 0.02)  # Increased from 0.01
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        # Set forget gate bias to 1
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.)

    def extract_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Extract modality-specific features from input.
        """
        # Split features by modality
        pose = features[:, :, :99]           # Pose landmarks
        hands = features[:, :, 99:225]       # Hand landmarks
        face = features[:, :, 225:1629]      # Face landmarks
        temporal = features[:, :, 1629:]     # Temporal features

        return pose, hands, face, temporal

    def forward(
        self,
        features: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            features: [B, T, 6516] input features
            input_lengths: [B] tensor of sequence lengths

        Returns:
            [T, B, vocab_size] log probabilities for CTC loss
        """
        B, T_orig, _ = features.shape

        # Extract modality-specific features
        pose, hands, face, temporal = self.extract_features(features)

        # Encode each modality
        pose_enc = self.pose_encoder(pose)          # [B, T, 64]
        hands_enc = self.hands_encoder(hands)       # [B, T, 128]
        face_enc = self.face_encoder(face)          # [B, T, 64]
        temporal_enc = self.temporal_encoder(temporal)  # [B, T, 128]

        # Concatenate encoded features
        x = torch.cat([pose_enc, hands_enc, face_enc, temporal_enc], dim=-1)  # [B, T, 384]

        # Reshape for Conv1d (expects [B, C, T])
        x = x.transpose(1, 2)  # [B, 384, T]

        # MobileNetV3 backbone (with stride, reduces to T/4)
        x = self.stem(x)  # [B, 16, T/2]
        for block in self.blocks:
            x = block(x)  # [B, 48, T/4] after all blocks

        # Project features
        x = self.feature_proj(x)  # [B, hidden_dim, T/4]

        # CRITICAL FIX: Use interpolation to restore exact temporal dimension
        # This ensures perfect alignment for CTC loss
        if x.size(2) != T_orig:
            x = F.interpolate(x, size=T_orig, mode='linear', align_corners=False)

        # Transpose back to [B, T, hidden_dim]
        x = x.transpose(1, 2)  # [B, T, hidden_dim]

        # BiLSTM temporal modeling
        if input_lengths is not None:
            # Pack sequences for efficiency
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            x, _ = self.lstm(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=T_orig)
        else:
            x, _ = self.lstm(x)

        # Output projection
        x = self.output_proj(x)  # [B, T, vocab_size]

        # Apply log_softmax for CTC loss
        x = F.log_softmax(x, dim=-1)

        # Transpose for CTC loss (expects [T, B, vocab_size])
        x = x.transpose(0, 1)

        return x

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_mobilenet_v3_fixed_model(
    vocab_size: int,
    dropout: float = 0.05,  # Much lower default
    **kwargs
) -> MobileNetV3SignLanguageFixed:
    """
    Factory function to create fixed MobileNetV3 model for sign language.
    """
    model = MobileNetV3SignLanguageFixed(
        vocab_size=vocab_size,
        dropout=dropout,
        **kwargs
    )

    # Log model statistics
    total_params = model.count_parameters()
    model_size_mb = total_params * 4 / 1024 / 1024  # Assuming float32

    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Target: < 100 MB [{'PASS' if model_size_mb < 100 else 'FAIL'}]")

    return model


if __name__ == "__main__":
    # Test the model
    vocab_size = 973  # From dataset
    model = create_mobilenet_v3_fixed_model(vocab_size)

    # Test forward pass
    batch_size = 2
    seq_length = 176
    feature_dim = 6516

    x = torch.randn(batch_size, seq_length, feature_dim)
    input_lengths = torch.tensor([seq_length, seq_length-10])

    with torch.no_grad():
        output = model(x, input_lengths)
        print(f"\nInput shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")