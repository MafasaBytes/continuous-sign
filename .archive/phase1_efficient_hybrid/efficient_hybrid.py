"""Efficient hybrid architecture for sign language recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution for 8-9x parameter reduction."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        # Depthwise convolution (spatial filtering)
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        # Pointwise convolution (channel mixing)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU6(inplace=True)  # Mobile-friendly activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [B, C, T] -> [B, C', T]"""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class InvertedResidualBlock(nn.Module):
    """MobileNetV2-style inverted residual with linear bottleneck."""

    def __init__(self, in_channels: int, out_channels: int, expansion: int = 4,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expansion != 1:
            # Expand
            layers.append(nn.Conv1d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # Depthwise
        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=stride,
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # Project (linear bottleneck - no activation)
        layers.extend([
            nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.layers(x)
        return self.layers(x)


class MultiScaleTemporalBlock(nn.Module):
    """Multi-scale temporal feature extraction."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Different kernel sizes for different temporal scales
        self.branch1 = DepthwiseSeparableConv1d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.branch2 = DepthwiseSeparableConv1d(in_channels, out_channels//4, kernel_size=7, padding=3)
        self.branch3 = DepthwiseSeparableConv1d(in_channels, out_channels//4, kernel_size=15, padding=7)
        # Dilated convolution for long-range
        self.branch4 = nn.Conv1d(in_channels, out_channels//4, kernel_size=7, padding=6, dilation=2)

        self.fusion = nn.Conv1d(out_channels, out_channels, 1)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] -> [B, C', T]"""
        # Multi-scale processing
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Concatenate and fuse
        multi_scale = torch.cat([b1, b2, b3, b4], dim=1)
        output = self.fusion(multi_scale)
        output = self.norm(output)
        return F.relu(output)


class CrossModalAttention(nn.Module):
    """Attention mechanism across pose, hands, and face modalities."""

    def __init__(self, pose_dim: int, hand_dim: int, face_dim: int,
                 hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.pose_proj = nn.Linear(pose_dim, hidden_dim)
        self.hand_proj = nn.Linear(hand_dim, hidden_dim)
        self.face_proj = nn.Linear(face_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, pose: torch.Tensor, hands: torch.Tensor,
                face: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose: [B, T, pose_dim]
            hands: [B, T, hand_dim]
            face: [B, T, face_dim]
        Returns:
            [B, T, hidden_dim]
        """
        # Project to common dimension
        pose_feat = self.pose_proj(pose)
        hand_feat = self.hand_proj(hands)
        face_feat = self.face_proj(face)

        # Stack for attention [B, 3, T, hidden_dim]
        modalities = torch.stack([pose_feat, hand_feat, face_feat], dim=1)
        B, M, T, D = modalities.shape

        # Reshape for attention [B*T, M, D]
        modalities = modalities.permute(0, 2, 1, 3).reshape(B*T, M, D)

        # Self-attention across modalities
        attended, _ = self.attention(modalities, modalities, modalities)
        attended = self.norm(attended + modalities)

        # Reshape back and fuse [B, T, M*D]
        attended = attended.reshape(B, T, M*D)
        fused = self.fusion(attended)

        return fused


class ResidualBiLSTM(nn.Module):
    """BiLSTM with residual connection for gradient flow."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=1,
            batch_first=True, bidirectional=True, dropout=0
        )
        self.dropout = nn.Dropout(dropout)

        # Projection for residual if dimensions don't match
        self.residual_proj = None
        lstm_output_dim = hidden_dim * 2  # bidirectional
        if input_dim != lstm_output_dim:
            self.residual_proj = nn.Linear(input_dim, lstm_output_dim)

        self.norm = nn.LayerNorm(lstm_output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Forward with residual connection."""
        residual = x

        # Pack if lengths provided
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        x, _ = self.lstm(x)

        # Unpack if packed
        if lengths is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)

        x = self.dropout(x)

        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        x = self.norm(x + residual)
        return x


class EfficientHybridModel(nn.Module):
    """
    Efficient hybrid model combining CNN and LSTM with mobile optimizations.
    Target: WER < 20% with ~35M parameters.
    """

    def __init__(self,
                 input_dim: int = 1024,
                 hidden_dim: int = 256,  # Default to smaller size
                 num_classes: int = 966,
                 dropout: float = 0.5):  # Higher default dropout
        super().__init__()

        # Stage 1: Efficient temporal feature extraction
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Multi-scale temporal processing (CNN)
        self.temporal_encoder = nn.ModuleList([
            InvertedResidualBlock(hidden_dim, hidden_dim, expansion=2, kernel_size=5),
            InvertedResidualBlock(hidden_dim, hidden_dim, expansion=3, kernel_size=9),
            InvertedResidualBlock(hidden_dim, hidden_dim, expansion=4, kernel_size=15),
        ])

        # Stage 2: Sequential modeling with residuals (LSTM)
        self.lstm_layers = nn.ModuleList([
            ResidualBiLSTM(hidden_dim, hidden_dim//2, dropout=dropout),  # 768 -> 768
            ResidualBiLSTM(hidden_dim, hidden_dim//2, dropout=dropout),  # 768 -> 768
            ResidualBiLSTM(hidden_dim, hidden_dim//2, dropout=dropout),  # 768 -> 768
        ])

        # Stage 3: Simpler output projection for smaller model
        # With hidden_dim=256, don't expand as much to prevent overfitting
        expansion = min(2, max(1, 512 // hidden_dim))  # Adaptive expansion
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.LayerNorm(hidden_dim * expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion, num_classes)
        )

        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def forward(self, features: torch.Tensor, lengths: torch.Tensor = None,
                stage: int = 2, blank_penalty: float = 0.0,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Args:
            features: [B, T, D]
            lengths: [B]
            stage: Training stage (unused, for compatibility)
            blank_penalty: Penalty to apply to blank token logits
            temperature: Temperature for softmax (higher = more uniform)
        Returns:
            [T, B, C] log probabilities for CTC
        """
        batch_size, seq_len, _ = features.shape

        # Input projection
        x = self.input_proj(features)
        x = self.input_norm(x)
        x = self.dropout(x)

        # Transpose for CNN: [B, T, D] -> [B, D, T]
        x_cnn = x.transpose(1, 2)

        # Multi-scale temporal encoding
        for temporal_block in self.temporal_encoder:
            x_cnn = temporal_block(x_cnn)

        # Back to [B, T, D]
        x = x_cnn.transpose(1, 2)

        # Sequential modeling with residuals
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, lengths)

        # Output projection
        output = self.output_proj(x)

        # Apply blank penalty to encourage non-blank predictions (gradient-safe)
        if blank_penalty != 0:
            # Create penalty tensor to avoid in-place operations
            penalty = torch.zeros_like(output)
            penalty[:, :, 0] = -blank_penalty
            output = output + penalty

        # Temperature scaling for calibration
        if temperature != 1.0:
            output = output / temperature

        # Log probabilities for CTC
        log_probs = F.log_softmax(output, dim=-1)

        # Transpose for CTC: [B, T, C] -> [T, B, C]
        log_probs = log_probs.transpose(0, 1)

        return log_probs

    def _initialize_weights(self):
        """Xavier/He initialization for better gradient flow."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Forget gate bias = 1
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EfficientHybridWithCrossModal(nn.Module):
    """
    Advanced version with cross-modal attention for pose/hands/face.
    For use with full MediaPipe features (not PCA reduced).
    """

    def __init__(self,
                 pose_dim: int = 258,      # 33*3 + 33*4 + 33 (xyz + quaternion + visibility)
                 hand_dim: int = 126,       # 21*3*2 (left and right hands)
                 face_dim: int = 1404,      # 468*3
                 hidden_dim: int = 768,
                 num_classes: int = 1295,
                 dropout: float = 0.3):
        super().__init__()

        # Cross-modal attention fusion
        self.cross_modal = CrossModalAttention(
            pose_dim, hand_dim, face_dim,
            hidden_dim, num_heads=16, dropout=dropout
        )

        # Rest of the architecture
        self.temporal_encoder = nn.ModuleList([
            InvertedResidualBlock(hidden_dim, hidden_dim, expansion=2, kernel_size=5),
            InvertedResidualBlock(hidden_dim, hidden_dim, expansion=3, kernel_size=9),
            InvertedResidualBlock(hidden_dim, hidden_dim, expansion=4, kernel_size=15),
        ])

        self.lstm_layers = nn.ModuleList([
            ResidualBiLSTM(hidden_dim, hidden_dim//2, dropout=dropout),
            ResidualBiLSTM(hidden_dim, hidden_dim//2, dropout=dropout),
            ResidualBiLSTM(hidden_dim, hidden_dim//2, dropout=dropout),
            ResidualBiLSTM(hidden_dim, hidden_dim//2, dropout=dropout),  # Deeper
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )

        self._initialize_weights()

    def forward(self, pose: torch.Tensor, hands: torch.Tensor,
                face: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pose: [B, T, pose_dim]
            hands: [B, T, hand_dim]
            face: [B, T, face_dim]
            lengths: [B]
        Returns:
            [T, B, C] log probabilities
        """
        # Cross-modal attention fusion
        x = self.cross_modal(pose, hands, face)

        # Temporal encoding
        x_cnn = x.transpose(1, 2)
        for block in self.temporal_encoder:
            x_cnn = block(x_cnn)
        x = x_cnn.transpose(1, 2)

        # Sequential modeling
        for lstm in self.lstm_layers:
            x = lstm(x, lengths)

        # Output
        output = self.output_proj(x)
        log_probs = F.log_softmax(output, dim=-1)
        return log_probs.transpose(0, 1)

    def _initialize_weights(self):
        """Initialize weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def create_efficient_model(model_type: str = 'hybrid',
                          input_dim: int = 1024,
                          hidden_dim: int = 768,
                          num_classes: int = 966,
                          dropout: float = 0.3,
                          **kwargs) -> nn.Module:
    """
    Factory function to create efficient models.

    Args:
        model_type: 'hybrid' or 'cross_modal'
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_classes: Vocabulary size
        dropout: Dropout rate
        **kwargs: Additional arguments for cross_modal model

    Returns:
        Model instance
    """
    if model_type == 'hybrid':
        return EfficientHybridModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )
    elif model_type == 'cross_modal':
        return EfficientHybridWithCrossModal(
            pose_dim=kwargs.get('pose_dim', 258),
            hand_dim=kwargs.get('hand_dim', 126),
            face_dim=kwargs.get('face_dim', 1404),
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")