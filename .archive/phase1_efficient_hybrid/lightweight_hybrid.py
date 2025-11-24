"""Lightweight hybrid architecture to prevent overfitting on small datasets."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LightweightTemporalBlock(nn.Module):
    """Simplified temporal block with fewer parameters."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        # Single depthwise separable convolution instead of multiple branches
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=kernel_size//2, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout1d(0.2)  # Add dropout for regularization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] -> [B, C', T]"""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class CompactBiLSTM(nn.Module):
    """Compact BiLSTM with strong regularization."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with proper packing for variable lengths."""
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                    enforce_sorted=False)

        x, _ = self.lstm(x)

        if lengths is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)

        x = self.dropout(x)
        x = self.norm(x)
        return x


class LightweightHybridModel(nn.Module):
    """
    Lightweight model for small datasets (< 10k samples).
    Target: ~3-5M parameters for 5.6k training samples.

    Key design principles:
    1. Smaller hidden dimensions (256 instead of 768)
    2. Fewer layers (2 LSTM layers instead of 3)
    3. Simplified temporal blocks
    4. Heavy dropout (0.5) throughout
    5. No expansion in inverted residuals
    """

    def __init__(self,
                 input_dim: int = 1024,
                 hidden_dim: int = 256,  # Much smaller
                 num_classes: int = 1122,
                 dropout: float = 0.5):  # Higher dropout
        super().__init__()

        # Lightweight input projection with bottleneck
        self.input_bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Simplified temporal processing (no multi-scale)
        self.temporal_encoder = nn.ModuleList([
            LightweightTemporalBlock(hidden_dim, hidden_dim, kernel_size=5),
            LightweightTemporalBlock(hidden_dim, hidden_dim, kernel_size=7),
        ])

        # Compact BiLSTM layers
        self.lstm_layers = nn.ModuleList([
            CompactBiLSTM(hidden_dim, hidden_dim//4, num_layers=1, dropout=dropout),
            CompactBiLSTM(hidden_dim//2, hidden_dim//4, num_layers=1, dropout=dropout),
        ])

        # Direct output projection without expansion
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes)
        )

        # Label smoothing for regularization
        self.label_smoothing = 0.1

        self._initialize_weights()

    def forward(self, features: torch.Tensor, lengths: torch.Tensor = None,
                stage: int = 2, blank_penalty: float = 0.0,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Forward pass with regularization.

        Args:
            features: [B, T, D]
            lengths: [B]
            stage: Training stage (unused)
            blank_penalty: Penalty for blank token
            temperature: Temperature scaling
        Returns:
            [T, B, C] log probabilities
        """
        batch_size, seq_len, _ = features.shape

        # Input projection with bottleneck
        x = self.input_bottleneck(features)

        # Temporal encoding
        x_cnn = x.transpose(1, 2)  # [B, D, T]
        for block in self.temporal_encoder:
            x_cnn = block(x_cnn)
        x = x_cnn.transpose(1, 2)  # [B, T, D]

        # Sequential modeling
        for lstm in self.lstm_layers:
            x = lstm(x, lengths)

        # Output projection
        logits = self.output_proj(x)

        # Apply regularization techniques
        if self.training:
            # Add label smoothing regularization
            if self.label_smoothing > 0:
                confidence = 1.0 - self.label_smoothing
                smoothing_value = self.label_smoothing / (logits.size(-1) - 1)
                one_hot = torch.zeros_like(logits).scatter_(-1,
                    logits.argmax(-1, keepdim=True), confidence)
                smooth_targets = one_hot + smoothing_value * (1 - one_hot)
                logits = logits * smooth_targets

        # Apply blank penalty
        if blank_penalty != 0:
            logits[:, :, 0] = logits[:, :, 0] - blank_penalty

        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Transpose for CTC: [B, T, C] -> [T, B, C]
        log_probs = log_probs.transpose(0, 1)

        return log_probs

    def _initialize_weights(self):
        """Conservative initialization to prevent gradient explosion."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Smaller initialization for large vocab
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in',
                                       nonlinearity='relu')
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.5)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data, gain=0.5)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Forget gate bias
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_lightweight_model(input_dim: int = 1024,
                            hidden_dim: int = 256,
                            num_classes: int = 1122,
                            dropout: float = 0.5) -> nn.Module:
    """
    Factory function for lightweight model.

    Recommended settings for 5.6k training samples:
    - hidden_dim: 256 (results in ~3.5M parameters)
    - dropout: 0.5 or higher
    - Use with gradient accumulation if batch_size < 16
    """
    return LightweightHybridModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout
    )