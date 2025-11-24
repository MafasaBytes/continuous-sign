"""CTC (Connectionist Temporal Classification) utilities."""

import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np


class CTCLoss(nn.Module):
    """CTC Loss wrapper with blank penalty to prevent blank collapse."""
    
    def __init__(self, blank_idx: int = 0, reduction: str = 'mean', blank_penalty: float = 0.0,
                 repetition_penalty: float = 0.0):
        """
        Initialize CTC Loss with optional blank and repetition penalties.
        
        Args:
            blank_idx: Index of blank token
            reduction: 'mean' or 'sum'
            blank_penalty: Penalty weight for blank predictions (0.0 = no penalty)
                          Higher values (0.1-0.5) discourage blank-heavy predictions
            repetition_penalty: Penalty weight for consecutive same-token predictions (0.0 = no penalty)
                               Higher values (0.1-0.5) discourage repetitive predictions
        """
        super().__init__()
        self.blank_idx = blank_idx
        self.blank_penalty = blank_penalty
        self.repetition_penalty = repetition_penalty
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=True)
    
    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor, 
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute CTC loss with optional blank penalty.
        
        Args:
            log_probs: [T, N, C] log probabilities (T=time, N=batch, C=vocab)
            targets: [N*S] target sequence (concatenated)
            input_lengths: [N] actual sequence lengths (can be on GPU, will be moved to CPU only for CTC loss)
            target_lengths: [N] target sequence lengths (can be on GPU, will be moved to CPU only for CTC loss)
        
        Returns:
            CTC loss value (with blank penalty if enabled)
        """
        # Keep lengths on GPU for penalty computations
        # Only move to CPU right before CTC loss call
        device = log_probs.device
        T = log_probs.shape[0]
        
        # Ensure input_lengths don't exceed sequence length (on GPU)
        input_lengths_gpu = input_lengths.long().clamp(max=T)
        
        # Compute penalties on GPU before moving lengths to CPU
        blank_penalty_term = None
        repetition_penalty_term = None
        
        # Add mild blank penalty to discourage excessive blank predictions
        # SIMPLIFIED: Only use blank penalty for overfit test, no other penalties
        # Research advisor: "regularize last, not first" - avoid penalty stacking during early training
        if self.blank_penalty > 0.0:
            # Compute average blank probability across all timesteps (all on GPU)
            blank_log_probs = log_probs[:, :, self.blank_idx]  # [T, N]
            blank_probs = torch.exp(blank_log_probs)  # [T, N]

            # Vectorized computation: mask by sequence lengths and compute weighted mean
            batch_size = blank_probs.shape[1]
            # Create mask: [T, N] where True indicates valid timesteps
            seq_indices = torch.arange(T, device=device).unsqueeze(1)  # [T, 1]
            length_mask = seq_indices < input_lengths_gpu.unsqueeze(0)  # [T, N]
            
            # Compute mean blank probability per sequence (vectorized)
            masked_blank_probs = blank_probs * length_mask.float()  # [T, N]
            seq_blank_probs = masked_blank_probs.sum(dim=0) / input_lengths_gpu.float().clamp(min=1)  # [N]
            blank_penalty_term = seq_blank_probs.mean()  # Scalar
        
        # Add repetition penalty to discourage consecutive same-token predictions
        # This addresses CTC's tendency to allow many-to-one alignment where same token
        # is predicted at all timesteps, which CTC collapses to single token
        if self.repetition_penalty > 0.0:
            probs = torch.exp(log_probs)  # [T, N, C]
            batch_size = probs.shape[1]
            
            # Get argmax predictions (all on GPU)
            argmax_preds = torch.argmax(probs, dim=2)  # [T, N]
            
            # Vectorized computation: find consecutive same tokens
            # Shift predictions by 1 timestep to compare with previous
            prev_preds = torch.cat([argmax_preds[0:1, :], argmax_preds[:-1, :]], dim=0)  # [T, N]
            
            # Find where current == previous and both are non-blank
            same_token = (argmax_preds == prev_preds)  # [T, N]
            non_blank = (argmax_preds != self.blank_idx)  # [T, N]
            consecutive_mask = same_token & non_blank  # [T, N]
            
            # Mask by sequence lengths
            seq_indices = torch.arange(T, device=device).unsqueeze(1)  # [T, 1]
            length_mask = seq_indices < input_lengths_gpu.unsqueeze(0)  # [T, N]
            # Don't count first timestep (no previous token)
            length_mask[0, :] = False
            
            # Count consecutive same tokens per sequence
            masked_consecutive = consecutive_mask & length_mask
            consecutive_counts = masked_consecutive.sum(dim=0).float()  # [N]
            valid_lengths = (input_lengths_gpu - 1).float().clamp(min=1)  # [N], exclude first timestep
            repetition_ratios = consecutive_counts / valid_lengths  # [N]
            repetition_penalty_term = repetition_ratios.mean()  # Scalar
        
        # Now move lengths to CPU only for CTC loss call
        input_lengths_cpu = input_lengths_gpu.cpu()
        target_lengths_cpu = target_lengths.cpu().long()
        
        # Standard CTC loss (requires CPU lengths)
        ctc_loss_value = self.ctc_loss(log_probs, targets, input_lengths_cpu, target_lengths_cpu)
        
        # Apply penalties computed on GPU
        if blank_penalty_term is not None:
            ctc_loss_value = ctc_loss_value + self.blank_penalty * (blank_penalty_term ** 2)
        
        if repetition_penalty_term is not None:
            ctc_loss_value = ctc_loss_value + self.repetition_penalty * (repetition_penalty_term ** 2)
        
        return ctc_loss_value


def ctc_decode(log_probs: torch.Tensor, blank_idx: int = 0, method: str = 'greedy',
                beam_width: int = 10) -> List[List[int]]:
    """
    Decode CTC output.

    Args:
        log_probs: [T, N, C] log probabilities
        blank_idx: Index of blank token
        method: 'greedy' or 'beam_search'
        beam_width: Beam width for beam search (default: 10)
        use_gpu: Use GPU-accelerated beam search if available (default: True)

    Returns:
        List of decoded sequences
    """
    if method == 'greedy':
        return ctc_greedy_decode(log_probs, blank_idx)

    elif method == 'beam_search':
        return ctc_beam_search_decode(log_probs, blank_idx, beam_width)

    else:
        raise ValueError(f"Unknown decode method: {method}")


def ctc_greedy_decode(log_probs: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
    """
    Greedy CTC decoding (GPU-accelerated).
    
    Args:
        log_probs: [T, N, C] log probabilities
        blank_idx: Index of blank token
    
    Returns:
        List of decoded sequences
    """
    # Handle both [T, N, C] and [N, T, C] formats
    if log_probs.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {log_probs.dim()}D")
    
    # Assume [T, N, C] format as per docstring
    T, N, C = log_probs.shape
    
    device = log_probs.device
    probs = torch.exp(log_probs)  # [T, N, C]
    predictions = torch.argmax(probs, dim=2)  # [T, N] - all on GPU
    
    # Vectorized CTC decoding on GPU
    # Remove blanks and collapse repeats
    # Strategy: shift predictions and compare to find transitions
    
    # Create mask for non-blank tokens
    non_blank_mask = (predictions != blank_idx)  # [T, N]
    
    # Find transitions: where token changes (or first non-blank)
    # Shift predictions by 1 to compare with previous
    prev_predictions = torch.cat([
        torch.full((1, N), blank_idx, device=device, dtype=predictions.dtype),
        predictions[:-1, :]
    ], dim=0)  # [T, N]
    
    # Token changes when: (current != previous) OR (current is first non-blank after blank)
    token_changes = (predictions != prev_predictions)  # [T, N]
    
    # Keep only non-blank tokens that represent transitions
    keep_mask = non_blank_mask & token_changes  # [T, N]
    
    # For each sequence, collect kept tokens
    decoded_sequences = []
    for n in range(N):
        # Get indices where we keep tokens
        kept_tokens = predictions[:, n][keep_mask[:, n]]  # [num_kept]
        # Convert to list (only move to CPU at the very end)
        decoded_sequences.append(kept_tokens.cpu().tolist())
    
    return decoded_sequences


def ctc_beam_search_decode(log_probs: torch.Tensor, blank_idx: int = 0, 
                          beam_width: int = 10) -> List[List[int]]:
    """
    Beam search CTC decoding with proper prefix beam search algorithm.
    
    Args:
        log_probs: [T, N, C] log probabilities
        blank_idx: Index of blank token
        beam_width: Beam width for search
    
    Returns:
        List of decoded sequences
    """
    # Handle both [T, N, C] and [N, T, C] formats
    if log_probs.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {log_probs.dim()}D")
    
    # Assume [T, N, C] format as per docstring
    T, N, C = log_probs.shape
    
    device = log_probs.device
    # Pre-compute log probabilities on GPU (more numerically stable than exp then log)
    # Add small epsilon to avoid log(0)
    log_prob_dist_all = log_probs  # Already log probabilities, [T, N, C]
    
    decoded_sequences = []
    
    for n in range(N):
        # Prefix beam search for sequence n
        # We maintain prefixes as (sequence, last_token, log_prob)
        # Use log probabilities to avoid numerical issues
        prefixes = {tuple(): (blank_idx, 0.0)}  # Start with empty prefix
        
        for t in range(T):
            # Get log probabilities for this timestep (keep on GPU)
            log_prob_dist = log_prob_dist_all[t, n, :]  # [C] - on GPU
            
            new_prefixes = {}
            
            for prefix_seq, (last_token, prefix_log_prob) in prefixes.items():
                # Batch compute all token log probabilities on GPU
                # Add prefix_log_prob to all token log probs at once
                token_log_probs = log_prob_dist + prefix_log_prob  # [C] - all on GPU
                
                # Process blank token first (most common case)
                blank_log_prob = token_log_probs[blank_idx].item()
                if prefix_seq not in new_prefixes:
                    new_prefixes[prefix_seq] = (last_token, blank_log_prob)
                else:
                    # Merge probabilities (log-sum-exp approximation: use max for speed)
                    old_log_prob = new_prefixes[prefix_seq][1]
                    if blank_log_prob > old_log_prob:
                        new_prefixes[prefix_seq] = (last_token, blank_log_prob)
                
                # Process non-blank tokens
                # Get all non-blank token log probs at once (vectorized on GPU)
                non_blank_mask = torch.arange(C, device=device) != blank_idx
                non_blank_log_probs = token_log_probs[non_blank_mask]  # [C-1] - on GPU
                non_blank_indices = torch.arange(C, device=device)[non_blank_mask]  # [C-1] - on GPU
                
                # Batch transfer to CPU (single transfer instead of C-1 individual transfers)
                non_blank_log_probs_cpu = non_blank_log_probs.cpu()
                non_blank_indices_cpu = non_blank_indices.cpu()
                
                # Iterate over non-blank tokens (dictionary operations require CPU)
                for idx in range(len(non_blank_indices_cpu)):
                    token_idx = non_blank_indices_cpu[idx].item()
                    new_log_prob = non_blank_log_probs_cpu[idx].item()
                    
                    if token_idx == last_token:
                        # Same as last: extend without adding (CTC collapse)
                        if prefix_seq not in new_prefixes:
                            new_prefixes[prefix_seq] = (token_idx, new_log_prob)
                        else:
                            old_log_prob = new_prefixes[prefix_seq][1]
                            if new_log_prob > old_log_prob:
                                new_prefixes[prefix_seq] = (token_idx, new_log_prob)
                    else:
                        # Different token: extend by adding new token
                        new_seq = prefix_seq + (token_idx,)
                        if new_seq not in new_prefixes:
                            new_prefixes[new_seq] = (token_idx, new_log_prob)
                        else:
                            old_log_prob = new_prefixes[new_seq][1]
                            if new_log_prob > old_log_prob:
                                new_prefixes[new_seq] = (token_idx, new_log_prob)
            
            # Keep top beam_width prefixes
            sorted_prefixes = sorted(new_prefixes.items(), key=lambda x: x[1][1], reverse=True)
            prefixes = dict(sorted_prefixes[:beam_width])
        
        # Get best sequence
        if prefixes:
            best_prefix = max(prefixes.items(), key=lambda x: x[1][1])
            decoded_sequences.append(list(best_prefix[0]))
        else:
            decoded_sequences.append([])
    
    return decoded_sequences


def prepare_ctc_targets(targets: List[List[int]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare targets for CTC loss.
    
    Args:
        targets: List of target sequences
        device: Device to place tensors on
    
    Returns:
        (targets_tensor, target_lengths)
        - targets_tensor: [N*S] concatenated targets
        - target_lengths: [N] target lengths
    """
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long, device=device)
    targets_flat = [token for seq in targets for token in seq]
    targets_tensor = torch.tensor(targets_flat, dtype=torch.long, device=device)
    
    return targets_tensor, target_lengths

