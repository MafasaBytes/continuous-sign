"""Forced alignment utilities for CTC training.

This module provides utilities for using frame-level alignments to help
prevent CTC blank collapse by providing temporal supervision.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


class ForcedAlignmentLoader:
    """Load and process forced alignment data from Phoenix dataset."""
    
    def __init__(self, alignment_file: Path, classes_file: Path):
        """
        Initialize alignment loader.
        
        Args:
            alignment_file: Path to train.alignment file
            classes_file: Path to trainingClasses.txt file
        """
        self.alignment_file = alignment_file
        self.classes_file = classes_file
        self.class_to_idx = self._load_classes()
        self.alignments = self._load_alignments()
    
    def _load_classes(self) -> Dict[str, int]:
        """Load class mapping from trainingClasses.txt."""
        class_to_idx = {}
        with open(self.classes_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                class_name = line.strip()
                if class_name:
                    class_to_idx[class_name] = idx
        return class_to_idx
    
    def _load_alignments(self) -> Dict[str, List[int]]:
        """Load frame-level alignments."""
        alignments = {}
        with open(self.alignment_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    video_id = parts[0]
                    frame_labels = [int(x) for x in parts[1:]]
                    alignments[video_id] = frame_labels
        return alignments
    
    def get_alignment(self, video_id: str) -> Optional[List[int]]:
        """Get frame-level alignment for a video."""
        return self.alignments.get(video_id)
    
    def convert_to_word_labels(self, alignment: List[int], 
                               word_to_class_map: Dict[str, int]) -> List[int]:
        """
        Convert frame-level class labels to word-level labels.
        
        Note: This is a simplified conversion. Full conversion would require
        mapping from 3694 class labels (3 states per sign) to word-level.
        """
        # Simplified: map class labels to word indices
        # This is approximate - full implementation would need proper mapping
        word_labels = []
        prev_label = None
        for label in alignment:
            if label != prev_label:
                # Map class label to word (simplified)
                word_idx = label // 3  # Approximate: 3 states per sign
                if word_idx < len(word_to_class_map):
                    word_labels.append(word_idx)
            prev_label = label
        return word_labels


def create_alignment_loss(alignment_loader: Optional[ForcedAlignmentLoader] = None,
                         alignment_weight: float = 0.1):
    """
    Create a loss function that combines CTC with alignment supervision.
    
    This helps prevent blank collapse by providing temporal supervision.
    """
    if alignment_loader is None:
        return None
    
    def alignment_supervision_loss(log_probs: torch.Tensor,
                                   alignments: List[List[int]],
                                   sequence_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment supervision loss.
        
        Args:
            log_probs: [T, N, C] log probabilities
            alignments: List of frame-level alignments for each sequence
            sequence_lengths: [N] actual sequence lengths
        
        Returns:
            Alignment loss
        """
        loss = 0.0
        batch_size = log_probs.shape[1]
        
        for n in range(batch_size):
            seq_len = sequence_lengths[n].item()
            alignment = alignments[n]
            
            if len(alignment) > 0:
                # Align to sequence length
                alignment = alignment[:seq_len]
                
                # Compute negative log likelihood for aligned frames
                for t, label in enumerate(alignment):
                    if t < seq_len:
                        loss -= log_probs[t, n, label]  # Negative log likelihood
        
        return loss / batch_size
    
    return alignment_supervision_loss

