from typing import List
import torch

def decode_predictions(log_probs: torch.Tensor,
                       lengths: torch.Tensor,
                       blank_idx: int = 0) -> List[List[int]]:
    """
    Greedy CTC decoding: removes blanks and repeats.
    log_probs: [T, B, C]
    """
    batch_size = log_probs.size(1)
    predictions = []

    for b in range(batch_size):
        seq_len = lengths[b].item()
        seq_log_probs = log_probs[:seq_len, b, :]
        _, pred_indices = seq_log_probs.max(dim=-1)
        pred_indices = pred_indices.cpu().numpy()

        decoded = []
        prev = blank_idx
        for idx in pred_indices:
            if idx != blank_idx and idx != prev:
                decoded.append(int(idx))
            prev = idx
        predictions.append(decoded)

    return predictions


