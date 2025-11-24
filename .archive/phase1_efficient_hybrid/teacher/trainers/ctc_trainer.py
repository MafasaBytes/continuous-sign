from typing import Dict, Optional
import time
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import os
import sys

# Add parent directory to path to allow imports from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from teacher.utils.decoding import decode_predictions
from utils.ctc import ctc_decode
from teacher.utils.metrics import compute_wer


def train_epoch(model: nn.Module, dataloader, criterion: nn.CTCLoss,
                optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                logger, blank_penalty: float = 0.0,
                time_mask_prob: float = 0.0, max_seq_len: Optional[int] = None,
                temperature: float = 1.0, gradient_clip: float = 10.0) -> Dict[str, float]:
                
    model.train()

    total_loss = 0.0
    total_samples = 0
    total_gradient_norm = 0.0
    batch_times = []

    pbar = tqdm(
        total=len(dataloader),
        desc=f"Epoch {epoch} - Training",
        leave=True,
        dynamic_ncols=True,
        position=0,
        mininterval=0.1,
        smoothing=0
    )
    for batch_idx, batch in enumerate(dataloader):
        start_time = time.time()

        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        if max_seq_len is not None:
            clipped_lengths = torch.clamp(input_lengths, max=max_seq_len)
            if features.size(1) > max_seq_len:
                features = features[:, :max_seq_len, :]
            input_lengths = clipped_lengths

        if time_mask_prob > 0:
            B, T, D = features.shape
            for i in range(B):
                seq_len = int(input_lengths[i].item())
                if seq_len > 0 and random.random() < time_mask_prob:
                    width = min(12, max(1, seq_len // 10))
                    start = random.randint(0, max(0, seq_len - width))
                    features[i, start:start+width, :] = 0.0

        log_probs = model(features, input_lengths, stage=2,
                          blank_penalty=blank_penalty, temperature=temperature)

        loss = criterion(log_probs, labels, input_lengths, target_lengths)
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}")
            continue

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        total_loss += float(loss.item()) * features.size(0)
        total_samples += features.size(0)
        total_gradient_norm += float(grad_norm)
        batch_times.append(time.time() - start_time)

        # Update a concise postfix every few batches to reduce flicker
        if batch_idx % 10 == 0:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}", grad=f"{float(grad_norm):.2f}", bp=f"{blank_penalty:.2f}")
        pbar.update(1)
        pbar.refresh()
    pbar.close()

    return {
        'train_loss': total_loss / total_samples if total_samples > 0 else 0.0,
        'gradient_norm': total_gradient_norm / max(1, len(dataloader)),
        'batch_time': float(np.mean(batch_times)) if batch_times else 0.0,
        'total_time': float(sum(batch_times))
    }


def validate(model: nn.Module, dataloader, criterion: nn.CTCLoss,
             vocab, device: torch.device, logger, epoch: int = 0,
             blank_penalty: float = 0.0, temperature: float = 1.0,
             decode_method: str = 'greedy', beam_width: int = 10) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []

    from collections import Counter
    frame_blank_frames = 0
    frame_total_frames = 0
    frame_token_counts = Counter()

    too_short_count = 0
    total_sequences = 0

    with torch.no_grad():
        pbar = tqdm(
            total=len(dataloader),
            desc="Validating",
            leave=True,              # keep validation bar visible
            dynamic_ncols=True,
            position=1,              # render below training bar
            mininterval=0.1,
            smoothing=0
        )
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            target_lengths = batch['target_lengths'].to(device)

            log_probs = model(features, input_lengths, stage=2,
                              blank_penalty=blank_penalty, temperature=temperature)

            B = features.size(0)
            for b in range(B):
                seq_len = input_lengths[b].item()
                frame_preds = log_probs[:seq_len, b, :].argmax(dim=-1)
                frame_blank_frames += (frame_preds == 0).sum().item()
                frame_total_frames += seq_len
                frame_token_counts.update(frame_preds.cpu().tolist())

            loss = criterion(log_probs, labels, input_lengths, target_lengths)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += float(loss.item()) * features.size(0)
                total_samples += features.size(0)

            if decode_method == 'greedy':
                predictions = decode_predictions(log_probs, input_lengths)
            else:
                predictions = ctc_decode(log_probs, blank_idx=0, method=decode_method, beam_width=beam_width)
            for b in range(len(predictions)):
                target_len = target_lengths[b].item()
                target = labels[b, :target_len].cpu().tolist()
                all_targets.append(target)
                all_predictions.append(predictions[b])

                T = input_lengths[b].item()
                L = target_len
                if T < (2 * L + 1):
                    too_short_count += 1
                total_sequences += 1

            # Show simple running stats in the bar
            if total_samples > 0:
                running_val = total_loss / total_samples
                pbar.set_postfix(val=f"{running_val:.4f}", blanks=f"{(frame_blank_frames/(frame_total_frames+1e-8))*100:.1f}%")
            pbar.update(1)
            pbar.refresh()
        pbar.close()

    wer = compute_wer(all_predictions, all_targets)
    frame_blank_ratio = (frame_blank_frames / frame_total_frames) * 100 if frame_total_frames > 0 else 0.0
    too_short_ratio = (too_short_count / total_sequences) * 100 if total_sequences > 0 else 0.0
    if frame_total_frames > 0 and len(frame_token_counts) > 0:
        most_common_token, most_common_count = frame_token_counts.most_common(1)[0]
        frame_top_token_ratio = (most_common_count / frame_total_frames) * 100
    else:
        most_common_token, frame_top_token_ratio = -1, 0.0

    return {
        'val_loss': total_loss / total_samples if total_samples > 0 else 0.0,
        'val_wer': float(wer),
        'blank_ratio': float(frame_blank_ratio),
        'unique_predictions': len(set(t for seq in all_predictions for t in seq)),
        'unique_nonblank_predictions': len(set(t for seq in all_predictions for t in seq if t != 0)),
        'frame_unique_predictions': len(frame_token_counts),
        'ctc_too_short_ratio': float(too_short_ratio),
        'frame_top_token_id': int(most_common_token),
        'frame_top_token_ratio': float(frame_top_token_ratio),
    }


