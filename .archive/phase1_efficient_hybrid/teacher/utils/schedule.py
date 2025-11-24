def get_warmup_lr(epoch: int, warmup_epochs: int, base_lr: float, start_lr: float = 1e-6) -> float:
    if epoch <= warmup_epochs:
        return start_lr + (base_lr - start_lr) * (epoch / max(1, warmup_epochs))
    return base_lr


def get_decaying_blank_penalty(epoch: int, start_penalty: float,
                               end_penalty: float, total_epochs: int) -> float:
    """Linear interpolation start→end (works for negatives)."""
    if total_epochs <= 0:
        return end_penalty
    e = max(0, min(epoch, total_epochs))
    return start_penalty + (end_penalty - start_penalty) * (e / total_epochs)


