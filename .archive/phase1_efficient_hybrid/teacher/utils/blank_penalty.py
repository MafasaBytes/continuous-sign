def adjust_blank_penalty(current_penalty: float,
                         blank_ratio: float,
                         unique_nb: int,
                         nb_target_low: int = 300,
                         nb_target_high: int = 500,
                         min_pen: float = -8.0,
                         max_pen: float = 0.0) -> float:
    """Adaptive controller for blank penalty based on validation stats."""
    if blank_ratio > 90.0:
        current_penalty -= 0.7
    elif blank_ratio > 80.0:
        current_penalty -= 0.4
    elif unique_nb > nb_target_high and blank_ratio < 60.0:
        current_penalty += 0.3
    elif unique_nb > nb_target_low and blank_ratio < 70.0:
        current_penalty += 0.2
    return float(max(min_pen, min(max_pen, current_penalty)))


