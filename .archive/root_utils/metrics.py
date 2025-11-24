"""Evaluation metrics for sign language recognition."""

import numpy as np
from typing import List, Tuple
from collections import Counter


def edit_distance(s1: List[str], s2: List[str]) -> int:
    """
    Compute Levenshtein edit distance between two sequences.
    
    Args:
        s1: First sequence
        s2: Second sequence
    
    Returns:
        Edit distance
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def word_error_rate(references: List[List[str]], hypotheses: List[List[str]]) -> float:
    """
    Compute Word Error Rate (WER).
    
    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = total words in reference
    
    Args:
        references: List of reference sequences
        hypotheses: List of hypothesis sequences
    
    Returns:
        Word Error Rate
    """
    total_errors = 0
    total_words = 0
    
    for ref, hyp in zip(references, hypotheses):
        errors = edit_distance(ref, hyp)
        total_errors += errors
        total_words += len(ref)
    
    return total_errors / total_words if total_words > 0 else 0.0


def sign_error_rate(references: List[List[str]], hypotheses: List[List[str]]) -> float:
    """
    Compute Sign Error Rate (SER) - same as WER but for isolated signs.
    
    Args:
        references: List of reference sequences
        hypotheses: List of hypothesis sequences
    
    Returns:
        Sign Error Rate
    """
    return word_error_rate(references, hypotheses)


def compute_wer(references: List[str], hypotheses: List[str]) -> Tuple[float, dict]:
    """
    Compute WER from string sequences.
    
    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings
    
    Returns:
        WER and detailed statistics
    """
    ref_tokens = [ref.split() for ref in references]
    hyp_tokens = [hyp.split() for hyp in hypotheses]
    
    wer = word_error_rate(ref_tokens, hyp_tokens)
    
    # Compute detailed statistics
    total_ref_words = sum(len(ref) for ref in ref_tokens)
    total_hyp_words = sum(len(hyp) for hyp in hyp_tokens)
    total_errors = sum(edit_distance(ref, hyp) for ref, hyp in zip(ref_tokens, hyp_tokens))
    
    stats = {
        'wer': wer,
        'total_reference_words': total_ref_words,
        'total_hypothesis_words': total_hyp_words,
        'total_errors': total_errors,
        'num_sequences': len(references)
    }
    
    return wer, stats


def compute_ser(references: List[str], hypotheses: List[str]) -> Tuple[float, dict]:
    """
    Compute SER from string sequences.
    
    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings
    
    Returns:
        SER and detailed statistics
    """
    return compute_wer(references, hypotheses)  # SER is same as WER for sign language


def bleu_score(references: List[List[str]], hypotheses: List[List[str]], n: int = 4) -> float:
    """
    Compute BLEU score (simplified version).
    
    Args:
        references: List of reference sequences
        hypotheses: List of hypothesis sequences
        n: Maximum n-gram order
    
    Returns:
        BLEU score
    """
    # Simplified BLEU - just for reference
    # Full BLEU implementation would require more sophisticated n-gram matching
    if len(references) != len(hypotheses):
        return 0.0
    
    matches = 0
    total = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref_set = set(ref)
        hyp_set = set(hyp)
        matches += len(ref_set & hyp_set)
        total += len(ref_set)
    
    return matches / total if total > 0 else 0.0

