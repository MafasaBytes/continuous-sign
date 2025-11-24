from typing import List

def edit_distance(seq1: List[int], seq2: List[int]) -> int:
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def compute_wer(predictions: List[List[int]], targets: List[List[int]]) -> float:
    total_errors = 0
    total_words = 0
    for pred, target in zip(predictions, targets):
        errors = edit_distance(pred, target)
        total_errors += errors
        total_words += len(target)
    if total_words == 0:
        return 0.0
    return (total_errors / total_words) * 100


