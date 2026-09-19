"""
Conformal prediction utilities: LAC and APS.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional


def lac_threshold(scores: np.ndarray, alpha: float) -> float:
    """LAC: q_alpha = ceil((n+1)*(1-alpha))/n quantile of calibration scores."""
    n = len(scores)
    if n == 0:
        return float('inf')
    q = np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
    return float(q)


def lac_predict(score: float, q_alpha: float) -> bool:
    """LAC: covered if score <= q_alpha."""
    return score <= q_alpha


def aps_threshold(prob_scores: np.ndarray, alpha: float) -> float:
    """APS: threshold on cumulative probability."""
    # prob_scores: (n, K) softmax probabilities for n cal examples, K classes
    # For each example, sort descending and find rank where cumsum crosses 1-alpha
    sorted_probs = -np.sort(-prob_scores, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    # Smallest k such that cumsum[k] >= 1-alpha; threshold = cumsum[k]
    thresholds = []
    for i in range(len(prob_scores)):
        # Find first k where cumsum >= 1-alpha
        idx = np.searchsorted(cumsum[i], 1 - alpha)
        if idx < len(sorted_probs[i]):
            thresholds.append(cumsum[i, idx])
        else:
            thresholds.append(cumsum[i, -1] + 1e-6)
    thresholds = np.array(thresholds)
    q = np.quantile(thresholds, min(1.0, np.ceil((len(prob_scores) + 1) * (1 - alpha)) / len(prob_scores)))
    return float(q)


def aps_prediction_set(probs: np.ndarray, q_alpha: float) -> List[int]:
    """APS: build prediction set for a single example's probabilities."""
    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    # Include classes until cumsum >= q_alpha
    k = np.searchsorted(cumsum, q_alpha) + 1
    k = min(k, len(probs))
    return sorted(sorted_idx[:k].tolist())


def domain_conditional_threshold(scores: np.ndarray, domains: np.ndarray, alpha: float) -> Dict[str, float]:
    """Per-domain LAC thresholds."""
    result = {}
    for d in np.unique(domains):
        mask = domains == d
        if mask.sum() < 10:
            # Too few: fall back to global
            result[d] = lac_threshold(scores, alpha)
        else:
            result[d] = lac_threshold(scores[mask], alpha)
    return result


def difficulty_binned_threshold(scores: np.ndarray, difficulty: np.ndarray, alpha: float, n_bins: int = 3) -> Dict[int, float]:
    """Difficulty-binned LAC thresholds."""
    quantiles = np.quantile(difficulty, np.linspace(0, 1, n_bins + 1))
    result = {}
    for b in range(n_bins):
        if b == n_bins - 1:
            mask = (difficulty >= quantiles[b]) & (difficulty <= quantiles[b + 1])
        else:
            mask = (difficulty >= quantiles[b]) & (difficulty < quantiles[b + 1])
        if mask.sum() < 10:
            result[b] = lac_threshold(scores, alpha)
        else:
            result[b] = lac_threshold(scores[mask], alpha)
    return result


def coverage_stats(covered: List[bool], correct: List[bool]) -> Dict[str, float]:
    """Compute coverage and risk from boolean lists."""
    covered = np.array(covered)
    correct = np.array(correct)
    n = len(covered)
    if n == 0:
        return {'coverage': 0.0, 'risk': 0.0, 'abstention': 1.0, 'n': 0}
    coverage = covered.mean()
    if covered.sum() == 0:
        risk = 1.0
    else:
        # Risk = P(wrong | covered)
        risk = 1 - correct[covered].mean() if covered.sum() > 0 else 1.0
    return {
        'coverage': float(coverage),
        'risk': float(risk),
        'abstention': float(1 - coverage),
        'n': n,
        'n_covered': int(covered.sum()),
        'n_correct_covered': int((correct & covered).sum()),
    }
