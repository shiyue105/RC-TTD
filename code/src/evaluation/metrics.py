"""
Evaluation metrics for RAG conformal decision policies.
"""
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics import roc_auc_score, brier_score_loss


def accuracy(correct: List[bool], answered: List[bool]) -> float:
    """Accuracy among answered."""
    correct = np.array(correct)
    answered = np.array(answered)
    if answered.sum() == 0:
        return 0.0
    return float(correct[answered].mean())


def coverage(answered: List[bool]) -> float:
    return float(np.array(answered).mean())


def abstention_rate(answered: List[bool]) -> float:
    return float(1 - np.array(answered).mean())


def selective_risk(correct: List[bool], answered: List[bool]) -> float:
    """P(wrong | answered)."""
    return 1.0 - accuracy(correct, answered)


def ece(confidences: List[float], correct: List[bool], n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    confidences = np.array(confidences)
    correct = np.array(correct).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece_val += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece_val)


def auroc(scores: List[float], wrong: List[bool]) -> float:
    """AUROC of score for predicting wrong=1. Higher score = more wrong."""
    scores = np.array(scores)
    wrong = np.array(wrong).astype(int)
    if len(np.unique(wrong)) < 2:
        return 0.5
    return float(roc_auc_score(wrong, scores))


def aurc(confidences: List[float], correct: List[bool]) -> float:
    """Area Under Risk-Coverage curve."""
    confidences = np.array(confidences)
    correct = np.array(correct).astype(int)
    n = len(confidences)
    # Sort by confidence descending
    order = np.argsort(-confidences)
    correct_sorted = correct[order]
    # Coverage from 1/n to 1
    cum_correct = np.cumsum(correct_sorted)
    coverage = np.arange(1, n + 1) / n
    risk = 1 - cum_correct / np.arange(1, n + 1)
    # AURC = integral of risk over coverage
    return float(np.trapz(risk, coverage))


def aurc_excess(confidences: List[float], correct: List[bool]) -> float:
    """Excess AURC = AURC - oracle AURC."""
    aurc_val = aurc(confidences, correct)
    # Oracle: rank by correctness (correct first), risk=0 until all correct covered
    correct = np.array(correct).astype(int)
    n = len(correct)
    n_correct = correct.sum()
    # Oracle: covers all correct first (risk=0), then incorrect (risk = n_wrong/n_each_step)
    oracle_risk = np.zeros(n)
    # After covering all correct, risk starts increasing
    if n_correct < n:
        # The remaining (n - n_correct) are wrong, risk at each step = 1
        oracle_risk[n_correct:] = 1.0
    coverage = np.arange(1, n + 1) / n
    oracle_aurc = float(np.trapz(oracle_risk, coverage))
    return aurc_val - oracle_aurc


def auarc(scores: List[float], correct: List[bool]) -> float:
    """Area Under Abstention-Risk curve. Lower is better.
    Abstention rate from 0 to 1; risk = P(wrong | not abstained)."""
    scores = np.array(scores)
    correct = np.array(correct).astype(int)
    n = len(scores)
    # Sort by score ascending (lowest score = most confident)
    order = np.argsort(scores)
    correct_sorted = correct[order]
    # As we abstain on more (higher scores), risk on remaining decreases
    abst_rates = np.linspace(0, 1, 50)
    risks = []
    for ar in abst_rates:
        n_keep = int(n * (1 - ar))
        if n_keep == 0:
            risks.append(0.0)
            continue
        # Keep n_keep lowest-score examples
        kept_correct = correct_sorted[:n_keep]
        if len(kept_correct) == 0:
            risks.append(0.0)
        else:
            risks.append(1 - kept_correct.mean())
    return float(np.trapz(risks, abst_rates))


def average_cost(actions: List[str], cost_model: Optional[Dict[str, float]] = None) -> float:
    """Average test-time cost per query."""
    if cost_model is None:
        cost_model = {'Answer': 1.0, 'RetrieveMore': 2.0, 'VerifyMore': 3.0, 'Abstain': 0.0}
    return float(np.mean([cost_model.get(a, 1.0) for a in actions]))


def cost_per_correct(actions: List[str], correct: List[bool], cost_model: Optional[Dict[str, float]] = None) -> float:
    """Total cost / # correct answers."""
    if cost_model is None:
        cost_model = {'Answer': 1.0, 'RetrieveMore': 2.0, 'VerifyMore': 3.0, 'Abstain': 0.0}
    total_cost = sum(cost_model.get(a, 1.0) for a in actions)
    n_correct = sum(correct)
    if n_correct == 0:
        return float('inf')
    return total_cost / n_correct


def set_size_avg(prediction_sets: List[List[int]]) -> float:
    """Average prediction set size for APS."""
    if not prediction_sets:
        return 0.0
    return float(np.mean([len(s) for s in prediction_sets]))


def all_metrics(results: Dict) -> Dict[str, float]:
    """Compute all metrics from a results dict with keys:
    'correct', 'answered', 'confidences', 'scores', 'actions', 'prediction_sets' (optional).
    """
    correct = results.get('correct', [])
    answered = results.get('answered', [])
    confidences = results.get('confidences', [])
    scores = results.get('scores', [])
    actions = results.get('actions', [])
    prediction_sets = results.get('prediction_sets', None)

    wrong = [not c for c in correct]
    m = {
        'accuracy': accuracy(correct, answered),
        'coverage': coverage(answered),
        'abstention': abstention_rate(answered),
        'selective_risk': selective_risk(correct, answered),
        'ece': ece(confidences, correct) if confidences else 0.0,
        'auroc': auroc(scores, wrong) if scores and len(scores) == len(correct) else 0.5,
        'aurc': aurc(confidences, correct) if confidences else 0.0,
        'e_aurc': aurc_excess(confidences, correct) if confidences else 0.0,
        'auarc': auarc(scores, correct) if scores and len(scores) == len(correct) else 0.0,
        'avg_cost': average_cost(actions) if actions else 1.0,
        'cost_per_correct': cost_per_correct(actions, correct) if actions else float('inf'),
    }
    if prediction_sets is not None:
        m['set_size_avg'] = set_size_avg(prediction_sets)
    return m
