"""
RC-TTD Learned Policy: Logistic regression-based action selection.

Trains a lightweight model on calibration data to predict the optimal action
based on RAG signals. This complements the rule-based policy by learning
optimal thresholds from data rather than using hand-tuned values.
"""
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from .rule_based import Instance, rcttd_rule_based

logger = logging.getLogger(__name__)


class LearnedRCTTD:
    """Learned RC-TTD policy using logistic regression for action prediction.

    Two-stage design (same as rule-based):
    - Stage 1: Conformal coverage via 1-p_max (LAC threshold)
    - Stage 2: Learned action selection for high-risk instances

    The learned model predicts P(correct | signals) and selects:
    - If P(correct) > tau_high → Answer (borderline safe)
    - Elif retrieval irrelevance is high → RetrieveMore
    - Elif P(correct) < tau_low → Abstain
    - Else → VerifyMore (default corrective action)
    """

    def __init__(self, alpha: float = 0.1, budget: float = 4.0):
        self.alpha = alpha
        self.budget = budget
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            penalty='l2', C=1.0, max_iter=1000, random_state=42
        )
        self.q_alpha = None  # Conformal threshold (from 1-p_max)
        self.tau_high = 0.7   # If P(correct) > tau_high → Answer
        self.tau_low = 0.3    # If P(correct) < tau_low → Abstain
        self.tau_ret = 0.4    # Retrieval irrelevance threshold
        self.cost_ret = 2.0
        self.cost_verif = 3.0
        self.is_trained = False

    def _extract_features(self, r: Dict) -> np.ndarray:
        """Extract feature vector from a RAG result dict."""
        return np.array([
            1.0 - r['p_max'],      # Nonconformity (probability)
            r.get('h_p', 0.0),      # Normalized entropy
            1.0 - r.get('rel_ret', 0.0),  # Retrieval irrelevance
            r.get('conf_e', 0.0),   # Evidence conflict
            1.0 - r.get('sup', 0.0),  # Support failure
            r.get('dis_v', 0.0),    # Verifier disagreement
            r.get('u_ans', 0.0),    # Self-consistency uncertainty
        ])

    def fit(self, cal_results: List[Dict]) -> None:
        """Train the learned policy on calibration results.

        Args:
            cal_results: List of result dicts from RAG pipeline.
                         Must contain 'correct' (bool) and signal fields.
        """
        # Stage 1: Compute conformal threshold from 1-p_max
        cal_scores = np.array([1.0 - r['p_max'] for r in cal_results])
        n = len(cal_scores)
        # LAC threshold: (1-alpha) quantile with finite-sample correction
        q_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        q_level = min(q_level, 1.0)
        self.q_alpha = float(np.quantile(cal_scores, q_level))
        logger.info(f"Learned RC-TTD: conformal threshold q_alpha={self.q_alpha:.4f}")

        # Stage 2: Train logistic regression to predict correctness
        X = np.array([self._extract_features(r) for r in cal_results])
        y = np.array([1 if r['correct'] else 0 for r in cal_results])

        # Handle class imbalance
        if len(np.unique(y)) < 2:
            logger.warning("Learned RC-TTD: only one class in calibration, using rule-based fallback")
            self.is_trained = False
            return

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Log feature importance (logistic regression coefficients)
        feat_names = ['1-p_max', 'h_p', '1-rel_ret', 'conf_e', '1-sup', 'dis_v', 'u_ans']
        coefs = self.model.coef_[0]
        logger.info("Learned RC-TTD feature coefficients:")
        for name, coef in sorted(zip(feat_names, coefs), key=lambda x: abs(x[1]), reverse=True):
            logger.info(f"  {name}: {coef:.4f}")

    def predict(self, inst: Instance) -> str:
        """Predict action for a test instance.

        Stage 1: Conformal coverage check (same as rule-based)
        Stage 2: Learned action selection
        """
        # Stage 1: Conformal coverage
        s_prob = 1.0 - inst.p_max
        if s_prob <= self.q_alpha:
            return 'Answer'

        # If model not trained, fall back to rule-based
        if not self.is_trained:
            return rcttd_rule_based(inst, self.q_alpha)

        # Stage 2: Learned action selection
        # Extract features and predict P(correct)
        r = {
            'p_max': inst.p_max,
            'h_p': inst.h_p,
            'rel_ret': inst.rel_ret,
            'conf_e': inst.conf_e,
            'sup': inst.sup,
            'dis_v': inst.dis_v,
            'u_ans': inst.u_ans,
        }
        x = self._extract_features(r).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        p_correct = float(self.model.predict_proba(x_scaled)[0, 1])

        # Action selection based on learned P(correct) and signals
        # High P(correct): answer despite being above conformal threshold
        if p_correct >= self.tau_high:
            return 'Answer'

        # Retrieval irrelevance dominant → RetrieveMore
        if (1 - inst.rel_ret) >= self.tau_ret and inst.budget_remaining >= self.cost_ret:
            return 'RetrieveMore'

        # Low P(correct) → Abstain
        if p_correct < self.tau_low:
            return 'Abstain'

        # Medium P(correct) with conflict → VerifyMore
        if inst.budget_remaining >= self.cost_verif:
            return 'VerifyMore'

        # Fallback
        return 'Abstain'

    def predict_batch(self, instances: List[Instance]) -> List[str]:
        """Predict actions for a batch of instances."""
        return [self.predict(inst) for inst in instances]
