"""
RC-TTD rule-based policy and baselines.
"""
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Instance:
    """A single test instance with all features."""
    query: str
    evidence: List[Dict]  # retrieved docs
    answer: str
    p_max: float  # answer confidence
    h_p: float  # normalized entropy
    u_ans: float  # answer uncertainty (self-consistency)
    rel_ret: float  # retrieval relevance (avg top-3 normalized score)
    conf_e: float  # evidence conflict (pairwise NLI contradiction rate)
    sup: float  # answer-evidence support (NLI entailment)
    dis_v: float  # verifier disagreement
    budget_remaining: float
    domain: str = "unknown"
    # Computed
    score: float = 0.0


def compute_nonconformity_score(features: Dict, weights: Optional[Dict] = None) -> float:
    """Compute retrieval-aware nonconformity score.
    High score = more risky (likely wrong).
    Primary signal: 1-p_max (model confidence).
    Secondary signals: retrieval relevance, evidence conflict, support, verifier disagreement.
    """
    if weights is None:
        weights = {
            '1-p_max': 0.35,
            '1-rel_ret': 0.15,
            'conf_e': 0.15,
            '1-sup': 0.20,
            'dis_v': 0.15,
        }
    s = (weights['1-p_max'] * (1 - features['p_max'])
         + weights['1-rel_ret'] * (1 - features['rel_ret'])
         + weights['conf_e'] * features['conf_e']
         + weights['1-sup'] * (1 - features['sup'])
         + weights['dis_v'] * features['dis_v'])
    return float(s)


def compute_features(answer: str, p_max: float, h_p: float, u_ans: float,
                     evidence: List[Dict], sup: float, conf_e: float,
                     dis_v: float) -> Dict:
    """Build feature dict for nonconformity score."""
    rel_ret = float(np.mean([e.get('normalized_score', 0.0) for e in evidence[:3]])) if evidence else 0.0
    return {
        'u_ans': u_ans,
        'rel_ret': rel_ret,
        'conf_e': conf_e,
        'sup': sup,
        'dis_v': dis_v,
        '1-rel_ret': 1 - rel_ret,
        '1-sup': 1 - sup,
        'p_max': p_max,
        'h_p': h_p,
    }


# ==================== BASELINES ====================

def baseline_always_answer(inst: Instance, **kwargs) -> str:
    return 'Answer'


def baseline_always_abstain(inst: Instance, **kwargs) -> str:
    return 'Abstain'


def baseline_static_threshold(inst: Instance, tau: float = 0.5, **kwargs) -> str:
    """Abstain if p_max < tau."""
    if inst.p_max >= tau:
        return 'Answer'
    return 'Abstain'


def baseline_always_retrieve(inst: Instance, budget: float = 4.0, **kwargs) -> str:
    """Always retrieve more if budget allows."""
    if inst.budget_remaining >= 2.0:
        return 'RetrieveMore'
    return 'Answer'


def baseline_always_verify(inst: Instance, budget: float = 4.0, **kwargs) -> str:
    """Always verify if budget allows."""
    if inst.budget_remaining >= 3.0:
        return 'VerifyMore'
    return 'Answer'


def baseline_lac(inst: Instance, q_alpha: float, **kwargs) -> str:
    """LAC: answer if nonconformity score <= q_alpha.
    Use 1 - p_max as score (probability-based nonconformity).
    """
    s_prob = 1 - inst.p_max
    if s_prob <= q_alpha:
        return 'Answer'
    return 'Abstain'


def baseline_aps(inst: Instance, q_alpha: float, **kwargs) -> str:
    """APS: answer if prediction set size = 1; else abstain.
    Approximation: if p_max - p_2nd >= q_alpha, answer; else abstain.
    """
    # Use entropy as proxy for set size
    if inst.h_p <= q_alpha:
        return 'Answer'
    return 'Abstain'


def baseline_entropy_abstain(inst: Instance, tau: float = 0.5, **kwargs) -> str:
    """Abstain if normalized entropy > tau."""
    if inst.h_p <= tau:
        return 'Answer'
    return 'Abstain'


def baseline_fixed_priority(inst: Instance, budget: float = 4.0, **kwargs) -> str:
    """Fixed priority: retrieve → verify → abstain."""
    if inst.budget_remaining >= 2.0:
        return 'RetrieveMore'
    if inst.budget_remaining >= 1.0:
        return 'Answer'
    return 'Abstain'


def baseline_random_budget(inst: Instance, budget: float = 4.0, rng=None, **kwargs) -> str:
    """Random action under budget."""
    if rng is None:
        rng = np.random.default_rng()
    actions = []
    if inst.budget_remaining >= 1.0:
        actions.append('Answer')
    if inst.budget_remaining >= 2.0:
        actions.append('RetrieveMore')
    if inst.budget_remaining >= 3.0:
        actions.append('VerifyMore')
    actions.append('Abstain')
    return rng.choice(actions)


# ==================== RC-TTD RULE-BASED ====================

def rcttd_rule_based(inst: Instance, q_alpha: float,
                     tau_ret: float = 0.4, tau_conf: float = 0.4,
                     tau_sup: float = 0.2, tau_verif: float = 0.3,
                     cost_ret: float = 2.0, cost_verif: float = 3.0,
                     use_two_stage: bool = True,
                     **kwargs) -> str:
    """RC-TTD rule-based policy — two-stage design.

    Stage 1 (conformal coverage guarantee): Use 1-p_max as primary nonconformity
    score. If 1-p_max <= q_alpha (LAC-style threshold), the instance is
    conformally safe → Answer. This guarantees 1-alpha marginal coverage.

    Stage 2 (corrective action selection): For instances above the conformal
    threshold, use the multi-signal retrieval-aware score to decide which
    corrective action to take (RetrieveMore / VerifyMore / Abstain) based on
    which risk signal is dominant.

    This decouples the coverage guarantee (from 1-p_max) from the action
    selection (from multi-signal score), avoiding the score distribution
    compression problem.

    Decision logic:
    1. If (1 - p_max) <= q_alpha: Answer (conformal safe)
    2. elif retrieval irrelevance dominant AND budget: RetrieveMore
    3. elif evidence conflict / verifier disagreement dominant AND budget: VerifyMore
    4. elif 1-p_max slightly above threshold (borderline): Answer
    5. else: Abstain
    """
    s_prob = 1.0 - inst.p_max  # LAC-style nonconformity

    if use_two_stage:
        # Stage 1: Conformal coverage check via 1-p_max
        if s_prob <= q_alpha:
            return 'Answer'
    else:
        # Legacy: use full score for conformal check
        if inst.score <= q_alpha:
            return 'Answer'

    # Stage 2: Corrective action selection via multi-signal analysis
    # 2. Retrieval irrelevance dominant → RetrieveMore
    if (1 - inst.rel_ret) >= tau_ret and inst.budget_remaining >= cost_ret:
        return 'RetrieveMore'

    # 3. Evidence conflict or verifier disagreement dominant → VerifyMore
    if (inst.conf_e >= tau_conf or inst.dis_v >= tau_verif) and inst.budget_remaining >= cost_verif:
        return 'VerifyMore'

    # 4. Borderline: 1-p_max slightly above threshold → Answer (avoid over-abstention)
    if s_prob <= q_alpha * 1.5:
        return 'Answer'

    # 5. High risk, no corrective signal → Abstain
    return 'Abstain'


def rcttd_ablation(inst: Instance, q_alpha: float, ablation: str = 'full', **kwargs) -> str:
    """RC-TTD with one signal removed for ablation.

    For ablations, the two-stage conformal check (Stage 1) always uses 1-p_max.
    The ablation only affects Stage 2 corrective action selection by zeroing
    the corresponding signal threshold.
    """
    if ablation == 'no_retrieval':
        # Disable retrieval signal: always skip RetrieveMore
        kwargs['tau_ret'] = 2.0  # impossible threshold → never triggers
    elif ablation == 'no_conflict':
        # Disable conflict signal: always skip VerifyMore from conflict
        kwargs['tau_conf'] = 2.0
    elif ablation == 'no_verifier':
        # Disable verifier signal
        kwargs['tau_verif'] = 2.0
    elif ablation == 'prob_only':
        # Disable all corrective actions → pure conformal abstention (like LAC)
        kwargs['tau_ret'] = 2.0
        kwargs['tau_conf'] = 2.0
        kwargs['tau_verif'] = 2.0
    # 'full' = no changes

    return rcttd_rule_based(inst, q_alpha, use_two_stage=True, **kwargs)


# ==================== POLICY REGISTRY ====================

POLICIES = {
    'always_answer': baseline_always_answer,
    'always_abstain': baseline_always_abstain,
    'static_threshold': baseline_static_threshold,
    'always_retrieve': baseline_always_retrieve,
    'always_verify': baseline_always_verify,
    'lac': baseline_lac,
    'aps': baseline_aps,
    'entropy_abstain': baseline_entropy_abstain,
    'fixed_priority': baseline_fixed_priority,
    'rcttd_rule': rcttd_rule_based,
}


def apply_policy(policy_name: str, inst: Instance, **kwargs) -> str:
    """Apply a named policy to an instance."""
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(POLICIES.keys())}")
    return POLICIES[policy_name](inst, **kwargs)
