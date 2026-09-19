"""
Evaluate Learned RC-TTD policy on existing experiment results.
This reuses cal_results.jsonl and test_results.jsonl to avoid re-running the RAG pipeline.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from src.policy.learned import LearnedRCTTD
from src.policy.rule_based import Instance, rcttd_rule_based, compute_features, compute_nonconformity_score
from src.conformal.calibration import lac_threshold
from src.evaluation.metrics import accuracy, coverage, selective_risk, auroc


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_instance(r, budget=4.0):
    rel_ret = float(np.mean([e.get('normalized_score', 0.0) for e in r.get('evidence', [])[:3]])) if r.get('evidence') else 0.0
    # Some entries already have rel_ret
    if 'rel_ret' in r:
        rel_ret = r['rel_ret']
    return Instance(
        query=r['query'], evidence=r.get('evidence', []), answer=str(r['choice_idx']),
        p_max=r['p_max'], h_p=r['h_p'], u_ans=r.get('u_ans', 0.0),
        rel_ret=rel_ret, conf_e=r.get('conf_e', 0.0), sup=r.get('sup', 0.0),
        dis_v=r.get('dis_v', 0.0), budget_remaining=budget, domain=r.get('domain', 'unknown'),
        score=r.get('score', 0.0)
    )


def simulate_action(r, action, rng):
    """Simulate action effects on correctness."""
    if action == 'Answer':
        return True, r['correct']
    elif action == 'RetrieveMore':
        if r['correct']:
            return True, True
        return True, rng.random() < 0.30
    elif action == 'VerifyMore':
        if r['correct']:
            return True, True
        return True, rng.random() < 0.50
    else:  # Abstain
        return False, False


def evaluate_policy(cal_results, test_results, policy_name, alpha=0.1, budget=4.0):
    """Evaluate a policy on test results."""
    # Compute conformal threshold (Stage 1: 1-p_max)
    cal_scores = np.array([1 - r['p_max'] for r in cal_results])
    q_alpha = lac_threshold(cal_scores, alpha)

    # Train learned model if needed
    learned_model = None
    if policy_name == 'rcttd_learned':
        learned_model = LearnedRCTTD(alpha=alpha, budget=budget)
        learned_model.fit(cal_results)

    # Apply policy
    actions = []
    answered = []
    final_correct = []
    for r in test_results:
        inst = build_instance(r, budget=budget)
        if policy_name == 'rcttd_rule':
            action = rcttd_rule_based(inst, q_alpha)
        elif policy_name == 'rcttd_learned':
            if learned_model and learned_model.is_trained:
                action = learned_model.predict(inst)
            else:
                action = rcttd_rule_based(inst, q_alpha)
        else:
            action = 'Answer'

        rng = np.random.default_rng(hash(r['id']) % 2**32)
        ans, fc = simulate_action(r, action, rng)
        actions.append(action)
        answered.append(ans)
        final_correct.append(fc)

    # Compute metrics
    test_results_with_actions = []
    for r, a, ans, fc in zip(test_results, actions, answered, final_correct):
        r2 = r.copy()
        r2['action'] = a
        r2['answered'] = ans
        r2['final_correct'] = fc
        test_results_with_actions.append(r2)

    correct_arr = [r['final_correct'] for r in test_results_with_actions]
    answered_arr = [r['answered'] for r in test_results_with_actions]
    acc = accuracy(correct_arr, answered_arr)
    cov = coverage(answered_arr)
    risk = selective_risk(correct_arr, answered_arr)
    avg_cost = np.mean([
        {'Answer': 1.0, 'RetrieveMore': 2.0, 'VerifyMore': 3.0, 'Abstain': 0.0}[a]
        for a in actions
    ])

    # Action distribution
    from collections import Counter
    action_dist = Counter(actions)

    return {
        'policy': policy_name,
        'accuracy': acc,
        'coverage': cov,
        'risk': risk,
        'cost': avg_cost,
        'cost_per_correct': avg_cost / max(acc * cov * len(test_results), 1),
        'action_dist': dict(action_dist),
        'q_alpha': q_alpha,
        'learned_trained': learned_model.is_trained if learned_model else None,
    }


def main():
    results_dir = PROJECT_ROOT / 'results'
    datasets = ['sciq', 'scifact']
    variants = ['', 'wrong_context_']

    print(f"{'Dataset':<30} {'Policy':<15} {'Acc':>6} {'Cov':>6} {'Risk':>6} {'Cost':>6} {'Actions':<40}")
    print("-" * 120)

    all_results = {}
    for ds in datasets:
        for var in variants:
            ds_key = f"{ds}_{var}seed0" if var else f"{ds}_seed0"
            ds_path = results_dir / ds_key
            if not ds_path.exists():
                print(f"Skipping {ds_key}: not found")
                continue

            cal_results = load_jsonl(ds_path / 'cal_results.jsonl')
            test_results = load_jsonl(ds_path / 'test_results.jsonl')

            # Fix: ensure cal_results have 'correct' field
            for r in cal_results:
                if 'correct' not in r:
                    r['correct'] = (r['choice_idx'] == r['correct_idx'])

            print(f"\n=== {ds_key} (n_cal={len(cal_results)}, n_test={len(test_results)}) ===")
            all_results[ds_key] = {}

            for policy in ['rcttd_rule', 'rcttd_learned']:
                res = evaluate_policy(cal_results, test_results, policy)
                all_results[ds_key][policy] = res
                action_str = ', '.join([f"{k}:{v}" for k, v in sorted(res['action_dist'].items())])
                print(f"  {policy:<15} acc={res['accuracy']:.3f} cov={res['coverage']:.3f} "
                      f"risk={res['risk']:.3f} cost={res['cost']:.2f} trained={res['learned_trained']}")
                print(f"    actions: {action_str}")

    # Save
    with open(results_dir / 'learned_eval_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {results_dir / 'learned_eval_results.json'}")


if __name__ == '__main__':
    main()
