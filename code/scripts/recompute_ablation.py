"""
Recompute ablation table using real corrective action results.
For each ablation variant, re-apply the ablated policy and use real corrective
action outcomes from the real_corrective experiment.
"""
import json
import numpy as np
import sys
from pathlib import Path
from collections import Counter

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from src.policy.rule_based import Instance, rcttd_rule_based, rcttd_ablation
from src.conformal.calibration import lac_threshold

RESULTS_DIR = PROJECT_ROOT / 'results'


def build_instance_from_result(r, budget=4.0):
    rel_ret = float(np.mean([e.get('normalized_score', 0.0) for e in r.get('evidence', [])[:3]])) if r.get('evidence') else 0.0
    return Instance(
        query=r['query'], evidence=r.get('evidence', []), answer=str(r['choice_idx']),
        p_max=r['p_max'], h_p=r['h_p'], u_ans=r['u_ans'],
        rel_ret=rel_ret, conf_e=r['conf_e'], sup=r['sup'], dis_v=r['dis_v'],
        budget_remaining=budget, domain=r.get('domain', 'unknown'), score=r.get('score', 0.0)
    )


def recompute_ablation(dataset_name, wrong_context=False, seed=0, alpha=0.1, budget=4.0):
    """Recompute ablation table for one dataset using real corrective action results."""
    base_dir = RESULTS_DIR / f"{dataset_name}{'_wrong_context' if wrong_context else ''}_seed{seed}_v2"
    corrective_dir = RESULTS_DIR / f"{dataset_name}{'_wrong_context' if wrong_context else ''}_seed{seed}_v2_real_corrective"

    # Load original test results
    test_results = []
    with open(base_dir / 'test_results.jsonl') as f:
        for line in f:
            test_results.append(json.loads(line))

    # Load real corrective action results (has final_correct for RM/VM)
    corrective_results = []
    corrective_map = {}
    if (corrective_dir / 'test_results_corrective.jsonl').exists():
        with open(corrective_dir / 'test_results_corrective.jsonl') as f:
            for line in f:
                r = json.loads(line)
                corrective_results.append(r)
                corrective_map[r['id']] = r

    # Load calibration results for threshold
    cal_results = []
    with open(base_dir / 'cal_results.jsonl') as f:
        for line in f:
            cal_results.append(json.loads(line))
    cal_scores = np.array([1 - r['p_max'] for r in cal_results])
    q_alpha = lac_threshold(cal_scores, alpha)

    # Define ablation variants
    ablations = ['full', 'no_retrieval', 'no_conflict', 'no_verifier', 'prob_only']

    results = {}
    for abl in ablations:
        action_counts = Counter()
        n_correct = 0
        n_answered = 0
        n_wrong = 0

        for r in test_results:
            inst = build_instance_from_result(r, budget)
            action = rcttd_ablation(inst, q_alpha, ablation=abl)
            action_counts[action] += 1

            cr = corrective_map.get(r['id'], {})

            if action == 'Answer':
                n_answered += 1
                if r['correct']:
                    n_correct += 1
                else:
                    n_wrong += 1
            elif action == 'RetrieveMore':
                n_answered += 1
                # Use real corrective result if available
                if cr and cr.get('action') == 'RetrieveMore' and cr.get('corrective_info', {}).get('executed'):
                    if cr['final_correct']:
                        n_correct += 1
                    else:
                        n_wrong += 1
                else:
                    # Corrective action not executed for this instance in real experiment
                    # (policy assigned different action). Fall back to original correctness.
                    if r['correct']:
                        n_correct += 1
                    else:
                        n_wrong += 1
            elif action == 'VerifyMore':
                n_answered += 1
                if cr and cr.get('action') == 'VerifyMore' and cr.get('corrective_info', {}).get('executed'):
                    if cr['final_correct']:
                        n_correct += 1
                    else:
                        n_wrong += 1
                else:
                    if r['correct']:
                        n_correct += 1
                    else:
                        n_wrong += 1
            else:  # Abstain
                pass

        n = len(test_results)
        accuracy = n_correct / n
        coverage = n_answered / n
        selective_risk = n_wrong / max(n_answered, 1)
        cost_map = {'Answer': 1.0, 'RetrieveMore': 2.0, 'VerifyMore': 3.0, 'Abstain': 0.0}
        avg_cost = np.mean([cost_map[rcttd_ablation(build_instance_from_result(r, budget), q_alpha, ablation=abl)] for r in test_results])

        results[abl] = {
            'accuracy': accuracy,
            'coverage': coverage,
            'selective_risk': selective_risk,
            'avg_cost': float(avg_cost),
            'actions': dict(action_counts),
        }

    return results


def main():
    datasets = [
        ('sciq', False, 'SciQ'),
        ('scifact', False, 'SciFact'),
    ]

    print("=" * 80)
    print("Ablation Table (Real Corrective Actions)")
    print("=" * 80)

    for dataset_name, wc, label in datasets:
        print(f"\n{label}:")
        results = recompute_ablation(dataset_name, wrong_context=wc)
        print(f"{'Variant':<15} {'Acc':>6} {'Cov':>6} {'Risk':>6} {'Cost':>6}")
        print("-" * 45)
        for abl in ['full', 'no_retrieval', 'no_conflict', 'no_verifier', 'prob_only']:
            r = results[abl]
            print(f"{abl:<15} {r['accuracy']:>6.3f} {r['coverage']:>6.3f} {r['selective_risk']:>6.3f} {r['avg_cost']:>6.2f}")

    # Generate LaTeX
    print("\n\n=== LaTeX Table ===")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"\caption{Ablation Study (Real Corrective Actions): Effect of removing each signal. ``Prob Only'' disables all corrective actions (equivalent to LAC).}")
    print(r"\label{tab:ablation}")
    print(r"\begin{tabular}{l|cccc|cccc}")
    print(r"\toprule")
    print(r"& \multicolumn{4}{c|}{\textbf{SciQ}} & \multicolumn{4}{c}{\textbf{SciFact}} \\")
    print(r"Variant & Acc & Cov & Risk & Cost & Acc & Cov & Risk & Cost \\")
    print(r"\midrule")

    sciq_results = recompute_ablation('sciq', wrong_context=False)
    scifact_results = recompute_ablation('scifact', wrong_context=False)

    labels = {
        'full': 'Full',
        'no_retrieval': 'No Retr',
        'no_conflict': 'No Conf',
        'no_verifier': 'No Verif',
        'prob_only': 'Prob Only',
    }

    for abl in ['full', 'no_retrieval', 'no_conflict', 'no_verifier', 'prob_only']:
        sq = sciq_results[abl]
        sf = scifact_results[abl]
        print(f"{labels[abl]} & {sq['accuracy']:.3f} & {sq['coverage']:.3f} & {sq['selective_risk']:.3f} & {sq['avg_cost']:.2f} & "
              f"{sf['accuracy']:.3f} & {sf['coverage']:.3f} & {sf['selective_risk']:.3f} & {sf['avg_cost']:.2f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == '__main__':
    main()
