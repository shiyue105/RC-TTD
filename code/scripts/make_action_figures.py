"""
Generate action distribution figure for RC-TTD paper.
Shows how RC-TTD distributes actions (Answer/RetrieveMore/VerifyMore/Abstain)
across different difficulty levels and datasets.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from src.policy.rule_based import Instance, rcttd_rule_based
from src.conformal.calibration import lac_threshold

FIGURES_DIR = PROJECT_ROOT / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS_ACTION = {
    'Answer': '#4CAF50',
    'RetrieveMore': '#FF9800',
    'VerifyMore': '#2196F3',
    'Abstain': '#F44336',
}


def load_results(results_dir: str):
    """Load cal and test results from a results directory."""
    results_path = PROJECT_ROOT / 'results' / results_dir
    cal_results = []
    with open(results_path / 'cal_results.jsonl') as f:
        for line in f:
            cal_results.append(json.loads(line))
    test_results = []
    with open(results_path / 'test_results.jsonl') as f:
        for line in f:
            test_results.append(json.loads(line))
    return cal_results, test_results


def build_instance(r, budget=4.0):
    """Build an Instance from a result dict."""
    return Instance(
        query=r['query'],
        evidence=r.get('evidence', []),
        answer=str(r['choice_idx']),
        p_max=r['p_max'],
        h_p=r['h_p'],
        u_ans=r.get('u_ans', 0.0),
        rel_ret=r.get('rel_ret', 0.0),
        conf_e=r.get('conf_e', 0.0),
        sup=r.get('sup', 0.0),
        dis_v=r.get('dis_v', 0.0),
        budget_remaining=budget,
        domain=r.get('domain', 'unknown'),
        score=r.get('score', 0.0),
    )


def get_actions(cal_results, test_results, alpha=0.1):
    """Apply RC-TTD policy and return actions for each test instance."""
    cal_scores = np.array([1 - r['p_max'] for r in cal_results])
    q_alpha = lac_threshold(cal_scores, alpha)

    actions = []
    for r in test_results:
        inst = build_instance(r)
        action = rcttd_rule_based(inst, q_alpha)
        actions.append(action)
    return actions, q_alpha


def fig_action_distribution():
    """Generate action distribution figure with 4 subplots."""
    datasets = [
        ('sciq_seed0_v2', 'SciQ (Easy)'),
        ('scifact_seed0_v2', 'SciFact (Hard)'),
        ('sciq_wrong_context_seed0_v2', 'SciQ (Wrong Context)'),
        ('scifact_wrong_context_seed0_v2', 'SciFact (Wrong Context)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (ds_dir, ds_label) in enumerate(datasets):
        ax = axes[idx // 2][idx % 2]

        try:
            cal_results, test_results = load_results(ds_dir)
            actions, q_alpha = get_actions(cal_results, test_results)
        except Exception as e:
            print(f"Error loading {ds_dir}: {e}")
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center')
            ax.set_title(ds_label)
            continue

        # Split by difficulty (p_max)
        p_maxes = [r['p_max'] for r in test_results]
        # 3 difficulty bins: hard (p_max < 0.5), medium (0.5-0.8), easy (>0.8)
        bins = [(0, 0.5, 'Hard'), (0.5, 0.8, 'Medium'), (0.8, 1.01, 'Easy')]

        x_labels = []
        action_types = ['Answer', 'RetrieveMore', 'VerifyMore', 'Abstain']
        data = {a: [] for a in action_types}

        for lo, hi, label in bins:
            mask = [(p >= lo and p < hi) for p in p_maxes]
            bin_actions = [a for a, m in zip(actions, mask) if m]
            n = len(bin_actions)
            x_labels.append(f'{label}\n(n={n})')
            counts = Counter(bin_actions)
            for a in action_types:
                data[a].append(counts.get(a, 0) / max(n, 1) * 100)

        # Stacked bar chart
        x = np.arange(len(x_labels))
        bottom = np.zeros(len(x_labels))
        for action in action_types:
            ax.bar(x, data[action], bottom=bottom, label=action,
                   color=COLORS_ACTION[action], alpha=0.85, edgecolor='white')
            bottom += np.array(data[action])

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel('Action Distribution (%)', fontsize=10)
        ax.set_title(ds_label, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.2)

    plt.suptitle('RC-TTD Action Distribution by Difficulty Level',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig7_action_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig7_action_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure saved: fig7_action_distribution.pdf")


def fig_action_by_signal():
    """Generate a figure showing which signals trigger corrective actions."""
    datasets = [
        ('sciq_seed0_v2', 'SciQ'),
        ('scifact_seed0_v2', 'SciFact'),
        ('sciq_wrong_context_seed0_v2', 'SciQ (WC)'),
        ('scifact_wrong_context_seed0_v2', 'SciFact (WC)'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), squeeze=False)

    for idx, (ds_dir, ds_label) in enumerate(datasets):
        ax = axes[0][idx]
        try:
            cal_results, test_results = load_results(ds_dir)
            actions, q_alpha = get_actions(cal_results, test_results)
        except Exception as e:
            print(f"Error loading {ds_dir}: {e}")
            continue

        # For each action, show average signal values
        action_types = ['Answer', 'RetrieveMore', 'VerifyMore', 'Abstain']
        signals = ['1-p_max', '1-rel_ret', 'conf_e', '1-sup', 'dis_v']
        signal_labels = ['1-p_max', '1-rel_ret', 'conf_e', '1-sup', 'dis_v']

        signal_values = {s: [] for s in signals}
        for r, a in zip(test_results, actions):
            signal_values['1-p_max'].append(1 - r['p_max'])
            signal_values['1-rel_ret'].append(1 - r.get('rel_ret', 0))
            signal_values['conf_e'].append(r.get('conf_e', 0))
            signal_values['1-sup'].append(1 - r.get('sup', 0))
            signal_values['dis_v'].append(r.get('dis_v', 0))

        # Group by action
        action_signal_means = {a: [] for a in action_types}
        for a in action_types:
            mask = [act == a for act in actions]
            for s in signals:
                vals = [v for v, m in zip(signal_values[s], mask) if m]
                action_signal_means[a].append(np.mean(vals) if vals else 0)

        # Heatmap
        data = np.array([action_signal_means[a] for a in action_types])
        im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(range(len(signals)))
        ax.set_xticklabels(signal_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(action_types)))
        ax.set_yticklabels(action_types, fontsize=9)
        ax.set_title(ds_label, fontsize=11, fontweight='bold')

        # Add text annotations
        for i in range(len(action_types)):
            for j in range(len(signals)):
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', fontsize=8)

    plt.suptitle('Average Signal Values by Action Type', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig8_signal_by_action.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig8_signal_by_action.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure saved: fig8_signal_by_action.pdf")


if __name__ == '__main__':
    fig_action_distribution()
    fig_action_by_signal()
    print("All action distribution figures generated.")
