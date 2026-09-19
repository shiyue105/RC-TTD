"""
Generate all figures and tables from experiment results.
"""
import os
import sys
import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / 'figures'
TABLES_DIR = PROJECT_ROOT / 'paper' / 'tables'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Color palette for methods
COLORS = {
    'always_answer': '#888888',
    'always_abstain': '#444444',
    'static_threshold': '#AAAAAA',
    'always_retrieve': '#FF9999',
    'always_verify': '#FF6666',
    'lac': '#66B2FF',
    'aps': '#3399FF',
    'entropy_abstain': '#99CCFF',
    'fixed_priority': '#FFCC99',
    'rcttd_rule': '#FF0000',
    'rcttd_rule_domain_conditional': '#CC0000',
}

LABELS = {
    'always_answer': 'Always Answer',
    'always_abstain': 'Always Abstain',
    'static_threshold': 'Static Threshold',
    'always_retrieve': 'Always Retrieve',
    'always_verify': 'Always Verify',
    'lac': 'LAC',
    'aps': 'APS',
    'entropy_abstain': 'Entropy Abstain',
    'fixed_priority': 'Fixed Priority',
    'rcttd_rule': 'RC-TTD (Rule)',
    'rcttd_rule_domain_conditional': 'RC-TTD (Domain-Cond.)',
}

# Abbreviated labels for compact figures
SHORT_LABELS = {
    'always_answer': 'AlwaysAns',
    'always_abstain': 'AlwaysAbs',
    'static_threshold': 'StaticThr',
    'always_retrieve': 'AlwaysRet',
    'always_verify': 'AlwaysVer',
    'lac': 'LAC',
    'aps': 'APS',
    'entropy_abstain': 'EntAbst',
    'fixed_priority': 'FixedPri',
    'rcttd_rule': 'RC-TTD',
    'rcttd_rule_domain_conditional': 'RC-TTD-DC',
}


def load_all_results(results_dir: Path) -> Dict:
    """Load all experiment results from results directory."""
    all_results = {}
    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        summary_file = dataset_dir / 'summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
            dataset_name = summary.get('dataset', dataset_dir.name)
            all_results[dataset_dir.name] = summary
            logger.info(f"Loaded {dataset_dir.name}: {len(summary.get('policies', {}))} policies")
    return all_results


def fig1_method_overview():
    """Figure 1: Method overview diagram (text-based, saved as PDF)."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('RC-TTD: Risk-Calibrated Test-Time Decision Policy for RAG', fontsize=14, fontweight='bold')

    # Draw boxes
    boxes = [
        (0.5, 4, 'Query x', '#E8E8FF'),
        (2.5, 4, 'Retrieve\n(BM25)', '#FFE8E8'),
        (4.5, 4, 'Generate\n(LLaMA-3.1)', '#E8FFE8'),
        (6.5, 4, 'Verify\n(DeBERTa-NLI)', '#FFF8E8'),
        (8.5, 4, 'Nonconformity\nScore s(x,y,E)', '#FFE8FF'),
        (10.5, 4, 'RC-TTD\nPolicy', '#FFCCCC'),
    ]
    for x, y, text, color in boxes:
        rect = plt.Rectangle((x-0.7, y-0.6), 1.4, 1.2, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw arrows
    for i in range(5):
        ax.annotate('', xy=(boxes[i+1][0]-0.7, 4), xytext=(boxes[i][0]+0.7, 4),
                    arrowprops=dict(arrowstyle='->', lw=1.5))

    # Draw 4 actions
    actions = [
        (2, 1.5, 'Answer\n(cost=1)', '#90EE90'),
        (4.5, 1.5, 'Retrieve More\n(cost=2)', '#FFD700'),
        (7, 1.5, 'Verify More\n(cost=3)', '#FFA500'),
        (9.5, 1.5, 'Abstain\n(cost=0)', '#FFB6C1'),
    ]
    for x, y, text, color in actions:
        rect = plt.Rectangle((x-0.8, y-0.5), 1.6, 1.0, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9)

    # Arrows from policy to actions
    for x, _, _, _ in actions:
        ax.annotate('', xy=(x, 2.0), xytext=(10.5, 3.4),
                    arrowprops=dict(arrowstyle='->', lw=1, color='gray'))

    # 5 signals
    signals = '5 Signals: u_ans | 1-rel_ret | conf_e | 1-sup | dis_v'
    ax.text(6, 5.2, signals, ha='center', fontsize=10, style='italic', color='blue')

    # Budget constraint
    ax.text(6, 0.5, 'Budget B = 4×  |  Risk target α = 0.1  |  Coverage ≥ 1-α',
            ha='center', fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_method_overview.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig1_method_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Figure 1 saved: fig1_method_overview.pdf")


def fig2_main_results_table(all_results: Dict):
    """Figure 2: Main results table (as figure for compactness)."""
    datasets = sorted(all_results.keys())
    methods = ['always_answer', 'static_threshold', 'lac', 'aps', 'entropy_abstain',
               'always_retrieve', 'always_verify', 'rcttd_rule', 'rcttd_rule_domain_conditional']

    # Build table data
    rows = []
    for method in methods:
        row = [SHORT_LABELS.get(method, method)]
        for ds in datasets:
            if ds in all_results and method in all_results[ds].get('policies', {}):
                m = all_results[ds]['policies'][method]
                row.append(f"{m['accuracy']:.3f}")
                row.append(f"{m['coverage']:.3f}")
                row.append(f"{m['selective_risk']:.3f}")
                row.append(f"{m['avg_cost']:.2f}")
            else:
                row.extend(['-', '-', '-', '-'])
        rows.append(row)

    # Create figure
    n_cols = 1 + 4 * len(datasets)
    col_labels = ['Method']
    for ds in datasets:
        col_labels.extend([f'{ds}\nAcc', f'{ds}\nCov', f'{ds}\nRisk', f'{ds}\nCost'])

    fig, ax = plt.subplots(figsize=(4 + 1.2 * len(datasets), 0.6 * len(methods) + 1))
    ax.axis('off')
    ax.set_title('Main Results: Accuracy / Coverage / Selective Risk / Avg Cost', fontsize=12, fontweight='bold')

    table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Highlight RC-TTD rows
    for i, method in enumerate(methods):
        if 'rcttd' in method:
            for j in range(n_cols):
                table[i+1, j].set_facecolor('#FFEEEE')
                table[i+1, j].set_text_props(fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig2_main_results.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig2_main_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Figure 2 saved: fig2_main_results.pdf")


def fig3_cost_risk_curve(all_results: Dict):
    """Figure 3: Cost vs Risk bar chart across methods."""
    datasets = sorted(all_results.keys())
    methods = ['always_answer', 'lac', 'aps', 'always_retrieve', 'always_verify', 'rcttd_rule']

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5), squeeze=False)
    for idx, ds in enumerate(datasets):
        ax = axes[0][idx]
        risk_vals = []
        cost_vals = []
        labels = []
        colors = []
        for m in methods:
            if ds in all_results and m in all_results[ds].get('policies', {}):
                metrics = all_results[ds]['policies'][m]
                risk_vals.append(metrics['selective_risk'])
                cost_vals.append(metrics['avg_cost'])
                labels.append(SHORT_LABELS.get(m, m))
                colors.append(COLORS.get(m, '#888888'))

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, risk_vals, width, label='Selective Risk', color=colors, alpha=0.7)
        ax.bar(x + width/2, cost_vals, width, label='Avg Cost', color=colors, alpha=0.4, hatch='//')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Value')
        ax.set_title(f'{ds}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Cost-Risk Trade-off Across Methods', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_cost_risk_curve.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig3_cost_risk_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Figure 3 saved: fig3_cost_risk_curve.pdf")


def fig4_ablation(all_results: Dict):
    """Figure 4: Ablation results."""
    datasets = sorted(all_results.keys())
    ablations = ['no_retrieval', 'no_conflict', 'no_verifier', 'prob_only']
    abl_labels = ['No Retrieval', 'No Conflict', 'No Verifier', 'Prob Only']

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5), squeeze=False)
    for idx, ds in enumerate(datasets):
        ax = axes[0][idx]
        risk_vals = []
        labels = ['Full\n(RC-TTD)'] + abl_labels
        colors = ['#FF0000'] + ['#FF9999', '#FFCC99', '#99CCFF', '#CCCCCC']

        # Full RC-TTD
        if ds in all_results and 'rcttd_rule' in all_results[ds].get('policies', {}):
            risk_vals.append(all_results[ds]['policies']['rcttd_rule']['selective_risk'])
        else:
            risk_vals.append(0)

        # Ablations
        for abl in ablations:
            if ds in all_results and 'ablations' in all_results[ds] and abl in all_results[ds]['ablations']:
                risk_vals.append(all_results[ds]['ablations'][abl]['selective_risk'])
            else:
                risk_vals.append(0)

        x = np.arange(len(labels))
        ax.bar(x, risk_vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Selective Risk @ 90% Coverage')
        ax.set_title(f'{ds}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Highlight full
        ax.bar(0, risk_vals[0], color='#FF0000', edgecolor='black', linewidth=1.5)

    plt.suptitle('Ablation: Effect of Removing Each Signal', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig4_ablation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig4_ablation.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Figure 4 saved: fig4_ablation.pdf")


def fig5_domain_conditional(all_results: Dict):
    """Figure 5: Domain-conditional vs marginal coverage."""
    datasets = sorted(all_results.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(datasets))
    width = 0.3

    marginal_cov = []
    dc_cov = []
    marginal_risk = []
    dc_risk = []

    for ds in datasets:
        if ds in all_results:
            if 'rcttd_rule' in all_results[ds].get('policies', {}):
                marginal_cov.append(all_results[ds]['policies']['rcttd_rule']['coverage'])
                marginal_risk.append(all_results[ds]['policies']['rcttd_rule']['selective_risk'])
            else:
                marginal_cov.append(0)
                marginal_risk.append(0)
            if 'rcttd_rule_domain_conditional' in all_results[ds].get('policies', {}):
                dc_cov.append(all_results[ds]['policies']['rcttd_rule_domain_conditional']['coverage'])
                dc_risk.append(all_results[ds]['policies']['rcttd_rule_domain_conditional']['selective_risk'])
            else:
                dc_cov.append(0)
                dc_risk.append(0)
        else:
            marginal_cov.append(0)
            dc_cov.append(0)
            marginal_risk.append(0)
            dc_risk.append(0)

    ax.bar(x - width/2, marginal_cov, width, label='Marginal CP - Coverage', color='#66B2FF')
    ax.bar(x + width/2, dc_cov, width, label='Domain-Cond. CP - Coverage', color='#FF6666')
    ax.axhline(y=0.9, color='green', linestyle='--', label='Target (1-α=0.9)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel('Coverage')
    ax.set_title('Marginal vs Domain-Conditional Coverage', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig5_domain_conditional.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig5_domain_conditional.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Figure 5 saved: fig5_domain_conditional.pdf")


def fig6_score_comparison(all_results: Dict):
    """Figure 6: AUROC comparison of nonconformity scores."""
    datasets = sorted(all_results.keys())
    score_types = ['prob_only', 'no_retrieval', 'no_conflict', 'no_verifier']
    score_labels = ['Prob Only\n(LAC-style)', 'No Retrieval', 'No Conflict', 'No Verifier', 'Full\n(5 signals)']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets))
    width = 0.15

    for i, (st, sl) in enumerate(zip(score_types, score_labels)):
        aurocs = []
        for ds in datasets:
            if ds in all_results and 'ablations' in all_results[ds] and st in all_results[ds]['ablations']:
                aurocs.append(all_results[ds]['ablations'][st]['auroc'])
            else:
                aurocs.append(0)
        ax.bar(x + (i - 2) * width, aurocs, width, label=sl, alpha=0.8)

    # Full score (from main RC-TTD)
    full_aurocs = []
    for ds in datasets:
        if ds in all_results and 'rcttd_rule' in all_results[ds].get('policies', {}):
            full_aurocs.append(all_results[ds]['policies']['rcttd_rule']['auroc'])
        else:
            full_aurocs.append(0)
    ax.bar(x + (3 - 2) * width, full_aurocs, width, label='Full (5 signals)', color='#FF0000', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel('AUROC (wrong answer prediction)')
    ax.set_title('Nonconformity Score Comparison: AUROC', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig6_score_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig6_score_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Figure 6 saved: fig6_score_comparison.pdf")


def generate_latex_table(all_results: Dict):
    """Generate LaTeX main results table."""
    datasets = sorted(all_results.keys())
    methods = ['always_answer', 'static_threshold', 'lac', 'aps', 'entropy_abstain',
               'always_retrieve', 'always_verify', 'rcttd_rule', 'rcttd_rule_domain_conditional']

    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\caption{Main Results: Accuracy, Coverage, Selective Risk, and Average Cost across datasets.}')
    lines.append(r'\label{tab:main_results}')
    col_spec = 'l' + 'cccc' * len(datasets)
    lines.append(r'\begin{tabular}{' + col_spec + r'}')
    lines.append(r'\toprule')

    # Header
    header = 'Method'
    for ds in datasets:
        header += f' & \\multicolumn{{4}}{{c}}{{{ds}}}'
    lines.append(header + r' \\')
    lines.append(r'\cmidrule(lr){2-5}' + r' \cmidrule(lr){6-9}' * (len(datasets) - 1) if len(datasets) > 1 else r'\cmidrule(lr){2-5}')

    subheader = ''
    for ds in datasets:
        subheader += ' & Acc & Cov & Risk & Cost'
    lines.append(subheader + r' \\')
    lines.append(r'\midrule')

    # Rows
    for method in methods:
        label = LABELS.get(method, method)
        row = label
        for ds in datasets:
            if ds in all_results and method in all_results[ds].get('policies', {}):
                m = all_results[ds]['policies'][method]
                row += f' & {m["accuracy"]:.3f} & {m["coverage"]:.3f} & {m["selective_risk"]:.3f} & {m["avg_cost"]:.2f}'
            else:
                row += ' & - & - & - & -'
        lines.append(row + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    table_path = TABLES_DIR / 'main_results.tex'
    with open(table_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"LaTeX table saved: {table_path}")


def generate_markdown_summary(all_results: Dict):
    """Generate Markdown summary of all results."""
    datasets = sorted(all_results.keys())
    methods = ['always_answer', 'static_threshold', 'lac', 'aps', 'entropy_abstain',
               'always_retrieve', 'always_verify', 'rcttd_rule', 'rcttd_rule_domain_conditional']

    lines = []
    lines.append('# Experimental Results Summary\n')
    lines.append(f'Generated from {len(datasets)} datasets.\n')

    for ds in datasets:
        lines.append(f'\n## {ds}\n')
        lines.append('| Method | Accuracy | Coverage | Selective Risk | AUROC | Avg Cost | Cost/Correct |')
        lines.append('|--------|----------|----------|----------------|-------|----------|--------------|')
        for method in methods:
            if ds in all_results and method in all_results[ds].get('policies', {}):
                m = all_results[ds]['policies'][method]
                label = LABELS.get(method, method)
                lines.append(f'| {label} | {m["accuracy"]:.3f} | {m["coverage"]:.3f} | '
                             f'{m["selective_risk"]:.3f} | {m["auroc"]:.3f} | {m["avg_cost"]:.2f} | '
                             f'{m["cost_per_correct"]:.2f} |')

        # Ablations
        if 'ablations' in all_results[ds]:
            lines.append('\n### Ablations\n')
            lines.append('| Ablation | Accuracy | Coverage | Risk | AUROC | Cost |')
            lines.append('|----------|----------|----------|------|-------|------|')
            for abl_name, abl_metrics in all_results[ds]['ablations'].items():
                lines.append(f'| {abl_name} | {abl_metrics["accuracy"]:.3f} | {abl_metrics["coverage"]:.3f} | '
                             f'{abl_metrics["selective_risk"]:.3f} | {abl_metrics["auroc"]:.3f} | '
                             f'{abl_metrics["avg_cost"]:.2f} |')

    summary_path = PROJECT_ROOT / 'results' / 'RESULTS_SUMMARY.md'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Markdown summary saved: {summary_path}")


def main():
    results_dir = PROJECT_ROOT / 'results'
    all_results = load_all_results(results_dir)

    if not all_results:
        logger.warning("No results found. Run experiments first.")
        return

    logger.info(f"Generating figures for {len(all_results)} datasets...")

    # Generate all figures
    fig1_method_overview()
    fig2_main_results_table(all_results)
    fig3_cost_risk_curve(all_results)
    fig4_ablation(all_results)
    fig5_domain_conditional(all_results)
    fig6_score_comparison(all_results)

    # Generate tables
    generate_latex_table(all_results)
    generate_markdown_summary(all_results)

    logger.info("All figures and tables generated.")


if __name__ == '__main__':
    main()
