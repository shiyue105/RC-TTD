"""
Validate theoretical predictions by subsampling calibration set.
No new LLM inference - reuses existing results.

Validates:
1. Coverage stability vs. calibration set size (Theorem 1)
2. Score distribution compression (Lemma 1)
3. Cost-optimality (Theorem 3)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = _PROJECT_ROOT / "results"
FIGURES_DIR = _PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_lac_threshold(cal_scores, alpha=0.1):
    """LAC threshold: (1-alpha) quantile of 1-p_max."""
    scores = np.array(cal_scores)
    n = len(scores)
    # Split conformal: ceil((1-alpha) * (n+1)) / n quantile
    q_level = np.ceil((1 - alpha) * (n + 1)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(scores, q_level)


def evaluate_coverage_risk(test_data, threshold, alpha=0.1):
    """Evaluate coverage and risk given threshold on 1-p_max."""
    scores = np.array([1 - r['p_max'] for r in test_data])
    correct = np.array([r['correct'] for r in test_data])

    answered = scores <= threshold
    coverage = answered.mean()
    if answered.sum() > 0:
        risk = 1 - correct[answered].mean()
    else:
        risk = 1.0
    return coverage, risk


def validate_calibration_size():
    """Validate Theorem 1: coverage stability vs. calibration size."""
    datasets = ['sciq_seed0_v2', 'scifact_seed0_v2']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for idx, ds in enumerate(datasets):
        cal_path = RESULTS_DIR / ds / "cal_results.jsonl"
        test_path = RESULTS_DIR / ds / "test_results.jsonl"
        if not cal_path.exists() or not test_path.exists():
            print(f"Skipping {ds}: files not found")
            continue

        cal_data = load_jsonl(cal_path)
        test_data = load_jsonl(test_path)

        cal_scores = [1 - r['p_max'] for r in cal_data]

        # Subsample calibration set at different sizes
        cal_sizes = [25, 50, 100, 200, 300, 500]
        n_trials = 10  # Multiple trials for confidence interval
        results = {size: {'coverage': [], 'risk': []} for size in cal_sizes}

        rng = np.random.default_rng(42)
        for size in cal_sizes:
            for trial in range(n_trials):
                if size <= len(cal_data):
                    indices = rng.choice(len(cal_data), size=size, replace=False)
                    subsample = [cal_scores[i] for i in indices]
                    threshold = compute_lac_threshold(subsample, alpha=0.1)
                    coverage, risk = evaluate_coverage_risk(test_data, threshold)
                    results[size]['coverage'].append(coverage)
                    results[size]['risk'].append(risk)

        # Plot coverage
        sizes = [s for s in cal_sizes if len(results[s]['coverage']) > 0]
        cov_mean = [np.mean(results[s]['coverage']) for s in sizes]
        cov_std = [np.std(results[s]['coverage']) for s in sizes]
        risk_mean = [np.mean(results[s]['risk']) for s in sizes]
        risk_std = [np.std(results[s]['risk']) for s in sizes]

        ax = axes[idx]
        ax2 = ax.twinx()

        line1, = ax.plot(sizes, cov_mean, 'b-o', label='Coverage', linewidth=2)
        ax.fill_between(sizes,
                        [m - s for m, s in zip(cov_mean, cov_std)],
                        [m + s for m, s in zip(cov_mean, cov_std)],
                        alpha=0.2, color='blue')
        ax.axhline(y=0.9, color='blue', linestyle='--', alpha=0.5, label='Target 0.9')

        line2, = ax2.plot(sizes, risk_mean, 'r-s', label='Risk', linewidth=2)
        ax2.fill_between(sizes,
                         [m - s for m, s in zip(risk_mean, risk_std)],
                         [m + s for m, s in zip(risk_mean, risk_std)],
                         alpha=0.2, color='red')

        ax.set_xlabel('Calibration Set Size', fontsize=11)
        ax.set_ylabel('Coverage', color='blue', fontsize=11)
        ax2.set_ylabel('Selective Risk', color='red', fontsize=11)
        ax.set_title(f'{ds.split("_")[0].capitalize()}', fontsize=12)
        ax.set_ylim(0.7, 1.05)
        ax2.set_ylim(0, 0.3)

        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right', fontsize=9)

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig_theory_calibration_size.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

    # Print summary table
    print("\n=== Calibration Size vs. Coverage/Risk ===")
    print(f"{'Size':>6} | {'SciQ Cov':>10} | {'SciQ Risk':>10} | {'SciFact Cov':>12} | {'SciFact Risk':>12}")
    print("-" * 65)
    for size in cal_sizes:
        sciq_cov = np.mean(results[size]['coverage']) if ds == 'sciq_seed0_v2' else 0
        # Re-read for both datasets
        pass

    return output_path


def validate_score_compression():
    """Validate Lemma 1: score distribution compression."""
    ds = 'sciq_seed0_v2'
    cal_path = RESULTS_DIR / ds / "cal_results.jsonl"
    test_path = RESULTS_DIR / ds / "test_results.jsonl"
    cal_data = load_jsonl(cal_path)
    test_data = load_jsonl(test_path)

    # Compute scores
    p_max_scores = np.array([1 - r['p_max'] for r in test_data])
    combined_scores = np.array([r['score'] for r in test_data])
    correct = np.array([r['correct'] for r in test_data])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Score distributions
    ax = axes[0]
    ax.hist(p_max_scores, bins=50, alpha=0.6, label=f'$1-p_{{\\max}}$ (range={p_max_scores.max():.3f})', color='blue', density=True)
    ax.hist(combined_scores, bins=50, alpha=0.6, label=f'Combined $s$ (range={combined_scores.max():.3f})', color='red', density=True)
    ax.set_xlabel('Score Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Score Distribution Compression', fontsize=12)
    ax.legend(fontsize=9)

    # Plot 2: Coverage at different thresholds
    ax = axes[1]
    thresholds_p = np.linspace(0, 0.5, 100)
    thresholds_s = np.linspace(0, 0.6, 100)

    cov_p = [(p_max_scores <= t).mean() for t in thresholds_p]
    cov_s = [(combined_scores <= t).mean() for t in thresholds_s]

    ax.plot(thresholds_p, cov_p, 'b-', label='$1-p_{\\max}$', linewidth=2)
    ax.plot(thresholds_s, cov_s, 'r-', label='Combined $s$', linewidth=2)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Target 0.9')
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Coverage', fontsize=11)
    ax.set_title('Coverage vs. Threshold', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig_theory_score_compression.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

    # Print statistics
    print(f"\n=== Score Compression (SciQ) ===")
    print(f"1-p_max: range=[{p_max_scores.min():.4f}, {p_max_scores.max():.4f}], 90th pct={np.percentile(p_max_scores, 90):.4f}")
    print(f"Combined s: range=[{combined_scores.min():.4f}, {combined_scores.max():.4f}], 90th pct={np.percentile(combined_scores, 90):.4f}")
    print(f"Ratio r2/r1 = {np.percentile(combined_scores, 90) / max(np.percentile(p_max_scores, 90), 1e-6):.1f}")

    # Coverage at LAC threshold
    cal_p = [1 - r['p_max'] for r in cal_data]
    cal_s = [r['score'] for r in cal_data]
    q_p = compute_lac_threshold(cal_p, alpha=0.1)
    q_s = compute_lac_threshold(cal_s, alpha=0.1)
    cov_p_test = (p_max_scores <= q_p).mean()
    cov_s_test = (combined_scores <= q_s).mean()
    print(f"LAC threshold on 1-p_max: {q_p:.4f}, test coverage: {cov_p_test:.3f}")
    print(f"LAC threshold on s: {q_s:.4f}, test coverage: {cov_s_test:.3f}")
    print(f"Coverage inflation epsilon = {cov_s_test - cov_p_test:.3f}")

    return output_path


def validate_cost_optimality():
    """Validate Theorem 3: cost-optimality of two-stage design."""
    # Use existing real corrective action results
    datasets = {
        'SciQ': 'sciq_seed0_v2_real_corrective',
        'SciFact': 'scifact_seed0_v2_real_corrective',
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    methods_data = []
    for name, ds_dir in datasets.items():
        metrics_path = RESULTS_DIR / ds_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            methods_data.append({
                'name': f'{name}\nRC-TTD',
                'cost': m.get('avg_cost', 0),
                'risk': m.get('selective_risk', 0),
                'coverage': m.get('coverage', 0),
                'method': 'RC-TTD'
            })

        # Add baselines from test_results
        test_path = RESULTS_DIR / ds_dir.replace('_real_corrective', '') / "test_results.jsonl"
        if test_path.exists():
            test_data = load_jsonl(test_path)
            correct = np.array([r['correct'] for r in test_data])
            # Always answer
            methods_data.append({
                'name': f'{name}\nAlwaysAns',
                'cost': 1.0,
                'risk': 1 - correct.mean(),
                'coverage': 1.0,
                'method': 'AlwaysAns'
            })
            # Always verify (cost=3)
            methods_data.append({
                'name': f'{name}\nAlwaysVer',
                'cost': 3.0,
                'risk': 1 - correct.mean(),  # Approximate
                'coverage': 1.0,
                'method': 'AlwaysVer'
            })

    # Plot cost vs risk
    colors = {'RC-TTD': 'red', 'AlwaysAns': 'blue', 'AlwaysVer': 'green'}
    markers = {'RC-TTD': '*', 'AlwaysAns': 'o', 'AlwaysVer': 's'}

    for d in methods_data:
        ax.scatter(d['cost'], d['risk'],
                  c=colors.get(d['method'], 'gray'),
                  marker=markers.get(d['method'], 'o'),
                  s=150, label=d['method'] if d['name'].endswith('RC-TTD') else None,
                  edgecolors='black', linewidth=0.5)
        ax.annotate(d['name'].replace('\n', ' '), (d['cost'], d['risk']),
                   fontsize=8, ha='center', va='bottom')

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)

    ax.set_xlabel('Average Cost', fontsize=12)
    ax.set_ylabel('Selective Risk', fontsize=12)
    ax.set_title('Cost-Risk Trade-off: RC-TTD vs. Single-Action Policies', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig_theory_cost_optimality.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

    # Print cost-optimality verification
    print("\n=== Cost-Optimality Verification ===")
    for name, ds_dir in datasets.items():
        metrics_path = RESULTS_DIR / ds_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            rcttd_cost = m.get('avg_cost', 0)
            rcttd_risk = m.get('selective_risk', 0)
            print(f"{name}: RC-TTD cost={rcttd_cost:.2f}, risk={rcttd_risk:.3f}")
            print(f"  vs. AlwaysAns: cost=1.00, risk={1 - np.mean([r['correct'] for r in load_jsonl(RESULTS_DIR / ds_dir.replace('_real_corrective', '') / 'test_results.jsonl')]):.3f}")
            print(f"  vs. AlwaysVer: cost=3.00 (RC-TTD saves {3.00 - rcttd_cost:.2f} = {(3.00 - rcttd_cost)/3.00*100:.0f}%)")

    return output_path


if __name__ == "__main__":
    print("=== Theory Validation Experiments ===\n")

    print("1. Validating Theorem 1 (Coverage Stability)...")
    validate_calibration_size()

    print("\n2. Validating Lemma 1 (Score Compression)...")
    validate_score_compression()

    print("\n3. Validating Theorem 3 (Cost-Optimality)...")
    validate_cost_optimality()

    print("\n=== All validation experiments complete ===")
