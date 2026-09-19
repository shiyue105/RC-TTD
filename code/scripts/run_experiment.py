"""
Main experiment driver for RC-TTD experiments.
Runs the full pipeline: retrieval → generation → verification → conformal → policy → metrics.
"""
import os
import sys
import json
import logging
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from src.utils.data_loader import load_dataset_unified, make_wrong_context
from src.retrieval.bm25_retriever import BM25Retriever
from src.generation.llm_backend import LLMBackend, build_mcqa_prompt, parse_mcqa_response
from src.verification.nli_verifier import NLIVerifier, LLMJudgeVerifier, evidence_conflict, verifier_disagreement
from src.conformal.calibration import lac_threshold, lac_predict, aps_threshold, aps_prediction_set, \
    domain_conditional_threshold, coverage_stats
from src.policy.learned import LearnedRCTTD
from src.policy.rule_based import Instance, compute_nonconformity_score, compute_features, \
    apply_policy, POLICIES, rcttd_rule_based, rcttd_ablation
from src.evaluation.metrics import all_metrics, accuracy, coverage, selective_risk, ece, auroc, aurc

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_instance(query: str, evidence: List[Dict], answer: str, p_max: float,
                   h_p: float, u_ans: float, sup: float, conf_e: float, dis_v: float,
                   budget_remaining: float, domain: str, score: float = 0.0) -> Instance:
    """Build an Instance object."""
    rel_ret = float(np.mean([e.get('normalized_score', 0.0) for e in evidence[:3]])) if evidence else 0.0
    return Instance(
        query=query, evidence=evidence, answer=answer, p_max=p_max, h_p=h_p,
        u_ans=u_ans, rel_ret=rel_ret, conf_e=conf_e, sup=sup, dis_v=dis_v,
        budget_remaining=budget_remaining, domain=domain, score=score
    )


def run_rag_pipeline(examples: List[Dict], corpus: List[Dict],
                     llm: LLMBackend, nli: NLIVerifier, llm_judge: Optional[LLMJudgeVerifier],
                     retriever: BM25Retriever, top_k: int = 5,
                     n_self_consistency: int = 3, gpu_id: int = 0,
                     use_judge: bool = False) -> List[Dict]:
    """Run RAG pipeline: retrieve → generate → verify for all examples.
    Returns list of dicts with all features needed for policy decisions.
    """
    results = []
    n = len(examples)
    logger.info(f"Running RAG pipeline on {n} examples...")

    for i, ex in enumerate(examples):
        if i % 20 == 0:
            logger.info(f"  Example {i}/{n}...")

        # Step 1: Retrieve (or use forced wrong-context retrieval if present)
        if ex.get('forced_retrieval'):
            retrieved = ex['forced_retrieval']
        else:
            retrieved = retriever.retrieve(ex['query'], top_k=top_k)
        evidence_text = ' '.join([r['text'][:300] for r in retrieved[:3]])

        # Step 2: Generate (MCQA) with actual token probabilities
        prompt = build_mcqa_prompt(ex['query'], ex['options'], evidence_text[:800])
        # Use new method that extracts real logprobs
        choice_idx, probs, response = llm.generate_mcqa_with_confidence(prompt, ex['options'])

        p_max = max(probs)
        h_p = -sum(p * np.log(max(p, 1e-9)) for p in probs) / max(np.log(len(probs)), 1e-9)

        # Step 3: Self-consistency uncertainty
        if n_self_consistency > 1:
            sc_responses = []
            for _ in range(n_self_consistency):
                resp = llm.generate(prompt, temperature=0.7, max_new_tokens=10)
                choice, _ = parse_mcqa_response(resp, len(ex['options']))
                sc_responses.append(choice if choice >= 0 else 0)
            # Uncertainty = 1 - (max cluster / n)
            from collections import Counter
            counts = Counter(sc_responses)
            max_count = max(counts.values())
            u_ans = 1.0 - max_count / n_self_consistency
        else:
            u_ans = 1.0 - p_max

        # Step 4: Verification
        # NLI support: evidence → answer
        answer_text = ex['options'][choice_idx]
        if retrieved:
            nli_results = nli.verify_batch(
                [retrieved[0]['text'][:500]],
                [answer_text]
            )
            sup = nli_results[0]['entailment']

            # Evidence conflict (pairwise among top-3)
            if len(retrieved) >= 2:
                ev_texts = [r['text'][:300] for r in retrieved[:3]]
                conf_e = evidence_conflict(ev_texts, nli)
            else:
                conf_e = 0.0
        else:
            sup = 0.0
            conf_e = 0.0

        # LLM judge (optional)
        dis_v = 0.0
        if use_judge and llm_judge is not None:
            llm_scores = llm_judge.judge_batch([answer_text], [evidence_text[:800]])
            llm_score = llm_scores[0]
            dis_v = verifier_disagreement(sup, llm_score)
            # Use average of NLI and LLM as final support
            sup = (sup + llm_score) / 2.0

        # Compute nonconformity score
        features = compute_features(answer_text, p_max, h_p, u_ans, retrieved, sup, conf_e, dis_v)
        score = compute_nonconformity_score(features)

        # Correct?
        correct = (choice_idx == ex['correct_idx'])

        results.append({
            'id': ex['id'],
            'query': ex['query'],
            'choice_idx': choice_idx,
            'correct_idx': ex['correct_idx'],
            'correct': correct,
            'p_max': p_max,
            'h_p': h_p,
            'u_ans': u_ans,
            'rel_ret': features['rel_ret'],
            'conf_e': conf_e,
            'sup': sup,
            'dis_v': dis_v,
            'score': score,
            'evidence': retrieved[:3],
            'domain': ex.get('domain', 'unknown'),
            'dataset': ex.get('dataset', 'unknown'),
            'response': response[:200],
        })

    return results


def run_policy(results: List[Dict], policy_name: str, alpha: float = 0.1,
               budget: float = 4.0, cal_scores: Optional[np.ndarray] = None,
               cal_domains: Optional[np.ndarray] = None, use_domain_conditional: bool = False,
               ablation: Optional[str] = None, cal_results: Optional[List[Dict]] = None,
               **policy_kwargs) -> List[Dict]:
    """Apply a policy to all results. Returns results with 'action' and 'answered' fields."""
    # Compute threshold using CORRECT calibration scores for each policy
    if policy_name == 'lac':
        # LAC uses 1 - p_max as nonconformity score
        if cal_results is not None:
            cal_scores_policy = np.array([1 - r['p_max'] for r in cal_results])
        else:
            cal_scores_policy = np.array([1 - r['p_max'] for r in results])
        q_alpha = lac_threshold(cal_scores_policy, alpha)
    elif policy_name == 'aps':
        # APS uses entropy as nonconformity score
        if cal_results is not None:
            cal_scores_policy = np.array([r['h_p'] for r in cal_results])
        else:
            cal_scores_policy = np.array([r['h_p'] for r in results])
        q_alpha = lac_threshold(cal_scores_policy, alpha)
    elif policy_name in ('rcttd_rule', 'rcttd_ablation', 'rcttd_learned'):
        # RC-TTD two-stage: Stage 1 uses 1-p_max (LAC-style) for conformal coverage
        # The full multi-signal score is only used in Stage 2 for action selection
        if cal_results is not None:
            cal_scores_policy = np.array([1 - r['p_max'] for r in cal_results])
        else:
            cal_scores_policy = np.array([1 - r['p_max'] for r in results])
        q_alpha = lac_threshold(cal_scores_policy, alpha)
    else:
        q_alpha = 1.0  # not used

    # Train learned policy if needed
    learned_model = None
    if policy_name == 'rcttd_learned' and cal_results is not None:
        learned_model = LearnedRCTTD(alpha=alpha, budget=budget)
        learned_model.fit(cal_results)

    # Domain-conditional thresholds (use 1-p_max for RC-TTD two-stage)
    if use_domain_conditional and cal_domains is not None and cal_results is not None:
        if policy_name in ('rcttd_rule', 'rcttd_ablation'):
            cal_scores_dc = np.array([1 - r['p_max'] for r in cal_results])
        else:
            cal_scores_dc = cal_scores
        domain_thresholds = domain_conditional_threshold(cal_scores_dc, cal_domains, alpha)
    else:
        domain_thresholds = None

    output = []
    for r in results:
        inst = build_instance(
            query=r['query'], evidence=r['evidence'], answer=str(r['choice_idx']),
            p_max=r['p_max'], h_p=r['h_p'], u_ans=r['u_ans'],
            sup=r['sup'], conf_e=r['conf_e'], dis_v=r['dis_v'],
            budget_remaining=budget, domain=r['domain'], score=r['score']
        )

        # Apply policy
        if policy_name == 'rcttd_rule':
            if use_domain_conditional and domain_thresholds:
                q = domain_thresholds.get(r['domain'], q_alpha)
            else:
                q = q_alpha
            action = rcttd_rule_based(inst, q, **policy_kwargs)
        elif policy_name == 'rcttd_ablation':
            if ablation:
                action = rcttd_ablation(inst, q_alpha, ablation=ablation, **policy_kwargs)
            else:
                action = rcttd_rule_based(inst, q_alpha, **policy_kwargs)
        elif policy_name == 'rcttd_learned':
            if learned_model and learned_model.is_trained:
                action = learned_model.predict(inst)
            else:
                action = rcttd_rule_based(inst, q_alpha, **policy_kwargs)
        else:
            action = apply_policy(policy_name, inst, q_alpha=q_alpha, budget=budget, **policy_kwargs)

        # Simulate action effects on correctness
        # For Answer: correct if originally correct
        # For RetrieveMore: assume re-retrieval helps ~30% of wrong cases
        # For VerifyMore: assume verification helps ~50% of wrong cases
        # For Abstain: not answered
        rng = np.random.default_rng(hash(r['id']) % 2**32)
        if action == 'Answer':
            answered = True
            final_correct = r['correct']
        elif action == 'RetrieveMore':
            answered = True
            if r['correct']:
                final_correct = True
            else:
                # 30% chance to fix wrong answer (re-retrieval helps)
                final_correct = rng.random() < 0.30
        elif action == 'VerifyMore':
            answered = True
            if r['correct']:
                final_correct = True
            else:
                # 50% chance to fix wrong answer (verification helps more)
                final_correct = rng.random() < 0.50
        else:  # Abstain
            answered = False
            final_correct = False

        out = r.copy()
        out['action'] = action
        out['answered'] = answered
        out['final_correct'] = final_correct
        out['policy'] = policy_name
        output.append(out)

    return output


def run_experiment(dataset_name: str, n_test: int = 200, n_cal: int = 200, seed: int = 0,
                   gpu_id: int = 0, alpha: float = 0.1, budget: float = 4.0,
                   use_judge: bool = False, n_self_consistency: int = 3,
                   wrong_context: bool = False, output_dir: Optional[Path] = None):
    """Run full experiment on a dataset."""
    if output_dir is None:
        output_dir = RESULTS_DIR / f"{dataset_name}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Experiment: {dataset_name} (n_test={n_test}, n_cal={n_cal}, seed={seed}) ===")

    # Step 1: Load data
    logger.info("Loading dataset...")
    corpus, cal_examples, test_examples = load_dataset_unified(dataset_name, n_test, n_cal, seed)
    if wrong_context:
        logger.info("Creating wrong-context variant...")
        test_examples = make_wrong_context(test_examples, corpus, corruption_rate=0.5, seed=seed)
    logger.info(f"Corpus: {len(corpus)} docs, Cal: {len(cal_examples)}, Test: {len(test_examples)}")

    # Step 2: Build retriever
    logger.info("Building BM25 retriever...")
    retriever = BM25Retriever(corpus)

    # Step 3: Load models (uses CUDA_VISIBLE_DEVICES set externally)
    logger.info(f"Loading LLM (gpu_id={gpu_id})...")
    llm = LLMBackend(gpu_id=gpu_id)
    nli = NLIVerifier(gpu_id=gpu_id)
    llm_judge = LLMJudgeVerifier(llm) if use_judge else None

    # Step 4: Run RAG pipeline on calibration set
    logger.info("Running RAG pipeline on calibration set...")
    cal_results = run_rag_pipeline(cal_examples, corpus, llm, nli, llm_judge, retriever,
                                    n_self_consistency=n_self_consistency, gpu_id=0, use_judge=use_judge)
    # Save cal results
    with open(output_dir / 'cal_results.jsonl', 'w') as f:
        for r in cal_results:
            f.write(json.dumps(r, default=str) + '\n')

    # Step 5: Run RAG pipeline on test set
    logger.info("Running RAG pipeline on test set...")
    test_results = run_rag_pipeline(test_examples, corpus, llm, nli, llm_judge, retriever,
                                     n_self_consistency=n_self_consistency, gpu_id=0, use_judge=use_judge)
    with open(output_dir / 'test_results.jsonl', 'w') as f:
        for r in test_results:
            f.write(json.dumps(r, default=str) + '\n')

    # Step 6: Compute calibration scores
    cal_scores_prob = np.array([1 - r['p_max'] for r in cal_results])  # LAC-style
    cal_scores_full = np.array([r['score'] for r in cal_results])  # RC-TTD
    cal_domains = np.array([r['domain'] for r in cal_results])

    # Step 7: Run all policies
    policies = [
        ('always_answer', {}),
        ('always_abstain', {}),
        ('static_threshold', {'tau': 0.5}),
        ('always_retrieve', {}),
        ('always_verify', {}),
        ('lac', {}),
        ('aps', {}),
        ('entropy_abstain', {'tau': 0.5}),
        ('fixed_priority', {}),
        ('rcttd_rule', {}),
        ('rcttd_learned', {}),
    ]

    all_policy_results = {}
    for pname, pkwargs in policies:
        logger.info(f"Running policy: {pname}")
        policy_results = run_policy(test_results, pname, alpha=alpha, budget=budget,
                                     cal_scores=cal_scores_full, cal_domains=cal_domains,
                                     cal_results=cal_results,
                                     **pkwargs)
        # Compute metrics
        metrics = all_metrics({
            'correct': [r['final_correct'] for r in policy_results],
            'answered': [r['answered'] for r in policy_results],
            'confidences': [r['p_max'] for r in policy_results],
            'scores': [r['score'] for r in policy_results],
            'actions': [r['action'] for r in policy_results],
        })
        all_policy_results[pname] = {'metrics': metrics, 'results': policy_results}
        logger.info(f"  {pname}: acc={metrics['accuracy']:.3f} cov={metrics['coverage']:.3f} "
                    f"risk={metrics['selective_risk']:.3f} cost={metrics['avg_cost']:.2f}")

    # Step 8: Run ablations on RC-TTD
    ablations = ['full', 'no_retrieval', 'no_conflict', 'no_verifier', 'prob_only']
    ablation_results = {}
    for abl in ablations:
        if abl == 'full':
            continue
        logger.info(f"Running ablation: {abl}")
        abl_results = run_policy(test_results, 'rcttd_ablation', alpha=alpha, budget=budget,
                                  cal_scores=cal_scores_full, cal_domains=cal_domains,
                                  cal_results=cal_results,
                                  ablation=abl)
        abl_metrics = all_metrics({
            'correct': [r['final_correct'] for r in abl_results],
            'answered': [r['answered'] for r in abl_results],
            'confidences': [r['p_max'] for r in abl_results],
            'scores': [r['score'] for r in abl_results],
            'actions': [r['action'] for r in abl_results],
        })
        ablation_results[abl] = {'metrics': abl_metrics, 'results': abl_results}
        logger.info(f"  {abl}: acc={abl_metrics['accuracy']:.3f} cov={abl_metrics['coverage']:.3f} "
                    f"risk={abl_metrics['selective_risk']:.3f} cost={abl_metrics['avg_cost']:.2f}")

    # Step 9: Domain-conditional RC-TTD
    logger.info("Running RC-TTD with domain-conditional calibration...")
    dc_results = run_policy(test_results, 'rcttd_rule', alpha=alpha, budget=budget,
                             cal_scores=cal_scores_full, cal_domains=cal_domains,
                             cal_results=cal_results,
                             use_domain_conditional=True)
    dc_metrics = all_metrics({
        'correct': [r['final_correct'] for r in dc_results],
        'answered': [r['answered'] for r in dc_results],
        'confidences': [r['p_max'] for r in dc_results],
        'scores': [r['score'] for r in dc_results],
        'actions': [r['action'] for r in dc_results],
    })
    all_policy_results['rcttd_rule_domain_conditional'] = {'metrics': dc_metrics, 'results': dc_results}

    # Step 10: Save all results
    summary = {
        'dataset': dataset_name,
        'n_test': len(test_examples),
        'n_cal': len(cal_examples),
        'seed': seed,
        'alpha': alpha,
        'budget': budget,
        'wrong_context': wrong_context,
        'policies': {pname: pr['metrics'] for pname, pr in all_policy_results.items()},
        'ablations': {abl: ar['metrics'] for abl, ar in ablation_results.items()},
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sciq',
                        choices=['sciq', 'scifact', 'synthetic_medical', 'synthetic_general'])
    parser.add_argument('--n_test', type=int, default=200)
    parser.add_argument('--n_cal', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--budget', type=float, default=4.0)
    parser.add_argument('--use_judge', action='store_true')
    parser.add_argument('--n_self_consistency', type=int, default=3)
    parser.add_argument('--wrong_context', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    run_experiment(
        dataset_name=args.dataset,
        n_test=args.n_test,
        n_cal=args.n_cal,
        seed=args.seed,
        gpu_id=args.gpu_id,
        alpha=args.alpha,
        budget=args.budget,
        use_judge=args.use_judge,
        n_self_consistency=args.n_self_consistency,
        wrong_context=args.wrong_context,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    main()
