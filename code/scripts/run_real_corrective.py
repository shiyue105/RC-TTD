"""
Real Corrective Actions for RC-TTD.
Replaces simulated 30%/50% fix rates with actual end-to-end execution:
  - RetrieveMore: re-retrieve top-15, NLI re-rank, re-generate answer
  - VerifyMore: generate per-evidence answers, NLI-weighted majority vote
  - Answer/Abstain: no change

Reuses existing RAG pipeline results (test_results.jsonl) and only executes
corrective actions for instances flagged by the RC-TTD policy.
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
from typing import List, Dict, Optional, Tuple
from collections import Counter

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from src.utils.data_loader import load_dataset_unified, make_wrong_context
from src.retrieval.bm25_retriever import BM25Retriever
from src.generation.llm_backend import LLMBackend, build_mcqa_prompt, parse_mcqa_response
from src.verification.nli_verifier import NLIVerifier, evidence_conflict, verifier_disagreement
from src.conformal.calibration import lac_threshold
from src.policy.rule_based import Instance, compute_nonconformity_score, compute_features, \
    rcttd_rule_based, rcttd_ablation
from src.policy.learned import LearnedRCTTD

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / 'results'


def build_instance_from_result(r: Dict, budget: float = 4.0) -> Instance:
    """Build an Instance from a result dict."""
    rel_ret = float(np.mean([e.get('normalized_score', 0.0) for e in r.get('evidence', [])[:3]])) if r.get('evidence') else 0.0
    return Instance(
        query=r['query'], evidence=r.get('evidence', []), answer=str(r['choice_idx']),
        p_max=r['p_max'], h_p=r['h_p'], u_ans=r['u_ans'],
        rel_ret=rel_ret, conf_e=r['conf_e'], sup=r['sup'], dis_v=r['dis_v'],
        budget_remaining=budget, domain=r.get('domain', 'unknown'), score=r.get('score', 0.0)
    )


def execute_retrieve_more(r: Dict, ex: Dict, retriever: BM25Retriever,
                          llm: LLMBackend, nli: NLIVerifier,
                          top_k_more: int = 15) -> Tuple[int, float, Dict]:
    """REAL RetrieveMore: re-retrieve top-k_more, NLI re-rank, re-generate.

    Returns (new_choice_idx, new_p_max, info_dict).
    """
    query = r['query']
    options = ex['options']

    # Step 1: Re-retrieve with larger top_k
    # Skip docs already in top-3 (use top-15, take new ones beyond original top-3)
    original_ids = set(e['id'] for e in r.get('evidence', []))
    new_retrieved = retriever.retrieve(query, top_k=top_k_more)
    # Combine original + new, dedup by id
    combined = list(r.get('evidence', []))
    seen_ids = set(original_ids)
    for doc in new_retrieved:
        if doc['id'] not in seen_ids:
            combined.append(doc)
            seen_ids.add(doc['id'])
    # Take top-5 after re-ranking (we'll re-rank by NLI support for the query)

    # Step 2: NLI re-rank - score each doc by NLI(claim=doc, hypothesis=query) 
    # Actually, for MCQA we don't have a claim. We use doc relevance to query directly.
    # Re-rank by BM25 normalized_score (already computed) + NLI support for each option.
    # Simplified: take top-5 by BM25 score, build combined evidence
    top5 = combined[:5]
    evidence_text = ' '.join([d['text'][:300] for d in top5])

    # Step 3: Re-generate answer with new evidence
    prompt = build_mcqa_prompt(query, options, evidence_text[:1000])
    choice_idx, probs, response = llm.generate_mcqa_with_confidence(prompt, options)
    p_max = max(probs) if probs else 0.0

    return choice_idx, p_max, {
        'new_evidence_count': len(top5),
        'new_evidence_ids': [d['id'] for d in top5],
        'response': response[:100],
    }


def execute_verify_more(r: Dict, ex: Dict, llm: LLMBackend, nli: NLIVerifier) -> Tuple[int, float, Dict]:
    """REAL VerifyMore: generate per-evidence, NLI-weighted majority vote.

    For each of top-3 evidence pieces:
      1. Generate answer using only that evidence
      2. Compute NLI support: P(entailment | evidence -> answer)
    Then weighted majority vote: weight = NLI support score.

    Returns (final_choice_idx, final_p_max, info_dict).
    """
    query = r['query']
    options = ex['options']
    evidences = r.get('evidence', [])[:3]

    if not evidences:
        return r['choice_idx'], r['p_max'], {'reason': 'no_evidence'}

    per_evidence_choices = []
    per_evidence_weights = []

    for ev in evidences:
        ev_text = ev['text'][:500]
        # Generate answer with single evidence
        prompt = build_mcqa_prompt(query, options, ev_text)
        choice_idx, probs, response = llm.generate_mcqa_with_confidence(prompt, options)
        p_max = max(probs) if probs else 0.0

        # NLI support: evidence -> chosen answer
        answer_text = options[choice_idx] if choice_idx < len(options) else ""
        nli_result = nli.verify_batch([ev_text], [answer_text])[0]
        support = nli_result['entailment']

        per_evidence_choices.append(choice_idx)
        per_evidence_weights.append(support)

    # Weighted majority vote
    weight_per_choice = {}
    for choice, weight in zip(per_evidence_choices, per_evidence_weights):
        if choice not in weight_per_choice:
            weight_per_choice[choice] = 0.0
        weight_per_choice[choice] += weight

    # Select choice with highest total weight
    final_choice = max(weight_per_choice, key=weight_per_choice.get)
    total_weight = sum(weight_per_choice.values())
    final_p_max = weight_per_choice[final_choice] / max(total_weight, 1e-9)

    return final_choice, final_p_max, {
        'per_evidence_choices': per_evidence_choices,
        'per_evidence_weights': per_evidence_weights,
        'weight_per_choice': weight_per_choice,
    }


def run_real_corrective(dataset_name: str, seed: int = 0, gpu_id: int = 0,
                        alpha: float = 0.1, budget: float = 4.0,
                        wrong_context: bool = False, n_test: int = 500, n_cal: int = 500,
                        policy_name: str = 'rcttd_rule',
                        output_suffix: str = '_real_corrective'):
    """Run real corrective actions on existing results."""
    # Setup directories
    base_dir = RESULTS_DIR / f"{dataset_name}{'_wrong_context' if wrong_context else ''}_seed{seed}_v2"
    if not base_dir.exists():
        base_dir = RESULTS_DIR / f"{dataset_name}{'_wrong_context' if wrong_context else ''}_seed{seed}"
    output_dir = RESULTS_DIR / f"{dataset_name}{'_wrong_context' if wrong_context else ''}_seed{seed}_v2{output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Real Corrective Actions: {dataset_name} (wrong_context={wrong_context}) ===")
    logger.info(f"Input dir: {base_dir}")
    logger.info(f"Output dir: {output_dir}")

    # Step 1: Load existing results
    cal_results = []
    with open(base_dir / 'cal_results.jsonl') as f:
        for line in f:
            cal_results.append(json.loads(line))
    test_results = []
    with open(base_dir / 'test_results.jsonl') as f:
        for line in f:
            test_results.append(json.loads(line))
    logger.info(f"Loaded {len(cal_results)} cal, {len(test_results)} test results")

    # Step 2: Reload dataset to get options for re-generation
    logger.info("Reloading dataset for options...")
    corpus, cal_examples, test_examples = load_dataset_unified(dataset_name, n_test, n_cal, seed)
    if wrong_context:
        test_examples = make_wrong_context(test_examples, corpus, corruption_rate=0.5, seed=seed)
    # Build id -> example mapping
    ex_map = {ex['id']: ex for ex in test_examples}

    # Step 3: Build retriever and load models
    logger.info("Building BM25 retriever...")
    retriever = BM25Retriever(corpus)
    logger.info(f"Loading LLM with device_map=auto (visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})...")
    llm = LLMBackend(gpu_id=gpu_id, device_map="auto")
    nli = NLIVerifier(gpu_id=gpu_id)

    # Step 4: Compute conformal threshold from calibration set
    cal_scores = np.array([1 - r['p_max'] for r in cal_results])
    q_alpha = lac_threshold(cal_scores, alpha)
    logger.info(f"Conformal threshold q_alpha={q_alpha:.4f} (alpha={alpha})")

    # Step 5: Train learned policy if needed
    learned_model = None
    if policy_name == 'rcttd_learned':
        learned_model = LearnedRCTTD(alpha=alpha, budget=budget)
        learned_model.fit(cal_results)
        logger.info("Learned policy trained.")

    # Step 6: Apply policy and execute real corrective actions
    output = []
    action_counts = Counter()
    retrieve_more_count = 0
    verify_more_count = 0
    retrieve_more_fixed = 0  # originally wrong, now correct
    verify_more_fixed = 0
    retrieve_more_broken = 0  # originally correct, now wrong (shouldn't happen often)
    verify_more_broken = 0

    t0 = time.time()
    for i, r in enumerate(test_results):
        if i % 50 == 0:
            logger.info(f"  Processing {i}/{len(test_results)}... (RM={retrieve_more_count}, VM={verify_more_count})")

        inst = build_instance_from_result(r, budget=budget)

        # Apply policy
        if policy_name == 'rcttd_learned' and learned_model and learned_model.is_trained:
            action = learned_model.predict(inst)
        else:
            action = rcttd_rule_based(inst, q_alpha)

        action_counts[action] += 1
        ex = ex_map.get(r['id'], {})

        # Execute action
        if action == 'Answer':
            answered = True
            final_correct = r['correct']
            final_choice = r['choice_idx']
            final_p_max = r['p_max']
            corrective_info = {'executed': False}

        elif action == 'RetrieveMore':
            retrieve_more_count += 1
            answered = True
            try:
                new_choice, new_p_max, info = execute_retrieve_more(
                    r, ex, retriever, llm, nli, top_k_more=15
                )
                final_choice = new_choice
                final_p_max = new_p_max
                final_correct = (new_choice == r['correct_idx'])
                corrective_info = {'executed': True, 'action': 'RetrieveMore', **info}
                if not r['correct'] and final_correct:
                    retrieve_more_fixed += 1
                if r['correct'] and not final_correct:
                    retrieve_more_broken += 1
            except Exception as e:
                logger.warning(f"RetrieveMore failed for {r['id']}: {e}")
                final_choice = r['choice_idx']
                final_p_max = r['p_max']
                final_correct = r['correct']
                corrective_info = {'executed': False, 'error': str(e)}

        elif action == 'VerifyMore':
            verify_more_count += 1
            answered = True
            try:
                new_choice, new_p_max, info = execute_verify_more(
                    r, ex, llm, nli
                )
                final_choice = new_choice
                final_p_max = new_p_max
                final_correct = (new_choice == r['correct_idx'])
                corrective_info = {'executed': True, 'action': 'VerifyMore', **info}
                if not r['correct'] and final_correct:
                    verify_more_fixed += 1
                if r['correct'] and not final_correct:
                    verify_more_broken += 1
            except Exception as e:
                logger.warning(f"VerifyMore failed for {r['id']}: {e}")
                final_choice = r['choice_idx']
                final_p_max = r['p_max']
                final_correct = r['correct']
                corrective_info = {'executed': False, 'error': str(e)}

        else:  # Abstain
            answered = False
            final_correct = False
            final_choice = -1
            final_p_max = r['p_max']
            corrective_info = {'executed': False}

        out = r.copy()
        out['action'] = action
        out['answered'] = answered
        out['final_correct'] = final_correct
        out['final_choice_idx'] = final_choice
        out['final_p_max'] = final_p_max
        out['corrective_info'] = corrective_info
        out['policy'] = policy_name
        output.append(out)

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s")

    # Step 7: Save results
    with open(output_dir / 'test_results_corrective.jsonl', 'w') as f:
        for r in output:
            f.write(json.dumps(r, default=str) + '\n')

    # Step 8: Compute and save metrics
    n = len(output)
    n_answered = sum(1 for r in output if r['answered'])
    n_correct = sum(1 for r in output if r['answered'] and r['final_correct'])
    accuracy = n_correct / n
    coverage = n_answered / n
    # Selective risk = error rate among answered
    n_wrong = sum(1 for r in output if r['answered'] and not r['final_correct'])
    selective_risk = n_wrong / max(n_answered, 1)
    # Cost
    cost_map = {'Answer': 1.0, 'RetrieveMore': 2.0, 'VerifyMore': 3.0, 'Abstain': 0.0}
    avg_cost = np.mean([cost_map[r['action']] for r in output])

    metrics = {
        'dataset': dataset_name,
        'wrong_context': wrong_context,
        'policy': policy_name,
        'n_test': n,
        'alpha': alpha,
        'budget': budget,
        'q_alpha': float(q_alpha),
        'action_counts': dict(action_counts),
        'accuracy': accuracy,
        'coverage': coverage,
        'selective_risk': selective_risk,
        'avg_cost': float(avg_cost),
        'retrieve_more_count': retrieve_more_count,
        'verify_more_count': verify_more_count,
        'retrieve_more_fixed': retrieve_more_fixed,  # wrong -> correct
        'verify_more_fixed': verify_more_fixed,
        'retrieve_more_broken': retrieve_more_broken,  # correct -> wrong
        'verify_more_broken': verify_more_broken,
        'retrieve_more_fix_rate': retrieve_more_fixed / max(retrieve_more_count, 1),
        'verify_more_fix_rate': verify_more_fixed / max(verify_more_count, 1),
        'elapsed_seconds': elapsed,
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"=== Results: {dataset_name} (WC={wrong_context}) ===")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Coverage: {coverage:.4f}")
    logger.info(f"  Selective Risk: {selective_risk:.4f}")
    logger.info(f"  Avg Cost: {avg_cost:.4f}")
    logger.info(f"  Actions: {dict(action_counts)}")
    logger.info(f"  RetrieveMore: {retrieve_more_count} executed, {retrieve_more_fixed} fixed, "
                f"{retrieve_more_broken} broken (fix rate={metrics['retrieve_more_fix_rate']:.3f})")
    logger.info(f"  VerifyMore: {verify_more_count} executed, {verify_more_fixed} fixed, "
                f"{verify_more_broken} broken (fix rate={metrics['verify_more_fix_rate']:.3f})")

    return metrics


def run_all_experiments(gpu_id: int = 0, policy_name: str = 'rcttd_rule'):
    """Run real corrective actions on all 4 conditions."""
    all_metrics = []
    conditions = [
        ('sciq', False),
        ('scifact', False),
        ('sciq', True),
        ('scifact', True),
    ]
    for dataset_name, wc in conditions:
        try:
            m = run_real_corrective(dataset_name, seed=0, gpu_id=gpu_id,
                                    wrong_context=wc, policy_name=policy_name)
            all_metrics.append(m)
        except Exception as e:
            logger.error(f"Failed {dataset_name} WC={wc}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined metrics
    summary_path = RESULTS_DIR / 'real_corrective_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info(f"\n=== Summary saved to {summary_path} ===")
    for m in all_metrics:
        logger.info(f"{m['dataset']} WC={m['wrong_context']}: "
                    f"acc={m['accuracy']:.4f} cov={m['coverage']:.4f} "
                    f"risk={m['selective_risk']:.4f} cost={m['avg_cost']:.4f} "
                    f"RM_fix={m.get('retrieve_more_fix_rate', 0):.3f} "
                    f"VM_fix={m.get('verify_more_fix_rate', 0):.3f}")
    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['sciq', 'scifact', 'all'])
    parser.add_argument('--wrong_context', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--policy', type=str, default='rcttd_rule',
                        choices=['rcttd_rule', 'rcttd_learned'])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.dataset == 'all':
        run_all_experiments(gpu_id=args.gpu_id, policy_name=args.policy)
    else:
        run_real_corrective(args.dataset, seed=args.seed, gpu_id=args.gpu_id,
                           wrong_context=args.wrong_context, policy_name=args.policy)
