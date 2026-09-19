"""
Unified data loader for RAG conformal decision experiments.
Supports: sciq, scifact, and synthetic datasets.
"""
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CACHE_DIR = _PROJECT_ROOT / "data" / "processed"


def load_sciq(n_test: int = 200, n_cal: int = 200, seed: int = 0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load SciQ dataset. Returns (corpus, cal_examples, test_examples).
    Each example: {query, options, correct_idx, evidence_text, domain}.
    Corpus: list of {id, text} for retrieval.
    """
    from datasets import load_dataset
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ds_train = load_dataset('sciq', split='train')
    ds_test = load_dataset('sciq', split='test')

    # Build corpus from all support texts (train + test)
    corpus = []
    for i, ex in enumerate(ds_train):
        if ex['support']:
            corpus.append({'id': f'train_{i}', 'text': ex['support'][:500]})
    for i, ex in enumerate(ds_test):
        if ex['support']:
            corpus.append({'id': f'test_{i}', 'text': ex['support'][:500]})
    logger.info(f"SciQ corpus: {len(corpus)} docs")

    # Build MCQA examples
    def make_example(ex, idx):
        options = [ex['correct_answer'], ex['distractor1'], ex['distractor2'], ex['distractor3']]
        # Shuffle
        rng = np.random.default_rng(seed + idx)
        order = rng.permutation(4)
        shuffled = [options[i] for i in order]
        correct_idx = int(np.where(order == 0)[0][0])
        return {
            'id': f'sciq_{idx}',
            'query': ex['question'],
            'options': shuffled,
            'correct_idx': correct_idx,
            'evidence_text': ex['support'][:500] if ex['support'] else "",
            'domain': 'science',
            'dataset': 'sciq',
        }

    # Use train split for calibration, test for testing
    rng = np.random.default_rng(seed)
    cal_indices = rng.choice(len(ds_train), size=min(n_cal, len(ds_train)), replace=False)
    cal_examples = [make_example(ds_train[int(i)], i) for i in cal_indices]
    test_indices = rng.choice(len(ds_test), size=min(n_test, len(ds_test)), replace=False)
    test_examples = [make_example(ds_test[int(i)], i) for i in test_indices]
    return corpus, cal_examples, test_examples


def load_scifact(n_test: int = 200, n_cal: int = 200, seed: int = 0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load SciFact from local data."""
    scifact_data_dir = _PROJECT_ROOT / "data" / "raw" / "scifact" / "data"
    if not scifact_data_dir.exists():
        raise FileNotFoundError(f"SciFact data not found at {scifact_data_dir}")

    # Load corpus
    corpus = []
    with open(scifact_data_dir / 'corpus.jsonl') as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get('title', '') + " " + ' '.join(doc.get('abstract', []))
            corpus.append({'id': str(doc['doc_id']), 'text': text[:800]})
    logger.info(f"SciFact corpus: {len(corpus)} docs")

    # Load claims
    def load_claims(path):
        examples = []
        if not path.exists():
            return examples
        with open(path) as f:
            for line in f:
                c = json.loads(line)
                examples.append(c)
        return examples

    train_claims = load_claims(scifact_data_dir / 'claims_train.jsonl')
    dev_claims = load_claims(scifact_data_dir / 'claims_dev.jsonl')

    # Convert to MCQA: for each claim, create 2 options (supported / not supported)
    def make_example(claim, idx, corpus_lookup):
        # Get evidence text if available
        evidence_text = ""
        evidence_doc_ids = []
        # evidence is a dict: {doc_id: [{"sentence": [...], "label": "SUPPORT"}]}
        evidence_dict = claim.get('evidence', {})
        if isinstance(evidence_dict, dict):
            for doc_id_str, ev_list in evidence_dict.items():
                evidence_doc_ids.append(str(doc_id_str))
        elif isinstance(evidence_dict, list):
            for ev in evidence_dict:
                evidence_doc_ids.append(str(ev.get('doc_id', '')))

        # Also check cited_doc_ids
        cited = claim.get('cited_doc_ids', [])
        for cid in cited:
            sid = str(cid)
            if sid not in evidence_doc_ids:
                evidence_doc_ids.append(sid)

        if evidence_doc_ids:
            for doc in corpus:
                if doc['id'] in evidence_doc_ids:
                    evidence_text = doc['text']
                    break

        # MCQA: 2 options - "Supported" / "Not Supported"
        # Determine correct answer from evidence
        has_support = bool(evidence_dict) or bool(cited)
        options = ["The claim is supported by evidence", "The claim is not supported by evidence"]
        correct_idx = 0 if has_support else 1
        # Shuffle
        rng = np.random.default_rng(seed + idx)
        order = rng.permutation(2)
        shuffled = [options[i] for i in order]
        correct_idx = int(np.where(order == correct_idx)[0][0])

        return {
            'id': f'scifact_{claim.get("id", idx)}',
            'query': claim['claim'],
            'options': shuffled,
            'correct_idx': correct_idx,
            'evidence_text': evidence_text,
            'domain': 'science',
            'dataset': 'scifact',
            'gold_doc_ids': evidence_doc_ids,
        }

    corpus_lookup = {d['id']: d for d in corpus}
    all_claims = train_claims + dev_claims
    rng = np.random.default_rng(seed)
    # Split: 50% cal, 50% test (since test set has no labels)
    indices = rng.permutation(len(all_claims))
    n_total = len(all_claims)
    cal_idx = indices[:min(n_cal, n_total // 2)]
    test_idx = indices[min(n_cal, n_total // 2):min(n_cal, n_total // 2) + min(n_test, n_total // 2)]
    cal_examples = [make_example(all_claims[int(i)], i, corpus_lookup) for i in cal_idx]
    test_examples = [make_example(all_claims[int(i)], i, corpus_lookup) for i in test_idx]
    return corpus, cal_examples, test_examples


def make_wrong_context(examples: List[Dict], corpus: List[Dict],
                       corruption_rate: float = 0.5, seed: int = 0) -> List[Dict]:
    """Create wrong-context variant: replace gold evidence with random docs.

    Stores forced_retrieval (list of random docs) so the RAG pipeline can
    inject misleading context instead of using BM25-retrieved evidence.
    """
    rng = np.random.default_rng(seed)
    corrupted = []
    for ex in examples:
        ex_copy = ex.copy()
        # Replace evidence with random docs (3 random docs to mimic top-k=3)
        if rng.random() < corruption_rate and corpus:
            random_docs = [corpus[int(i)] for i in rng.integers(0, len(corpus), size=3)]
            ex_copy['forced_retrieval'] = random_docs
            ex_copy['evidence_text'] = ' '.join([d['text'][:300] for d in random_docs])
            ex_copy['wrong_context'] = True
        else:
            ex_copy['wrong_context'] = False
        corrupted.append(ex_copy)
    return corrupted


def make_synthetic_dataset(name: str, n_examples: int = 500, seed: int = 0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Generate synthetic RAG dataset for testing.
    Creates claims with known correct answers and varying difficulty.
    """
    rng = np.random.default_rng(seed)
    # Generate a corpus of "facts"
    facts = []
    templates = [
        "The {subject} is {value} in {domain}.",
        "Research shows that {subject} {value} during {period}.",
        "According to {source}, {subject} has {value} properties.",
        "The {subject} was {value} by {agent}.",
        "{subject} are {value} in {location}.",
    ]
    subjects = ['cancer', 'protein', 'virus', 'treatment', 'drug', 'vaccine', 'gene', 'cell',
                'reaction', 'compound', 'enzyme', 'hormone', 'bacteria', 'tissue', 'organ']
    values = ['discovered', 'effective', 'dangerous', 'safe', 'common', 'rare', 'increasing',
              'decreasing', 'linked', 'isolated', 'synthesized', 'approved']
    domains = ['medicine', 'biology', 'chemistry', 'physics', 'astronomy']
    sources = ['a 2020 study', 'recent research', 'clinical trials', 'the WHO', 'scientists']
    periods = ['2020', 'the pandemic', 'the last decade', 'recent years']
    agents = ['researchers', 'scientists', 'doctors', 'the team']
    locations = ['Europe', 'Asia', 'tropical regions', 'laboratories', 'hospitals']

    for i in range(1000):
        template = rng.choice(templates)
        fact = template.format(
            subject=rng.choice(subjects),
            value=rng.choice(values),
            domain=rng.choice(domains),
            source=rng.choice(sources),
            period=rng.choice(periods),
            agent=rng.choice(agents),
            location=rng.choice(locations),
        )
        facts.append({'id': f'fact_{i}', 'text': fact})

    # Split corpus
    corpus = facts
    # Generate examples
    examples = []
    for i in range(n_examples):
        # Pick a fact as the basis
        fact_idx = rng.integers(len(facts))
        fact = facts[fact_idx]
        # Create a claim (query) related to the fact
        claim = f"Is the following true: {fact['text']}"
        # MCQA: Yes / No / Uncertain
        correct = rng.integers(3)
        options = ["Yes, it is true", "No, it is false", "Uncertain, evidence is mixed"]
        ex = {
            'id': f'{name}_{i}',
            'query': claim,
            'options': options,
            'correct_idx': int(correct),
            'evidence_text': fact['text'] if rng.random() < 0.7 else "",
            'domain': rng.choice(domains),
            'dataset': name,
        }
        examples.append(ex)

    # Split into cal/test
    n_cal = min(200, n_examples // 3)
    cal = examples[:n_cal]
    test = examples[n_cal:n_cal + n_examples // 2]
    return corpus, cal, test


def load_dataset_unified(name: str, n_test: int = 200, n_cal: int = 200,
                          seed: int = 0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Unified loader. Returns (corpus, cal_examples, test_examples)."""
    if name == 'sciq':
        return load_sciq(n_test, n_cal, seed)
    elif name == 'scifact':
        return load_scifact(n_test, n_cal, seed)
    elif name.startswith('synthetic_'):
        return make_synthetic_dataset(name, n_test + n_cal + 200, seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")
