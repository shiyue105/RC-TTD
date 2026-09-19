"""
NLI verifier using DeBERTa-v3-large-MNLI and LLM-judge for verification.
"""
import os
import torch
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CACHE_DIR = _PROJECT_ROOT / "cache" / "nli_cache"


class NLIVerifier:
    """DeBERTa NLI verifier for (answer, evidence) support scoring."""

    # Label mapping for cross-encoder/nli-deberta-v3-large
    LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, device: str = "cuda",
                 gpu_id: int = 0, max_length: int = 512):
        self.model_name = model_name
        self.device = f"cuda:{gpu_id}" if device == "cuda" else "cpu"
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        logger.info(f"Loading NLI model {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        logger.info(f"NLI model loaded. Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    def _cache_key(self, premise: str, hypothesis: str) -> str:
        key_str = f"{premise[:500]}|||{hypothesis[:500]}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{key}.json"

    def verify_batch(self, premises: List[str], hypotheses: List[str],
                     use_cache: bool = True) -> List[Dict]:
        """Verify support: premise=evidence, hypothesis=answer.
        Returns list of {entailment, neutral, contradiction} prob dicts."""
        assert len(premises) == len(hypotheses)
        results = [None] * len(premises)
        to_compute = []
        for i, (p, h) in enumerate(zip(premises, hypotheses)):
            if use_cache:
                key = self._cache_key(p, h)
                cache_file = self._cache_path(key)
                if cache_file.exists():
                    try:
                        with open(cache_file) as f:
                            results[i] = json.load(f)
                        continue
                    except Exception:
                        pass
            to_compute.append((i, p, h))

        if to_compute:
            new_results = self._verify_uncached([t[1] for t in to_compute],
                                                [t[2] for t in to_compute])
            for (idx, p, h), res in zip(to_compute, new_results):
                results[idx] = res
                if use_cache:
                    key = self._cache_key(p, h)
                    cache_file = self._cache_path(key)
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump(res, f)
                    except Exception:
                        pass

        return results

    def _verify_uncached(self, premises: List[str], hypotheses: List[str]) -> List[Dict]:
        results = []
        batch_size = 16
        for start in range(0, len(premises), batch_size):
            batch_p = premises[start:start + batch_size]
            batch_h = hypotheses[start:start + batch_size]
            inputs = self.tokenizer(batch_p, batch_h, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors='pt').to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for prob in probs:
                results.append({
                    'entailment': float(prob[0]),
                    'neutral': float(prob[1]),
                    'contradiction': float(prob[2]),
                })
        return results

    def support_score(self, evidence: str, answer: str) -> float:
        """Single support score = P(entailment)."""
        res = self.verify_batch([evidence], [answer])[0]
        return res['entailment']

    def support_scores_batch(self, evidences: List[str], answer: str) -> List[float]:
        """Support scores for one answer across multiple evidences."""
        hyps = [answer] * len(evidences)
        results = self.verify_batch(evidences, hyps)
        return [r['entailment'] for r in results]


def evidence_conflict(evidences: List[str], verifier: NLIVerifier) -> float:
    """Pairwise NLI conflict score among top-3 evidences."""
    if len(evidences) < 2:
        return 0.0
    # Take top-3
    evidences = evidences[:3]
    n_pairs = 0
    n_contradict = 0
    for i in range(len(evidences)):
        for j in range(i + 1, len(evidences)):
            res = verifier.verify_batch([evidences[i]], [evidences[j]])[0]
            n_pairs += 1
            if res['contradiction'] > 0.5:
                n_contradict += 1
    return n_contradict / max(n_pairs, 1)


class LLMJudgeVerifier:
    """Use LLM as a verifier for (answer, evidence) support."""

    def __init__(self, llm_backend):
        self.llm = llm_backend

    def judge_batch(self, answers: List[str], evidences: List[str]) -> List[float]:
        """Return support scores in [0, 1] from LLM judge."""
        prompts = []
        for a, e in zip(answers, evidences):
            prompt = f"""You are a factuality judge. Given an answer and evidence, judge if the answer is supported by the evidence.

Evidence: {e[:1500]}

Answer: {a}

Reply with a single number from 0 to 10, where:
- 10 = fully supported
- 5 = partially supported
- 0 = not supported or contradicted

Score:"""
            prompts.append(prompt)
        responses = self.llm.generate_batch(prompts, use_cache=True, temperature=0.0, max_new_tokens=10)
        scores = []
        for resp in responses:
            try:
                # Extract first number
                import re
                nums = re.findall(r'\d+', resp)
                if nums:
                    score = int(nums[0]) / 10.0
                    score = max(0.0, min(1.0, score))
                else:
                    score = 0.5
            except Exception:
                score = 0.5
            scores.append(score)
        return scores


def verifier_disagreement(nli_score: float, llm_score: float) -> float:
    """Disagreement between NLI and LLM judge."""
    return abs(nli_score - llm_score)
