"""Smoke test: verify pipeline end-to-end with 3 examples."""
import sys
import os
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_CODE_DIR)
sys.path.insert(0, _CODE_DIR)

os.environ['HF_HOME'] = os.path.join(_PROJECT_ROOT, 'cache', 'huggingface')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

print("=== RC-TTD Smoke Test ===", flush=True)

from src.utils.data_loader import load_dataset_unified
from src.retrieval.bm25_retriever import BM25Retriever
from src.generation.llm_backend import LLMBackend, build_mcqa_prompt, parse_mcqa_response
from src.verification.nli_verifier import NLIVerifier

# 1. Load data
print("Loading SciQ (3 examples)...", flush=True)
corpus, cal, test = load_dataset_unified('sciq', n_test=3, n_cal=3, seed=0)
print(f"  corpus={len(corpus)}, cal={len(cal)}, test={len(test)}", flush=True)

# 2. Build retriever
print("Building BM25 retriever...", flush=True)
retriever = BM25Retriever(corpus)
print("  BM25 OK", flush=True)

# 3. Load LLM
print("Loading LLaMA-3.1-8B-Instruct on GPU 0...", flush=True)
llm = LLMBackend(gpu_id=0)
print("  LLM OK", flush=True)

# 4. Test generation
ex = test[0]
retrieved = retriever.retrieve(ex['query'], top_k=3)
evidence_text = ' '.join([r['text'][:200] for r in retrieved[:3]])
prompt = build_mcqa_prompt(ex['query'], ex['options'], evidence_text[:600])
print(f"Query: {ex['query'][:80]}", flush=True)
print(f"Generating...", flush=True)
response = llm.generate(prompt, temperature=0.0, max_new_tokens=10)
print(f"  Response: {response!r}", flush=True)
choice_idx, _ = parse_mcqa_response(response, len(ex['options']))
print(f"  Choice: {choice_idx}, Correct: {ex['correct_idx']}", flush=True)

# 5. Load NLI
print("Loading NLI verifier...", flush=True)
nli = NLIVerifier(gpu_id=0)
print("  NLI OK", flush=True)
sup = nli.support_score(evidence_text[:500], ex['options'][0])
print(f"  NLI support for option 0: {sup:.3f}", flush=True)

print("=== SMOKE TEST PASSED ===", flush=True)
