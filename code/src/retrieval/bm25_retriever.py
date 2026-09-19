"""
BM25 retriever for RAG.
"""
import logging
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def tokenize_simple(text: str) -> List[str]:
    """Simple tokenizer: lowercase + split on non-alphanumeric."""
    import re
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


class BM25Retriever:
    """BM25 retriever over a corpus of documents."""

    def __init__(self, corpus: List[Dict[str, str]], doc_text_key: str = "text"):
        """
        corpus: list of dicts, each with 'id' and doc_text_key.
        """
        self.corpus = corpus
        self.doc_text_key = doc_text_key
        self.doc_ids = [d.get('id', str(i)) for i, d in enumerate(corpus)]
        self.doc_texts = [d[doc_text_key] for d in corpus]
        logger.info(f"Tokenizing {len(self.doc_texts)} docs...")
        self.tokenized = [tokenize_simple(t) for t in self.doc_texts]
        self.bm25 = BM25Okapi(self.tokenized)
        logger.info("BM25 index built.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k docs for query."""
        q_tokens = tokenize_simple(query)
        scores = self.bm25.get_scores(q_tokens)
        # Get top-k indices
        top_k = min(top_k, len(scores))
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_idx:
            results.append({
                'id': self.doc_ids[idx],
                'text': self.doc_texts[idx],
                'score': float(scores[idx]),
                'normalized_score': 0.0,  # Will be set externally
            })
        # Normalize scores to [0, 1]
        if results:
            raw_scores = [r['score'] for r in results]
            max_s = max(raw_scores) if max(raw_scores) > 0 else 1.0
            min_s = min(raw_scores)
            denom = max(max_s - min_s, 1e-6)
            for r in results:
                r['normalized_score'] = (r['score'] - min_s) / denom
        return results


class ContrieverRetriever:
    """Contriever retriever using sentence-transformers."""

    def __init__(self, corpus: List[Dict[str, str]], doc_text_key: str = "text",
                 model_name: str = "facebook/contriever", device: str = "cuda"):
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer, AutoModel
            self.corpus = corpus
            self.doc_text_key = doc_text_key
            self.doc_ids = [d.get('id', str(i)) for i, d in enumerate(corpus)]
            self.doc_texts = [d[doc_text_key] for d in corpus]
            logger.info(f"Loading Contriever model {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()
            self.device = device
            # Encode corpus
            self._encode_corpus()
        except Exception as e:
            logger.error(f"Failed to load Contriever: {e}")
            raise

    def _encode_corpus(self):
        import torch
        logger.info(f"Encoding {len(self.doc_texts)} corpus docs...")
        embeddings = []
        batch_size = 32
        with torch.no_grad():
            for start in range(0, len(self.doc_texts), batch_size):
                batch = self.doc_texts[start:start + batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True,
                                        max_length=512, return_tensors='pt').to(self.device)
                outputs = self.model(**inputs)
                # Mean pooling
                attention_mask = inputs['attention_mask']
                last_hidden = outputs.last_hidden_state
                masked = last_hidden * attention_mask.unsqueeze(-1)
                summed = masked.sum(1)
                counts = attention_mask.sum(1, keepdim=True).clamp(min=1e-9)
                emb = summed / counts
                embeddings.append(emb.cpu().numpy())
        self.corpus_embeddings = np.vstack(embeddings)
        # Normalize
        norms = np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True)
        self.corpus_embeddings = self.corpus_embeddings / np.maximum(norms, 1e-9)
        logger.info(f"Corpus encoded. Shape: {self.corpus_embeddings.shape}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        import torch
        with torch.no_grad():
            inputs = self.tokenizer([query], padding=True, truncation=True,
                                    max_length=512, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            attention_mask = inputs['attention_mask']
            last_hidden = outputs.last_hidden_state
            masked = last_hidden * attention_mask.unsqueeze(-1)
            summed = masked.sum(1)
            counts = attention_mask.sum(1, keepdim=True).clamp(min=1e-9)
            query_emb = (summed / counts).cpu().numpy()
        # Normalize
        query_emb = query_emb / max(np.linalg.norm(query_emb), 1e-9)
        # Cosine similarity
        scores = (self.corpus_embeddings @ query_emb.T).flatten()
        top_k = min(top_k, len(scores))
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_idx:
            results.append({
                'id': self.doc_ids[idx],
                'text': self.doc_texts[idx],
                'score': float(scores[idx]),
                'normalized_score': float(max(scores[idx], 0.0)),  # cosine sim in [-1,1]
            })
        return results


class EnsembleRetriever:
    """Ensemble BM25 + Contriever with reciprocal rank fusion."""

    def __init__(self, retrievers: List, weights: Optional[List[float]] = None):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        # Get top 2*top_k from each, fuse with RRF
        all_results = [r.retrieve(query, top_k=2 * top_k) for r in self.retrievers]
        # Reciprocal Rank Fusion
        rrf_scores = {}
        text_to_doc = {}
        for weight, results in zip(self.weights, all_results):
            for rank, doc in enumerate(results):
                text = doc['text'][:200]  # use first 200 chars as key
                if text not in rrf_scores:
                    rrf_scores[text] = 0.0
                    text_to_doc[text] = doc
                rrf_scores[text] += weight / (60 + rank + 1)
        # Sort by fused score
        sorted_texts = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
        results = []
        max_score = max(s for _, s in sorted_texts) if sorted_texts else 1.0
        for text, score in sorted_texts:
            doc = text_to_doc[text].copy()
            doc['score'] = float(score)
            doc['normalized_score'] = float(score / max(max_score, 1e-9))
            results.append(doc)
        return results


def build_index_from_corpus(corpus: List[Dict[str, str]], retriever_type: str = "bm25",
                             **kwargs) -> object:
    """Factory to build retriever."""
    if retriever_type == "bm25":
        return BM25Retriever(corpus, **kwargs)
    elif retriever_type == "contriever":
        return ContrieverRetriever(corpus, **kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
