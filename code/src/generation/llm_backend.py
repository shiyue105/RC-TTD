"""
LLM backend using HuggingFace transformers for LLaMA-3.1-8B-Instruct.
"""
import os
import torch
import logging
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Default model path (local)
DEFAULT_MODEL_PATH = "/data/public_models/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"

# Cache directory for LLM responses
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CACHE_DIR = _PROJECT_ROOT / "cache" / "llm_responses"


class LLMBackend:
    """HuggingFace transformers backend for LLaMA-3.1-8B-Instruct."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = "cuda",
                 gpu_id: int = 0, dtype=torch.float16, max_new_tokens: int = 256,
                 device_map: str = None):
        self.model_path = model_path
        self.device = f"cuda:{gpu_id}" if device == "cuda" else "cpu"
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map  # If None, use self.device; if "auto", use auto
        self.tokenizer = None
        self.model = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        dm = self.device_map if self.device_map else self.device
        logger.info(f"Loading LLM from {self.model_path} with device_map={dm}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=dm,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info(f"LLM loaded. Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    def _cache_key(self, prompt: str, **kwargs) -> str:
        key_str = prompt + json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{key}.json"

    def generate_batch(self, prompts: List[str], use_cache: bool = True,
                       temperature: float = 0.0, max_new_tokens: Optional[int] = None) -> List[str]:
        """Generate responses for a batch of prompts. Caches results."""
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        results = [None] * len(prompts)
        to_generate = []  # indices to generate
        prompts_to_gen = []

        for i, p in enumerate(prompts):
            if use_cache:
                key = self._cache_key(p, temperature=temperature, max_new_tokens=max_new_tokens)
                cache_file = self._cache_path(key)
                if cache_file.exists():
                    try:
                        with open(cache_file) as f:
                            results[i] = json.load(f)['response']
                        continue
                    except Exception:
                        pass
            to_generate.append(i)
            prompts_to_gen.append(p)

        if prompts_to_gen:
            new_responses = self._generate_uncached(prompts_to_gen, temperature, max_new_tokens)
            for idx, resp in zip(to_generate, new_responses):
                results[idx] = resp
                if use_cache:
                    key = self._cache_key(prompts[idx], temperature=temperature, max_new_tokens=max_new_tokens)
                    cache_file = self._cache_path(key)
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump({'response': resp, 'prompt': prompts[idx][:200]}, f)
                    except Exception:
                        pass

        return results

    def _get_model_device(self):
        """Get the device where the model's first parameter lives (for device_map='auto')."""
        try:
            return next(self.model.parameters()).device
        except (StopIteration, Exception):
            return self.device

    def _generate_uncached(self, prompts: List[str], temperature: float, max_new_tokens: int) -> List[str]:
        responses = []
        target_device = self._get_model_device()
        # Apply chat template
        batch_size = 4 if temperature == 0 else 1  # Smaller batch for sampling
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            # Apply chat template
            messages_batch = [[{"role": "user", "content": p}] for p in batch]
            texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in messages_batch]
            # Tokenize
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                    max_length=1024).to(target_device)
            with torch.no_grad():
                do_sample = temperature > 0
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=max(temperature, 0.1) if do_sample else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            # Decode only new tokens
            for i, out in enumerate(output):
                input_len = inputs['input_ids'][i].shape[0]
                new_tokens = out[input_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                responses.append(text.strip())
        return responses

    def get_first_token_logprobs(self, prompt: str) -> Dict[int, float]:
        """Get log probabilities for the first generated token.
        Returns a dict mapping token_id -> probability.
        """
        target_device = self._get_model_device()
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(target_device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]  # last token logits
        probs = torch.softmax(logits, dim=-1)
        return {tok_id: prob.item() for tok_id, prob in enumerate(probs)}

    def generate_mcqa_with_confidence(self, prompt: str, options: List[str],
                                      use_cache: bool = True) -> tuple:
        """Generate MCQA response with actual token probabilities.
        Returns (choice_idx, probs_list, response_text).
        probs_list is the probability for each option (normalized over option letters).
        """
        # Get cached or generated response
        response = self.generate(prompt, temperature=0.0, max_new_tokens=10, use_cache=use_cache)

        # Get first-token logprobs
        logprobs = self.get_first_token_logprobs(prompt)

        # Find token IDs for option letters (A, B, C, D, ...)
        n_options = len(options)
        option_letters = [chr(65 + i) for i in range(n_options)]
        option_probs = []
        for letter in option_letters:
            # Try "A", " A", "(A" etc.
            candidates = [letter, f" {letter}", f"({letter}", f"{letter})"]
            max_prob = 0.0
            for cand in candidates:
                tok_ids = self.tokenizer.encode(cand, add_special_tokens=False)
                if tok_ids:
                    prob = logprobs.get(tok_ids[0], 0.0)
                    max_prob = max(max_prob, prob)
            option_probs.append(max_prob)

        # Parse choice from response
        choice_idx = -1
        for char in response.strip():
            if char.isalpha():
                idx = ord(char.upper()) - 65
                if 0 <= idx < n_options:
                    choice_idx = idx
                    break

        # Normalize option probabilities
        total = sum(option_probs)
        if total > 0:
            probs = [p / total for p in option_probs]
        else:
            # Fallback: use uniform
            probs = [1.0 / n_options] * n_options

        if choice_idx == -1:
            choice_idx = 0

        return choice_idx, probs, response

    def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_batch([prompt], **kwargs)[0]


def build_mcqa_prompt(question: str, options: List[str], evidence: str = "") -> str:
    """Build an MCQA prompt in URAG style."""
    prompt = ""
    if evidence:
        prompt += f"Evidence: {evidence}\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += f"({chr(65 + i)}) {opt}\n"
    prompt += "\nAnswer with the letter of the correct option. Just give the letter, nothing else.\n"
    prompt += "Answer:"
    return prompt


def parse_mcqa_response(response: str, n_options: int) -> tuple:
    """Parse MCQA response. Returns (choice_index, confidence_dict)."""
    response = response.strip()
    # Look for first letter A-Z
    for char in response:
        if char.isalpha():
            idx = ord(char.upper()) - 65
            if 0 <= idx < n_options:
                # Build uniform confidence (will be refined externally)
                probs = [0.1] * n_options
                probs[idx] = 0.9
                return idx, probs
    # Fallback: no parseable answer
    return -1, [1.0 / n_options] * n_options


def build_open_prompt(question: str, evidence: str = "") -> str:
    """Build an open-ended QA prompt."""
    prompt = ""
    if evidence:
        prompt += f"Based on the following evidence: {evidence}\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Give a brief answer (one sentence).\nAnswer:"
    return prompt
