"""Unified OpenAI-compatible chat client for every backend (proxy / ollama /
deepseek / ark), with reasoning-model handling, retry, and on-disk caching.

All victims, judges, and the attacker talk to models through `Client.chat`.
Responses are cached (keyed by backend+model+messages+decoding params) so that
re-runs and the judge never re-bill an identical call, and `--resume` is free.
"""
from __future__ import annotations
import os, time, pathlib
import concurrent.futures as _cf
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .common import load_yaml, load_secrets, sha1, append_jsonl, read_jsonl, ROOT

CACHE_PATH = ROOT / "data" / "cache" / "llm_cache.jsonl"

# A hung upstream (e.g. a proxy that dribbles keep-alive bytes) can defeat the
# per-request socket timeout and stall a call indefinitely, which wedges the
# whole pipeline because the judge ensemble blocks on all three judges. Run every
# network call on a shared worker pool and enforce a hard wall-clock deadline; on
# timeout we abandon the call (its thread dies on the tighter socket read-timeout)
# and raise, so judge.py scores it 0 and run.py logs an error rather than hanging.
_NET_POOL = _cf.ThreadPoolExecutor(max_workers=32, thread_name_prefix="net")
HARD_DEADLINE_S = 150


class LLMError(RuntimeError):
    pass


class Client:
    def __init__(self, models_cfg: dict | None = None, use_cache: bool = True):
        load_secrets()
        self.cfg = models_cfg or load_yaml("models.yaml")
        self.backends = self.cfg["backends"]
        self.use_cache = use_cache
        self._cache: dict[str, dict] = {}
        if use_cache:
            for rec in read_jsonl(CACHE_PATH):
                self._cache[rec["key"]] = rec["resp"]

    # -- backend resolution -------------------------------------------------
    def _backend(self, name: str) -> dict:
        b = self.backends[name]
        key = os.environ.get(b["key_env"], "")
        return {"base_url": b["base_url"].rstrip("/"), "key": key}

    # -- one chat call ------------------------------------------------------
    def chat(
        self,
        backend: str,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        reasoning: bool = False,
        seed: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Return the assistant text. Empty-string content falls back to
        reasoning_content (reasoning models sometimes leave content empty)."""
        if backend == "hf_local":
            # in-process 4-bit HF generation (the GRPO-trained attacker adapter);
            # not cached (stochastic) and never billed.
            return _hf_local_generate(model, messages, temperature, max_tokens)

        cache_key = sha1(backend, model, messages, temperature, max_tokens, reasoning, seed, stop)
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]["text"]

        fut = _NET_POOL.submit(self._call, backend, model, messages,
                               temperature, max_tokens, reasoning, seed, stop)
        try:
            text, raw = fut.result(timeout=HARD_DEADLINE_S)
        except _cf.TimeoutError:
            raise LLMError(f"hard deadline {HARD_DEADLINE_S}s exceeded for {backend}/{model}")
        if self.use_cache:
            self._cache[cache_key] = {"text": text}
            append_jsonl(CACHE_PATH, {"key": cache_key, "backend": backend, "model": model,
                                       "resp": {"text": text}})
        return text

    @retry(reraise=True, stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=2, min=2, max=8),
           retry=retry_if_exception_type((requests.RequestException, LLMError)))
    def _call(self, backend, model, messages, temperature, max_tokens, reasoning, seed, stop):
        b = self._backend(backend)
        url = f"{b['base_url']}/chat/completions"
        headers = {"Authorization": f"Bearer {b['key'] or 'x'}", "Content-Type": "application/json"}
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if stop:
            payload["stop"] = stop
        new_api = reasoning or model.startswith(("gpt-5", "o1", "o3", "o4"))
        if new_api:
            # o1/o3/o4/gpt-5 family + reasoning models: the Chat Completions API
            # rejects max_tokens (use max_completion_tokens) and a custom temperature
            # (omit it -> defaults to 1). Reasoning tokens count against the budget.
            payload["max_completion_tokens"] = max(max_tokens, 4096)
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
            if seed is not None:
                payload["seed"] = seed
        r = requests.post(url, headers=headers, json=payload, timeout=(10, 45))
        if r.status_code != 200:
            # 429 / 5xx -> retry; 4xx auth/model errors -> raise immediately
            if r.status_code in (429,) or r.status_code >= 500:
                raise LLMError(f"{r.status_code}: {r.text[:200]}")
            raise LLMError(f"non-retryable {r.status_code}: {r.text[:300]}")
        data = r.json()
        if "error" in data:
            raise LLMError(str(data["error"])[:300])
        msg = data["choices"][0].get("message", {}) or {}
        text = (msg.get("content") or "").strip()
        if not text:
            text = (msg.get("reasoning_content") or "").strip()
        return text, data


# ---- in-process HF backend for the GRPO-trained attacker (4-bit + LoRA) ------
_HF_CACHE: dict = {}
BASE_ATTACKER = "Qwen/Qwen2.5-7B-Instruct"


def _hf_local_generate(adapter_dir: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    import torch
    if adapter_dir not in _HF_CACHE:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(BASE_ATTACKER)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        m = AutoModelForCausalLM.from_pretrained(BASE_ATTACKER, quantization_config=bnb, device_map="auto")
        # adapter_dir == "BASE" -> untrained base attacker on the SAME 4-bit backend,
        # so untrained-vs-trained MTP differs only by the LoRA adapter (clean ablation).
        if adapter_dir != "BASE":
            m = PeftModel.from_pretrained(m, adapter_dir)
        m.eval()
        _HF_CACHE[adapter_dir] = (tok, m)
    tok, m = _HF_CACHE[adapter_dir]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(m.device)
    import torch as _t
    with _t.no_grad():
        out = m.generate(**ids, max_new_tokens=max_tokens, do_sample=temperature > 0,
                         temperature=max(temperature, 0.01), top_p=0.95,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# module-level singleton for convenience
_default: Client | None = None


def get_client(use_cache: bool = True) -> Client:
    global _default
    if _default is None:
        _default = Client(use_cache=use_cache)
    return _default
