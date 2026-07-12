"""Victim-model registry: one uniform interface over commercial (proxy) and
open-source (Ollama) victims. `Victim.respond(messages)` returns the reply."""
from __future__ import annotations
from .common import load_yaml
from .llm_client import get_client


class Victim:
    def __init__(self, key: str, spec: dict):
        self.key = key
        self.backend = spec["backend"]
        self.model = spec["model"]
        self.paper_name = spec.get("paper_name", key)
        self.uniform = spec.get("uniform", False)
        self.reasoning = spec.get("reasoning", False)
        self.note = spec.get("note", "")

    def respond(self, messages: list[dict], *, temperature: float = 0.0,
                max_tokens: int = 1024, seed: int | None = None) -> str:
        return get_client().chat(self.backend, self.model, messages,
                                 temperature=temperature, max_tokens=max_tokens,
                                 reasoning=self.reasoning, seed=seed)


def load_victims(cfg: dict | None = None) -> dict[str, Victim]:
    cfg = cfg or load_yaml("models.yaml")
    return {k: Victim(k, v) for k, v in cfg["victims"].items()}


def get_victim(key: str) -> Victim:
    return load_victims()[key]
