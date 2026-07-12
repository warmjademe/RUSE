"""Shared utilities: config loading, seeded RNG, JSONL I/O, hashing."""
from __future__ import annotations
import os, json, hashlib, random, pathlib, threading
from typing import Any, Iterable

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, "r") as f:
        return yaml.safe_load(f)


def load_secrets(path: str | None = None) -> None:
    """Populate os.environ from configs/secrets.env (KEY=VALUE lines)."""
    p = pathlib.Path(path) if path else (CONFIG_DIR / "secrets.env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def sha1(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def rng(seed: int) -> random.Random:
    return random.Random(seed)


# ---- append-only JSONL with a thread lock (safe for the pilot) ----------
_locks: dict[str, threading.Lock] = {}


def _lock(path: str) -> threading.Lock:
    return _locks.setdefault(path, threading.Lock())


def append_jsonl(path: str | pathlib.Path, record: dict) -> None:
    path = str(path)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _lock(path):
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | pathlib.Path) -> list[dict]:
    path = pathlib.Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # tolerate a torn line from a concurrent appender (JSONL is written one
            # full line at a time, so at most the last line can be mid-write)
            continue
    return out


def iter_jsonl(path: str | pathlib.Path) -> Iterable[dict]:
    path = pathlib.Path(path)
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
