"""Dataset loaders for AdvBench / HarmBench / SORRY-Bench.

We evaluate a fixed random sample of N=20 harmful queries per dataset (seed in
configs/experiment.yaml) so ASR carries a Wilson CI rather than a bare point
estimate. Full-set N is recorded for the table captions.

Source of truth = local CSVs in data/sources/ (fetched from the public GitHub
releases; the HF versions of these datasets are gated). Falls back to a small
bundled benign sample only if a source file is missing.

  advbench.csv   : llm-attacks harmful_behaviors.csv  (columns: goal,target) -> 520
  harmbench.csv  : HarmBench behaviors text all        (Behavior,FunctionalCategory,...) -> 200 standard
  sorrybench.csv : optional; {question} or {turns} column                    -> 440

Each item: {"id", "query", "dataset", "category"}.
"""
from __future__ import annotations
import os, csv, pathlib
from .common import load_yaml, rng, ROOT

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
SRC = ROOT / "data" / "sources"
BUNDLED = ROOT / "data" / "bundled"


def _read_csv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_pool(dataset: str) -> list[dict]:
    if dataset == "advbench":
        rows = _read_csv(SRC / "advbench.csv")
        return [{"id": f"advbench-{i}", "query": r["goal"].strip(), "dataset": "advbench",
                 "category": None} for i, r in enumerate(rows) if r.get("goal")]
    if dataset == "harmbench":
        rows = _read_csv(SRC / "harmbench.csv")
        rows = [r for r in rows if r.get("FunctionalCategory") == "standard"]  # N=200 textual
        return [{"id": f"harmbench-{i}", "query": r["Behavior"].strip(), "dataset": "harmbench",
                 "category": r.get("SemanticCategory")} for i, r in enumerate(rows) if r.get("Behavior")]
    if dataset == "sorrybench":
        p = SRC / "sorrybench.csv"
        if not p.exists():
            raise FileNotFoundError("data/sources/sorrybench.csv missing (HF gated); provide it to run SORRY-Bench")
        rows = _read_csv(p)
        key = "question" if rows and "question" in rows[0] else ("turns" if rows and "turns" in rows[0] else "prompt")
        return [{"id": f"sorrybench-{i}", "query": str(r[key]).strip(), "dataset": "sorrybench",
                 "category": r.get("category")} for i, r in enumerate(rows) if r.get(key)]
    raise ValueError(dataset)


def _bundled(dataset: str) -> list[dict]:
    p = BUNDLED / f"{dataset}.txt"
    if not p.exists():
        return []
    return [{"id": f"{dataset}-{i}", "query": q.strip(), "dataset": dataset, "category": None}
            for i, q in enumerate(p.read_text().splitlines()) if q.strip()]


def load_sample(dataset: str, exp_cfg: dict | None = None) -> list[dict]:
    exp_cfg = exp_cfg or load_yaml("experiment.yaml")
    spec = exp_cfg["datasets"][dataset]
    n, seed = spec["n_sample"], spec["seed"]
    try:
        pool = _load_pool(dataset)
        assert pool, "empty pool"
    except Exception as e:  # noqa: BLE001
        print(f"[data] source load failed for {dataset} ({e}); using bundled benign fallback")
        pool = _bundled(dataset)
    if not pool:
        raise RuntimeError(f"no items for {dataset}")
    idx = list(range(len(pool)))
    rng(seed).shuffle(idx)
    return [pool[i] for i in idx[:n]]
