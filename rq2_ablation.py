"""RQ2: ablation variants on AdvBench, N=100, same judge/budget/victims as RQ1
(Table rq2). The full-RUSE column reuses RQ1 runs/<ds>/<victim>/mtp.jsonl.

Variants (each keeps a subset of RUSE's three components; rc derived from the RQ1
budget _RC_EVAL so only the ablated component differs):
  sftonly    : fine-tuned generator, best-of-1  (n_pool=0, n_free=1, gens=0)
  basebandit : UN-tuned base generator + single bandit round (adapter=BASE, gens=0)
  banditonly : fine-tuned generator + single bandit round, no evolution (gens=0)
  [full = RUSE = RQ1, not re-run here]

Uses the HF attacker (GPU) -> run after RQ1 frees the GPU. gemma3/mistral victims
are OpenRouter (API); the others as configured.

Usage:
  MTP_ATTACKER_ADAPTER=runs_sft/attacker \
  python rq2_ablation.py --dataset advbench [--victims ...] [--variants ...] [--limit N]
Outputs runs_ablation/<victim>/<variant>__<ds>.jsonl (resumable).
"""
from __future__ import annotations
import argparse, os, time

from mtp.common import load_yaml, append_jsonl, read_jsonl, ROOT
from mtp.victims import load_victims
from mtp.data import load_sample
from mtp.judge import get_judge
from mtp.attack_evolve import evolve_task, _RC_EVAL

DEFAULT_VICTIMS = ["llama3", "gemma3", "mistral", "gpt-4o-mini", "deepseek"]


def _variants():
    base = dict(_RC_EVAL)                       # RQ1 budget: n_pool16 n_free12 n_free_evo4 elite5 k_per3 gens4
    trained = os.environ.get("MTP_ATTACKER_ADAPTER", "runs_sft/attacker")
    return {
        # SFT-only: generator emits one prompt per goal, no search, no evolution
        "sftonly":    (trained, {**base, "n_pool": 0, "n_free": 1, "n_free_evo": 0, "gens": 0}),
        # Base+Bandit: un-tuned base generator, single bandit round (gen0), no evolution
        "basebandit": ("BASE",  {**base, "gens": 0}),
        # Bandit-only: fine-tuned generator, single bandit round (gen0), no evolution
        "banditonly": (trained, {**base, "gens": 0}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="advbench")
    ap.add_argument("--victims", default=",".join(DEFAULT_VICTIMS))
    ap.add_argument("--variants", default="sftonly,basebandit,banditonly")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ds = args.dataset
    exp = load_yaml("experiment.yaml")
    tau = load_yaml("models.yaml")["judges"].get("success_threshold", 5)
    V = load_victims()
    J = get_judge()
    items = load_sample(ds, exp)
    if args.limit:
        items = items[:args.limit]
    VARS = _variants()
    OUT = ROOT / "runs_ablation"
    assert os.environ.get("MTP_ATTACKER_ADAPTER"), "set MTP_ATTACKER_ADAPTER=runs_sft/attacker"

    for vk in args.victims.split(","):
        victim = V[vk]
        for name in args.variants.split(","):
            adapter, rc = VARS[name]
            outp = OUT / vk / f"{name}__{ds}.jsonl"
            outp.parent.mkdir(parents=True, exist_ok=True)
            done = {r["query_id"] for r in read_jsonl(outp)} if outp.exists() else set()
            for it in items:
                if it["id"] in done:
                    continue
                t0 = time.time()
                try:
                    best, calls = evolve_task(it, victim, adapter, J, tau, rc, log_fn=lambda r: None)
                    rec = {"query_id": it["id"], "query": it["query"], "variant": name, "victim": vk,
                           "success": bool(best["score"] >= tau), "final_score": best["score"],
                           "strategy": best.get("tag", ""), "n_victim_calls": calls,
                           "seconds": round(time.time() - t0, 1)}
                except Exception as e:  # noqa: BLE001
                    rec = {"query_id": it["id"], "query": it["query"], "variant": name, "victim": vk,
                           "success": False, "final_score": 0.0, "error": str(e)[:200],
                           "seconds": round(time.time() - t0, 1)}
                append_jsonl(outp, rec)
                print(f"  {vk}/{name} {it['id']}: {'OK' if rec['success'] else 'no'} "
                      f"S={rec.get('final_score')} ({rec['seconds']}s)", flush=True)
    print("ablation done", flush=True)


if __name__ == "__main__":
    main()
