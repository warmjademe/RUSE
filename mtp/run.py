"""Orchestrator: run attacks x victims x datasets x R replicates, resumable.

Each attack rollout appends one record to runs/<dataset>/<victim>/<attack>.jsonl:
  {rep, query_id, query, success, final_score, strategy, n_victim_calls, turns:[...]}
Re-running skips (query_id, rep) pairs already present, so it is safe to resume
after an interruption or to extend R.

Examples
--------
# pilot: 1 dataset, 1 open + 1 commercial victim, all attacks, R=1
python -m mtp.run --datasets advbench --victims gpt-3.5,mistral --attacks direct,paps,pair,mtp --reps 1
# full uniform re-eval as configured
python -m mtp.run --all
"""
from __future__ import annotations
import argparse, time, pathlib
from dataclasses import asdict

from .common import load_yaml, append_jsonl, read_jsonl, ROOT
from .victims import load_victims
from .data import load_sample
from .attack_mtp import run_mtp, AttackResult
from . import attack_baselines as B
from .defenses import DefendedVictim, REGISTRY as DEFENSES

RUNS = ROOT / "runs"


def _dispatch(attack: str, query: str, victim, cfg) -> AttackResult:
    if attack == "mtp":
        return run_mtp(query, victim, cfg=cfg)
    if attack in B.REGISTRY:
        return B.REGISTRY[attack](query, victim)
    # gcg-t / stinger / xjailbreak: ingest external prompts and replay uniformly
    prompt = _external_prompt(attack, query)
    if prompt is None:
        raise RuntimeError(f"no ingested prompt for {attack}/{query[:40]}; "
                           f"run baselines_external first (see README)")
    return B.replay(attack, query, victim, prompt)


_EXT_CACHE: dict[str, dict[str, str]] = {}


def _external_prompt(attack: str, query: str) -> str | None:
    if attack not in _EXT_CACHE:
        p = ROOT / "data" / "external_prompts" / f"{attack}.jsonl"
        _EXT_CACHE[attack] = {r["query"]: r["prompt"] for r in read_jsonl(p)}
    return _EXT_CACHE[attack].get(query)


def _done(path: pathlib.Path) -> set[tuple[str, int]]:
    return {(r["query_id"], r["rep"]) for r in read_jsonl(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="advbench")
    ap.add_argument("--victims", default="gpt-3.5,mistral")
    ap.add_argument("--attacks", default="direct,paps,pair,mtp")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--defense", default="none", choices=list(DEFENSES), help="wrap victims in this defense")
    ap.add_argument("--all", action="store_true", help="use configs/experiment.yaml lists")
    args = ap.parse_args()

    mcfg = load_yaml("models.yaml")
    exp = load_yaml("experiment.yaml")
    victims = load_victims(mcfg)

    if args.all:
        datasets = list(exp["datasets"])
        attacks = exp["attacks"]
        vics = list(victims)
        reps = exp["replicates"]
    else:
        datasets = args.datasets.split(",")
        attacks = args.attacks.split(",")
        vics = args.victims.split(",")
        reps = args.reps or exp["replicates"]

    for ds in datasets:
        items = load_sample(ds, exp)
        print(f"[{ds}] {len(items)} queries")
        for vk in vics:
            victim = victims[vk]
            if args.defense != "none":
                victim = DefendedVictim(victim, args.defense)
            pathkey = vk if args.defense == "none" else f"{vk}+{args.defense}"
            for attack in attacks:
                path = RUNS / ds / pathkey / f"{attack}.jsonl"
                done = _done(path)
                for rep in range(reps):
                    for it in items:
                        if (it["id"], rep) in done:
                            continue
                        t0 = time.time()
                        try:
                            res = _dispatch(attack, it["query"], victim, mcfg)
                            rec = asdict(res)
                        except Exception as e:  # noqa: BLE001 — log & continue
                            rec = {"attack": attack, "query": it["query"], "victim": vk,
                                   "success": False, "final_score": 0.0, "error": str(e)[:300],
                                   "turns": [], "n_victim_calls": 0}
                        rec.update({"query_id": it["id"], "rep": rep, "dataset": ds,
                                    "seconds": round(time.time() - t0, 2)})
                        append_jsonl(path, rec)
                        ok = "OK" if rec.get("success") else "no"
                        print(f"  {ds}/{vk}/{attack} rep{rep} {it['id']}: "
                              f"{ok} S={rec.get('final_score')} ({rec['seconds']}s)")


if __name__ == "__main__":
    main()
