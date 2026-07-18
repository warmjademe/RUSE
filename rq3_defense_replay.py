"""RQ3: replay RUSE's winning prompts through each prompt-level defense, re-judge,
report defended ASR over the same N goals (Table rq3).

Method (matches the manuscript): for each victim, take the winning attack prompt of
every goal RUSE solved in RQ1 (runs/<ds>/<victim>/mtp.jsonl), re-issue it as a single
turn through the DefendedVictim, re-judge, and count how many still succeed. The
defended ASR is (# still-succeeding) / N over the SAME N goals, so goals RUSE never
solved contribute 0 and the "No defense" row equals the RQ1 ASR.

GPU-free: no attacker is used (stored prompts are replayed). Safe to run alongside a
GPU job for API/OpenRouter victims; the local-ollama victim (llama3) does hit the GPU.

Usage:
  python rq3_defense_replay.py --dataset advbench [--victims ...] [--defenses ...] [--limit N] [--workers 6]
Outputs runs_defense/<victim>__<defense>.jsonl (resumable) + runs_defense/summary_<ds>.json
"""
from __future__ import annotations
import argparse, json
from concurrent.futures import ThreadPoolExecutor

from mtp.common import load_yaml, append_jsonl, read_jsonl, ROOT
from mtp.victims import load_victims
from mtp.judge import get_judge
from mtp.defenses import DefendedVictim, REGISTRY

DEFAULT_VICTIMS = ["llama3", "gemma3", "mistral", "gpt-4o-mini", "deepseek"]
# the four prompt-level defenses reported in Table rq3
DEFAULT_DEFENSES = ["prompt_detection", "prompt_perturbation", "system_prompt_guard", "constitutional"]


def _winning(ds: str, victim: str):
    """Return ({query_id: (query, winning_prompt)} for RQ1-solved goals, N_total)."""
    p = ROOT / "runs" / ds / victim / "mtp.jsonl"
    recs = list(read_jsonl(p))
    wins = {}
    for r in recs:
        if r.get("success") and r.get("turns"):
            wins[r["query_id"]] = (r.get("query", ""), r["turns"][0].get("prompt", ""))
    return wins, len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="advbench")
    ap.add_argument("--victims", default=",".join(DEFAULT_VICTIMS))
    ap.add_argument("--defenses", default=",".join(DEFAULT_DEFENSES))
    ap.add_argument("--limit", type=int, default=0, help="cap winning prompts per cell (debug)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    ds = args.dataset
    tau = load_yaml("models.yaml")["judges"].get("success_threshold", 5)
    V = load_victims()
    J = get_judge()
    OUT = ROOT / "runs_defense"
    summary: dict = {}

    for vk in args.victims.split(","):
        wins, N = _winning(ds, vk)
        if N == 0:
            print(f"[{vk}] no RQ1 {ds} data, skip", flush=True)
            continue
        summary.setdefault(vk, {})["N"] = N
        summary[vk]["none"] = round(len(wins) / N, 3)   # No-defense row = RQ1 ASR
        print(f"[{vk}] RQ1 solved {len(wins)}/{N} (no-defense ASR={len(wins)/N:.2f})", flush=True)
        items = list(wins.items())
        if args.limit:
            items = items[:args.limit]
        for dfn in args.defenses.split(","):
            if dfn not in REGISTRY:
                print(f"  !! unknown defense {dfn}", flush=True); continue
            dv = DefendedVictim(V[vk], dfn)
            outp = OUT / f"{vk}__{dfn}__{ds}.jsonl"
            done = {r["query_id"] for r in read_jsonl(outp)} if outp.exists() else set()

            def one(item, dv=dv, dfn=dfn, outp=outp):
                qid, (q, prompt) = item
                try:
                    resp = dv.respond([{"role": "user", "content": prompt}], temperature=0.0)
                    sc = J.score(q, resp).score
                    rec = {"query_id": qid, "victim": vk, "defense": dfn, "judge_score": sc,
                           "success": bool(sc >= tau)}
                except Exception as e:  # noqa: BLE001
                    rec = {"query_id": qid, "victim": vk, "defense": dfn, "judge_score": 0.0,
                           "success": False, "error": str(e)[:200]}
                append_jsonl(outp, rec)
                return rec["success"]

            todo = [it for it in items if it[0] not in done]
            if args.workers > 1 and todo:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    list(ex.map(one, todo))
            else:
                for it in todo:
                    one(it)
            succ = sum(1 for r in read_jsonl(outp) if r.get("success"))
            summary[vk][dfn] = round(succ / N, 3)   # defended ASR over all N goals
            print(f"  [{vk}] {dfn}: {succ}/{N} = {succ/N:.2f}", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / f"summary_{ds}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
