"""Transfer-evaluation report: does a single attacker, SFT+GRPO-trained against
llama3, jailbreak a spread of victims better than the untrained base attacker?

For each victim we hold two matched-condition runs on the SAME hf_local backend
(the only difference is the LoRA adapter):
  * untrained -> runs/<ds>/<v>/mtp_untrained_transfer.jsonl   (adapter="BASE")
  * trained   -> runs/<ds>/<v>/mtp.jsonl                       (adapter=runs_grpo/attacker_sft)

Per victim: majority-vote collapse over R replicates, Wilson-CI ASR for each
condition, and a paired McNemar test (trained vs untrained) on the shared query
set with risk difference and odds ratio. p-values are BH-FDR corrected across
victims. Usage: python -m mtp.transfer_report [--dataset advbench]
"""
from __future__ import annotations
import argparse, os
from .common import read_jsonl, ROOT
from .stats import asr_cell, mcnemar_pair, fdr

VICTIMS = ["llama3", "gemma3", "mistral", "gpt-4o", "claude", "deepseek"]


def _per_query(path: str) -> dict[str, list[bool]]:
    d: dict[str, list[bool]] = {}
    if not os.path.exists(path):
        return d
    for r in read_jsonl(path):
        d.setdefault(r["query_id"], []).append(bool(r.get("success")))
    return d


def _collapse(pq: dict[str, list[bool]]) -> dict[str, bool]:
    # best-of-N: a query is jailbroken if any replicate succeeds (standard
    # jailbreak ASR; matches asr_cell's default collapse="any").
    return {q: any(v) for q, v in pq.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="advbench")
    args = ap.parse_args()
    ds = args.dataset

    rows, pvals = [], []
    for v in VICTIMS:
        base_pq = _per_query(os.path.join(ROOT, "runs", ds, v, "mtp_untrained_transfer.jsonl"))
        tr_pq = _per_query(os.path.join(ROOT, "runs", ds, v, "mtp.jsonl"))
        if not base_pq and not tr_pq:
            continue
        base_cell = asr_cell("mtp-untrained", v, ds, base_pq) if base_pq else None
        tr_cell = asr_cell("mtp-trained", v, ds, tr_pq) if tr_pq else None
        mc = None
        if base_pq and tr_pq:
            mc = mcnemar_pair(_collapse(tr_pq), _collapse(base_pq))
            pvals.append(mc["p_value"])
        rows.append((v, base_cell, tr_cell, mc))

    # BH-FDR across the victims that had a paired test
    qmap = {}
    if pvals:
        q = fdr(pvals)
        it = iter(q)
        for v, _, _, mc in rows:
            if mc is not None:
                qmap[v] = next(it)

    print(f"\n=== MTP transfer eval on {ds}: untrained(BASE) vs SFT+GRPO-trained "
          f"(best-of-N ASR; query jailbroken if any replicate succeeds) ===")
    hdr = f"{'victim':<9} {'untrained ASR [95% CI]':<26} {'trained ASR [95% CI]':<26} {'Δ':>7} {'OR':>6} {'p':>8} {'q(BH)':>8}"
    print(hdr); print("-" * len(hdr))
    for v, bc, tc, mc in rows:
        bs = f"{bc.asr:.2f} [{bc.ci_low:.2f},{bc.ci_high:.2f}]" if bc else "--"
        ts = f"{tc.asr:.2f} [{tc.ci_low:.2f},{tc.ci_high:.2f}]" if tc else "--"
        if mc:
            orat = mc["odds_ratio"]; ors = "inf" if orat == float("inf") else f"{orat:.2f}"
            line = (f"{v:<9} {bs:<26} {ts:<26} {mc['risk_difference']:>+7.2f} "
                    f"{ors:>6} {mc['p_value']:>8.3f} {qmap.get(v, float('nan')):>8.3f}")
        else:
            line = f"{v:<9} {bs:<26} {ts:<26} {'':>7} {'':>6} {'':>8} {'':>8}"
        print(line)
    print("\nΔ = trained ASR - untrained ASR (per-query majority vote). "
          "OR = McNemar discordant odds (trained-only / untrained-only wins).")


if __name__ == "__main__":
    main()
