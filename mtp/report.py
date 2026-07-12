"""Aggregate runs/*.jsonl into ASR tables (with Wilson CI), the MTP-vs-baseline
McNemar/BH-FDR significance table, and LaTeX (booktabs) output for the paper.

    python -m mtp.report --datasets advbench,harmbench
Writes results/asr_<dataset>.csv, results/sig_<dataset>.csv, and
results/tables_<dataset>.tex.
"""
from __future__ import annotations
import argparse, collections, pathlib

import pandas as pd

from .common import load_yaml, read_jsonl, ROOT
from .stats import asr_cell, mcnemar_pair, fdr, cell_to_dict

RUNS = ROOT / "runs"
RESULTS = ROOT / "results"


def _collect(ds: str, victims: list[str], attacks: list[str]):
    """-> success[(attack,victim)][query_id] = list[bool] over reps."""
    succ = collections.defaultdict(lambda: collections.defaultdict(list))
    for vk in victims:
        for attack in attacks:
            for r in read_jsonl(RUNS / ds / vk / f"{attack}.jsonl"):
                succ[(attack, vk)][r["query_id"]].append(bool(r.get("success")))
    return succ


def build(ds: str, victims: list[str], attacks: list[str], alpha=0.05):
    succ = _collect(ds, victims, attacks)
    # ASR cells
    cells = []
    for (attack, vk), perq in succ.items():
        cells.append(cell_to_dict(asr_cell(attack, vk, ds, perq, alpha)))
    asr_df = pd.DataFrame(cells)

    # significance: MTP vs each baseline per victim (majority-collapsed binary)
    def collapse(perq):
        return {q: (sum(v) > len(v) / 2) for q, v in perq.items()}
    rows, praw = [], []
    for vk in victims:
        if ("mtp", vk) not in succ:
            continue
        mtp = collapse(succ[("mtp", vk)])
        for attack in attacks:
            if attack == "mtp" or (attack, vk) not in succ:
                continue
            res = mcnemar_pair(mtp, collapse(succ[(attack, vk)]))
            res.update({"victim": vk, "baseline": attack})
            rows.append(res)
            praw.append(res["p_value"])
    sig_df = pd.DataFrame(rows)
    if not sig_df.empty:
        sig_df["q_value_bh"] = fdr(praw, alpha)
        sig_df["sig_0.05"] = sig_df["q_value_bh"] < 0.05
    return asr_df, sig_df


def to_latex(asr_df: pd.DataFrame, victims: list[str], attacks: list[str], ds: str) -> str:
    """ASR table: rows=victims, cols=attacks, cell = 'ASR [lo,hi]'."""
    piv = {}
    for _, r in asr_df.iterrows():
        piv[(r["victim"], r["attack"])] = f"{r['asr']:.2f} [{r['ci_low']:.2f},{r['ci_high']:.2f}]"
    header = " & ".join(["Model"] + attacks) + r" \\"
    lines = [r"\begin{tabular}{l" + "c" * len(attacks) + "}", r"\toprule", header, r"\midrule"]
    for vk in victims:
        cells = [piv.get((vk, a), "--") for a in attacks]
        lines.append(vk + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    cap = (f"% ASR on {ds} (N per cell in results/asr_{ds}.csv), 95%% Wilson CI in brackets; "
           f"all cells re-run under identical victim/query/judge conditions.")
    return cap + "\n" + "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="advbench")
    ap.add_argument("--victims", default=None)
    ap.add_argument("--attacks", default=None)
    args = ap.parse_args()
    exp = load_yaml("experiment.yaml")
    mcfg = load_yaml("models.yaml")
    attacks = args.attacks.split(",") if args.attacks else exp["attacks"]
    victims = args.victims.split(",") if args.victims else list(mcfg["victims"])
    RESULTS.mkdir(exist_ok=True)
    for ds in args.datasets.split(","):
        # keep only victims that actually have runs
        present = [v for v in victims if (RUNS / ds / v).exists()]
        asr_df, sig_df = build(ds, present, attacks)
        asr_df.to_csv(RESULTS / f"asr_{ds}.csv", index=False)
        sig_df.to_csv(RESULTS / f"sig_{ds}.csv", index=False)
        (RESULTS / f"tables_{ds}.tex").write_text(to_latex(asr_df, present, attacks, ds))
        print(f"[{ds}] wrote asr_{ds}.csv ({len(asr_df)} cells), sig_{ds}.csv ({len(sig_df)} tests), tables_{ds}.tex")


if __name__ == "__main__":
    main()
