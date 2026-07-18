"""Regenerate the RQ1 ASR tables (and a machine-readable summary) directly from
runs/*.jsonl, so the manuscript numbers are never hand-transcribed.

For each dataset it emits, per victim row:
  ASR + Wilson 95% CI for the 5 baselines and RUSE, and the BH-FDR-adjusted
  exact-McNemar q for RUSE vs the strongest baseline (by point ASR) in that row.
Cohen's h between RUSE and the strongest baseline is written to the JSON summary
for the prose. Best-of-N (any) collapse over replicates, N = number of goals.

Usage:  python make_tables.py [runs_dir]  ->  writes results/tables_*.tex, results/summary.json
"""
from __future__ import annotations
import json, math, pathlib, sys
from collections import defaultdict

from mtp.common import read_jsonl
from mtp.stats import wilson, mcnemar_pair, fdr

RUNS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
OUT = pathlib.Path("results"); OUT.mkdir(exist_ok=True)

DATASETS = ["advbench", "harmbench", "sorrybench"]
# paper column order and labels; keys are the attack ids in runs/
BASE_COLS = [("gcg-t", "GCG-T"), ("pair", "PAIR"), ("paps", "PAPs"),
             ("stinger", "Stinger"), ("xjailbreak", "xJailbreak")]
RUSE_KEY = "mtp"
VICTIM_ORDER = [("llama3", "Llama 3"), ("gemma3", "Gemma 3"),
                ("mistral", "Mistral Nemo"), ("gpt-4o-mini", "GPT-4o mini"),
                ("deepseek", "DeepSeek")]


def cohen_h(p1: float, p2: float) -> float:
    phi = lambda p: 2 * math.asin(math.sqrt(max(0.0, min(1.0, p))))
    return phi(p1) - phi(p2)


def load_success(ds: str) -> dict:
    """(victim, attack) -> {query_id: success_bool} (best-of-N over reps)."""
    out: dict = defaultdict(lambda: defaultdict(list))
    base = RUNS / ds
    if not base.exists():
        return {}
    for vdir in base.iterdir():
        if not vdir.is_dir() or "+" in vdir.name:   # skip defended victims
            continue
        for f in vdir.glob("*.jsonl"):
            atk = f.stem
            for r in read_jsonl(f):
                out[(vdir.name, atk)][r["query_id"]].append(bool(r.get("success")))
    # collapse best-of-N
    return {k: {q: any(v) for q, v in qd.items()} for k, qd in out.items()}


def fmt_ci(lo: float, hi: float) -> str:
    s = lambda x: f"{x:.2f}".lstrip("0") if x < 1 else "1.00"
    return f"[{s(lo)},{s(hi)}]"


def fmt_q(q: float) -> str:
    return "$<$.001" if q < 0.001 else f"{q:.3f}".lstrip("0")


def build(ds: str, succ: dict, summary: dict) -> str | None:
    rows_tex = []
    for vkey, vlabel in VICTIM_ORDER:
        if (vkey, RUSE_KEY) not in succ:
            continue
        ruse = succ[(vkey, RUSE_KEY)]
        n = len(ruse)
        cells = []
        base_asr = {}
        for akey, _ in BASE_COLS:
            d = succ.get((vkey, akey), {})
            s = sum(d.get(q, False) for q in ruse)          # align on RUSE's goal set
            asr, lo, hi = wilson(s, n)
            base_asr[akey] = asr
            cells.append(rf"\makecell{{{asr:.2f}\\{{\scriptsize{fmt_ci(lo, hi)}}}}}")
        rs = sum(ruse.values()); rasr, rlo, rhi = wilson(rs, n)
        cells.append(rf"\makecell{{\textbf{{{rasr:.2f}}}\\{{\scriptsize{fmt_ci(rlo, rhi)}}}}}")
        # strongest baseline for this row
        strong = max(base_asr, key=base_asr.get)
        mc = mcnemar_pair(ruse, succ.get((vkey, strong), {}))
        summary.setdefault(ds, {})[vkey] = {
            "n": n, "ruse_asr": rasr, "ruse_ci": [rlo, rhi],
            "baseline_asr": base_asr, "strongest_baseline": strong,
            "mcnemar_p": mc["p_value"], "cohen_h_vs_strongest": cohen_h(rasr, base_asr[strong]),
        }
        rows_tex.append((vkey, vlabel, cells, mc["p_value"]))
    if not rows_tex:
        return None
    # BH-FDR across the per-row RUSE-vs-strongest comparisons within this dataset
    qs = fdr([r[3] for r in rows_tex])
    lines = []
    for (vkey, vlabel, cells, _p), q in zip(rows_tex, qs):
        summary[ds][vkey]["mcnemar_q"] = q
        lines.append(f"{vlabel:<12} & " + " & ".join(cells) + f" & {fmt_q(q)} \\\\")
    return "\n".join(lines)


def main():
    summary: dict = {}
    for ds in DATASETS:
        succ = load_success(ds)
        if not succ:
            print(f"[{ds}] no runs")
            continue
        body = build(ds, succ, summary)
        if body is None:
            print(f"[{ds}] no RUSE rows")
            continue
        (OUT / f"tables_{ds}.tex").write_text(body + "\n")
        print(f"[{ds}] wrote results/tables_{ds}.tex")
        print(body)
        print()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote results/summary.json")


if __name__ == "__main__":
    main()
