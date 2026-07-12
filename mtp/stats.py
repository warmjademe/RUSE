"""Statistical analysis for the revision (reviewer R1.1 / R3.2 / R3.3).

Provides, for every (attack x victim x dataset) cell:
  - ASR point estimate with a 95% Wilson confidence interval (E2/E4)
  - MTP vs each baseline: McNemar exact paired test on identical queries (E3)
  - BH-FDR correction across the family of comparisons (E3)
  - effect sizes: risk difference and odds ratio (E3)

Inputs are the per-item success flags collected in runs/*.jsonl. Across R
replicates a query counts as success if its majority of replicates succeed
(and we also report mean ASR +/- std across replicates).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

import numpy as np
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


@dataclass
class ASRCell:
    attack: str
    victim: str
    dataset: str
    n: int
    successes: int
    asr: float
    ci_low: float
    ci_high: float
    asr_mean_over_reps: float | None = None
    asr_std_over_reps: float | None = None


def wilson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    lo, hi = proportion_confint(successes, n, alpha=alpha, method="wilson")
    return successes / n, float(lo), float(hi)


def asr_cell(attack, victim, dataset, per_query_success: dict[str, list[bool]],
             alpha: float = 0.05, collapse: str = "any") -> ASRCell:
    """per_query_success: query_id -> list of R replicate success bools.

    collapse: how R attempts on one query fold into a per-query jailbreak flag.
      "any"      -> best-of-N: the query is jailbroken if ANY attempt succeeds
                    (standard jailbreak ASR; the attacker gets N tries).
      "majority" -> jailbroken only if >half the attempts succeed (much stricter;
                    at R=2 this needs BOTH, which badly understates attack success).
    Per-replicate ASRs (mean/std) are kept regardless for the variance report.
    """
    qids = sorted(per_query_success)
    if collapse == "majority":
        collapsed = [int(sum(v) > len(v) / 2) for v in (per_query_success[q] for q in qids)]
    else:  # "any" == best-of-N
        collapsed = [int(any(v)) for v in (per_query_success[q] for q in qids)]
    n = len(collapsed)
    s = int(sum(collapsed))
    asr, lo, hi = wilson(s, n, alpha)
    R = max((len(v) for v in per_query_success.values()), default=1)
    rep_asrs = [np.mean([per_query_success[q][r] for q in qids if len(per_query_success[q]) > r])
                for r in range(R)]
    return ASRCell(attack, victim, dataset, n, s, asr, lo, hi,
                   float(np.mean(rep_asrs)), float(np.std(rep_asrs)))


def mcnemar_pair(mtp_success: dict[str, bool], base_success: dict[str, bool],
                 exact: bool = True) -> dict:
    """Paired MTP-vs-baseline test on the shared query set."""
    qids = sorted(set(mtp_success) & set(base_success))
    b = sum(1 for q in qids if mtp_success[q] and not base_success[q])   # MTP win
    c = sum(1 for q in qids if not mtp_success[q] and base_success[q])   # baseline win
    a = sum(1 for q in qids if mtp_success[q] and base_success[q])
    d = sum(1 for q in qids if not mtp_success[q] and not base_success[q])
    table = [[a, b], [c, d]]
    res = mcnemar(table, exact=exact)
    p_mtp = sum(1 for q in qids if mtp_success[q]) / len(qids) if qids else 0.0
    p_base = sum(1 for q in qids if base_success[q]) / len(qids) if qids else 0.0
    rd = p_mtp - p_base
    # odds ratio from discordant pairs (McNemar OR = b/c), guarded
    orat = (b / c) if c else float("inf") if b else 1.0
    return {"n": len(qids), "b_mtp_only": b, "c_base_only": c,
            "statistic": float(res.statistic), "p_value": float(res.pvalue),
            "risk_difference": rd, "odds_ratio": orat}


def fdr(pvals: list[float], alpha: float = 0.05) -> list[float]:
    if not pvals:
        return []
    _, q, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    return [float(x) for x in q]


def cell_to_dict(c: ASRCell) -> dict:
    return asdict(c)
