#!/usr/bin/env python3
import json, glob, sys
base = sys.argv[1] if len(sys.argv) > 1 else "runs"
rows = {}
for f in glob.glob(f"{base}/*/*/*.jsonl"):
    p = f.split("/"); ds, vic, atk = p[-3], p[-2], p[-1][:-6]
    recs = [json.loads(l) for l in open(f) if l.strip()]
    if not recs: continue
    asr = sum(1 for r in recs if r.get("success")) / len(recs)
    rows[(ds, vic, atk)] = (round(asr, 3), len(recs))
for k in sorted(rows):
    a, n = rows[k]
    print(f"{k[0]:11} {k[1]:12} {k[2]:11} ASR={a:.3f} n={n}")
print(f"# {len(rows)} cells done")
