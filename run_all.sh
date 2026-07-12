#!/bin/bash
# Master driver: run the full uniform re-eval (N=20/dataset, R=3, T=4),
# aggregate, Bark-notify. Resumable (run.py skips done cells). All judges are
# on the proxy (off-GPU); the 4090 only serves open-source victims + attacker.
# Commercial victims run FIRST (fast, network-bound; map to the paper tables),
# open-source victims SECOND (slower, local GPU) as added breadth.
set -u
BARK="https://api.day.app/<BARK_KEY>"
cd ~/TongBu/mtp_rebuttal/source_codes || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mtp
export HF_ENDPOINT=https://hf-mirror.com

COMM="gpt-3.5,gpt-4o,claude,deepseek"
OPEN="llama3,gemma3,mistral"

run() { echo "[`date`] RUN $*"; python -m mtp.run "$@" 2>&1 | tail -3; }

echo "[`date`] ===== PHASE 1: commercial victims (AdvBench+HarmBench), R=2 (cost) ====="
run --datasets advbench,harmbench --victims "$COMM" --attacks direct,paps,pair,mtp --reps 2
python -m mtp.report --datasets advbench,harmbench 2>&1 | tail -6
curl -s -G "$BARK" --data-urlencode "title=MTP阶段1完成" --data-urlencode "body=commercial victims done (AdvBench+HarmBench)" >/dev/null 2>&1

echo "[`date`] ===== PHASE 2: open-source victims (AdvBench+HarmBench) ====="
run --datasets advbench,harmbench --victims "$OPEN" --attacks direct,paps,pair,mtp --reps 3

# SORRY-Bench skipped: HF dataset is gated (no public CSV yet). Add
# data/sources/sorrybench.csv and re-enable to refresh Table 3.

echo "[`date`] ===== aggregate ====="
python -m mtp.report --datasets advbench,harmbench 2>&1 | tail -12

N=$(find runs -name '*.jsonl' | xargs cat 2>/dev/null | wc -l)
MSG="MTP rebuttal ALL experiments done. $N rollout records. results/ has asr_*.csv sig_*.csv tables_*.tex"
echo "[`date`] $MSG"
curl -s -G "$BARK" --data-urlencode "title=MTP实验全部跑完" --data-urlencode "body=$MSG" >/dev/null 2>&1
echo "[`date`] ALL DONE"
