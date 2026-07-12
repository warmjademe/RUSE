# RUSE — reproduction pipeline (PONE-D-26-17682)

Reproducible re-implementation of **RUSE** (Reusable Unalignment and Strategy
Evolution), the two-stage jailbreak studied in the PLOS ONE major revision. The
manuscript labels the method RUSE; the code namespace is `mtp/` for historical
reasons (the project began as "multi-turn persuasion").

RUSE has two stages:

1. **Unalignment (offline, once).** Aggregate multiple public jailbreak and
   persuasion datasets and QLoRA-fine-tune an open-source Qwen 2.5 7B Instruct on
   them, turning it into a reusable attack-prompt generator that produces fluent
   attack prompts instead of refusing. Trained a single time on one GPU and
   reused for every victim.
2. **Per-target search (online).** For each harmful goal, treat a set of
   persuasion strategies as the arms of a multi-armed bandit, generate a
   population of candidate prompts, probe them in parallel to locate the framing
   the victim does not refuse, score each with an LLM judge, and evolve variants
   of the strongest framing over successive rounds until the victim complies.
   Success is declared on the judge score (best-of-N, score >= tau).

The repository also runs the uniform **baseline testbed** — all attacks under one
identical victim/goal/judge protocol — and reports **Wilson 95% CIs, McNemar
paired tests, and BH-FDR** for the RQ1 comparison.

## Research questions -> where they are produced
| RQ | Question | Entry point | Output |
|---|---|---|---|
| RQ1 | Effectiveness of RUSE vs five prior attacks | `mtp.attack_evolve` (RUSE) + `mtp.run` / `mtp.report` (baselines) | `runs_search/<v>_<ds>_evolve*.jsonl`, `runs/<ds>/<v>/<attack>.jsonl` |
| RQ2 | Contribution of each component (unalignment / bandit / evolution) | `mtp.attack_evolve` with ablation flags | `runs_search/<v>_<ds>_ABL-{sftonly,basebandit,banditonly}_evolve*.jsonl` |
| RQ3 | Robustness to four prompt-level defenses | `rq3_filter.py` (replay winners through each defense) | `runs_rq3/<v>_<ds>_rq3.json` |

## Layout
```
configs/   models.yaml (victim/judge routing), experiment.yaml (reps, tau, stats),
           secrets.env(.template)   [secrets.env is git-ignored, never committed]
mtp/       common, llm_client, victims, data, persuasion, judge
           sft_data, sft_train      -> Stage 1 unalignment (build corpus + QLoRA SFT)
           attack_evolve            -> Stage 2 per-target bandit search + evolution (core)
           attack_baselines, run    -> uniform baseline testbed (gcg-t/pair/paps/stinger/xjailbreak)
           defenses, rq3_filter*    -> RQ3 prompt-level defenses (replay winners; rq3_filter.py at repo root)
           stats, report            -> Wilson CI, McNemar, BH-FDR, LaTeX tables
           attack_mtp, attack_loop  -> earlier single-/multi-turn drivers (superseded)
           grpo_train, ppo_train, transfer_report -> optional RL stage (not used for the reported runs)
agg_baselines.py                    -> per-cell ASR over runs/<ds>/<v>/<attack>.jsonl
rq3_filter.py                       -> RQ3 defense-replay driver
runs/         baseline-testbed rollouts, per (dataset/victim/attack)   (resumable)
runs_search/  attack_evolve rollouts: <v>_<ds>_evolve.jsonl (+ _solved.jsonl),
              ablations <v>_<ds>_ABL-{sftonly,basebandit,banditonly}_evolve.jsonl,
              defense filters <v>_<ds>_DEF-{prompt_detection,prompt_perturbation}_evolve.jsonl
runs_rq3/     RQ3 defense-filter summaries: <v>_<ds>_rq3.json
results/      asr_*.csv, tables_*.tex
data/         sft/ (SFT corpus), external_prompts/ (gcg-t/stinger/xjailbreak seeds)
```

## Victims, judge, and benchmarks
Victims: **Llama 3, Gemma 3 12B, Mistral Nemo 12B** (open-source, served locally
through Ollama) and **GPT-4o mini, DeepSeek v4-flash** (commercial, API). Every
backend speaks the OpenAI `/v1/chat/completions` protocol (`configs/models.yaml`):
open-source victims and the 4-bit SFT attacker on `ollama` at
`127.0.0.1:11434`; commercial victims and the judge on a `proxy` endpoint (key in
`configs/secrets.env`, `OPENAI_PROXY_KEY`).

Judge: a single **GPT-5.4-mini** grader on a 0--10 compliance scale, success
threshold **tau = 5**, validated against two human experts (Cohen's kappa = 0.92).

Benchmarks: **AdvBench** (520 goals), **HarmBench** (400), **SORRY-Bench** (440);
the reported runs use the full benchmarks. Obtain the raw goal corpora from their
original public releases (they are not redistributed here).

## Setup (conda env `mtp`)
```bash
conda activate mtp
cp configs/secrets.env.template configs/secrets.env   # fill OPENAI_PROXY_KEY
ollama pull llama3-mtp gemma3-mtp mistral-mtp qwen-mtp   # local victims + attacker (or base tags + Modelfiles)
```

## Stage 1 — unalignment (build the reusable attacker)
```bash
python -m mtp.sft_data                 # aggregate public datasets -> data/sft/pool.jsonl
python -m mtp.sft_train                # QLoRA SFT of Qwen 2.5 7B Instruct -> runs_sft/attacker
```

## Stage 2 — per-target search (RQ1, RUSE full method)
```bash
export MTP_ATTACKER_ADAPTER=runs_sft/attacker
python -m mtp.attack_evolve --victim llama3 --dataset advbench \
       --gens 8 --n-free 16 --n-pool 16 --log runs_search/llama3_advbench_evolve.jsonl
```
Key flags: `--gens` evolution rounds, `--n-pool` strategy-conditioned seeds,
`--n-free` free-form seeds, `--k-per` children per parent, `--defense`, `--resume`.

### RQ2 ablations (one component removed at a time)
```bash
# SFT-only   : unalignment only (best-of-1, no search, no evolution)
python -m mtp.attack_evolve --victim llama3 --dataset advbench --n-pool 0 --n-free 1 --gens 0 \
       --log runs_search/llama3_advbench_ABL-sftonly_evolve.jsonl
# Base+Bandit: bandit only on the un-tuned generator, single round
MTP_ATTACKER_ADAPTER=BASE python -m mtp.attack_evolve --victim llama3 --dataset advbench --gens 0 \
       --n-free 16 --log runs_search/llama3_advbench_ABL-basebandit_evolve.jsonl
# Bandit-only: unalignment + bandit, single round (no evolution)
python -m mtp.attack_evolve --victim llama3 --dataset advbench --gens 0 --n-free 16 \
       --log runs_search/llama3_advbench_ABL-banditonly_evolve.jsonl
# Full = all three (the Stage 2 command above, gens > 0)
```

## RQ1 baseline testbed (uniform conditions)
Five prior attacks — **GCG-T, PAIR, PAPs, Stinger, xJailbreak** — are replayed
under the same victims/goals/judge as RUSE (`mtp/attack_baselines.py`):
```bash
python -m mtp.run --datasets advbench,harmbench,sorrybench \
       --victims llama3,gemma3,mistral,gpt-4o-mini,deepseek \
       --attacks gcg-t,pair,paps,stinger,xjailbreak --reps 1
python agg_baselines.py runs                                    # per-cell ASR
python -m mtp.report --datasets advbench,harmbench,sorrybench   # Wilson CI, McNemar, BH-FDR, tables
```
`run.py` resumes by (query_id, rep), so interrupted runs continue without
recomputation. GCG-T / Stinger / xJailbreak consume optimiser-produced seeds from
`data/external_prompts/<attack>.jsonl` and replay them under the shared judge.

## RQ3 — robustness to prompt-level defenses
RQ3 does not re-run the search: it **replays each RUSE winner through every
defense and re-judges** (a defense can only turn a success into a failure), so the
defended ASR is the fraction of previously successful attacks that still succeed.
```bash
python rq3_filter.py --victim llama3 --infile runs_search/llama3_advbench_evolve_solved.jsonl \
       --denom <n_goals> --out runs_rq3/llama3_advbench_rq3.json
```
Four defenses (`--defenses`, default all four), matching the paper:
- `prompt_detection` — lightweight lexical jailbreak / instruction-override detector (no model download),
- `prompt_perturbation` — single-copy SmoothLLM-style character perturbation + re-query,
- `system_prompt_guard` — a defensive system message,
- `constitutional` — input + output LLM safety classifiers.
The full multi-copy / perplexity variants live in `mtp/defenses.py`; `rq3_filter.py`
uses the simplified, download-free versions.

## Statistics
`mtp/stats.py` and `mtp/report.py` compute a Wilson 95% confidence interval for
every ASR, McNemar's paired test for RUSE against the strongest baseline per
victim, and Benjamini--Hochberg FDR control across the family of comparisons.
Success is best-of-N and the judge is pinned to GPT-5.4-mini.

## Data availability and redaction (public release)
This is the **desensitized** artifact for a defensive/robustness study. Before
release the following redaction was applied:
- Rollout JSONL (`runs/*.jsonl`, `runs_search/*.jsonl`): the crafted jailbreak
  prompt and the victim's harmful response are replaced with a stub
  `[REDACTED sha256:<hash> len:<n>]`. The public goal id/text, the chosen
  persuasion strategy, the judge score, and the success flag are kept so the
  experimental record stays auditable.
- Aggregate summaries (`runs_rq3/*.json`): judge scores and success/blocked flags
  only — no prompts or responses.
- Excluded entirely: real API keys (`configs/secrets.env`; use
  `configs/secrets.env.template`), the unalignment QLoRA adapter (`runs_sft/`), the
  raw harmful-goal corpora (obtain AdvBench / HarmBench / SORRY-Bench from their
  original public releases), and the LLM response cache.

## Responsible use
This code reproduces a jailbreak attack for defensive and robustness research
only. The harmful behaviors come from public benchmarks (AdvBench, HarmBench,
SORRY-Bench); the crafted attack prompts and harmful completions are withheld per
the redaction above. Secrets live only in `configs/secrets.env` (git-ignored) and
are never committed.
