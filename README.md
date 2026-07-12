# RUSE — reproduction pipeline (PONE-D-26-17682)

Reproducible re-implementation of **RUSE** (Reusable Unalignment and Strategy
Evolution), the two-stage jailbreak studied in the PLOS ONE major revision. The
manuscript labels the method RUSE; the code namespace is `mtp/` for historical
reasons (the project began as "multi-turn persuasion").

RUSE has two stages:

1. **Unalignment (offline, once).** SFT a QLoRA adapter on an open-source
   Qwen 2.5 7B Instruct using multi-source jailbreak/persuasion demonstrations,
   turning it into a reusable attack-prompt generator. Trained a single time and
   reused for every victim.
2. **Per-target search (online).** For each harmful goal, generate a diverse
   population of candidate prompts (bandit-guided persuasion-strategy selection +
   free-form), score each with an LLM judge, and evolve the strongest framings over
   several rounds. Success is declared strictly on the judge score (>= tau).

The repository also runs the uniform **baseline testbed** (all attacks under one
identical victim/query/judge protocol, with Wilson CI + McNemar + BH-FDR), used for
the RQ1 comparison.

## Research questions → where they are produced
| RQ | Question | Entry point | Output |
|---|---|---|---|
| RQ1 | Effectiveness of RUSE vs prior attacks | `mtp.attack_evolve` (full) + `mtp.run` / `mtp.report` (baselines) | `runs_search/*_evolve_solved.jsonl`, `results/asr_*.csv` |
| RQ2 | Contribution of each component (unalignment / bandit / evolution) | `mtp.attack_evolve` with ablation flags | `runs_search/*_ABL-*_evolve.jsonl` |
| RQ3 | Robustness to prompt-level defenses | `rq3_filter.py` (replay winners through each defense) | `runs_rq3/*_advbench_rq3.json` |

## Layout
```
configs/   models.yaml (victim/judge routing), experiment.yaml (N, reps, tau, stats),
           secrets.env(.template)   [secrets.env is git-ignored, never committed]
mtp/       common, llm_client, victims, data, persuasion, judge,
           sft_data, sft_train      -> Stage 1 unalignment (build corpus + QLoRA SFT)
           attack_evolve            -> Stage 2 per-target bandit search + evolution (core)
           attack_mtp, attack_loop  -> earlier single-/multi-turn attack drivers
           attack_baselines, run    -> uniform baseline testbed (direct/paps/pair/...)
           defenses                 -> RQ3 prompt-level defenses (wrap a victim)
           grpo_train, ppo_train    -> optional RL stage (not used for the reported runs)
           stats, report, transfer_report
rq3_filter.py                       -> RQ3: replay Full-method winners through each defense
runs/         uniform-testbed rollouts, per (dataset/victim/attack)   (resumable, git-ignored)
runs_search/  attack_evolve rollouts: <victim>_<dataset>_evolve.jsonl (+ _solved.jsonl),
              ablations <victim>_<dataset>_ABL-{sftonly,basebandit,banditonly}_evolve.jsonl
runs_rq3/     RQ3 defense-filter summaries: <victim>_<dataset>_rq3.json
runs_sft/     the unalignment QLoRA adapter (runs_sft/attacker)        (git-ignored, large)
results/      asr_*.csv, sig_*.csv, tables_*.tex
data/         sft/ (SFT corpus), external_prompts/ (gcg-t/stinger/xjailbreak), cache/
```

## Backends and models
Every backend speaks the OpenAI `/v1/chat/completions` protocol (`configs/models.yaml`):
- **proxy** — commercial victims (`gpt-4o-mini`, `deepseek`, plus cited-only cells)
  and the judge. Key in `configs/secrets.env` (`OPENAI_PROXY_KEY`).
- **ollama** `127.0.0.1:11434` — open-source victims (`llama3-mtp`, `gemma3-mtp`,
  `mistral-mtp`) and the 4-bit SFT attacker.

Victims reported in the tables: **Llama 3, Gemma 3 12B, Mistral Nemo 12B**
(open-source, Ollama) and **GPT-4o mini, DeepSeek v4-flash** (commercial, API).
Judge: a single **GPT-5.4-mini** grader on an integer 0-10 compliance scale;
success threshold **tau = 5**. Model substitutions vs the manuscript are recorded
in `models.yaml` `note:` fields; non-uniform cited cells stay labelled.

## Setup (NAS, conda env `mtp`)
```bash
conda activate mtp
cp configs/secrets.env.template configs/secrets.env   # fill OPENAI_PROXY_KEY
# open-source victims + attacker served locally:
ollama pull llama3-mtp gemma3-mtp mistral-mtp qwen-mtp   # (or base tags + Modelfiles)
```

## Stage 1 — unalignment (build the reusable attacker)
```bash
# build the SFT corpus: harvest successful turns + rejection sampling + public datasets
python -m mtp.sft_data                        # -> data/sft/pool.jsonl
# QLoRA SFT of Qwen 2.5 7B Instruct on the corpus
python -m mtp.sft_train                        # -> runs_sft/attacker
```

## Stage 2 — per-target search (RQ1 full method)
```bash
export MTP_ATTACKER_ADAPTER=runs_sft/attacker
python -m mtp.attack_evolve --victim llama3 --dataset advbench --target 20 \
       --gens 8 --n-free 16 --n-pool 16 --log runs_search/llama3_advbench_evolve.jsonl
```
Key flags: `--target` goals, `--gens` evolution rounds, `--n-pool` strategy-conditioned
seeds, `--n-free` free-form seeds, `--k-per` children per parent, `--defense` (wrap the
victim), `--resume`.

### RQ2 ablations (same script, one component removed at a time)
```bash
# SFT-only  = unalignment only (best-of-1, no search, no evolution)
python -m mtp.attack_evolve --victim llama3 --dataset advbench --target 20 \
       --n-pool 0 --n-free 1 --gens 0 --log runs_search/llama3_advbench_ABL-sftonly_evolve.jsonl
# Base+Bandit = bandit only, native (un-tuned) Qwen generator, single round
MTP_ATTACKER_ADAPTER=BASE python -m mtp.attack_evolve --victim llama3 --dataset advbench \
       --target 20 --gens 0 --n-free 16 --log runs_search/llama3_advbench_ABL-basebandit_evolve.jsonl
# Bandit-only = unalignment + bandit, single round (no evolution)
MTP_ATTACKER_ADAPTER=runs_sft/attacker python -m mtp.attack_evolve --victim llama3 \
       --dataset advbench --target 20 --gens 0 --n-free 16 \
       --log runs_search/llama3_advbench_ABL-banditonly_evolve.jsonl
# Full = all three (the Stage 2 command above, gens > 0)
```

## RQ3 — robustness to prompt-level defenses
RQ3 does not re-run the search. It **replays each Full-method winner through every
defense and re-judges** (a defense can only turn a success into a failure), so the
defended ASR is the fraction of previously-successful attacks that still succeed.

```bash
python rq3_filter.py --victim llama3 \
       --infile runs_search/llama3_advbench_evolve_solved.jsonl \
       --denom 20 --out runs_rq3/llama3_advbench_rq3.json
```
Defenses (`--defenses`, default all four): `prompt_detection` (lightweight lexical
jailbreak / instruction-override detector — no model download), `prompt_perturbation`
(single-copy SmoothLLM-style character perturbation + re-query), `system_prompt_guard`
(defensive system message), `constitutional` (input + output LLM safety classifiers).
The full multi-copy / perplexity variants live in `mtp/defenses.py`; `rq3_filter.py`
uses the simplified download-free versions. Host drivers `rq3_nas.sh`
(llama3/gemma3/mistral) and `rq3_win.sh` (gpt-4o-mini/deepseek) run all victims for a
host and Bark-notify on completion.

## Baseline testbed (RQ1 comparison, uniform conditions)
```bash
# pilot
python -m mtp.run --datasets advbench --victims gpt-4o-mini,mistral --attacks direct,paps,pair,mtp --reps 1
# full uniform re-eval + aggregation (Wilson CI, McNemar, BH-FDR, LaTeX tables)
bash run_all.sh
python -m mtp.report --datasets advbench,harmbench,sorrybench
```
External-optimiser baselines (`gcg-t`, `stinger`, `xjailbreak`) are produced in their
own repos and ingested as `data/external_prompts/<attack>.jsonl`
(`{"query": ..., "prompt": ...}`), then replayed uniformly.

## Reproducibility
- Sample N = 20 goals/dataset (seed `20260709`); full-benchmark sizes recorded in
  `experiment.yaml` (`full_n`) and table captions.
- Victim decoding fixed (`victim_temperature: 0.0`); attacker sampling
  `attacker_temperature: 1.0`; multi-turn budget `max_turns: 10`; `success_threshold: 5`.
- Single judge pinned to GPT-5.4-mini; replicates `R = 3` for the stochastic testbed.
- All model substitutions and non-uniform cells are annotated in `models.yaml`.

## Data availability and redaction (public release)
This repository is the **desensitized** artifact for a defensive/robustness study.
To avoid distributing directly usable attack content, the following redaction was
applied before release:

- **Rollout JSONL** (`runs_search/*.jsonl`): the crafted jailbreak prompt
  (`prompt` / `attack_prompt`) and the victim's harmful response
  (`victim_response`) are replaced with a stub `[REDACTED sha256:<hash> len:<n>]`.
  The stub keeps a content hash and length for integrity/dedup checks but never
  the text. The public benchmark goal id and text (`query`, `query_id`), the
  chosen persuasion strategy (`tag`), the judge score, and the success flag are
  kept so the experimental record remains auditable.
- **Aggregate results** (`runs_rq3/*.json`): per-goal judge scores and
  success/blocked flags only — no prompts or responses.
- **Excluded entirely:** real API keys (`configs/secrets.env`; use
  `configs/secrets.env.template`), the unalignment QLoRA adapter (`runs_sft/`),
  the raw harmful-goal benchmark corpora (`data/sources/`, `data/bundled/` —
  obtain AdvBench and HarmBench from their original public releases), and the LLM
  response cache.

## Responsible use
This code reproduces a jailbreak attack for defensive/robustness research only.
The harmful behaviors come from public benchmarks (AdvBench, HarmBench,
SORRY-Bench); the crafted attack prompts and harmful completions are withheld per
the redaction above. Public jailbreak datasets are used under their licenses.
Secrets live only in `configs/secrets.env` (git-ignored) and are never committed.
