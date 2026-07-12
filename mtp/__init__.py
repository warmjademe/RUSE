"""MTP revision pipeline (PONE-D-26-17682).

Modules:
  common          config / seeded RNG / JSONL I/O
  llm_client      unified OpenAI-compatible client (proxy/ollama/deepseek/ark)
  victims         victim registry (commercial + open-source)
  data            AdvBench / HarmBench / SORRY-Bench samplers (N=20, seeded)
  persuasion      six social-influence principles (Phase 1 templates)
  judge           three-judge LLM-as-judge, Eq. 5 weighted aggregation
  attack_mtp      MTP two-phase attack (strategy selection + multi-turn)
  attack_baselines direct / paps / pair (+ replay for gcg-t/stinger/xjailbreak)
  ppo_train       PPO/QLoRA attacker fine-tuning (GPU)
  run             orchestrator (attacks x victims x datasets x replicates)
  stats           Wilson CI, McNemar, BH-FDR, effect sizes
  report          aggregate runs -> ASR + significance tables (LaTeX)
"""
__version__ = "0.1.0"
