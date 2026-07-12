"""Attacker fine-tuning with PPO + QLoRA (Methodology, Step 4).

Runs on the RTX 4090. Trains Qwen 2.5 7B (4-bit QLoRA) as the attacker policy;
the reward is the three-judge aggregated score S (Eq. 5), normalised to [0,1],
computed over a multi-turn rollout (attack_mtp.run_mtp) whose episode reward is
max_t R(q, r_t). Hyperparameters below match the manuscript (E5/R2.1/R3.4):

    base model         Qwen/Qwen2.5-7B-Instruct
    quantisation       4-bit NF4 (QLoRA)
    LoRA               r=8, alpha=16, dropout=0.05, target=attn+mlp proj
    PPO                lr=1e-4, batch_size=8, epochs=5
    entropy coef beta  0.01
    reward baseline b  0.5   (value/critic head baseline; the PPO "value model")
    clip eps           0.2
    reward             S/10 in [0,1], S = sum w_i s_i (w_i = 1/3), tau=5

The trl PPO API changes across releases; this is the reference configuration.
Adapt the trainer call to the installed trl version, then serve the merged
adapter to Ollama as `qwen2.5-mtp:7b` (see README) so run.py picks it up.
"""
from __future__ import annotations
import argparse

HPARAMS = dict(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    lora_r=8, lora_alpha=16, lora_dropout=0.05,
    lr=1e-4, batch_size=8, ppo_epochs=5,
    entropy_coef=0.01, reward_baseline=0.5, clip_range=0.2,
    reward_norm=10.0, tau=5, max_turns=5,
)


def build_models():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(HPARAMS["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(HPARAMS["base_model"], quantization_config=bnb,
                                                 device_map="auto")
    lora = LoraConfig(r=HPARAMS["lora_r"], lora_alpha=HPARAMS["lora_alpha"],
                      lora_dropout=HPARAMS["lora_dropout"], bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    return tok, model


def reward_from_rollout(query: str, victim_key: str) -> float:
    """Episode reward = max_t S(q, r_t)/10, via the shared multi-turn rollout."""
    from .victims import get_victim
    from .attack_mtp import run_mtp
    res = run_mtp(query, get_victim(victim_key))
    return res.final_score / HPARAMS["reward_norm"]


def main():
    ap = argparse.ArgumentParser(description="PPO/QLoRA attacker training (reference config)")
    ap.add_argument("--victim", default="mistral", help="victim to train the attacker against")
    ap.add_argument("--dataset", default="advbench")
    ap.add_argument("--dry-run", action="store_true", help="print config and exit")
    args = ap.parse_args()
    print("PPO/QLoRA hyperparameters:")
    for k, v in HPARAMS.items():
        print(f"  {k:16s} = {v}")
    if args.dry_run:
        return
    # NOTE: instantiate trl PPOTrainer for the installed version, loop over
    # load_sample(args.dataset), compute reward_from_rollout(...), and step.
    raise SystemExit("Wire the trl PPOTrainer loop for the installed trl version "
                     "(see module docstring); run with --dry-run to inspect config.")


if __name__ == "__main__":
    main()
