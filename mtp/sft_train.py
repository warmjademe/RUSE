"""Attacker SFT warm-start (reverse-alignment) -- the stage that was missing.

Cold-start GRPO from the aligned base attacker gave a near-zero reward signal
(flat before/after ASR: 0.10 -> 0.15, McNemar p=1.0). The standard recipe is
SFT-then-RL: first supervise the policy on (context -> successful jailbreak prompt)
demonstrations so it emits attack-style prompts, THEN let GRPO optimise against a
specific victim. This trains a QLoRA adapter on the demonstrations built by
mtp.sft_data (data/sft/pool.jsonl), in the SAME prompt format the attacker uses at
RL/eval time (mtp.attack_mtp), which also removes the earlier train/eval skew.

Base = Qwen 2.5 7B Instruct: the paper's attacker (Sec. Implementation), the base
that grpo_train.py and llm_client.hf_local already use, and the model that generated
the SFT data -- so the SFT adapter drops straight into GRPO (grpo_train --init-adapter).

Loss is computed on the assistant completion only (completion_only_loss), so the
policy learns to GENERATE the attack prompt given the context, not to memorise the
fixed system/user scaffold.

Usage:
  python -m mtp.sft_train --smoke
  python -m mtp.sft_train --data data/sft/pool.jsonl --epochs 3 --out runs_sft/attacker
"""
from __future__ import annotations
import argparse, os, json
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path

from .common import ROOT

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_pc_dataset(paths: str, limit: int | None = None):
    """Load one or more comma-separated JSONL pools (e.g. pool.jsonl,public.jsonl)
    into TRL prompt/completion conversational format: prompt = [system, user],
    completion = [assistant]. TRL applies the chat template and (with
    completion_only_loss) masks the prompt tokens. Deduplicated by (query, completion)."""
    from datasets import Dataset
    import hashlib
    rows, seen = [], set()
    for path in [p.strip() for p in str(paths).split(",") if p.strip()]:
        fp = Path(path)
        if not fp.exists():
            print(f"[sft] warn: {path} missing, skipping")
            continue
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = r.get("messages")
            if not msgs or len(msgs) < 2 or msgs[-1]["role"] != "assistant":
                continue
            if not (msgs[-1].get("content") or "").strip():
                continue
            key = hashlib.sha1((r.get("query", "") + "|" + msgs[-1]["content"]).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"prompt": msgs[:-1], "completion": [msgs[-1]]})
            if limit and len(rows) >= limit:
                return Dataset.from_list(rows)
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "sft" / "pool.jsonl") + "," +
                    str(ROOT / "data" / "sft" / "public.jsonl"),
                    help="comma-separated JSONL pools (verified pool + public source C)")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default="runs_sft/attacker")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing (faster when VRAM-abundant and compute-bound)")
    ap.add_argument("--smoke", action="store_true", help="16 examples, 2 steps, to verify it runs")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                                 device_map="auto", dtype=torch.bfloat16)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])

    ds = load_pc_dataset(args.data, limit=16 if args.smoke else None)
    print(f"[sft] base={BASE_MODEL} dataset={len(ds)} completion_only_loss=True", flush=True)
    if len(ds) == 0:
        raise SystemExit("[sft] empty dataset -- run mtp.sft_data --harvest/--generate first")

    cfg = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=(2 if args.smoke else args.batch),
        gradient_accumulation_steps=(1 if args.smoke else args.grad_accum),
        num_train_epochs=(1 if args.smoke else args.epochs),
        max_steps=(2 if args.smoke else -1),
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs=(None if args.no_grad_ckpt else {"use_reentrant": False}),
        logging_steps=1,
        save_steps=10_000,
        max_length=args.max_len,
        packing=False,
        completion_only_loss=True,
        report_to=[],
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    print(f"[sft] saved adapter to {args.out}", flush=True)
    print("[sft] next: grpo_train --init-adapter " + args.out, flush=True)


if __name__ == "__main__":
    main()
