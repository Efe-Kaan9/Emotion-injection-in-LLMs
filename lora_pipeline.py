"""
lora_pipeline.py — LoRA Baseline for Emotional Steering Comparison
====================================================================

Standalone LoRA fine-tuning + inference pipeline that serves as a
trainable parameter baseline against the KV-Cache injection method.

Modes:
  --mode train   Fine-tune a LoRA adapter on the 560-sample synthetic dataset
  --mode test    Run inference (50 neutral prompts × 6 emotions = 300 combos)
  --mode all     Train then test sequentially

Models (--model):
  phi4   → microsoft/Phi-4-mini-instruct
  qwen   → Qwen/Qwen2.5-1.5B-Instruct

Usage:
    python lora_pipeline.py --mode train --model phi4 --epochs 2
    python lora_pipeline.py --mode test  --model phi4
    python lora_pipeline.py --mode all   --model qwen --epochs 2

CRITICAL: This script is completely standalone. No existing project
files are modified. The output JSON format matches generation_results_*.json
so that analyze_results.py can parse it without modification.
"""

from __future__ import annotations

import os
import gc
import json
import random
import argparse
import csv
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

# Import the SAME prompt list used by all other evaluation scripts     venv\Scripts\activate.bat
# so the LoRA baseline is evaluated on identical inputs — fair comparison.
from generate_data import GENERATION_ITEMS

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR      = "./.hf_cache"
LORA_CKPT_DIR  = "./lora_checkpoints"
NOTEBOOK_PATH  = "KVinjectionWithPhi4.ipynb"   # source of 560-sample dataset

MAX_NEW_TOKENS = 100
TEMPERATURE    = 0.7
MAX_TOTAL_LEN  = 160   # prompt (≤80) + response (≤80) tokens

MODELS = {
    "phi4": "microsoft/Phi-4-mini-instruct",
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt list for the test phase
# ─────────────────────────────────────────────────────────────────────────────
# GENERATION_ITEMS is imported from generate_data.py — identical to the list
# used for KV-Cache Variant-1 and Variant-2 evaluation.
# Each item: {"prompt": str, "emotion_text": str, "target_emotion": str}
# Total: 50 items (10 emotions × 5 prompts each)
# This ensures a fair, apples-to-apples comparison with the KV-Cache baselines.


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def lora_ckpt_dir(model_key: str) -> str:
    path = os.path.join(LORA_CKPT_DIR, model_key)
    os.makedirs(path, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading — extract example_pairs from the training notebook
# Mirrors the exact logic in main.py:load_example_pairs (no import to avoid
# circular dependency)
# ─────────────────────────────────────────────────────────────────────────────

def load_example_pairs(notebook_path: str = NOTEBOOK_PATH) -> List[Dict]:
    """
    Extract example_pairs from KVinjectionWithPhi4.ipynb.
    Falls back to a small dummy dataset if the notebook is not found,
    so the script can be tested without the full project.
    """
    if not os.path.isfile(notebook_path):
        print(f"  [WARN] {notebook_path} not found — using 10-sample dummy set.")
        return [
            {"input": f"Sample prompt {i}", "output": f"Sample response {i}"}
            for i in range(10)
        ]

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        if "example_pairs = [" not in src or '"input"' not in src:
            continue
        # ── locate the list literal ────────────────────────────────────
        start         = src.index("example_pairs = [")
        bracket_count = 0
        started       = False
        end           = start
        for i, ch in enumerate(src[start:], start):
            if ch == "[":
                bracket_count += 1; started = True
            elif ch == "]":
                bracket_count -= 1
            if started and bracket_count == 0:
                end = i + 1; break
        local_ns = {}
        exec(src[start:end], {}, local_ns)
        pairs = local_ns["example_pairs"]
        print(f"  [OK] Loaded {len(pairs)} example_pairs from {notebook_path}")
        return pairs

    raise FileNotFoundError(f"Could not find example_pairs in {notebook_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset — Concatenated Causal LM format (prompt masked with -100)
# ─────────────────────────────────────────────────────────────────────────────

class EmotionPairDataset(Dataset):
    """
    Each item: full_ids = [prompt_ids | response_ids | pad]
    Labels   : [-100 × prompt_len | response_ids | -100 × pad_len]
    """

    def __init__(self, pairs: List[Dict], tokenizer,
                 max_prompt: int = 80, max_response: int = 80):
        self.samples   = []
        self.tokenizer = tokenizer
        pad_id = (tokenizer.pad_token_id
                  if tokenizer.pad_token_id is not None
                  else tokenizer.eos_token_id)
        max_total = max_prompt + max_response

        for p in pairs:
            prompt   = p.get("input",  p.get("prompt", ""))
            response = p.get("output", p.get("response", ""))

            p_ids = tokenizer(prompt,   add_special_tokens=True,
                              truncation=True, max_length=max_prompt)["input_ids"]
            r_ids = tokenizer(response, add_special_tokens=False,
                              truncation=True, max_length=max_response)["input_ids"]
            if r_ids and r_ids[-1] != tokenizer.eos_token_id:
                r_ids = r_ids + [tokenizer.eos_token_id]

            seq     = p_ids + r_ids
            seq_len = min(len(seq), max_total)
            pad_len = max_total - seq_len
            seq     = seq[:max_total]

            input_ids      = seq + [pad_id] * pad_len
            attention_mask = [1] * seq_len + [0] * pad_len
            p_len          = len(p_ids)
            r_len          = min(len(r_ids), max_total - p_len)
            labels         = ([-100] * p_len
                              + input_ids[p_len: p_len + r_len]
                              + [-100] * pad_len)

            self.samples.append({
                "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels":         torch.tensor(labels,         dtype=torch.long),
            })

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# LoRA configuration helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_lora_target_modules(model_name: str) -> List[str]:
    """
    Return standard attention projection names for each architecture.
    These are the modules LoRA patches into.
    """
    name = model_name.lower()
    if "phi" in name:
        # Phi-4-mini uses MHA with qkv_proj / o_proj
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "qwen" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    # Generic fallback
    return ["q_proj", "v_proj"]


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train(model_key: str, num_epochs: int = 2,
          batch_size: int = 2, lr: float = 5e-5, seed: int = 42):
    """
    Fine-tune a LoRA adapter on the 560-sample synthetic dataset.
    Saves adapter weights + loss CSV to lora_checkpoints/{model_key}/.
    """
    set_seed(seed)
    model_name = MODELS[model_key]
    save_dir   = lora_ckpt_dir(model_key)
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f"  LoRA TRAIN  —  {model_name}")
    print(f"  Epochs: {num_epochs}  |  batch: {batch_size}  |  lr: {lr}")
    print("=" * 70)

    # ── 1. Load dataset ────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset...")
    pairs = load_example_pairs()
    random.shuffle(pairs)
    split       = int(len(pairs) * 0.9)
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]
    print(f"  Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    # ── 2. Load base model (4-bit) ─────────────────────────────────────────
    print("\n[2/4] Loading base model (4-bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=CACHE_DIR, device_map="auto",
        quantization_config=bnb, torch_dtype=torch.float16,
    )
    print(f"  [OK] {model_name}")

    # ── 3. Attach LoRA adapter ─────────────────────────────────────────────
    print("\n[3/4] Attaching LoRA adapter...")
    # Model bazlı hedef modül seçimi (HATA BURADAYDI)
    if "phi-4" in MODELS[model_key].lower():
        # Phi-4-mini için tipik katman isimleri
        target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    else:
        # Qwen-2.5 (Llama mimarisi) için standart isimler
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── 4. Build datasets & train ──────────────────────────────────────────
    print("\n[4/4] Training...")
    train_ds = EmotionPairDataset(train_pairs, tokenizer)
    val_ds   = EmotionPairDataset(val_pairs,   tokenizer)

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=lr,
        fp16=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",          # disable W&B / tensorboard
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, padding=True, label_pad_token_id=-100
        ),
    )

    train_result = trainer.train()

    # ── Save final adapter ────────────────────────────────────────────────
    final_dir = os.path.join(save_dir, "final_adapter")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n  [OK] LoRA adapter saved to {final_dir}")

    # ── Save loss CSV ──────────────────────────────────────────────────────
    log_history = trainer.state.log_history
    csv_path    = os.path.join(save_dir, "loss_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss",
                                               "eval_loss", "epoch"])
        writer.writeheader()
        for entry in log_history:
            writer.writerow({
                "step":       entry.get("step",       ""),
                "train_loss": entry.get("loss",       ""),
                "eval_loss":  entry.get("eval_loss",  ""),
                "epoch":      entry.get("epoch",      ""),
            })
    print(f"  [OK] Loss history saved to {csv_path}")

    del model; clear_gpu()
    print("\n[DONE] LoRA training complete.")
    return final_dir


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: TEST
# ─────────────────────────────────────────────────────────────────────────────

def test(model_key: str, seed: int = 42) -> str:
    """
    Run inference using the trained LoRA adapter on GENERATION_ITEMS —
    the exact same 50 prompt+emotion pairs used for KV-Cache Variant-1
    and Variant-2 evaluation in generate_data.py.

    Total records: 50 (no inner emotion loop; each item has its own target_emotion).

    Output JSON format matches generation_results_*.json so that
    analyze_results.py can parse it without modification.
    """
    set_seed(seed)
    model_name  = MODELS[model_key]
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_dir = os.path.join(lora_ckpt_dir(model_key), "final_adapter")
    output_path = f"lora_generation_results_{model_key}.json"

    print("=" * 70)
    print(f"  LoRA TEST  —  {model_name}")
    print(f"  Adapter : {adapter_dir}")
    print(f"  Dataset : GENERATION_ITEMS ({len(GENERATION_ITEMS)} items, "
          f"identical to KV-Cache evaluation)")
    print("=" * 70)

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(
            f"LoRA adapter not found at {adapter_dir}. "
            f"Run --mode train first."
        )

    # ── Load base model (4-bit) ────────────────────────────────────────────
    print("\n[1/2] Loading base model + LoRA adapter (4-bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=CACHE_DIR, device_map="auto",
        quantization_config=bnb, torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    print("  [OK] Model + LoRA adapter loaded")

    # ── Inference loop ─────────────────────────────────────────────────────
    # Iterate over GENERATION_ITEMS — the exact same 50 items used in
    # generate_data.py for Variant-1 and Variant-2 evaluation.
    # Each item provides its own prompt AND target_emotion, so no
    # inner loop over emotions is needed (unlike NEUTRAL_PROMPTS × TARGET_EMOTIONS).
    # ── Inference loop ─────────────────────────────────────────────────────
    print("\n[2/2] Running inference on GENERATION_ITEMS...")
    results   = []
    total     = len(GENERATION_ITEMS)

    for record_id, item in enumerate(GENERATION_ITEMS):
        prompt         = item["prompt"]
        emotion        = item["target_emotion"]
        emotion_text   = item["emotion_text"]

        # 1. VANILLA ÜRETİMİ (Adaptör devre dışı, sadece nötr prompt)
        vanilla_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        
        with model.disable_adapter():
            with torch.no_grad():
                vanilla_ids = model.generate(
                    **vanilla_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
        vanilla_text = tokenizer.decode(vanilla_ids[0], skip_special_tokens=True)

        # 2. LORA ÜRETİMİ (Adaptör devrede, system prompt kullanılıyor)
        full_prompt = (
            f"Rewrite the following sentence conveying the "
            f"emotion of {emotion}: {prompt}"
        )
        
        lora_inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        
        with torch.no_grad():
            out_ids = model.generate(
                **lora_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # ── Output format matches generation_results_*.json ───────────────
        results.append({
            "record_id":      record_id,
            "prompt":         prompt,
            "emotion":        emotion,
            "target_emotion": emotion,
            "emotion_text":   emotion_text,
            "variant":        "lora",
            "alpha":          None,       
            "layer_config":   None,
            "generated_text": generated_text,
            "vanilla_text":   vanilla_text,  # Artık JSD hesabını bozmayacak gerçek baz metin
            "model":          model_key,
        })

        if (record_id + 1) % 10 == 0 or (record_id + 1) == total:
            print(f"  [{record_id+1:3d}/{total}] {emotion:15s} | {prompt[:50]}...")

        clear_gpu()
    # ── Save ──────────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  [OK] {len(results)} records saved to {output_path}")

    del model, base_model; clear_gpu()
    print("\n[DONE] LoRA inference complete.")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LoRA Baseline Pipeline for Emotion Steering"
    )
    parser.add_argument("--mode",   required=True,
                        choices=["train", "test", "all"],
                        help="Pipeline stage to execute")
    parser.add_argument("--model",  required=True,
                        choices=list(MODELS.keys()),
                        help="Target model key")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (train mode only)")
    parser.add_argument("--batch",  type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--lr",     type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    if args.mode == "train":
        train(args.model, num_epochs=args.epochs,
              batch_size=args.batch, lr=args.lr, seed=args.seed)

    elif args.mode == "test":
        out = test(args.model, seed=args.seed)
        print(f"\n  Results saved to: {out}")

    elif args.mode == "all":
        train(args.model, num_epochs=args.epochs,
              batch_size=args.batch, lr=args.lr, seed=args.seed)
        out = test(args.model, seed=args.seed)
        print(f"\n  Results saved to: {out}")


if __name__ == "__main__":
    main()
