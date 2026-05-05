"""
main.py -- Facade Script: Train + Generate + Evaluate
======================================================

Orchestrates the full Q1 publication pipeline:
  1. Train Var1 & Var2 projectors for Qwen2.5 (Phi-4 already trained)
  2. Generate steered text for BOTH models (sequential, one at a time)
  3. Evaluate: Linguistic (CPU) + Mechanistic (GPU sequential)

Usage:
    python main.py --step train-qwen
    python main.py --step generate --model phi4
    python main.py --step generate --model qwen2.5
    python main.py --step eval-linguistic --model phi4
    python main.py --step eval-mechanistic --model phi4
    python main.py --step all-eval --model phi4
    python main.py --step full-pipeline

Hardware: RTX 3060 (6GB VRAM), 16GB RAM
"""

from __future__ import annotations
import os, sys, json, gc, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    BitsAndBytesConfig, AutoConfig,
)
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from projector_agnostic import ModelConfig, create_projector

# =============================================================
# Constants
# =============================================================
CACHE_DIR      = "./.hf_cache"
CHECKPOINT_DIR = "./checkpoints"
ENCODER_NAME   = "distilbert-base-uncased"
NUM_EMOTIONS   = 28
PREFIX_LEN     = 4
MAX_SEQ_LEN    = 128
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "phi4":    "microsoft/Phi-4-mini-instruct",
    "qwen2.5": "Qwen/Qwen2.5-1.5B-Instruct",
}

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()

def plot_training_history(history: dict, title: str, save_path: str):
    """Saves a high-res (300 DPI) training loss plot for the paper."""
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.plot(epochs, history["train_loss"], 'b-', label='Training Loss', marker='o')
    if "val_loss" in history and len(history["val_loss"]) > 0:
        plt.plot(epochs, history["val_loss"], 'r-', label='Validation Loss', marker='s')
        
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved high-res training plot to {save_path}")

def ckpt_path(model_key, variant):
    """Model-specific checkpoint paths so Phi-4 and Qwen don't collide."""
    return os.path.join(CHECKPOINT_DIR, f"variant{variant}_projector_{model_key}.pt")

# =============================================================
# Extract example_pairs from notebook (no manual copy needed)
# =============================================================
def load_example_pairs(notebook_path="KVinjectionWithPhi4.ipynb"):
    """Extract example_pairs from the notebook by finding and exec-ing the cell."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "example_pairs = [" in src and '"input"' in src and '"output"' in src:
            # Extract just the example_pairs assignment
            start = src.index("example_pairs = [")
            # Find the matching closing bracket
            bracket_count = 0
            started = False
            end = start
            for i, ch in enumerate(src[start:], start):
                if ch == '[': 
                    bracket_count += 1
                    started = True
                elif ch == ']': 
                    bracket_count -= 1
                if started and bracket_count == 0:
                    end = i + 1
                    break
            pairs_code = src[start:end]
            local_ns = {}
            exec(pairs_code, {}, local_ns)
            pairs = local_ns["example_pairs"]
            print(f"  [OK] Loaded {len(pairs)} example_pairs from notebook")
            return pairs

    raise FileNotFoundError("Could not find example_pairs in notebook")

# =============================================================
# Emotion Extractor — matches DistilBertFineTune.ipynb EXACTLY
# Architecture: DistilBERT → [CLS] token → Linear(768, 28)
# =============================================================
class EmotionExtractor(torch.nn.Module):
    def __init__(self, encoder_name="distilbert-base-uncased", num_emotions=28):
        super().__init__()
        # Senin eğitimde kullandığın mimari (HuggingFace cache dizini ile birlikte)
        self.encoder = AutoModel.from_pretrained(encoder_name, cache_dir="./.hf_cache")
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_emotions)
        
        # SADECE SCRIPTLER İÇİN GEREKLİ OLAN KISIM: 
        # Modeli inference ve KV-Cache eğitimi için donduruyoruz ki ağırlıklar bozulmasın
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, **kwargs):
        # **kwargs sayesinde beklenmeyen argümanlar (token_type_ids vb.) hata vermez
        out = self.encoder(input_ids, attention_mask=attention_mask)
        
        # İŞTE BURASI HAYAT KURTARAN KISIM: Senin eğitimde kullandığın Mean Pooling!
        pooled = out.last_hidden_state.mean(dim=1) 
        
        logits = self.classifier(pooled)
        return logits


def get_emotion_embedding(text, emotion_ext, tokenizer, device):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    ).to(device)
    with torch.no_grad():
        logits = emotion_ext(**inputs)
        return torch.sigmoid(logits)


def load_emotion_extractor(device):
    """Load the fine-tuned EmotionExtractor from local checkpoint.pt.

    Uses [CLS] pooling to match DistilBertFineTune.ipynb training.
    Handles both {'model_state_dict': ...} and raw state-dict formats.
    """
    enc_tok = AutoTokenizer.from_pretrained(ENCODER_NAME, cache_dir=CACHE_DIR)
    emo_ext = EmotionExtractor(
        encoder_name=ENCODER_NAME, num_emotions=NUM_EMOTIONS
    ).to(device)
    emo_ext.eval()

    ckpt = "checkpoint.pt"
    if os.path.isfile(ckpt):
        sd    = torch.load(ckpt, map_location="cpu")
        state = sd.get("model_state_dict", sd)   # handles both formats
        emo_ext.load_state_dict(state)
        print("  [OK] EmotionExtractor loaded from checkpoint.pt (CLS pooling)")
    else:
        print("  [WARN] checkpoint.pt not found — using untrained EmotionExtractor!")

    return emo_ext, enc_tok

# =============================================================
# Dataset — Concatenated Causal LM (FIXED)
# Prompt tokens are masked with -100; only response tokens
# contribute to the Cross-Entropy loss.
# =============================================================
MAX_PROMPT_LEN   = 80
MAX_RESPONSE_LEN = 80
MAX_TOTAL_LEN    = MAX_PROMPT_LEN + MAX_RESPONSE_LEN  # 160

class EmotionalResponseDataset(Dataset):
    def __init__(self, pairs, tokenizer, emotion_ext, enc_tok, device=DEVICE):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.emotions  = []
        emotion_ext.eval()
        with torch.no_grad():
            for p in pairs:
                e = get_emotion_embedding(p["input"], emotion_ext, enc_tok, device)
                self.emotions.append(e.cpu())

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p   = self.pairs[idx]
        tok = self.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        # --- Tokenize prompt (with BOS if model uses it) ---
        prompt_ids = tok(
            p["input"],
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_PROMPT_LEN,
        )["input_ids"]

        # --- Tokenize response (no BOS; keep EOS as stop signal) ---
        response_ids = tok(
            p["output"],
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_RESPONSE_LEN,
        )["input_ids"]
        # Append EOS so the model learns to stop
        if response_ids and response_ids[-1] != tok.eos_token_id:
            response_ids = response_ids + [tok.eos_token_id]

        # --- Concatenate and pad to MAX_TOTAL_LEN ---
        seq      = prompt_ids + response_ids
        seq_len  = len(seq)
        pad_len  = MAX_TOTAL_LEN - seq_len
        if pad_len < 0:          # truncate from response end if too long
            seq     = seq[:MAX_TOTAL_LEN]
            seq_len = MAX_TOTAL_LEN
            pad_len = 0

        input_ids      = seq + [pad_id] * pad_len
        attention_mask = [1]   * seq_len + [0] * pad_len

        # --- Labels: mask prompt and padding with -100 ---
        prompt_len = len(prompt_ids)
        resp_len   = len(response_ids) if seq_len == len(seq) else MAX_TOTAL_LEN - prompt_len
        resp_len   = min(resp_len, MAX_TOTAL_LEN - prompt_len)
        labels     = ([-100] * prompt_len
                      + input_ids[prompt_len : prompt_len + resp_len]
                      + [-100] * pad_len)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
            "emotion":        self.emotions[idx].squeeze(0),
            # keep prompt length so training loop can split correctly
            "prompt_len":     torch.tensor(prompt_len,     dtype=torch.long),
        }

# =============================================================
# KV-Cache helpers
# =============================================================
def prepend_prefix_to_cache(past_kv, k_prefix, v_prefix):
    new_cache = DynamicCache()
    for layer_idx in range(len(past_kv)):
        # Version-agnostic extraction (Legacy Tuples vs mid-HF key_cache vs newer HF layers)
        if hasattr(past_kv, "key_cache"):
            k_l = past_kv.key_cache[layer_idx]
            v_l = past_kv.value_cache[layer_idx]
        elif hasattr(past_kv, "layers"):
            k_l = past_kv.layers[layer_idx].keys
            v_l = past_kv.layers[layer_idx].values
        else:
            k_l = past_kv[layer_idx][0]
            v_l = past_kv[layer_idx][1]
        k_new = torch.cat([k_prefix.to(k_l.device).to(k_l.dtype), k_l], dim=2)
        v_new = torch.cat([v_prefix.to(v_l.device).to(v_l.dtype), v_l], dim=2)
        new_cache.update(k_new, v_new, layer_idx=layer_idx)
    return new_cache

# =============================================================
# TRAINING (model-agnostic -- works for any HF causal LM)
# =============================================================
import csv as _csv

def _save_loss_csv(history: dict, model_key: str, variant: int):
    """Save per-epoch loss to CSV for smooth convergence plots."""
    path = f"loss_history_v{variant}_{model_key}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for i, (tr, vl) in enumerate(
            zip(history["train_loss"], history["val_loss"]), start=1
        ):
            writer.writerow({"epoch": i, "train_loss": f"{tr:.6f}", "val_loss": f"{vl:.6f}"})
    print(f"  [OK] Loss history saved to {path}")


def train_variant(model_key, variant, num_epochs=2, batch_size=2, lr=1e-4):
    """Train Var1 or Var2 projector for the given model.

    FIXED: Uses concatenated prompt+response with -100 masked labels so that
    Cross-Entropy is computed only over response tokens (proper Causal LM).
    """
    model_name = MODELS[model_key]
    save_path  = ckpt_path(model_key, variant)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"TRAINING VARIANT {variant} for {model_name}")
    print("=" * 70)

    # ── Load training pairs ──────────────────────────────────────────────
    pairs = load_example_pairs()
    random.shuffle(pairs)
    split       = int(len(pairs) * 0.9)
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]

    # ── Decoder (4-bit quantised, fully frozen) ──────────────────────────
    print("\n[1/4] Loading decoder...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    dec_tok = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    if dec_tok.pad_token is None:
        dec_tok.pad_token = dec_tok.eos_token
    dec_tok.padding_side = "right"

    decoder = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=CACHE_DIR, device_map="auto",
        quantization_config=bnb_cfg, torch_dtype=torch.float16,
    )
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False
    print(f"  [OK] {model_name} loaded and frozen")

    # ── Emotion extractor ────────────────────────────────────────────────
    print("[2/4] Loading emotion extractor...")
    emo_ext, enc_tok = load_emotion_extractor(DEVICE)

    # ── Projector ────────────────────────────────────────────────────────
    print("[3/4] Initialising projector...")
    mc = ModelConfig.from_hf_config(decoder.config, model_name=model_name)
    print(mc.summary())
    projector = create_projector(mc, variant=variant, device=DEVICE)
    projector.train()

    # Var2: load frozen Var1 base weights
    if variant == 2:
        v1_path = ckpt_path(model_key, 1)
        if os.path.isfile(v1_path):
            v1_sd   = torch.load(v1_path, map_location="cpu")["model_state_dict"]
            base_sd = {k: v for k, v in v1_sd.items() if k.startswith("proj_")}
            projector.load_state_dict(base_sd, strict=False)
            for n, p in projector.named_parameters():
                if n.startswith("proj_"):
                    p.requires_grad = False
            print(f"  [OK] Loaded + froze Var1 base from {v1_path}")

    # ── Datasets ─────────────────────────────────────────────────────────
    print("[4/4] Building datasets...")
    train_ds = EmotionalResponseDataset(train_pairs, dec_tok, emo_ext, enc_tok)
    val_ds   = EmotionalResponseDataset(val_pairs,   dec_tok, emo_ext, enc_tok)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── Optimiser & loss ─────────────────────────────────────────────────
    trainable = [p for p in projector.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=lr)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=-100)   # -100 masks prompt & pad
    scaler    = torch.amp.GradScaler(device=DEVICE)
    best_val  = float("inf")
    history   = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        projector.train()
        total_loss = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs}"):
            full_ids  = batch["input_ids"].to(DEVICE)       # (B, MAX_TOTAL_LEN)
            full_mask = batch["attention_mask"].to(DEVICE)  # (B, MAX_TOTAL_LEN)
            labels    = batch["labels"].to(DEVICE)          # (B, MAX_TOTAL_LEN)
            emo       = batch["emotion"].to(DEVICE)          # (B, 28)
            p_len     = batch["prompt_len"]                  # (B,) — CPU tensor

            # ── Step 1: Encode prompt portion → KV cache (no grad) ──────
            # Use the maximum prompt length in this batch to keep it simple
            max_p = int(p_len.max().item())
            prompt_ids  = full_ids[:, :max_p]
            prompt_mask = full_mask[:, :max_p]

            with torch.no_grad():
                prompt_out = decoder(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    use_cache=True,
                )
                past_kv = prompt_out.past_key_values

            # ── Step 2: Project emotion → KV prefix ──────────────────────
            with torch.amp.autocast(device_type=DEVICE):
                if variant == 1:
                    k_pre, v_pre = projector(emo)
                else:
                    k_pre, v_pre, _ = projector(emo)

            mod_kv = prepend_prefix_to_cache(past_kv, k_pre, v_pre)

            # ── Step 3: Decode response tokens with emotion-modulated KV ─
            # Feed only the response slice; labels already aligned to full seq
            response_ids  = full_ids[:,  max_p:]    # (B, resp_len)
            response_mask = full_mask[:, max_p:]    # (B, resp_len)
            response_labs = labels[:,    max_p:]    # (B, resp_len)

            # Build combined attention mask: past (prompt+prefix) + response
            prefix_len  = k_pre.size(2)             # number of prefix tokens added
            past_length = max_p + prefix_len
            combined_mask = torch.cat([
                torch.ones(full_ids.size(0), past_length, device=DEVICE, dtype=full_mask.dtype),
                response_mask,
            ], dim=1)

            with torch.amp.autocast(device_type=DEVICE):
                out    = decoder(
                    input_ids=response_ids,
                    attention_mask=combined_mask,
                    past_key_values=mod_kv,
                    use_cache=False,
                )
                # Shift: predict token[t+1] from logits[t]
                shift_logits = out.logits[:, :-1, :].contiguous()
                shift_labels = response_labs[:, 1:].contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()

        torch.cuda.empty_cache()
        avg_train = total_loss / max(len(train_dl), 1)
        history["train_loss"].append(avg_train)

        # ── Validation ───────────────────────────────────────────────────
        projector.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                full_ids  = batch["input_ids"].to(DEVICE)
                full_mask = batch["attention_mask"].to(DEVICE)
                labels    = batch["labels"].to(DEVICE)
                emo       = batch["emotion"].to(DEVICE)
                p_len     = batch["prompt_len"]

                max_p       = int(p_len.max().item())
                prompt_ids  = full_ids[:, :max_p]
                prompt_mask = full_mask[:, :max_p]

                prompt_out = decoder(
                    input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=True
                )
                past_kv = prompt_out.past_key_values

                if variant == 1:
                    k_pre, v_pre = projector(emo)
                else:
                    k_pre, v_pre, _ = projector(emo)
                mod_kv = prepend_prefix_to_cache(past_kv, k_pre, v_pre)

                response_ids  = full_ids[:,  max_p:]
                response_mask = full_mask[:, max_p:]
                response_labs = labels[:,    max_p:]
                prefix_len    = k_pre.size(2)
                past_length   = max_p + prefix_len
                combined_mask = torch.cat([
                    torch.ones(full_ids.size(0), past_length, device=DEVICE, dtype=full_mask.dtype),
                    response_mask,
                ], dim=1)

                out = decoder(
                    input_ids=response_ids,
                    attention_mask=combined_mask,
                    past_key_values=mod_kv,
                    use_cache=False,
                )
                shift_logits = out.logits[:, :-1, :].contiguous()
                shift_labels = response_labs[:, 1:].contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                val_loss += loss.item()

        avg_val = val_loss / max(len(val_dl), 1)
        history["val_loss"].append(avg_val)

        print(f"  Epoch {epoch+1}/{num_epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {"epoch": epoch, "model_state_dict": projector.state_dict(),
                 "train_loss": avg_train, "val_loss": avg_val},
                save_path,
            )
            print(f"  [OK] Best checkpoint saved → {save_path}")

    # ── Save CSV & cleanup ────────────────────────────────────────────────
    _save_loss_csv(history, model_key, variant)
    del decoder, emo_ext
    clear_gpu()
    print(f"\n[DONE] Training Variant {variant} for {model_key} complete.")
    return history


# =============================================================
# CLI DISPATCH
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="KV-Cache Emotion Steering Pipeline")
    parser.add_argument("--step", required=True,
        choices=["train-qwen", "generate", "eval-linguistic",
                 "eval-mechanistic", "all-eval", "full-pipeline"])
    parser.add_argument("--model", choices=["phi4", "qwen2.5"], default="phi4")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    set_seed(42)

    if args.step == "train-qwen":
        print("\n>>> STEP: Train Qwen2.5 Var1 & Var2\n")
        # h1 = train_variant("qwen2.5", variant=1, num_epochs=args.epochs)
        # plot_training_history(h1, "Qwen2.5 - Variant 1 (Basic) Training Loss", "qwen2.5_var1_loss.png")
        # clear_gpu()
        
        # User already successfully trained Var1, proceeding to Var2 directly
        h2 = train_variant("qwen2.5", variant=2, num_epochs=args.epochs)
        plot_training_history(h2, "Qwen2.5 - Variant 2 (Modulated) Training Loss", "qwen2.5_var2_loss.png")
        clear_gpu()

    elif args.step == "generate":
        # Patch generate_data checkpoint paths for the selected model
        import generate_data as gd
        # Override checkpoint paths to model-specific ones
        gd.CHECKPOINT_DIR = CHECKPOINT_DIR
        out = f"generation_results_{args.model}.json"
        gd.run_generation(args.model, out)

    elif args.step == "eval-linguistic":
        import eval_linguistic as el
        inp = f"generation_results_{args.model}.json"
        sys.argv = ["eval_linguistic.py", "--input", inp]
        el.main()

    elif args.step == "eval-mechanistic":
        import eval_mechanistic as em
        inp = f"generation_results_{args.model}.json"
        sys.argv = ["eval_mechanistic.py", "--input", inp, "--model", args.model]
        em.main()

    elif args.step == "all-eval":
        print(f"\n>>> Running ALL evaluation for {args.model}\n")
        inp = f"generation_results_{args.model}.json"
        # Linguistic (CPU)
        import eval_linguistic as el
        sys.argv = ["eval_linguistic.py", "--input", inp]
        el.main()
        # Mechanistic (GPU sequential)
        import eval_mechanistic as em
        sys.argv = ["eval_mechanistic.py", "--input", inp, "--model", args.model]
        em.main()

    elif args.step == "full-pipeline":
            print("\n" + "=" * 70)
            print("FULL PIPELINE: Train Both -> Generate Both -> Evaluate Both")
            print("=" * 70)

            # 1. Train Phi-4 (Bunu biz ekledik)
            print("\n[1/6] Training Phi-4 Var1...")
            h_phi1 = train_variant("phi4", variant=1, num_epochs=args.epochs)
            plot_training_history(h_phi1, "Phi-4 - Variant 1 (Basic) Training Loss", "phi4_var1_loss.png")
            clear_gpu()

            print("\n[2/6] Training Phi-4 Var2...")
            h_phi2 = train_variant("phi4", variant=2, num_epochs=args.epochs)
            plot_training_history(h_phi2, "Phi-4 - Variant 2 (Modulated) Training Loss", "phi4_var2_loss.png")
            clear_gpu()

            # 2. Train Qwen
            print("\n[3/6] Training Qwen2.5 Var1...")
            h_qwen1 = train_variant("qwen2.5", variant=1, num_epochs=args.epochs)
            plot_training_history(h_qwen1, "Qwen2.5 - Variant 1 (Basic) Training Loss", "qwen2.5_var1_loss.png")
            clear_gpu()

            print("\n[4/6] Training Qwen2.5 Var2...")
            h_qwen2 = train_variant("qwen2.5", variant=2, num_epochs=args.epochs)
            plot_training_history(h_qwen2, "Qwen2.5 - Variant 2 (Modulated) Training Loss", "qwen2.5_var2_loss.png")
            clear_gpu()

            # 3. Generate for both models
            import generate_data as gd
            for mk in ["phi4", "qwen2.5"]:
                print(f"\n[5/6] Generating for {mk}...")
                gd.run_generation(mk, f"generation_results_{mk}.json")
                clear_gpu()

            # 4. Linguistic eval (CPU, both)
            import eval_linguistic as el
            for mk in ["phi4", "qwen2.5"]:
                print(f"\n[6/6] Linguistic eval for {mk}...")
                sys.argv = ["eval_linguistic.py", "--input", f"generation_results_{mk}.json",
                             "--output", f"eval_linguistic_{mk}.json",
                             "--csv", f"eval_linguistic_{mk}.csv"]
                el.main()

            # 5. Mechanistic eval (GPU, both)
            import eval_mechanistic as em
            for mk in ["phi4", "qwen2.5"]:
                print(f"\n[7/7] Mechanistic eval for {mk}...")
                sys.argv = ["eval_mechanistic.py", "--input", f"generation_results_{mk}.json",
                             "--model", mk, "--output", f"eval_mechanistic_{mk}.json"]
                em.main()
                clear_gpu()

            print("\n" + "=" * 70)
            print("[DONE] Full pipeline complete! All loss graphs and JSONs are saved.")
            print("=" * 70)

if __name__ == "__main__":
    main()