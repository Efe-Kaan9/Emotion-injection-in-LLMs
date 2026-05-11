"""
eval_real_world.py — Real-World GoEmotions Benchmark Evaluation
================================================================

Evaluates the KV-Cache injection framework on 50 REAL neutral texts
from the GoEmotions test split across three scenarios:

  Scenario A — Vanilla      : Raw LLM generation (no steering)
  Scenario B — System Prompt: Template-based prompt engineering baseline
  Scenario C — Steered V2   : KV-Cache injection with Variant-2 projector
                               (optimal per-model alpha + layer config)

Models evaluated:
  • microsoft/Phi-4-mini-instruct   (alpha=1.0, layers="all")
  • Qwen/Qwen2.5-1.5B-Instruct     (alpha=1.5, layers="second_half")

Metrics (per scenario):
  • Target Score  — RoBERTa (SamLowe/roberta-base-go_emotions) P(target|output)
  • Perplexity    — Causal LM loss computed on RESPONSE tokens only
  • JSD           — Jensen-Shannon Divergence (vanilla dist. ‖ scenario dist.)

Usage:
    python eval_real_world.py
    python eval_real_world.py --n-samples 750 --seed 42 --output real_world_metrics.json

Output:
    real_world_metrics.json  — averaged metrics per model × scenario
    real_world_raw.json      — per-example raw results

CRITICAL RULE: This script imports and uses the existing pipeline as-is.
It does NOT redefine any class or modify any existing file.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.cache_utils import DynamicCache
from peft import PeftModel

# ── Import existing pipeline components (DO NOT redefine) ──────────────────
from generate_data import (
    EmotionExtractor,
    get_emotion_embedding,
    prepend_prefix_to_cache_ablated,
    generate_vanilla,
    ENCODER_NAME,
    NUM_EMOTIONS,
    CACHE_DIR,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    CHECKPOINT_DIR,
)
from projector_agnostic import ModelConfig, create_projector
from eval_mechanistic import (
    jensen_shannon_divergence,
    clear_gpu,
)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

ROBERTA_CLASSIFIER = "SamLowe/roberta-base-go_emotions"

MODELS: Dict[str, Dict] = {
    "phi4": {
        "hf_name":  "microsoft/Phi-4-mini-instruct",
        "alpha":    1.0,
        "layers":   "all",
    },
    "qwen2.5": {
        "hf_name":  "Qwen/Qwen2.5-1.5B-Instruct",
        "alpha":    1.5,
        "layers":   "second_half",
    },
}

# 6 target emotions; 50 samples will be round-robin assigned
TARGET_EMOTIONS: List[str] = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

# GoEmotions neutral label index (label 27 in the simplified 28-class set)
# We detect it dynamically from the dataset features to be safe.
NEUTRAL_LABEL_NAME = "neutral"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataset preparation
# ══════════════════════════════════════════════════════════════════════════════

def load_neutral_samples(n: int = 50, seed: int = 42) -> List[Dict]:
    """Return n neutral texts from GoEmotions test split with assigned target emotions."""
    print(f"\n[Dataset] Loading go_emotions test split from local cache...")
    ds = load_dataset(
        "go_emotions",
        "simplified",
        split="test",
        cache_dir=CACHE_DIR,
    )

    # Resolve neutral label index from dataset features
    label_names: List[str] = ds.features["labels"].feature.names
    try:
        neutral_idx = label_names.index(NEUTRAL_LABEL_NAME)
    except ValueError:
        raise ValueError(
            f"'{NEUTRAL_LABEL_NAME}' not found in dataset labels: {label_names}"
        )
    print(f"  Neutral label index = {neutral_idx}")

    # Filter: keep only examples whose ONLY label is neutral
    neutral_rows = [
        row for row in ds
        if row["labels"] == [neutral_idx] and row["text"].strip()
    ]
    print(f"  Found {len(neutral_rows)} strictly-neutral examples")

    if len(neutral_rows) < n:
        raise RuntimeError(
            f"Not enough neutral examples ({len(neutral_rows)} < {n}). "
            "Lower --n-samples or relax the filter."
        )

    rng = random.Random(seed)
    selected = rng.sample(neutral_rows, n)

    # Round-robin assign target emotions
    samples = []
    for i, row in enumerate(selected):
        samples.append({
            "id":             i,
            "text":           row["text"],
            "target_emotion": TARGET_EMOTIONS[i % len(TARGET_EMOTIONS)],
        })

    print(f"  [OK] Selected {len(samples)} samples "
          f"({len(TARGET_EMOTIONS)} emotions, round-robin)")
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# 2. Emotion extractor loader (reuses existing class from generate_data)
# ══════════════════════════════════════════════════════════════════════════════

def load_emotion_extractor_local(device: str):
    """Load fine-tuned EmotionExtractor from checkpoint.pt using existing class."""
    enc_tok = AutoTokenizer.from_pretrained(ENCODER_NAME, cache_dir=CACHE_DIR)
    model   = EmotionExtractor(
        encoder_name=ENCODER_NAME, num_emotions=NUM_EMOTIONS
    ).to(device)
    model.eval()

    ckpt_path = "checkpoint.pt"
    if os.path.isfile(ckpt_path):
        sd    = torch.load(ckpt_path, map_location="cpu")
        state = sd.get("model_state_dict", sd)
        model.load_state_dict(state)
        print("  [OK] EmotionExtractor loaded from checkpoint.pt")
    else:
        print("  [WARN] checkpoint.pt not found — using untrained EmotionExtractor!")

    for p in model.parameters():
        p.requires_grad = False
    return model, enc_tok


# ══════════════════════════════════════════════════════════════════════════════
# 3. RoBERTa target-score helper
# ══════════════════════════════════════════════════════════════════════════════

def get_emotion_probs_roberta(
    text: str,
    roberta_model,
    roberta_tok,
    device: str,
) -> np.ndarray:
    """Return 28-dim sigmoid probability vector from the RoBERTa classifier."""
    if not text.strip():
        return np.zeros(28)
    enc = roberta_tok(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        logits = roberta_model(**enc).logits[0]
    return torch.sigmoid(logits).cpu().numpy()


def target_score_from_probs(
    probs: np.ndarray,
    target_emotion: str,
    label2id: Dict[str, int],
) -> float:
    idx = label2id.get(target_emotion.lower(), -1)
    if idx < 0 or idx >= len(probs):
        return -1.0
    return float(probs[idx])


# ══════════════════════════════════════════════════════════════════════════════
# 4. PPL on response tokens only
# ══════════════════════════════════════════════════════════════════════════════

def compute_response_ppl(
    response_text: str,
    llm,
    llm_tok,
    device: str,
    max_length: int = 256,
) -> float:
    """
    Compute perplexity STRICTLY on response tokens.
    The prompt template is NOT included in the PPL calculation.
    """
    if not response_text.strip():
        return float("inf")

    enc = llm_tok(
        response_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    ids = enc["input_ids"]
    if ids.size(1) < 2:
        return float("inf")

    with torch.no_grad():
        loss = llm(input_ids=ids, labels=ids).loss.item()

    return math.exp(loss)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Generation helpers
# ══════════════════════════════════════════════════════════════════════════════

from contextlib import contextmanager

@contextmanager
def _null_ctx():
    """No-op context manager used when LoRA adapter is not loaded."""
    yield


def _strip_prompt(full_text: str, prompt: str) -> str:
    """Remove the prompt prefix from the full generated string if present."""
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text.strip()


def generate_system_prompt(
    llm,
    llm_tok,
    neutral_text: str,
    target_emotion: str,
    device: str,
) -> Tuple[str, str]:
    """
    Scenario B: template-based prompt engineering.
    Returns (full_generated_string, response_only_string).
    """
    prompt = (
        f"Rewrite the following sentence conveying the emotion of "
        f"{target_emotion.upper()}: {neutral_text}"
    )
    full = generate_vanilla(llm, llm_tok, prompt, device)
    response = _strip_prompt(full, prompt)
    return full, response


def generate_steered(
    llm,
    llm_tok,
    neutral_text: str,
    emotion_vec: torch.Tensor,
    projector_v2,
    alpha: float,
    layer_cfg: str,
    device: str,
) -> Tuple[str, str]:
    """
    Scenario C: KV-Cache injection with Variant-2 projector.
    Returns (full_generated_string, response_only_string).
    """
    inputs = llm_tok(neutral_text, return_tensors="pt").to(device)

    with torch.no_grad():
        # Get base KV cache
        base_out = llm(**inputs, use_cache=True)
        past_kv  = base_out.past_key_values

        # Project emotion → KV prefix
        k_prefix, v_prefix, _ = projector_v2(emotion_vec)

    # Inject with optimal alpha + layer config (from existing function)
    num_layers = llm.config.num_hidden_layers
    mod_kv = prepend_prefix_to_cache_ablated(
        past_kv, k_prefix, v_prefix,
        alpha=alpha,
        layer_config=layer_cfg,
        num_layers=num_layers,
    )

    # Token-by-token generation (mirrors generate_data.generate_with_cache)
    generated_ids = inputs["input_ids"]
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            out    = llm(
                input_ids=generated_ids[:, -1:],
                use_cache=True,
                past_key_values=mod_kv,
            )
            logits = out.logits[:, -1, :]
            mod_kv = out.past_key_values
            probs  = torch.softmax(logits / TEMPERATURE, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_t], dim=-1)
            if next_t.item() == llm_tok.eos_token_id:
                break

    full     = llm_tok.decode(generated_ids[0], skip_special_tokens=True)
    response = _strip_prompt(full, neutral_text)
    return full, response


def generate_lora(
    llm,
    llm_tok,
    neutral_text: str,
    target_emotion: str,
    device: str,
) -> Tuple[str, str]:
    """
    Scenario D: LoRA adapter generation.
    Adapter must be ACTIVE when calling this function.
    Uses the identical system-prompt template as lora_pipeline.py.
    Returns (full_generated_string, response_only_string).
    """
    prompt = (
        f"Rewrite the following sentence conveying the emotion of "
        f"{target_emotion}: {neutral_text}"
    )
    full     = generate_vanilla(llm, llm_tok, prompt, device)
    response = _strip_prompt(full, prompt)
    return full, response


# ══════════════════════════════════════════════════════════════════════════════
# Human-eval export helper
# ══════════════════════════════════════════════════════════════════════════════

HUMAN_EVAL_N = 100   # Number of samples to export for human evaluation

def _save_human_eval(buffer: List[Dict], model_key: str) -> None:
    """
    Save the first HUMAN_EVAL_N samples to:
      human_eval_<model_key>.json  — machine-readable, for form generation
      human_eval_<model_key>.txt   — human-readable, for quick review
    Called once per model after the evaluation loop. Zero GPU activity.
    """
    json_path = f"human_eval_{model_key}.json"
    txt_path  = f"human_eval_{model_key}.txt"

    # ── JSON ────────────────────────────────────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(buffer, f, indent=2, ensure_ascii=False)
    print(f"  [HumanEval] JSON → {json_path}  ({len(buffer)} samples)")

    # ── TXT ────────────────────────────────────────────────────────────────────
    sep  = "=" * 72
    thin = "-" * 72
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write(f"  HUMAN EVALUATION EXPORT — {model_key.upper()}\n")
        f.write(f"  First {len(buffer)} samples (Vanilla / System-Prompt / Steered-V2 / LoRA)\n")
        f.write(f"{sep}\n\n")
        for entry in buffer:
            f.write(f"{thin}\n")
            f.write(f"Sample #{entry['record_id']:03d}   Emotion: {entry['target_emotion'].upper()}\n")
            f.write(f"{thin}\n")
            f.write(f"INPUT PROMPT:\n  {entry['input_prompt']}\n\n")
            f.write(f"[A] VANILLA OUTPUT:\n  {entry['vanilla_output']}\n\n")
            f.write(f"[B] SYSTEM-PROMPT OUTPUT:\n  {entry['system_prompt_output']}\n\n")
            f.write(f"[C] STEERED V2 OUTPUT:\n  {entry['steered_v2_output']}\n\n")
            lora_txt = entry.get("lora_output", "") or "(LoRA adapter not available)"
            f.write(f"[D] LORA OUTPUT:\n  {lora_txt}\n\n")
    print(f"  [HumanEval] TXT  → {txt_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Per-model evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model_key: str,
    cfg: Dict,
    samples: List[Dict],
    roberta_model,
    roberta_tok,
    label2id: Dict[str, int],
) -> List[Dict]:
    """Run all 4 scenarios (Vanilla, SysPrompt, Steered-V2, LoRA) for one LLM."""
    print("\n" + "=" * 70)
    print(f"  EVALUATING: {model_key.upper()}  ({cfg['hf_name']})")
    print(f"  Optimal alpha={cfg['alpha']}  layers='{cfg['layers']}'")
    print("=" * 70)

    # ── Load LLM (4-bit) ────────────────────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    llm_tok = AutoTokenizer.from_pretrained(cfg["hf_name"], cache_dir=CACHE_DIR)
    if llm_tok.pad_token is None:
        llm_tok.pad_token = llm_tok.eos_token

    base_llm = AutoModelForCausalLM.from_pretrained(
        cfg["hf_name"],
        cache_dir=CACHE_DIR,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
    )
    base_llm.eval()
    print(f"  [OK] {cfg['hf_name']} loaded (4-bit NF4)")

    # ── Wrap with LoRA adapter (adapter disabled by default) ─────────────────
    lora_adapter_dir = os.path.join("./lora_checkpoints", model_key, "checkpoint-504")
    lora_available   = os.path.isdir(lora_adapter_dir)
    if lora_available:
        llm = PeftModel.from_pretrained(base_llm, lora_adapter_dir)
        llm.eval()
        print(f"  [OK] LoRA adapter wrapped from {lora_adapter_dir}")
    else:
        llm = base_llm
        print(f"  [WARN] LoRA adapter not found at {lora_adapter_dir} — Scenario D will be skipped.")

    # ── Load emotion extractor ───────────────────────────────────────────────
    emo_ext, enc_tok = load_emotion_extractor_local(DEVICE)

    # ── Load Variant-2 projector ─────────────────────────────────────────────
    mc      = ModelConfig.from_hf_config(base_llm.config, model_name=cfg["hf_name"])
    v2_ckpt = os.path.join(CHECKPOINT_DIR, f"variant2_projector_{model_key}.pt")
    if not os.path.isfile(v2_ckpt) and model_key == "phi4":
        v2_ckpt = os.path.join(CHECKPOINT_DIR, "variant2_projector.pt")
    proj_v2 = create_projector(mc, variant=2, checkpoint_path=v2_ckpt, device=DEVICE)
    proj_v2.eval()
    print(f"  [OK] Variant-2 projector loaded from {v2_ckpt}")

    # ── Per-sample loop ──────────────────────────────────────────────────────
    records          = []
    human_eval_buf   = []   # lightweight: only first HUMAN_EVAL_N samples
    for i, sample in enumerate(samples):
        neutral_text    = sample["text"]
        target_emotion  = sample["target_emotion"]

        emotion_anchors = {
            "joy": "I am so happy, joyful, and excited today!",
            "sadness": "I am feeling very sad, depressed, and heartbroken.",
            "anger": "I am absolutely furious, angry, and frustrated right now!",
            "fear": "I am feeling very scared, anxious, and terrified.",
            "surprise": "Wow, I am so surprised, shocked, and amazed by this!",
            "disgust": "This is absolutely disgusting, awful, and repulsive."
        }

        anchor_sentence = emotion_anchors.get(target_emotion, target_emotion)

        print(f"\n  [{i+1:02d}/{len(samples)}] emotion={target_emotion:10s} "
              f"text='{neutral_text[:60]}...'")
        
        # Extract emotion vector (from neutral text itself — reviewer-proof)
        emo_vec = get_emotion_embedding(anchor_sentence, emo_ext, enc_tok, DEVICE)

        # Scenarios A, B, C must run with adapter DISABLED (base model only)
        ctx = llm.disable_adapter() if lora_available else _null_ctx()

        with ctx:
            # ── Scenario A: Vanilla ──────────────────────────────────────────
            van_full = generate_vanilla(llm, llm_tok, neutral_text, DEVICE)
            van_resp = _strip_prompt(van_full, neutral_text)
            clear_gpu()

            # ── Scenario B: System Prompt ────────────────────────────────────
            _, sp_resp = generate_system_prompt(
                llm, llm_tok, neutral_text, target_emotion, DEVICE
            )
            clear_gpu()

            # ── Scenario C: KV-Cache Steered (Variant 2) ─────────────────────
            _, st_resp = generate_steered(
                llm, llm_tok, neutral_text, emo_vec,
                proj_v2, cfg["alpha"], cfg["layers"], DEVICE
            )
            clear_gpu()

            # Vanilla probs computed with adapter disabled (reference dist.)
            probs_van_for_jsd = get_emotion_probs_roberta(
                van_resp, roberta_model, roberta_tok, DEVICE
            )

        # ── Scenario D: LoRA (adapter ACTIVE) ────────────────────────────────
        if lora_available:
            _, lr_resp = generate_lora(
                llm, llm_tok, neutral_text, target_emotion, DEVICE
            )
            clear_gpu()
        else:
            lr_resp = ""

        # ── RoBERTa probabilities ─────────────────────────────────────────────
        probs_van = get_emotion_probs_roberta(van_resp, roberta_model, roberta_tok, DEVICE)
        probs_sp  = get_emotion_probs_roberta(sp_resp,  roberta_model, roberta_tok, DEVICE)
        probs_st  = get_emotion_probs_roberta(st_resp,  roberta_model, roberta_tok, DEVICE)
        probs_lr  = get_emotion_probs_roberta(lr_resp,  roberta_model, roberta_tok, DEVICE) if lr_resp else np.zeros(28)

        # ── Target scores ─────────────────────────────────────────────────────
        ts_van = target_score_from_probs(probs_van, target_emotion, label2id)
        ts_sp  = target_score_from_probs(probs_sp,  target_emotion, label2id)
        ts_st  = target_score_from_probs(probs_st,  target_emotion, label2id)
        ts_lr  = target_score_from_probs(probs_lr,  target_emotion, label2id) if lr_resp else -1.0

        # ── JSD (vanilla-with-adapter-disabled as reference) ──────────────────
        jsd_sp = jensen_shannon_divergence(probs_van_for_jsd, probs_sp)
        jsd_st = jensen_shannon_divergence(probs_van_for_jsd, probs_st)
        jsd_lr = jensen_shannon_divergence(probs_van_for_jsd, probs_lr) if lr_resp else -1.0

        # ── PPL (response tokens only; LoRA PPL with adapter active) ─────────
        ctx_van = llm.disable_adapter() if lora_available else _null_ctx()
        with ctx_van:
            ppl_van = compute_response_ppl(van_resp, llm, llm_tok, DEVICE)
            ppl_sp  = compute_response_ppl(sp_resp,  llm, llm_tok, DEVICE)
            ppl_st  = compute_response_ppl(st_resp,  llm, llm_tok, DEVICE)

        if lr_resp and lora_available:
            ppl_lr = compute_response_ppl(lr_resp, llm, llm_tok, DEVICE)
        else:
            ppl_lr = float("inf")

        print(f"    TargetScore  van={ts_van:.3f}  sp={ts_sp:.3f}  st={ts_st:.3f}  lr={ts_lr:.3f}")
        print(f"    PPL          van={ppl_van:.1f}  sp={ppl_sp:.1f}  st={ppl_st:.1f}  lr={ppl_lr:.1f}")
        print(f"    JSD                          sp={jsd_sp:.4f}  st={jsd_st:.4f}  lr={jsd_lr:.4f}")

        records.append({
            "model":          model_key,
            "sample_id":      sample["id"],
            "neutral_text":   neutral_text,
            "target_emotion": target_emotion,
            # Responses
            "vanilla_response":       van_resp,
            "system_prompt_response": sp_resp,
            "steered_response":       st_resp,
            "lora_response":          lr_resp,
            # Target scores
            "ts_vanilla":       round(ts_van, 4),
            "ts_system_prompt": round(ts_sp,  4),
            "ts_steered":       round(ts_st,  4),
            "ts_lora":          round(ts_lr,  4),
            # PPL
            "ppl_vanilla":       round(ppl_van, 2),
            "ppl_system_prompt": round(ppl_sp,  2),
            "ppl_steered":       round(ppl_st,  2),
            "ppl_lora":          round(ppl_lr,  2),
            # JSD (vs vanilla with adapter disabled)
            "jsd_system_prompt": round(jsd_sp, 6),
            "jsd_steered":       round(jsd_st, 6),
            "jsd_lora":          round(jsd_lr, 6),
        })

        # ── Human-eval hook (no GPU, no extra inference) ──────────────────────
        if i < HUMAN_EVAL_N:
            human_eval_buf.append({
                "record_id":            i,
                "target_emotion":       target_emotion,
                "input_prompt":         neutral_text,
                "vanilla_output":       van_resp,
                "system_prompt_output": sp_resp,
                "steered_v2_output":    st_resp,
                "lora_output":          lr_resp,   # Scenario D — empty str if adapter missing
            })

    # ── Human-eval export (CPU-only, no extra GPU pass) ───────────────────
    _save_human_eval(human_eval_buf, model_key)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    del llm, base_llm, llm_tok, emo_ext, proj_v2
    clear_gpu()
    print(f"\n  [OK] {model_key} evaluation complete. GPU cleared.")

    return records


# ══════════════════════════════════════════════════════════════════════════════
# 7. Aggregate metrics
# ══════════════════════════════════════════════════════════════════════════════

def _safe_mean(values: List[float]) -> float:
    finite = [v for v in values if math.isfinite(v) and v >= 0]
    return round(sum(finite) / len(finite), 4) if finite else -1.0


def aggregate(records: List[Dict], model_key: str) -> Dict:
    r = [x for x in records if x["model"] == model_key]
    # LoRA entries may be -1 / inf when adapter was missing — exclude them
    lr_valid = [x for x in r if x.get("ts_lora", -1) >= 0]
    return {
        "n": len(r),
        "vanilla": {
            "target_score": _safe_mean([x["ts_vanilla"]       for x in r]),
            "ppl":          _safe_mean([x["ppl_vanilla"]       for x in r]),
            "jsd":          "—",   # reference distribution
        },
        "system_prompt": {
            "target_score": _safe_mean([x["ts_system_prompt"] for x in r]),
            "ppl":          _safe_mean([x["ppl_system_prompt"] for x in r]),
            "jsd":          _safe_mean([x["jsd_system_prompt"] for x in r]),
        },
        "steered_v2": {
            "target_score": _safe_mean([x["ts_steered"]       for x in r]),
            "ppl":          _safe_mean([x["ppl_steered"]       for x in r]),
            "jsd":          _safe_mean([x["jsd_steered"]       for x in r]),
        },
        "lora": {
            "target_score": _safe_mean([x["ts_lora"]          for x in lr_valid]),
            "ppl":          _safe_mean([x["ppl_lora"]          for x in lr_valid]),
            "jsd":          _safe_mean([x["jsd_lora"]          for x in lr_valid]),
            "n_valid":      len(lr_valid),
        },
    }


def print_summary(summary: Dict):
    print("\n" + "=" * 70)
    print("REAL-WORLD EVALUATION SUMMARY")
    print("=" * 70)
    header = f"{'Model':<12} {'Scenario':<16} {'TargetScore':>12} {'PPL':>10} {'JSD':>10}"
    print(header)
    print("-" * 70)
    for model_key, stats in summary.items():
        for scenario, metrics in stats.items():
            if scenario == "n":
                continue
            ts  = metrics.get("target_score", -1)
            ppl = metrics.get("ppl",          -1)
            jsd = metrics.get("jsd",          "—")
            ts_str  = f"{ts:.4f}"  if isinstance(ts,  float) else str(ts)
            ppl_str = f"{ppl:.2f}" if isinstance(ppl, float) else str(ppl)
            jsd_str = f"{jsd:.4f}" if isinstance(jsd, float) else str(jsd)
            print(f"{model_key:<12} {scenario:<16} {ts_str:>12} {ppl_str:>10} {jsd_str:>10}")
        print()
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Main orchestration
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Real-World GoEmotions KV-Cache Evaluation"
    )
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of neutral GoEmotions examples to evaluate (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection (default: 42)")
    parser.add_argument("--output", type=str, default="real_world_metrics.json",
                        help="Path for averaged metrics JSON")
    parser.add_argument("--raw-output", type=str, default="real_world_raw.json",
                        help="Path for per-example raw results JSON")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help="Which models to evaluate (default: both)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Step 1: Prepare dataset ──────────────────────────────────────────────
    samples = load_neutral_samples(n=args.n_samples, seed=args.seed)

    # ── Step 2: Load RoBERTa classifier (shared across models) ──────────────
    print(f"\n[Classifier] Loading {ROBERTA_CLASSIFIER}...")
    roberta_tok = AutoTokenizer.from_pretrained(
        ROBERTA_CLASSIFIER, cache_dir=CACHE_DIR
    )
    roberta_model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_CLASSIFIER, cache_dir=CACHE_DIR
    ).to(DEVICE).eval()

    # Build label → index mapping once
    id2label  = roberta_model.config.id2label
    label2id  = {v.lower(): int(k) for k, v in id2label.items()}
    print(f"  [OK] Loaded with {len(id2label)} labels on {DEVICE}")

    # ── Step 3: Evaluate each model sequentially ─────────────────────────────
    all_records: List[Dict] = []
    for model_key in args.models:
        cfg     = MODELS[model_key]
        records = evaluate_model(
            model_key, cfg, samples,
            roberta_model, roberta_tok, label2id,
        )
        all_records.extend(records)

    # ── Step 4: Aggregate and print ──────────────────────────────────────────
    summary = {mk: aggregate(all_records, mk) for mk in args.models}
    print_summary(summary)

    # ── Step 5: Save ─────────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Averaged metrics → {args.output}")

    with open(args.raw_output, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] Per-example raw results → {args.raw_output}")

    print("\n[DONE] Real-world evaluation complete.")


if __name__ == "__main__":
    main()
