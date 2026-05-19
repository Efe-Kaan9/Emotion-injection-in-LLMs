"""
baseline_eval.py — Standalone System-Prompt Baseline Evaluation
================================================================

PURPOSE:
    Compares KV-Cache injection against a pure system-prompt baseline.
    Uses ONLY instruction-following ("Rewrite this sentence with [emotion]")
    WITHOUT any KV-Cache injection.

SCOPE:
    - 50 base prompts × 24 emotion labels = 1200 outputs per model
    - Both Phi-4-mini and Qwen2.5-1.5B are evaluated sequentially
    - Metrics: PPL, Target Score (RoBERTa), Dist-1/Dist-2, JSD vs Vanilla

OUTPUT:
    system_prompt_results.json  — completely independent from existing results

Usage:
    python baseline_eval.py
    python baseline_eval.py --model phi4
    python baseline_eval.py --model qwen2.5
    python baseline_eval.py --model all       # runs both sequentially (default)
"""

from __future__ import annotations

import os
import gc
import json
import math
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

# ──────────────────────────────────────────────────────────────
# Configuration  (mirrors existing project — DO NOT change)
# ──────────────────────────────────────────────────────────────

CACHE_DIR = "./.hf_cache"
EMOTION_CLASSIFIER = "SamLowe/roberta-base-go_emotions"

MODELS: Dict[str, str] = {
    "phi4":    "microsoft/Phi-4-mini-instruct",
    "qwen2.5": "Qwen/Qwen2.5-1.5B-Instruct",
}

MAX_NEW_TOKENS = 100
TEMPERATURE    = 0.7

# GoEmotions 28-label set (same order as the RoBERTa classifier)
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# ──────────────────────────────────────────────────────────────
# 10 base sentences (identical to the existing generation items)
# ──────────────────────────────────────────────────────────────

BASE_PROMPTS: List[Dict] = [
    # ── JOY (5 prompts) ────────────────────────────────────────────────
    {"prompt": "Tell me about the weather today.",
     "emotion_text": "I am so happy and excited!", "target_emotion": "joy"},
    {"prompt": "What are some fun things to do on a Saturday?",
     "emotion_text": "Life is wonderful and full of possibilities!", "target_emotion": "joy"},
    {"prompt": "Describe what a perfect morning looks like.",
     "emotion_text": "Everything is going great and I couldn't be happier!", "target_emotion": "joy"},
    {"prompt": "Tell me about your favourite season.",
     "emotion_text": "I feel pure bliss and delight every single day!", "target_emotion": "joy"},
    {"prompt": "How would you describe a good friendship?",
     "emotion_text": "I am overjoyed by the people around me!", "target_emotion": "joy"},

    # ── SADNESS (5 prompts) ─────────────────────────────────────────────
    {"prompt": "What should I have for dinner?",
     "emotion_text": "I feel really sad and lonely tonight.", "target_emotion": "sadness"},
    {"prompt": "What do you think about rainy days?",
     "emotion_text": "Everything feels hopeless and grey.", "target_emotion": "sadness"},
    {"prompt": "Describe what it means to grow old.",
     "emotion_text": "I miss the people who are no longer here.", "target_emotion": "sadness"},
    {"prompt": "What is the hardest thing about saying goodbye?",
     "emotion_text": "My heart aches and I feel completely broken.", "target_emotion": "sadness"},
    {"prompt": "Tell me about a time something ended.",
     "emotion_text": "I feel so empty and grief-stricken.", "target_emotion": "sadness"},

    # ── ANGER (5 prompts) ───────────────────────────────────────────────
    {"prompt": "Explain how computers work.",
     "emotion_text": "This makes me furious and I can't stand it!", "target_emotion": "anger"},
    {"prompt": "What do you think about traffic jams?",
     "emotion_text": "I am absolutely livid right now!", "target_emotion": "anger"},
    {"prompt": "How should workplaces handle unfairness?",
     "emotion_text": "I am outraged by the injustice I see!", "target_emotion": "anger"},
    {"prompt": "What is your opinion on people breaking rules?",
     "emotion_text": "This is completely unacceptable and enraging!", "target_emotion": "anger"},
    {"prompt": "Describe what happens when technology fails.",
     "emotion_text": "I am so frustrated I could scream!", "target_emotion": "anger"},

    # ── FEAR (5 prompts) ────────────────────────────────────────────────
    {"prompt": "What is the meaning of life?",
     "emotion_text": "I'm terrified of what might happen next.", "target_emotion": "fear"},
    {"prompt": "What do you think happens after death?",
     "emotion_text": "The unknown fills me with dread and terror.", "target_emotion": "fear"},
    {"prompt": "How do you deal with big decisions?",
     "emotion_text": "I am paralyzed with anxiety and fear of failing.", "target_emotion": "fear"},
    {"prompt": "Describe a dark, stormy night.",
     "emotion_text": "Every shadow makes my heart pound with fright.", "target_emotion": "fear"},
    {"prompt": "What is the scariest part of starting something new?",
     "emotion_text": "I am overwhelmed with panic and uncertainty.", "target_emotion": "fear"},

    # ── GRATITUDE (5 prompts) ───────────────────────────────────────────
    {"prompt": "Tell me a story about a cat.",
     "emotion_text": "I'm so grateful for everything you've done.", "target_emotion": "gratitude"},
    {"prompt": "What makes a good teacher?",
     "emotion_text": "I am so thankful for all the mentors in my life.", "target_emotion": "gratitude"},
    {"prompt": "How important is it to help others?",
     "emotion_text": "I deeply appreciate every act of kindness.", "target_emotion": "gratitude"},
    {"prompt": "Describe what home means to you.",
     "emotion_text": "I feel so blessed and thankful for everything I have.", "target_emotion": "gratitude"},
    {"prompt": "What do you appreciate most about nature?",
     "emotion_text": "My heart is full of gratitude for the beauty around me.", "target_emotion": "gratitude"},

    # ── CURIOSITY (5 prompts) ───────────────────────────────────────────
    {"prompt": "How does the internet work?",
     "emotion_text": "I'm so curious and want to learn everything!", "target_emotion": "curiosity"},
    {"prompt": "What is quantum computing?",
     "emotion_text": "I am fascinated and eager to understand how this works!", "target_emotion": "curiosity"},
    {"prompt": "Explain the theory of evolution.",
     "emotion_text": "My mind is buzzing with questions and wonder!", "target_emotion": "curiosity"},
    {"prompt": "What is at the bottom of the ocean?",
     "emotion_text": "I am endlessly curious about the mysteries of the deep!", "target_emotion": "curiosity"},
    {"prompt": "How are languages related to each other?",
     "emotion_text": "I find this topic endlessly fascinating and want to explore it!", "target_emotion": "curiosity"},

    # ── PRIDE (5 prompts) ───────────────────────────────────────────────
    {"prompt": "What makes a good leader?",
     "emotion_text": "I feel so proud of what I've accomplished.", "target_emotion": "pride"},
    {"prompt": "Describe the feeling of finishing a long project.",
     "emotion_text": "I worked so hard and I am immensely proud of myself!", "target_emotion": "pride"},
    {"prompt": "What does it mean to build something from scratch?",
     "emotion_text": "I feel a deep sense of achievement and dignity.", "target_emotion": "pride"},
    {"prompt": "Tell me about the importance of craftsmanship.",
     "emotion_text": "My work reflects my values and I stand by it with pride!", "target_emotion": "pride"},
    {"prompt": "What does winning feel like?",
     "emotion_text": "I am beaming with pride at what we have achieved!", "target_emotion": "pride"},

    # ── DISAPPOINTMENT (5 prompts) ──────────────────────────────────────
    {"prompt": "Suggest a weekend activity.",
     "emotion_text": "I'm really disappointed with how things turned out.", "target_emotion": "disappointment"},
    {"prompt": "What do you do when plans fall through?",
     "emotion_text": "I had such high hopes and now I just feel let down.", "target_emotion": "disappointment"},
    {"prompt": "Describe the feeling of expecting something that doesn't come.",
     "emotion_text": "I feel deflated and let down by the outcome.", "target_emotion": "disappointment"},
    {"prompt": "How do you recover when you fail an important exam?",
     "emotion_text": "I am so disheartened by my results.", "target_emotion": "disappointment"},
    {"prompt": "What is the hardest part of trusting someone?",
     "emotion_text": "I feel so let down and betrayed.", "target_emotion": "disappointment"},

    # ── LOVE (5 prompts) ────────────────────────────────────────────────
    {"prompt": "Describe a beautiful sunset.",
     "emotion_text": "I love you more than words can express.", "target_emotion": "love"},
    {"prompt": "What makes a relationship last?",
     "emotion_text": "I am deeply in love and feel completely at peace.", "target_emotion": "love"},
    {"prompt": "Write about what it feels like to hold a newborn baby.",
     "emotion_text": "My heart overflows with unconditional love.", "target_emotion": "love"},
    {"prompt": "Describe the bond between a parent and child.",
     "emotion_text": "I feel a warmth and tenderness that words cannot capture.", "target_emotion": "love"},
    {"prompt": "What does home mean to you?",
     "emotion_text": "Home is wherever I feel loved and cherished.", "target_emotion": "love"},

    # ── CONFUSION (5 prompts) ───────────────────────────────────────────
    {"prompt": "How do airplanes fly?",
     "emotion_text": "I'm confused and don't understand anything.", "target_emotion": "confusion"},
    {"prompt": "Explain what blockchain technology is.",
     "emotion_text": "I have no idea what is going on and feel totally lost.", "target_emotion": "confusion"},
    {"prompt": "What is the difference between machine learning and AI?",
     "emotion_text": "I am completely baffled and nothing makes sense to me.", "target_emotion": "confusion"},
    {"prompt": "Describe what happens during a software bug.",
     "emotion_text": "I am disoriented and cannot figure out the problem at all.", "target_emotion": "confusion"},
    {"prompt": "How do stock markets actually work?",
     "emotion_text": "The more I read, the more puzzled and overwhelmed I become.", "target_emotion": "confusion"},
]

# All 24 non-neutral emotion labels used for sweeping
TARGET_EMOTIONS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "remorse", "sadness",
]



# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def clear_gpu() -> None:
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) with epsilon-clipping."""
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JSD(P || Q) ∈ [0, ln2].  Symmetric, bounded."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def distinct_n(texts: List[str], n: int) -> float:
    """Dist-N = |unique n-grams| / |total n-grams|."""
    all_ngrams: List[Tuple] = []
    for text in texts:
        tokens = text.lower().split()
        all_ngrams += [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def compute_ppl(text: str, model, tokenizer, device: str, max_length: int = 512) -> float:
    """Perplexity of a single text using causal LM cross-entropy."""
    if not text.strip():
        return float("inf")
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    ids = enc["input_ids"]
    if ids.size(1) < 2:
        return float("inf")
    with torch.no_grad():
        loss = model(input_ids=ids, labels=ids).loss.item()
    return math.exp(loss)


# ──────────────────────────────────────────────────────────────
# Step 1: Generate system-prompt outputs + vanilla baselines
# ──────────────────────────────────────────────────────────────

def generate_all(model_key: str) -> List[Dict]:
    """
    For each (prompt × emotion) pair generate:
      - system_text  : LLM output steered via system prompt
      - vanilla_text : LLM output with no instruction (plain prompt only)

    Returns 240 records with raw text, ready for metric computation.
    """
    model_name = MODELS[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"  [GENERATE] {model_name}  |  device={device}")
    print(f"{'='*70}")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
    ).eval()
    print(f"  [OK] {model_name} loaded (4-bit NF4)")

    records  : List[Dict] = []
    record_id = 0
    total     = len(BASE_PROMPTS) * len(TARGET_EMOTIONS)

    for item in BASE_PROMPTS:
        prompt = item["prompt"]

        # --- Vanilla (no emotion instruction) ---
        vanilla_text = _generate_plain(model, tokenizer, prompt, device)
        torch.cuda.empty_cache()

        for emotion in TARGET_EMOTIONS:
            record_id += 1

            # System-prompt instruction
            sys_prompt = (
                f"Rewrite the following sentence conveying the emotion of {emotion}: {prompt}"
            )
            sys_text = _generate_plain(model, tokenizer, sys_prompt, device)
            torch.cuda.empty_cache()

            records.append({
                "record_id":     record_id,
                "model":         model_key,
                "prompt":        prompt,
                "target_emotion": emotion,
                "system_text":   sys_text,
                "vanilla_text":  vanilla_text,
            })

            if record_id % 20 == 0 or record_id == total:
                print(f"  [{record_id:3d}/{total}] emotion={emotion:15s} | prompt={prompt[:35]}...")

    del model, tokenizer
    clear_gpu()
    print(f"  [DONE] Generation complete — {len(records)} records")
    return records


def _generate_plain(model, tokenizer, prompt: str, device: str) -> str:
    """Simple greedy/sampling generation without KV manipulation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────
# Step 2: Emotion metrics (RoBERTa target score + JSD)
# ──────────────────────────────────────────────────────────────

def compute_emotion_metrics(records: List[Dict]) -> List[Dict]:
    """
    Loads RoBERTa GoEmotions classifier, computes for each record:
      - target_score   : P(target_emotion | system_text)
      - vanilla_score  : P(target_emotion | vanilla_text)
      - jsd            : JSD(vanilla_dist || system_dist)
    Model is deleted after this step.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print("  [STEP A] RoBERTa Emotion Scores + JSD")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(EMOTION_CLASSIFIER, cache_dir=CACHE_DIR)
    clf = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_CLASSIFIER, cache_dir=CACHE_DIR
    ).to(device).eval()

    id2label  = clf.config.id2label
    label2id  = {v.lower(): int(k) for k, v in id2label.items()}
    print(f"  [OK] RoBERTa loaded — {len(id2label)} labels on {device}")

    def classify(text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(len(id2label))
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            return torch.sigmoid(clf(**inputs).logits[0]).cpu().numpy()

    emotion_results: List[Dict] = []

    for i, rec in enumerate(records):
        p_sys     = classify(rec["system_text"])
        p_vanilla = classify(rec["vanilla_text"])
        emotion   = rec["target_emotion"].lower()

        idx            = label2id.get(emotion, -1)
        target_score   = float(p_sys[idx])     if idx >= 0 else -1.0
        vanilla_score  = float(p_vanilla[idx]) if idx >= 0 else -1.0
        jsd            = jensen_shannon_divergence(p_vanilla, p_sys)

        top3_idx = np.argsort(p_sys)[::-1][:3]
        top3 = [
            {"label": id2label.get(int(j), "?"), "score": round(float(p_sys[j]), 4)}
            for j in top3_idx
        ]

        emotion_results.append({
            "record_id":          rec["record_id"],
            "target_score":       round(target_score, 4),
            "vanilla_score":      round(vanilla_score, 4),
            "delta_target_score": round(target_score - vanilla_score, 4),
            "jsd":                round(jsd, 6),
            "top3_emotions":      top3,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(records):
            print(f"  [{i+1:3d}/{len(records)}] {emotion:15s} "
                  f"score={target_score:.4f}  JSD={jsd:.4f}")

    del clf, tok
    clear_gpu()
    print("  [OK] Step A complete — model deleted")
    return emotion_results


# ──────────────────────────────────────────────────────────────
# Step 3: Perplexity (PPL)
# ──────────────────────────────────────────────────────────────

def compute_ppl_metrics(records: List[Dict], model_key: str) -> List[Dict]:
    """
    Loads the target LLM (4-bit), computes PPL for system_text and vanilla_text.
    Model is deleted after this step.
    """
    model_name = MODELS[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  [STEP B] Perplexity — {model_name}")
    print(f"{'='*60}")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok   = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
    ).eval()
    print(f"  [OK] {model_name} loaded (4-bit NF4)")

    ppl_results: List[Dict] = []

    for i, rec in enumerate(records):
        ppl_sys     = compute_ppl(rec["system_text"],  model, tok, device)
        ppl_vanilla = compute_ppl(rec["vanilla_text"], model, tok, device)

        ppl_results.append({
            "record_id":   rec["record_id"],
            "ppl_system":  round(ppl_sys, 2),
            "ppl_vanilla": round(ppl_vanilla, 2),
            "ppl_delta":   round(ppl_sys - ppl_vanilla, 2),
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(records):
            print(f"  [{i+1:3d}/{len(records)}] "
                  f"PPL_sys={ppl_sys:.2f}  PPL_vanilla={ppl_vanilla:.2f}")

        if (i + 1) % 30 == 0:
            torch.cuda.empty_cache()

    del model, tok
    clear_gpu()
    print("  [OK] Step B complete — model deleted")
    return ppl_results


# ──────────────────────────────────────────────────────────────
# Step 4: Distinct-N (CPU, no model needed)
# ──────────────────────────────────────────────────────────────

def compute_distinct_metrics(records: List[Dict]) -> Dict:
    """
    Groups records by target_emotion and computes Dist-1 / Dist-2
    over system_text outputs. Returns aggregate summary.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        groups[rec["target_emotion"]].append(rec["system_text"])

    per_emotion: List[Dict] = []
    for emotion, texts in groups.items():
        per_emotion.append({
            "emotion": emotion,
            "n":       len(texts),
            "dist_1":  round(distinct_n(texts, 1), 4),
            "dist_2":  round(distinct_n(texts, 2), 4),
        })

    all_texts = [rec["system_text"] for rec in records]
    return {
        "overall_dist_1": round(distinct_n(all_texts, 1), 4),
        "overall_dist_2": round(distinct_n(all_texts, 2), 4),
        "per_emotion":    per_emotion,
    }


# ──────────────────────────────────────────────────────────────
# Step 5: Merge all metrics into one record list
# ──────────────────────────────────────────────────────────────

def merge_metrics(
    records:        List[Dict],
    emotion_mets:   List[Dict],
    ppl_mets:       List[Dict],
) -> List[Dict]:
    """Left-join on record_id."""
    em_map  = {r["record_id"]: r for r in emotion_mets}
    ppl_map = {r["record_id"]: r for r in ppl_mets}

    merged = []
    for rec in records:
        rid = rec["record_id"]
        row = {
            "record_id":      rid,
            "model":          rec["model"],
            "prompt":         rec["prompt"],
            "target_emotion": rec["target_emotion"],
            "system_text":    rec["system_text"],
            "vanilla_text":   rec["vanilla_text"],
        }
        if rid in em_map:
            row.update({k: v for k, v in em_map[rid].items() if k != "record_id"})
        if rid in ppl_map:
            row.update({k: v for k, v in ppl_map[rid].items() if k != "record_id"})
        merged.append(row)
    return merged


# ──────────────────────────────────────────────────────────────
# Aggregate summary (mean ± std per model)
# ──────────────────────────────────────────────────────────────

def build_summary(records: List[Dict]) -> Dict:
    """Compute mean ± std for the key numeric metrics."""
    def stats(vals: List[float]) -> Dict:
        arr = np.array([v for v in vals if v is not None and not np.isinf(v)])
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std":  round(float(np.std(arr)),  4),
            "n":    int(len(arr)),
        }

    return {
        "target_score":       stats([r.get("target_score",   0) for r in records]),
        "vanilla_score":      stats([r.get("vanilla_score",  0) for r in records]),
        "delta_target_score": stats([r.get("delta_target_score", 0) for r in records]),
        "jsd":                stats([r.get("jsd",            0) for r in records]),
        "ppl_system":         stats([r.get("ppl_system",     0) for r in records]),
        "ppl_vanilla":        stats([r.get("ppl_vanilla",    0) for r in records]),
        "ppl_delta":          stats([r.get("ppl_delta",      0) for r in records]),
    }


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def run_pipeline(model_key: str) -> Dict:
    """Full evaluation pipeline for one model. Returns result dict."""
    print(f"\n{'#'*70}")
    print(f"##  BASELINE EVAL — {model_key.upper():8s}  ({MODELS[model_key]})")
    print(f"{'#'*70}")

    # 1. Generate texts
    records = generate_all(model_key)

    # 2. Emotion metrics (loads & deletes RoBERTa)
    emotion_mets = compute_emotion_metrics(records)

    # 3. PPL metrics (loads & deletes target LLM)
    ppl_mets = compute_ppl_metrics(records, model_key)

    # 4. Distinct-N (CPU only)
    distinct_mets = compute_distinct_metrics(records)

    # 5. Merge
    merged = merge_metrics(records, emotion_mets, ppl_mets)

    # 6. Summary
    summary = build_summary(merged)
    summary["distinct"] = distinct_mets

    print(f"\n{'─'*60}")
    print(f"  SUMMARY for {model_key}")
    for k, v in summary.items():
        if k != "distinct":
            print(f"    {k:25s}: {v['mean']:.4f} ± {v['std']:.4f}")
    print(f"    {'overall_dist_1':25s}: {distinct_mets['overall_dist_1']}")
    print(f"    {'overall_dist_2':25s}: {distinct_mets['overall_dist_2']}")
    print(f"{'─'*60}\n")

    return {"records": merged, "summary": summary}


def main():
    parser = argparse.ArgumentParser(
        description="System-Prompt Baseline Evaluation (no KV-Cache injection)"
    )
    parser.add_argument(
        "--model",
        choices=["phi4", "qwen2.5", "all"],
        default="all",
        help="Which model(s) to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="system_prompt_results.json",
        help="Output JSON file (default: system_prompt_results.json)",
    )
    args = parser.parse_args()

    model_keys = ["phi4", "qwen2.5"] if args.model == "all" else [args.model]

    final_output: Dict = {}
    for mk in model_keys:
        final_output[mk] = run_pipeline(mk)
        clear_gpu()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"[DONE] Results saved → {args.output}")
    print(f"       Models evaluated: {model_keys}")
    print(f"       Records per model: 240  (10 prompts × 24 emotions)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
