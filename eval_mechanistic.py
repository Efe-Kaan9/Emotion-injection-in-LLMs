"""
eval_mechanistic.py — Phase 3: GPU-Based Sequential Evaluation
===============================================================

Two-step sequential evaluation on RTX 3060 (6 GB VRAM):

  Step A — Emotion Control + JSD
    • Load SamLowe/roberta-base-go_emotions
    • For each record: compute target emotion score + JSD(vanilla ‖ steered)
    • Delete model, clear VRAM

  Step B — Perplexity (PPL)
    • Load the base LLM (phi-4-mini or qwen2.5)
    • For each record: compute PPL of steered text
    • Delete model, clear VRAM

  Final: merge results and save.

Usage:
    python eval_mechanistic.py --input generation_results_phi4.json --model phi4
    python eval_mechanistic.py --input generation_results_qwen2.5.json --model qwen2.5
"""

from __future__ import annotations

import os
import sys
import json
import gc
import argparse
import math
from typing import List, Dict

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModel,
    BitsAndBytesConfig,
)
from scipy.spatial.distance import cosine

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

CACHE_DIR = "./.hf_cache"
EMOTION_CLASSIFIER = "SamLowe/roberta-base-go_emotions"

MODELS = {
    "phi4": "microsoft/Phi-4-mini-instruct",
    "qwen2.5": "Qwen/Qwen2.5-1.5B-Instruct",
}

# GoEmotions label mapping (28 labels — same order as the classifier)
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# DistilBERT EmotionExtractor
ENCODER_NAME = "distilbert-base-uncased"
NUM_EMOTIONS = 28

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


def load_distilbert_extractor(device: str):
    """Load the fine-tuned DistilBERT EmotionExtractor from checkpoint.pt.

    Uses [CLS] pooling to match DistilBertFineTune.ipynb.
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
        print("  [OK] DistilBERT EmotionExtractor loaded from checkpoint.pt (CLS pooling)")
    else:
        print("  [WARN] checkpoint.pt not found — using untrained EmotionExtractor!")

    return emo_ext, enc_tok

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    '''Return cosine similarity between two vectors (1 - cosine distance).'''
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return 1.0 - float(cosine(vec1, vec2))


def clear_gpu():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ──────────────────────────────────────────────────────────────
# Jensen–Shannon Divergence (CPU, numpy)
# ──────────────────────────────────────────────────────────────

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) with epsilon for numerical stability."""
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Returns a value in [0, ln(2)] ≈ [0, 0.693].
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# ──────────────────────────────────────────────────────────────
# Step A: Emotion Classification + JSD
# ──────────────────────────────────────────────────────────────

def step_a_emotion_jsd(records: List[Dict]) -> List[Dict]:
    """
    Load GoEmotions classifier, compute:
      - target_emotion_score: P(target emotion | steered text)
      - vanilla_emotion_score: P(target emotion | vanilla text)
      - jsd: JSD between full probability distributions of vanilla vs steered
    """
    print("\n" + "=" * 60)
    print("STEP A: Emotion Classification + JSD")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  Loading {EMOTION_CLASSIFIER}...")
    tokenizer = AutoTokenizer.from_pretrained(
        EMOTION_CLASSIFIER, cache_dir=CACHE_DIR
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_CLASSIFIER, cache_dir=CACHE_DIR
    ).to(device).eval()

    # Build label → index mapping from model config
    id2label = model.config.id2label
    label2id = {v.lower(): int(k) for k, v in id2label.items()}
    print(f"  [OK] Loaded classifier with {len(id2label)} labels on {device}")

    results = []

    for i, rec in enumerate(records):
        target_emotion = rec.get("target_emotion", "neutral").lower()
        emotion_text = rec.get("emotion_text", "")
        steered_text = rec.get("generated_text", "")
        vanilla_text = rec.get("vanilla_text", "")

        # Classify emotion_text (reference)
        with torch.no_grad():
            inputs_ref = tokenizer(emotion_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            probs_ref = torch.sigmoid(model(**inputs_ref).logits[0]).cpu().numpy()

        # Classify steered text
        with torch.no_grad():
            inputs_s = tokenizer(steered_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            probs_s = torch.sigmoid(model(**inputs_s).logits[0]).cpu().numpy()

        # Classify vanilla text
        with torch.no_grad():
            inputs_v = tokenizer(vanilla_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            probs_v = torch.sigmoid(model(**inputs_v).logits[0]).cpu().numpy()

        # Target emotion score
        target_idx = label2id.get(target_emotion, -1)
        if target_idx >= 0 and target_idx < len(probs_s):
            steered_score = float(probs_s[target_idx])
            vanilla_score = float(probs_v[target_idx])
        else:
            steered_score = -1.0
            vanilla_score = -1.0

        # JSD between vanilla and steered distributions
        jsd = jensen_shannon_divergence(probs_v, probs_s)

        # RoBERTa Cosine Similarity (Input vs Vanilla/Steered)
        sim_v = compute_cosine_similarity(probs_ref, probs_v)
        sim_s = compute_cosine_similarity(probs_ref, probs_s)

        # Top-3 emotions for steered output
        top3_idx = np.argsort(probs_s)[::-1][:3]
        top3 = [
            {"label": id2label.get(int(idx), "?"), "score": round(float(probs_s[idx]), 4)}
            for idx in top3_idx
        ]

        results.append({
            "record_id": rec.get("record_id", i),
            "target_emotion": target_emotion,
            "steered_target_score": round(steered_score, 4),
            "vanilla_target_score": round(vanilla_score, 4),
            "delta_target_score": round(steered_score - vanilla_score, 4),
            "jsd": round(jsd, 6),
            "roberta_sim_vanilla": round(sim_v, 4),
            "roberta_sim_steered": round(sim_s, 4),
            "roberta_sim_delta": round(sim_s - sim_v, 4),
            "steered_top3": top3,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            print(f"  [{i+1:3d}/{len(records)}] target={target_emotion:15s} "
                  f"score={steered_score:.4f} JSD={jsd:.4f}")

    # Cleanup
    del model, tokenizer
    clear_gpu()
    print("  [OK] Step A complete -- model deleted, VRAM cleared")

    return results

# ──────────────────────────────────────────────────────────────
# Step A.2: DistilBERT Cosine Similarity
# ──────────────────────────────────────────────────────────────

def step_a2_distilbert_sim(records: List[Dict]) -> List[Dict]:
    """Load your fine-tuned DistilBERT EmotionExtractor and compute Cosine Similarities."""
    print("\n" + "=" * 60)
    print("STEP A.2: DistilBERT Cosine Similarity")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"  Loading DistilBERT EmotionExtractor on {device}...")
    model, tokenizer = load_distilbert_extractor(device)

    results = []

    for i, rec in enumerate(records):
        emotion_text = rec.get("emotion_text", "")
        steered_text = rec.get("generated_text", "")
        vanilla_text = rec.get("vanilla_text", "")

        def get_vec(text):
            if not text.strip():
                return np.zeros(NUM_EMOTIONS)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                return torch.sigmoid(model(**inputs)[0]).cpu().numpy()

        vec_ref = get_vec(emotion_text)
        vec_s = get_vec(steered_text)
        vec_v = get_vec(vanilla_text)

        sim_v = compute_cosine_similarity(vec_ref, vec_v)
        sim_s = compute_cosine_similarity(vec_ref, vec_s)

        results.append({
            "record_id": rec.get("record_id", i),
            "distilbert_sim_vanilla": round(sim_v, 4),
            "distilbert_sim_steered": round(sim_s, 4),
            "distilbert_sim_delta": round(sim_s - sim_v, 4),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            print(f"  [{i+1:3d}/{len(records)}] DistilBERT Delta: {sim_s - sim_v:+.4f}")

    del model, tokenizer
    clear_gpu()
    print("  [OK] Step A.2 complete -- model deleted, VRAM cleared")

    return results


# ──────────────────────────────────────────────────────────────
# Step B: Perplexity
# ──────────────────────────────────────────────────────────────

def step_b_perplexity(records: List[Dict], model_key: str) -> List[Dict]:
    """
    Load the base LLM and compute perplexity of steered text.

    PPL = exp(cross-entropy loss) where the model predicts each token
    given all previous tokens (causal LM objective).
    """
    print("\n" + "=" * 60)
    print("STEP B: Perplexity (PPL)")
    print("=" * 60)

    model_name = MODELS[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  Loading {model_name} (4-bit)...")
    bnb_config = BitsAndBytesConfig(
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
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"  [OK] {model_name} loaded on {device}")

    results = []

    for i, rec in enumerate(records):
        steered_text = rec.get("generated_text", "")
        vanilla_text = rec.get("vanilla_text", "")

        # PPL for steered text
        ppl_steered = _compute_ppl(steered_text, model, tokenizer, device)
        ppl_vanilla = _compute_ppl(vanilla_text, model, tokenizer, device)

        results.append({
            "record_id": rec.get("record_id", i),
            "ppl_steered": round(ppl_steered, 2),
            "ppl_vanilla": round(ppl_vanilla, 2),
            "ppl_delta": round(ppl_steered - ppl_vanilla, 2),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            print(f"  [{i+1:3d}/{len(records)}] "
                  f"PPL_steered={ppl_steered:.2f} PPL_vanilla={ppl_vanilla:.2f}")

        # Periodic cache cleanup
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    # Cleanup
    del model, tokenizer
    clear_gpu()
    print("  [OK] Step B complete -- model deleted, VRAM cleared")

    return results


def _compute_ppl(
    text: str,
    model,
    tokenizer,
    device: str,
    max_length: int = 512,
) -> float:
    """Compute perplexity of a single text using causal LM loss."""
    if not text.strip():
        return float("inf")

    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_ids = encodings["input_ids"]

    if input_ids.size(1) < 2:
        return float("inf")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss.item()

    return math.exp(loss)


# ──────────────────────────────────────────────────────────────
# Main: run A → B sequentially, merge results
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: GPU sequential evaluation (JSD + PPL)"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to generation results JSON from Phase 1",
    )
    parser.add_argument(
        "--model", choices=list(MODELS.keys()), default="phi4",
        help="Which LLM was used for generation (for PPL computation)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: eval_mechanistic_results.json)",
    )
    parser.add_argument(
        "--skip-a", action="store_true",
        help="Skip Step A (emotion + JSD)",
    )
    parser.add_argument(
        "--skip-b", action="store_true",
        help="Skip Step B (perplexity)",
    )
    args = parser.parse_args()

    output_path = args.output or "eval_mechanistic_results.json"

    # Load records
    print(f"Loading data from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records")

    # Step A (RoBERTa Emotion + JSD)
    if not args.skip_a:
        emotion_results = step_a_emotion_jsd(records)
        distilbert_results = step_a2_distilbert_sim(records)
        # Merge DistilBERT results directly into emotion_results
        for e_res, d_res in zip(emotion_results, distilbert_results):
            e_res.update({k: v for k, v in d_res.items() if k != "record_id"})
    else:
        emotion_results = [{"record_id": r.get("record_id", i)} for i, r in enumerate(records)]
        print("\n  [SKIPPED] Step A (emotion + JSD)")

    # Step B
    if not args.skip_b:
        ppl_results = step_b_perplexity(records, args.model)
    else:
        ppl_results = [{"record_id": r.get("record_id", i)} for i, r in enumerate(records)]
        print("\n  [SKIPPED] Step B (perplexity)")

    # Merge into original records
    merged = []
    for rec, emo, ppl in zip(records, emotion_results, ppl_results):
        merged_rec = {**rec}
        merged_rec.update({k: v for k, v in emo.items() if k != "record_id"})
        merged_rec.update({k: v for k, v in ppl.items() if k != "record_id"})
        merged.append(merged_rec)

    # Overall summary
    valid_emo = [r for r in merged if r.get("steered_target_score", -1) >= 0]
    valid_ppl = [r for r in merged if r.get("ppl_steered", float("inf")) < float("inf")]

    summary = {}
    if valid_emo:
        summary["avg_steered_target_score"] = round(
            sum(r["steered_target_score"] for r in valid_emo) / len(valid_emo), 4
        )
        summary["avg_vanilla_target_score"] = round(
            sum(r["vanilla_target_score"] for r in valid_emo) / len(valid_emo), 4
        )
        summary["avg_jsd"] = round(
            sum(r["jsd"] for r in valid_emo) / len(valid_emo), 6
        )
        summary["avg_roberta_sim_delta"] = round(
            sum(r.get("roberta_sim_delta", 0.0) for r in valid_emo) / len(valid_emo), 4
        )
        summary["avg_distilbert_sim_delta"] = round(
            sum(r.get("distilbert_sim_delta", 0.0) for r in valid_emo) / len(valid_emo), 4
        )
    if valid_ppl:
        summary["avg_ppl_steered"] = round(
            sum(r["ppl_steered"] for r in valid_ppl) / len(valid_ppl), 2
        )
        summary["avg_ppl_vanilla"] = round(
            sum(r["ppl_vanilla"] for r in valid_ppl) / len(valid_ppl), 2
        )

    # Print summary
    print("\n" + "=" * 60)
    print("MECHANISTIC EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:35s}: {v}")
    print("=" * 60)

    # Save
    output_data = {
        "summary": summary,
        "records": merged,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved {len(merged)} merged records to {output_path}")


if __name__ == "__main__":
    main()
