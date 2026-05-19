""" table 4
eval_mechanistic_clean.py
=========================
Response-only mechanistic evaluation script.

Re-calculates Target Score, JSD, and Perplexity (PPL) for ALL methods
(Vanilla, Variant1, Variant2, SystemPrompt, LoRA) using ONLY the clean
response text – not the echo of the input prompt.

Usage:
    python eval_mechanistic_clean.py --model phi4   
    python eval_mechanistic_clean.py --model qwen2.5

Output:
    - Prints a Table-4-style summary to the terminal
    - Saves table4_clean_metrics_phi4.json (or qwen2.5) alongside this script

NOTE: All source JSON files are opened read-only. Nothing is modified.
"""

import argparse
import gc
import json
import math
import os

# pyrefly: ignore [missing-import]
import torch
from scipy.spatial.distance import jensenshannon
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import AutoModelForCausalLM

# --- PHI-4 REMOTE CODE IMPORT HATASI İÇİN DİNAMİK YAMALAR ---
import transformers.cache_utils
if not hasattr(transformers.cache_utils, "SlidingWindowCache"):
    class SlidingWindowCache:
        pass
    transformers.cache_utils.SlidingWindowCache = SlidingWindowCache

import transformers.utils
if not hasattr(transformers.utils, "LossKwargs"):
    from typing import TypedDict
    class LossKwargs(TypedDict, total=False):
        pass
    transformers.utils.LossKwargs = LossKwargs
# ---------------------------------------------------------

# ─────────────────────────── constants ────────────────────────────────────────

MODEL_CONFIGS = {
    "phi4": {
        "hf_name": "microsoft/Phi-4-mini-instruct",
        "mech_json": "eval_mechanistic_phi4.json",
        "gen_json":  "generation_results_phi4.json",
        "lora_json": "lora_generation_results_phi4.json",
        "sp_key":    "phi4",
        # Optimal ablation settings for the Core 50 Set (Variant 2)
        "core_alpha":  1.0,
        "core_layer":  "all",
    },
    "qwen2.5": {
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "mech_json": "eval_mechanistic_qwen2.5.json",
        "gen_json":  "generation_results_qwen2.5.json",
        "lora_json": "lora_generation_results_qwen.json",
        "sp_key":    "qwen2.5",
        # Optimal ablation settings for the Core 50 Set (Variant 2)
        "core_alpha":  1.5,
        "core_layer":  "second_half",
    },
}

ROBERTA_MODEL = "SamLowe/roberta-base-go_emotions"
EMOTION_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise","neutral",
]

# ─────────────────────────── text cleaning ────────────────────────────────────

def clean_generation(raw_text: str, prompt_text: str, emotion_text: str) -> str:
    """
    Strip any echoed instruction prefix from the generated text, leaving only
    the model's actual response. Handles both UPPER and title-case emotion.
    This is the canonical function provided in the task specification.
    """
    if not isinstance(raw_text, str):
        return ""
    t = raw_text.strip()
    p = str(prompt_text).strip()
    e = str(emotion_text).strip()

    t1 = f"Rewrite the following sentence conveying the emotion of {e.upper()}: {p}"
    t2 = f"Rewrite the following sentence conveying the emotion of {e}: {p}"

    if t.startswith(t1):
        t = t[len(t1):].strip()
    elif t.startswith(t2):
        t = t[len(t2):].strip()
    if t.startswith(p):
        t = t[len(p):].strip()

    return t.strip()

# ─────────────────────────── data loading ─────────────────────────────────────

def load_core_set(cfg: dict) -> list[dict]:
    """
    Read eval_mechanistic_*.json and filter to exactly the 50-sample Core Set
    that corresponds to Variant 2 at the optimal alpha and layer_config.
    Returns a list of dicts, each with keys: prompt, target_emotion, emotion_text.
    """
    with open(cfg["mech_json"], "r", encoding="utf-8") as f:
        mech = json.load(f)

    records = mech["records"]
    core = [
        r for r in records
        if str(r.get("variant")) == "2"
        and float(r.get("alpha", -1)) == cfg["core_alpha"]
        and r.get("layer_config") == cfg["core_layer"]
    ]
    if not core:
        raise ValueError(
            f"Core Set filter returned 0 records. "
            f"Check alpha={cfg['core_alpha']}, layer={cfg['core_layer']}, variant=2 "
            f"in {cfg['mech_json']}"
        )
    print(f"  Core Set size: {len(core)} records (Var2, alpha={cfg['core_alpha']}, layer={cfg['core_layer']})")
    # Build a lookup set of (prompt, target_emotion) tuples for fast membership test
    return core


def build_core_index(core: list[dict]) -> set[tuple]:
    return {(r["prompt"], r["target_emotion"]) for r in core}


def load_all_methods(cfg: dict, core_index: set[tuple]) -> dict[str, list[dict]]:
    """
    Load and filter all generation sources to the Core Set.
    Returns a dict keyed by method name with lists of cleaned records.
    Each record: {prompt, target_emotion, emotion_text, clean_text, clean_vanilla}
    """
    results: dict[str, list[dict]] = {
        "vanilla":    [],
        "variant1":   [],
        "variant2":   [],
        "system_prompt": [],
        "lora":       [],
    }

    # ── generation_results (Var1 + Var2 + their vanilla baseline) ──────────
    with open(cfg["gen_json"], "r", encoding="utf-8") as f:
        gen_records = json.load(f)

    for r in gen_records:
        key = (r["prompt"], r["target_emotion"])
        if key not in core_index:
            continue

        # 🚨 HAYAT KURTARAN DÜZELTME: Sadece Şampiyon Parametreleri (50 adet) al!
        # Yoksa 600 tane ablasyon kaydını ortalamaya sokar ve metrikleri patlatır.
        variant = str(r.get("variant", ""))
        if variant in ["1", "2"]:
            r_alpha = float(r.get("alpha", -1))
            r_layer = r.get("layer_config", "")
            if r_alpha != cfg["core_alpha"] or r_layer != cfg["core_layer"]:
                continue

        emotion_text = r.get("emotion_text", "")

        emotion_text = r.get("emotion_text", "")
        clean_v = clean_generation(r.get("vanilla_text", ""), r["prompt"], emotion_text)
        clean_g = clean_generation(r.get("generated_text", ""), r["prompt"], emotion_text)

        base = {
            "prompt":         r["prompt"],
            "target_emotion": r["target_emotion"],
            "emotion_text":   emotion_text,
            "clean_text":     clean_g,
            "clean_vanilla":  clean_v,
        }

        variant = str(r.get("variant", ""))
        if variant == "1":
            results["variant1"].append(base)
        elif variant == "2":
            results["variant2"].append(base)

    # Deduplicate vanilla from variant2 records (same prompts)
    seen_vanilla: set[tuple] = set()
    for r in gen_records:
        key = (r["prompt"], r["target_emotion"])
        if key not in core_index or key in seen_vanilla:
            continue
        if str(r.get("variant")) == "2":
            emotion_text = r.get("emotion_text", "")
            clean_v = clean_generation(r.get("vanilla_text", ""), r["prompt"], emotion_text)
            results["vanilla"].append({
                "prompt":         r["prompt"],
                "target_emotion": r["target_emotion"],
                "emotion_text":   emotion_text,
                "clean_text":     clean_v,
                "clean_vanilla":  clean_v,
            })
            seen_vanilla.add(key)

    # ── system_prompt_results ─────────────────────────────────────────────
    with open("system_prompt_results.json", "r", encoding="utf-8") as f:
        sp_all = json.load(f)

    sp_records = sp_all[cfg["sp_key"]]["records"]
    for r in sp_records:
        key = (r["prompt"], r["target_emotion"])
        if key not in core_index:
            continue
        emotion_text = r.get("target_emotion", "")   # SP records have no emotion_text
        clean_g = clean_generation(r.get("system_text", ""), r["prompt"], emotion_text)
        clean_v = clean_generation(r.get("vanilla_text", ""), r["prompt"], emotion_text)
        results["system_prompt"].append({
            "prompt":         r["prompt"],
            "target_emotion": r["target_emotion"],
            "emotion_text":   emotion_text,
            "clean_text":     clean_g,
            "clean_vanilla":  clean_v,
        })

    # ── lora_generation_results ───────────────────────────────────────────
    with open(cfg["lora_json"], "r", encoding="utf-8") as f:
        lora_records = json.load(f)

    for r in lora_records:
        key = (r["prompt"], r["target_emotion"])
        if key not in core_index:
            continue
        # LoRA records use target_emotion (e.g. 'joy') as the emotion in the template,
        # NOT the emotion_text sentence. Use target_emotion for both cleaning passes.
        lora_emotion = r.get("target_emotion", "")
        clean_g = clean_generation(r.get("generated_text", ""), r["prompt"], lora_emotion)
        clean_v = clean_generation(r.get("vanilla_text", ""), r["prompt"], lora_emotion)
        results["lora"].append({
            "prompt":         r["prompt"],
            "target_emotion": r["target_emotion"],
            "emotion_text":   lora_emotion,
            "clean_text":     clean_g,
            "clean_vanilla":  clean_v,
        })

    for method, recs in results.items():
        print(f"  {method:15s}: {len(recs)} records after filtering")

    return results

# ─────────────────────────── Step A: RoBERTa ──────────────────────────────────

def run_roberta(all_methods: dict[str, list[dict]]) -> dict[str, dict]:
    """
    Load RoBERTa once, compute Target Score and JSD for every method.
    Returns per-method dicts: {target_scores: [...], jsds: [...]}
    """
    print("\n[Step A] Loading RoBERTa classifier …")
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline(
        "text-classification",
        model=ROBERTA_MODEL,
        top_k=None,
        device=device,
        truncation=True,
        max_length=512,
    )

    def get_raw_dist(text: str) -> list[float]:
        """Return raw 28-dim sigmoid probabilities without artificial normalization."""
        # EĞER METİN BOŞSA (Mode collapse), NaN hatasını önlemek için boşluk karakteri ver
        if not text or str(text).strip() == "":
            text = " "
            
        preds = clf(text[:1024])[0]
        label2score = {p["label"]: p["score"] for p in preds}
        return [label2score.get(lbl, 0.0) for lbl in EMOTION_LABELS]

    method_scores: dict[str, dict] = {}

    for method, records in all_methods.items():
        print(f"  Scoring {method} …", end=" ", flush=True)
        target_scores, jsds = [], []

        for rec in records:
            emotion = rec["target_emotion"].lower()
            try:
                emo_idx = EMOTION_LABELS.index(emotion)
            except ValueError:
                emo_idx = -1

            dist_gen     = get_raw_dist(rec["clean_text"])
            dist_vanilla = get_raw_dist(rec["clean_vanilla"])

            ts  = dist_gen[emo_idx] if emo_idx >= 0 else 0.0
            jsd = float(jensenshannon(dist_gen, dist_vanilla) ** 2)   # JSD (squared)

            target_scores.append(ts)
            jsds.append(jsd)

        method_scores[method] = {
            "target_scores": target_scores,
            "jsds":          jsds,
        }
        avg_ts  = sum(target_scores) / len(target_scores) if target_scores else 0.0
        avg_jsd = sum(jsds) / len(jsds) if jsds else 0.0
        print(f"TargetScore={avg_ts:.4f}  JSD={avg_jsd:.4f}")

    # Clean up GPU memory
    del clf
    gc.collect()
    torch.cuda.empty_cache()
    return method_scores

# ─────────────────────────── Step B: Perplexity ───────────────────────────────

def calc_ppl_for_texts(model, tokenizer, texts: list[str]) -> list[float]:
    """
    Compute per-text PPL using the base LLM on response-only token sequences.
    PPL = exp(cross-entropy loss over the whole response).
    """
    ppls = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            if not text or not text.strip():
                ppls.append(float("inf"))
                continue
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_ids = enc["input_ids"].to(model.device)
            if input_ids.shape[1] < 2:
                ppls.append(float("inf"))
                continue
            try:
                out = model(input_ids=input_ids, labels=input_ids)
                ppl = math.exp(out.loss.item())
            except Exception:
                ppl = float("inf")
            ppls.append(ppl)
    return ppls


def run_perplexity(cfg: dict, all_methods: dict[str, list[dict]]) -> dict[str, list[float]]:
    """
    Load the 4-bit base LLM once, compute PPL for every method's clean texts.
    """
    print(f"\n[Step B] Loading base LLM ({cfg['hf_name']}) in 4-bit …")
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"], cache_dir="./.hf_cache")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_name"],
        cache_dir="./.hf_cache",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    ppl_results: dict[str, list[float]] = {}
    for method, records in all_methods.items():
        texts = [r["clean_text"] for r in records]
        print(f"  Computing PPL for {method} ({len(texts)} texts) …", end=" ", flush=True)
        ppls = calc_ppl_for_texts(model, tokenizer, texts)
        # Report median (robust to outlier explosions) alongside mean
        valid = [p for p in ppls if math.isfinite(p)]
        if valid:
            valid_sorted = sorted(valid)
            n = len(valid_sorted)
            median = valid_sorted[n // 2] if n % 2 else (valid_sorted[n//2-1]+valid_sorted[n//2])/2
            avg    = sum(valid) / len(valid)
        else:
            median = avg = float("inf")
        print(f"mean={avg:.2f}  median={median:.2f}  ({len(valid)}/{len(ppls)} valid)")
        ppl_results[method] = ppls

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return ppl_results

# ─────────────────────────── aggregation & output ─────────────────────────────

def safe_mean(values: list[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    return sum(valid) / len(valid) if valid else float("nan")

def safe_median(values: list[float]) -> float:
    valid = sorted(v for v in values if math.isfinite(v))
    if not valid:
        return float("nan")
    n = len(valid)
    return valid[n // 2] if n % 2 else (valid[n//2-1]+valid[n//2])/2


def build_table(
    model_name: str,
    all_methods: dict[str, list[dict]],
    roberta_scores: dict[str, dict],
    ppl_results: dict[str, list[float]],
) -> dict:
    """Aggregate and print the Table 4 summary."""

    METHOD_ORDER = ["vanilla", "system_prompt", "lora", "variant1", "variant2"]
    METHOD_LABELS = {
        "vanilla":       "Vanilla (no steering)",
        "system_prompt": "System Prompt",
        "lora":          "LoRA Baseline",
        "variant1":      "KV-Inject Var 1 (Linear)",
        "variant2":      "KV-Inject Var 2 (Gated)  ← Ours",
    }

    rows = []
    for m in METHOD_ORDER:
        if m not in all_methods or not all_methods[m]:
            continue
        ts_list  = roberta_scores[m]["target_scores"]
        jsd_list = roberta_scores[m]["jsds"]
        ppl_list = ppl_results[m]

        row = {
            "method":           m,
            "n":                len(ts_list),
            "target_score_mean": safe_mean(ts_list),
            "jsd_mean":          safe_mean(jsd_list),
            "ppl_mean":          safe_mean(ppl_list),
            "ppl_median":        safe_median(ppl_list),
        }
        rows.append(row)

    # ── Print formatted table ──────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"  TABLE 4 — Clean Response-Only Metrics  [{model_name.upper()}]  (Core Set = {rows[0]['n'] if rows else 0} samples)")
    print("=" * 90)
    hdr = f"{'Method':<38} {'N':>4}  {'Target↑':>10}  {'JSD↑':>8}  {'PPL↓(mean)':>12}  {'PPL↓(med)':>11}"
    print(hdr)
    print("-" * 90)
    for row in rows:
        label = METHOD_LABELS.get(row["method"], row["method"])
        print(
            f"  {label:<36} {row['n']:>4}  "
            f"{row['target_score_mean']:>10.4f}  "
            f"{row['jsd_mean']:>8.4f}  "
            f"{row['ppl_mean']:>12.2f}  "
            f"{row['ppl_median']:>11.2f}"
        )
    print("=" * 90)

    output = {
        "model": model_name,
        "note":  "Response-only evaluation. Metrics computed on clean text (prompt prefix stripped).",
        "rows":  rows,
    }
    return output


def save_results(output: dict, model_name: str):
    fname = f"table4_clean_metrics_{model_name}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {fname}")


# ─────────────────────────── main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Response-only mechanistic evaluation")
    parser.add_argument(
        "--model", required=True, choices=list(MODEL_CONFIGS.keys()),
        help="Which model to evaluate: phi4 or qwen2.5"
    )
    args = parser.parse_args()
    model_name = args.model
    cfg = MODEL_CONFIGS[model_name]

    print(f"\n{'='*60}")
    print(f"  eval_mechanistic_clean.py  —  model: {model_name}")
    print(f"{'='*60}")

    # ── Step 0: Build Core Set ─────────────────────────────────────────────
    print("\n[Step 0] Identifying 50-sample Core Set …")
    core = load_core_set(cfg)
    core_index = build_core_index(core)

    # ── Step 0b: Load and filter all methods ──────────────────────────────
    print("\n[Step 0b] Loading & filtering all generation sources …")
    all_methods = load_all_methods(cfg, core_index)

    # ── Step A: RoBERTa (Target Score + JSD) ──────────────────────────────
    roberta_scores = run_roberta(all_methods)

    # ── Step B: Perplexity ────────────────────────────────────────────────
    ppl_results = run_perplexity(cfg, all_methods)

    # ── Aggregate & print ─────────────────────────────────────────────────
    output = build_table(model_name, all_methods, roberta_scores, ppl_results)

    # ── Save ──────────────────────────────────────────────────────────────
    save_results(output, model_name)


if __name__ == "__main__":
    main()
