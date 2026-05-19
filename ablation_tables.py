"""
ablation_tables.py
=========================
Re-calculates Ablation metrics (Alpha and Layer) STRICTLY on response-only tokens.
Groups 600 Variant 2 samples and outputs Table 9 and Table 10 directly.
python ablation_tables.py --model phi4

=============================================================
  ABLATION RESULTS [Clean Response-Only] - PHI4
=============================================================

--- TABLE 9: Alpha Ablation (Averaged across layers) ---
 alpha  Target_Score     PPL    JSD
0.5000        0.0976  7.5081 0.2149
1.0000        0.0967  5.5686 0.2042
1.5000        0.0908 55.6461 0.1956
2.0000        0.1000  5.9344 0.2006

--- TABLE 10: Layer Ablation (Averaged across alphas) ---
layer_config  Target_Score     PPL    JSD
         all        0.1276  7.5551 0.2049
  first_half        0.0713 42.3965 0.2119
 second_half        0.0901  5.2454 0.1947
=============================================================

--------------------------------------------------------------

python ablation_tables.py --model qwen2.5

=============================================================
  ABLATION RESULTS [Clean Response-Only] - QWEN2.5
=============================================================

--- TABLE 9: Alpha Ablation (Averaged across layers) ---
 alpha  Target_Score     PPL    JSD
0.5000        0.0611 14.8361 0.2120
1.0000        0.0757 13.6587 0.2172
1.5000        0.0811 15.6130 0.2096
2.0000        0.0767 15.6185 0.2057

--- TABLE 10: Layer Ablation (Averaged across alphas) ---
layer_config  Target_Score     PPL    JSD
         all        0.0726 15.4743 0.1995
  first_half        0.0668 15.2463 0.2291
 second_half        0.0816 14.0740 0.2048
=============================================================
"""

import argparse
import gc
import json
import math
import os
import pandas as pd
import numpy as np

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

MODEL_CONFIGS = {
    "phi4": {
        "hf_name": "microsoft/Phi-4-mini-instruct",
        "gen_json": "generation_results_phi4.json"
    },
    "qwen2.5": {
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "gen_json": "generation_results_qwen2.5.json"
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

def clean_generation(raw_text: str, prompt_text: str, emotion_text: str) -> str:
    if not isinstance(raw_text, str): return ""
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

def load_ablation_records(cfg: dict) -> list[dict]:
    with open(cfg["gen_json"], "r", encoding="utf-8") as f:
        records = json.load(f)
    
    # SADECE Variant 2 alınıyor (Ablasyon tablosu Var 2 için)
    var2_records = [r for r in records if str(r.get("variant")) == "2"]
    
    clean_records = []
    for r in var2_records:
        emo = r.get("emotion_text", "")
        clean_gen = clean_generation(r.get("generated_text", ""), r["prompt"], emo)
        clean_van = clean_generation(r.get("vanilla_text", ""), r["prompt"], emo)
        
        clean_records.append({
            "target_emotion": r["target_emotion"],
            "alpha": float(r["alpha"]),
            "layer_config": r["layer_config"],
            "clean_gen": clean_gen,
            "clean_van": clean_van
        })
    print(f"  Loaded {len(clean_records)} Variant 2 records for Ablation.")
    return clean_records

def run_roberta(records: list[dict]) -> list[dict]:
    print("\n[Step A] Loading RoBERTa classifier …")
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("text-classification", model=ROBERTA_MODEL, top_k=None, device=device, truncation=True, max_length=512)

    def get_raw_dist(text: str) -> list[float]:
        if not text or str(text).strip() == "": text = " "
        preds = clf(text[:1024])[0]
        label2score = {p["label"]: p["score"] for p in preds}
        return [label2score.get(lbl, 0.0) for lbl in EMOTION_LABELS]

    print("  Scoring records ...", flush=True)
    for i, rec in enumerate(records):
        emo_idx = EMOTION_LABELS.index(rec["target_emotion"].lower()) if rec["target_emotion"].lower() in EMOTION_LABELS else -1
        dist_gen = get_raw_dist(rec["clean_gen"])
        dist_van = get_raw_dist(rec["clean_van"])
        
        rec["target_score"] = dist_gen[emo_idx] if emo_idx >= 0 else 0.0
        rec["jsd"] = float(jensenshannon(dist_gen, dist_van) ** 2)
        
        if (i+1) % 100 == 0: print(f"    {i+1}/600 scored.")

    del clf
    gc.collect()
    torch.cuda.empty_cache()
    return records

def run_perplexity(cfg: dict, records: list[dict]) -> list[dict]:
    print(f"\n[Step B] Loading base LLM ({cfg['hf_name']}) in 4-bit …")
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"], cache_dir="./.hf_cache")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_name"], cache_dir="./.hf_cache", quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    print("  Computing PPL ...", flush=True)
    with torch.no_grad():
        for i, rec in enumerate(records):
            text = rec["clean_gen"]
            if not text.strip():
                rec["ppl"] = float("nan")
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc["input_ids"].to(model.device)
            if input_ids.shape[1] < 2:
                rec["ppl"] = float("nan")
                continue
            try:
                out = model(input_ids=input_ids, labels=input_ids)
                rec["ppl"] = math.exp(out.loss.item())
            except Exception:
                rec["ppl"] = float("nan")
                
            if (i+1) % 100 == 0: print(f"    {i+1}/600 PPL computed.")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return records

def print_tables(records: list[dict], model_name: str):
    df = pd.DataFrame(records)
    
    print(f"\n=============================================================")
    print(f"  ABLATION RESULTS [Clean Response-Only] - {model_name.upper()}")
    print(f"=============================================================")
    
    # ALPHA ABLATION (Table 9)
    print("\n--- TABLE 9: Alpha Ablation (Averaged across layers) ---")
    alpha_df = df.groupby("alpha").agg(
        Target_Score=("target_score", "mean"),
        PPL=("ppl", lambda x: np.nanmean(x)),
        JSD=("jsd", "mean")
    ).reset_index()
    print(alpha_df.to_string(index=False, float_format="%.4f"))

    # LAYER ABLATION (Table 10)
    print("\n--- TABLE 10: Layer Ablation (Averaged across alphas) ---")
    layer_df = df.groupby("layer_config").agg(
        Target_Score=("target_score", "mean"),
        PPL=("ppl", lambda x: np.nanmean(x)),
        JSD=("jsd", "mean")
    ).reset_index()
    print(layer_df.to_string(index=False, float_format="%.4f"))
    print("=============================================================\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()
    
    cfg = MODEL_CONFIGS[args.model]
    records = load_ablation_records(cfg)
    records = run_roberta(records)
    records = run_perplexity(cfg, records)
    print_tables(records, args.model)

if __name__ == "__main__":
    main()