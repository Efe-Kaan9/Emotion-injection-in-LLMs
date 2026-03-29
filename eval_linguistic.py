"""
eval_linguistic.py — Phase 2: CPU-Only Linguistic Evaluation
=============================================================

Reads the JSON output from generate_data.py and computes:
  - Dist-1  (distinct unigrams ratio)
  - Dist-2  (distinct bigrams ratio)
  - Self-BLEU (intra-group generation diversity)

ZERO VRAM — this script does not import torch or use GPU at all.

Usage:
    python eval_linguistic.py --input generation_results_phi4.json
    python eval_linguistic.py --input generation_results_phi4.json --output eval_ling.json
"""

from __future__ import annotations

import os
import sys
import json
import csv
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple
from itertools import combinations

# ──────────────────────────────────────────────────────────────
# NLTK setup (CPU-only)
# ──────────────────────────────────────────────────────────────

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    print("ERROR: nltk is required. Install with:  pip install nltk")
    sys.exit(1)

# Ensure punkt tokeniser is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ──────────────────────────────────────────────────────────────
# Tokenisation helper
# ──────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercasing tokenisation."""
    return text.lower().split()


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def distinct_n(texts: List[str], n: int) -> float:
    """
    Compute Distinct-N across a list of texts.

    Dist-N = |unique n-grams| / |total n-grams|

    A higher value means the model produces more varied language.
    """
    total_ngrams: List[Tuple[str, ...]] = []
    for text in texts:
        tokens = tokenize(text)
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        total_ngrams.extend(ngrams)

    if len(total_ngrams) == 0:
        return 0.0

    return len(set(total_ngrams)) / len(total_ngrams)


def self_bleu(texts: List[str]) -> float:
    """
    Compute Self-BLEU for a group of texts.

    Self-BLEU measures how similar the generated texts are to each other.
    Lower Self-BLEU = more diverse outputs.

    For each text, compute its BLEU score against all OTHER texts in the group,
    then average.
    """
    if len(texts) < 2:
        return 0.0

    smoother = SmoothingFunction().method1
    tokenized = [tokenize(t) for t in texts]
    scores = []

    for i, hypothesis in enumerate(tokenized):
        references = [tokenized[j] for j in range(len(tokenized)) if j != i]
        if len(hypothesis) == 0:
            scores.append(0.0)
            continue
        try:
            score = sentence_bleu(
                references,
                hypothesis,
                weights=(0.5, 0.5),  # bigram-weighted BLEU
                smoothing_function=smoother,
            )
            scores.append(score)
        except (ValueError, ZeroDivisionError):
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


# ──────────────────────────────────────────────────────────────
# Group-wise evaluation
# ──────────────────────────────────────────────────────────────

def evaluate(records: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Group records by (target_emotion, alpha, layer_config, variant) and
    compute Dist-1, Dist-2, Self-BLEU for each group.

    Returns:
        group_results: per-group metrics
        overall_summary: aggregated means
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    vanilla_texts: List[str] = []

    for rec in records:
        key = (
            rec.get("target_emotion", "unknown"),
            rec.get("alpha", 1.0),
            rec.get("layer_config", "all"),
            rec.get("variant", 1),
        )
        groups[str(key)].append(rec.get("generated_text", ""))
        vanilla_texts.append(rec.get("vanilla_text", ""))

    group_results = []
    for key_str, texts in groups.items():
        d1 = distinct_n(texts, 1)
        d2 = distinct_n(texts, 2)
        sb = self_bleu(texts)
        group_results.append({
            "group_key": key_str,
            "num_texts": len(texts),
            "dist_1": round(d1, 4),
            "dist_2": round(d2, 4),
            "self_bleu": round(sb, 4),
        })

    # Also compute metrics for vanilla baseline
    vanilla_unique = list(set(vanilla_texts))
    vanilla_d1 = distinct_n(vanilla_unique, 1)
    vanilla_d2 = distinct_n(vanilla_unique, 2)
    vanilla_sb = self_bleu(vanilla_unique)

    # Overall summary
    if group_results:
        avg_d1 = sum(g["dist_1"] for g in group_results) / len(group_results)
        avg_d2 = sum(g["dist_2"] for g in group_results) / len(group_results)
        avg_sb = sum(g["self_bleu"] for g in group_results) / len(group_results)
    else:
        avg_d1 = avg_d2 = avg_sb = 0.0

    overall = {
        "steered_avg_dist1": round(avg_d1, 4),
        "steered_avg_dist2": round(avg_d2, 4),
        "steered_avg_self_bleu": round(avg_sb, 4),
        "vanilla_dist1": round(vanilla_d1, 4),
        "vanilla_dist2": round(vanilla_d2, 4),
        "vanilla_self_bleu": round(vanilla_sb, 4),
        "num_groups": len(group_results),
        "total_records": len(records),
    }

    return group_results, overall


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: CPU-only linguistic evaluation (Dist-1, Dist-2, Self-BLEU)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to generation results JSON from Phase 1",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: eval_linguistic_results.json)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output CSV summary path (default: eval_linguistic_summary.csv)",
    )
    args = parser.parse_args()

    output_path = args.output or "eval_linguistic_results.json"
    csv_path = args.csv or "eval_linguistic_summary.csv"

    # Load data
    print(f"Loading data from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records")

    # Evaluate
    print("\nComputing linguistic metrics (CPU only)...")
    group_results, overall = evaluate(records)

    # Print summary
    print("\n" + "=" * 60)
    print("LINGUISTIC EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total records:         {overall['total_records']}")
    print(f"  Number of groups:      {overall['num_groups']}")
    print(f"  Steered avg Dist-1:    {overall['steered_avg_dist1']:.4f}")
    print(f"  Steered avg Dist-2:    {overall['steered_avg_dist2']:.4f}")
    print(f"  Steered avg Self-BLEU: {overall['steered_avg_self_bleu']:.4f}")
    print(f"  Vanilla Dist-1:        {overall['vanilla_dist1']:.4f}")
    print(f"  Vanilla Dist-2:        {overall['vanilla_dist2']:.4f}")
    print(f"  Vanilla Self-BLEU:     {overall['vanilla_self_bleu']:.4f}")
    print("=" * 60)

    # Save JSON
    output_data = {
        "overall_summary": overall,
        "group_results": group_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Full results saved to {output_path}")

    # Save CSV summary
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=group_results[0].keys())
        writer.writeheader()
        writer.writerows(group_results)
    print(f"✅ CSV summary saved to {csv_path}")


if __name__ == "__main__":
    main()
