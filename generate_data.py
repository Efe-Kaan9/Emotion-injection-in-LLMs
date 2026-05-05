"""
generate_data.py — Phase 1: Offline Data Generation with Ablations
===================================================================

Generates steered text for all combinations of:
  (prompt × emotion × alpha × layer_config × variant)

and saves results to a JSON file for downstream evaluation.

Q1-Grade Ablation Axes:
  - Alpha scaling:    {0.5, 1.0, 1.5, 2.0}
  - Layer-wise ablation: {"all", "first_half", "second_half"}
  - Variant:          {1 (Basic), 2 (Modulated)}

Memory Budget: RTX 3060 (6 GB VRAM), 16 GB RAM
  - 4-bit quantised LLM
  - batch size = 1
  - torch.no_grad() everywhere
  - torch.cuda.empty_cache() between records

Usage:
    python generate_data.py --model phi4
    python generate_data.py --model qwen2.5
    python generate_data.py --model phi4 --dry-run     # shapes only, no GPU
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
)
from transformers.cache_utils import DynamicCache

from projector_agnostic import ModelConfig, create_projector

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

MODELS = {
    "phi4": "microsoft/Phi-4-mini-instruct",
    "qwen2.5": "Qwen/Qwen2.5-1.5B-Instruct",
}

CACHE_DIR = "./.hf_cache"
CHECKPOINT_DIR = "./checkpoints"
ENCODER_NAME = "distilbert-base-uncased"
NUM_EMOTIONS = 28

ALPHA_VALUES = [0.5, 1.0, 1.5, 2.0]
LAYER_CONFIGS = ["all", "first_half", "second_half"]

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
PREFIX_LEN = 4

# GoEmotions label set (28 classes) — used for naming only
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# ──────────────────────────────────────────────────────────────
# Prompts & target emotions for generation
# ──────────────────────────────────────────────────────────────

GENERATION_ITEMS: List[Dict] = [
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



# ──────────────────────────────────────────────────────────────
# Emotion Extractor (mirrors notebook — frozen, no code change)
# ──────────────────────────────────────────────────────────────

class EmotionExtractor(nn.Module):
    """Text → 28-dim emotion distribution (sigmoid multi-label)."""

    def __init__(self, encoder_name: str = ENCODER_NAME, num_emotions: int = NUM_EMOTIONS):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name, cache_dir=CACHE_DIR)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_emotions)

    def forward(self, input_ids, attention_mask, **kwargs):
        # **kwargs absorbs token_type_ids that some tokenizers produce
        out = self.encoder(input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)


def get_emotion_embedding(
    text: str,
    emotion_extractor: nn.Module,
    tokenizer,
    device: str,
) -> torch.Tensor:
    """Extract emotion distribution from text. Returns (1, num_emotions)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = emotion_extractor(**inputs)
        return torch.sigmoid(logits)


# ──────────────────────────────────────────────────────────────
# Modified cache injection with layer-wise ablation + alpha
# ──────────────────────────────────────────────────────────────

def prepend_prefix_to_cache_ablated(
    past_kv: DynamicCache,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    alpha: float = 1.0,
    layer_config: str = "all",
    num_layers: int = 32,
) -> DynamicCache:
    """
    Prepend emotion KV prefix with alpha scaling and layer selection.

    Parameters
    ----------
    alpha : float
        Scaling factor applied to the prefix before concatenation.
        alpha=1.0 is the default (no scaling).
    layer_config : {"all", "first_half", "second_half"}
        Which layers receive the emotion prefix.
    """
    half = num_layers // 2

    if layer_config == "first_half":
        inject_layers = set(range(0, half))
    elif layer_config == "second_half":
        inject_layers = set(range(half, num_layers))
    else:  # "all"
        inject_layers = set(range(num_layers))

    # Apply alpha scaling
    k_scaled = k_prefix * alpha
    v_scaled = v_prefix * alpha

    new_cache = DynamicCache()
    for layer_idx in range(len(past_kv)):
        if hasattr(past_kv, "key_cache"):
            k_layer = past_kv.key_cache[layer_idx]
            v_layer = past_kv.value_cache[layer_idx]
        elif hasattr(past_kv, "layers"):
            k_layer = past_kv.layers[layer_idx].keys
            v_layer = past_kv.layers[layer_idx].values
        else:
            k_layer = past_kv[layer_idx][0]
            v_layer = past_kv[layer_idx][1]

        if layer_idx in inject_layers:
            k_new = torch.cat(
                [k_scaled.to(k_layer.device).to(k_layer.dtype), k_layer], dim=2
            )
            v_new = torch.cat(
                [v_scaled.to(v_layer.device).to(v_layer.dtype), v_layer], dim=2
            )
        else:
            # No injection — still need matching sequence length
            # Use zeros so attention to prefix positions is neutral
            zeros_k = torch.zeros_like(k_scaled).to(k_layer.device).to(k_layer.dtype)
            zeros_v = torch.zeros_like(v_scaled).to(v_layer.device).to(v_layer.dtype)
            k_new = torch.cat([zeros_k, k_layer], dim=2)
            v_new = torch.cat([zeros_v, v_layer], dim=2)

        new_cache.update(k_new, v_new, layer_idx=layer_idx)

    return new_cache


# ──────────────────────────────────────────────────────────────
# Autoregressive generation with modified cache
# ──────────────────────────────────────────────────────────────

def generate_with_cache(
    decoder,
    decoder_tokenizer,
    input_ids: torch.Tensor,
    past_kv: DynamicCache,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """Token-by-token generation using a pre-modified KV cache."""
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = decoder(
                input_ids=generated[:, -1:],
                use_cache=True,
                past_key_values=past_kv,
            )
            logits = out.logits[:, -1, :]
            past_kv = out.past_key_values
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == decoder_tokenizer.eos_token_id:
                break
    return decoder_tokenizer.decode(generated[0], skip_special_tokens=True)


def generate_vanilla(decoder, decoder_tokenizer, prompt: str, device: str) -> str:
    """Baseline generation — no KV injection."""
    inputs = decoder_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = decoder.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.95,
            pad_token_id=decoder_tokenizer.eos_token_id,
        )
    return decoder_tokenizer.decode(output[0], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────

def run_generation(model_key: str, output_path: str, dry_run: bool = False):
    model_name = MODELS[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f"  Phase 1: Data Generation -- {model_name}")
    print(f"  Device:  {device}")
    print(f"  Output:  {output_path}")
    print("=" * 70)

    # ---- 1. Load emotion extractor (small, stays on GPU) ----
    print("\n[1/4] Loading emotion extractor...")
    enc_tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME, cache_dir=CACHE_DIR)
    emotion_ext = EmotionExtractor().to(device).eval()
    emo_ckpt_path = "checkpoint.pt"
    if os.path.isfile(emo_ckpt_path):
        ckpt = torch.load(emo_ckpt_path, map_location="cpu")
        emotion_ext.load_state_dict(ckpt["model_state_dict"])
        print("  [OK] Loaded pre-trained emotion extractor")
    else:
        print("  [WARN] Using untrained emotion extractor")

    # ---- 2. Load LLM (4-bit quantised) ----
    print("\n[2/4] Loading LLM...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    dec_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    if dec_tokenizer.pad_token is None:
        dec_tokenizer.pad_token = dec_tokenizer.eos_token

    decoder = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )
    decoder.eval()
    print(f"  [OK] {model_name} loaded (4-bit)")

    # ---- 3. Build ModelConfig + projectors ----
    print("\n[3/4] Initialising projectors...")
    mc = ModelConfig.from_hf_config(decoder.config, model_name=model_name)
    print(mc.summary())

    # Checkpoint paths — model-specific to avoid collisions
    v1_ckpt = os.path.join(CHECKPOINT_DIR, f"variant1_projector_{model_key}.pt")
    v2_ckpt = os.path.join(CHECKPOINT_DIR, f"variant2_projector_{model_key}.pt")
    # Fallback: try the original Phi-4 checkpoint names
    if not os.path.isfile(v1_ckpt) and model_key == "phi4":
        v1_ckpt = os.path.join(CHECKPOINT_DIR, "variant1_projector.pt")
    if not os.path.isfile(v2_ckpt) and model_key == "phi4":
        v2_ckpt = os.path.join(CHECKPOINT_DIR, "variant2_projector.pt")

    projector_v1 = create_projector(
        mc, variant=1, checkpoint_path=v1_ckpt, device=device
    )
    projector_v2 = create_projector(
        mc, variant=2, checkpoint_path=v2_ckpt, device=device
    )

    # ---- 4. Generation loop ----
    print("\n[4/4] Starting generation loop...")
    total_combos = (
        len(GENERATION_ITEMS) * len(ALPHA_VALUES) * len(LAYER_CONFIGS) * 2
    )
    print(f"  Total records: {total_combos}")

    if dry_run:
        print("  [DRY RUN] -- skipping actual generation")
        dummy = torch.randn(1, NUM_EMOTIONS, device=device)
        k, v = projector_v1(dummy)
        print(f"  Var1 prefix shape: K={k.shape}, V={v.shape}")
        k2, v2, g = projector_v2(dummy)
        print(f"  Var2 prefix shape: K={k2.shape}, V={v2.shape}, gates={g.shape}")
        print("  [PASS] Dry run complete -- shapes OK")
        return

    results = []
    record_idx = 0

    for item in GENERATION_ITEMS:
        prompt = item["prompt"]
        emotion_text = item["emotion_text"]
        target_emotion = item["target_emotion"]

        # Generate VANILLA output (once per prompt)
        vanilla_text = generate_vanilla(decoder, dec_tokenizer, prompt, device)
        torch.cuda.empty_cache()

        # Extract emotion vector (once per prompt + emotion_text)
        emotion_vec = get_emotion_embedding(
            emotion_text, emotion_ext, enc_tokenizer, device
        )

        for alpha in ALPHA_VALUES:
            for layer_cfg in LAYER_CONFIGS:
                for variant_id in [1, 2]:
                    record_idx += 1
                    t0 = time.time()

                    # Get projector output
                    projector = projector_v1 if variant_id == 1 else projector_v2
                    with torch.no_grad():
                        if variant_id == 1:
                            k_prefix, v_prefix = projector(emotion_vec)
                        else:
                            k_prefix, v_prefix, _ = projector(emotion_vec)

                    # Initial forward to get base cache
                    inputs = dec_tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = decoder(**inputs, use_cache=True)
                        past_kv = out.past_key_values

                    # Inject with ablation
                    modified_cache = prepend_prefix_to_cache_ablated(
                        past_kv, k_prefix, v_prefix,
                        alpha=alpha,
                        layer_config=layer_cfg,
                        num_layers=mc.num_hidden_layers,
                    )

                    # Generate
                    steered_text = generate_with_cache(
                        decoder, dec_tokenizer, inputs["input_ids"],
                        modified_cache,
                    )

                    # Store record
                    record = {
                        "record_id": record_idx,
                        "model": model_key,
                        "prompt": prompt,
                        "emotion_text": emotion_text,
                        "target_emotion": target_emotion,
                        "alpha": alpha,
                        "layer_config": layer_cfg,
                        "variant": variant_id,
                        "generated_text": steered_text,
                        "vanilla_text": vanilla_text,
                    }
                    results.append(record)

                    elapsed = time.time() - t0
                    print(
                        f"  [{record_idx:3d}/{total_combos}] "
                        f"v{variant_id} | α={alpha} | layers={layer_cfg:12s} | "
                        f"{target_emotion:15s} | {elapsed:.1f}s"
                    )

                    # Memory cleanup
                    del past_kv, modified_cache, k_prefix, v_prefix, out
                    torch.cuda.empty_cache()

        # Cleanup per-prompt
        del emotion_vec
        torch.cuda.empty_cache()

    # ---- Save results ----
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved {len(results)} records to {output_path}")


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1: Generate steered text with ablations"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="phi4",
        help="Which LLM to use (default: phi4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: generation_results_{model}.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only test shapes, no GPU generation",
    )
    args = parser.parse_args()

    output_path = args.output or f"generation_results_{args.model}.json"
    run_generation(args.model, output_path, dry_run=args.dry_run)
