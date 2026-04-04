import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm

# =========================================================================
# Q1 Journal Data Augmentation (Algorithmic Mapping over GoEmotions)
# =========================================================================

# GoEmotions label integer mapping
GO_EMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Semantic domains to create rich prompt inputs
SEMANTIC_DOMAINS = [
    "technology", "art", "daily life", "politics", "sports", 
    "science", "movies", "food", "travel", "work"
]

# Generic context prompts that can fit almost any statement
GENERIC_PROMPTS = [
    "What are your thoughts on this?",
    "Can you share your perspective?",
    "How does this make you feel?",
    "Tell me about your reaction to the recent events.",
    "Could you elaborate on your current mood regarding this?",
    "What is your stance on the matter?",
    "How would you describe your feelings today?",
    "Can you give me your honest opinion?",
    "What's your reaction to the news?",
    "Please express your thoughts on the situation."
]

def generate_augmented_dataset(target_size=2500, output_path="augmented_dataset.jsonl"):
    print(f"Loading 'go_emotions' (simplified) dataset from HuggingFace cache...")
    try:
        dataset = load_dataset("go_emotions", "simplified", cache_dir="./.hf_cache", download_mode="reuse_dataset_if_exists")
    except Exception as e:
        print("Failed to load generic 'go_emotions'. Trying 'google-research-datasets'...")
        dataset = load_dataset("google-research-datasets/go_emotions", "simplified", cache_dir="./.hf_cache", download_mode="reuse_dataset_if_exists")
        
    pure_records = []
    
    # Extract records with pure (single) emotions
    print("Filtering pure emotional signals...")
    for split in ["train", "validation", "test"]:
        for record in tqdm(dataset[split], desc=f"Scanning {split}"):
            labels = record["labels"]
            if len(labels) == 1: # Pure emotion
                emo_id = labels[0]
                emo_name = GO_EMOTIONS_LABELS[emo_id]
                text = record["text"]
                # Skip extremely short or long texts
                if 15 < len(text) < 150:
                    pure_records.append((emo_name, text))
                    
    # Shuffle aggressively for randomness
    random.seed(42)
    random.shuffle(pure_records)
    
    # Sample down to target size
    sampled = pure_records[:target_size] if len(pure_records) > target_size else pure_records
    
    final_pairs = []
    for (emo_name, emotion_text) in sampled:
        # Algorithmic pairing
        domain = random.choice(SEMANTIC_DOMAINS)
        context_prompt = random.choice(GENERIC_PROMPTS)
        
        # User input simulates someone asking a domain-specific conversational query
        fake_input = f"[{domain.capitalize()}] {context_prompt}"
        
        # The output is simply the real human GoEmotion text (which acts as the assistant expressing the emotion naturally)
        fake_output = f"{emotion_text.capitalize()}"
        
        final_pairs.append({
            "target_emotion": emo_name,
            "emotion_text": emotion_text,
            "input": fake_input,
            "output": fake_output
        })
        
    print(f"\nCreated {len(final_pairs)} highly diverse pairs.")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in final_pairs:
            f.write(json.dumps(pair) + "\n")
            
    print(f"[SUCCESS] Augmented dataset written to {output_path}")
    print(f"Sample: {json.dumps(final_pairs[0], indent=2)}")

if __name__ == "__main__":
    generate_augmented_dataset(2500, "augmented_dataset.jsonl")
