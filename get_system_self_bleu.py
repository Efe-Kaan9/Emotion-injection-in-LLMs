import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# NLTK punkt indir (eğer yoksa)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_self_bleu(texts):
    if len(texts) < 2:
        return 0.0
        
    # Metinleri token'lara ayır (küçük harf yaparak)
    tokenized_texts = [text.lower().split() for text in texts if str(text).strip()]
    
    smoothie = SmoothingFunction().method1
    scores = []
    
    print(f"Hesaplanıyor... ({len(tokenized_texts)} cümle)")
    for i, hyp in enumerate(tokenized_texts):
        # Kendisi hariç diğer tüm metinleri referans kabul et (Mode Collapse ölçümü)
        refs = tokenized_texts[:i] + tokenized_texts[i+1:]
        
        # Eğer hipotez çok kısaysa hata almamak için
        if len(hyp) == 0:
            continue
            
        score = sentence_bleu(refs, hyp, smoothing_function=smoothie)
        scores.append(score)
        
    if not scores:
        return 0.0
        
    return sum(scores) / len(scores)

def main():
    file_name = "system_prompt_results.json"
    
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"HATA: {file_name} bulunamadı!")
        return

    print("--- SYSTEM PROMPT SELF-BLEU SONUÇLARI ---")
    
    for model_key in ["phi4", "qwen2.5"]:
        if model_key in data:
            records = data[model_key].get("records", [])
            # Sadece system prompt çıktılarını alıyoruz
            system_texts = [r.get("system_text", "") for r in records]
            
            print(f"\n{model_key.upper()} için Self-BLEU hesaplanıyor...")
            self_bleu = calculate_self_bleu(system_texts)
            print(f">>> {model_key.upper()} System Prompt Self-BLEU: {self_bleu:.4f}")

if __name__ == "__main__":
    main()