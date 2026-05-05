import json
import random

def extract_examples():
    file_name = "eval_mechanistic_phi4.json"
    
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    records = data.get("records", data)
    
    print("\n--- JSON İÇİNDEKİ GERÇEK ANAHTARLAR (DEBUG İÇİN) ---")
    print(list(records[0].keys()))
    print("-" * 60 + "\n")
    
    # Filtreleme: Variant 2 ve layer_config='all'
    valid_records = [r for r in records if str(r.get("variant")) == "2" and str(r.get("layer_config")) == "all"]
    
    if not valid_records:
        print("DİKKAT: Variant 2 ve layer_config='all' olan veri bulunamadı! Rastgele seçiliyor...")
        valid_records = records
        
    random.shuffle(valid_records)
    
    print("--- MAKALEYE YAPIŞTIRILACAK LATEX KODLARI ---\n")
    
    for item in valid_records[:5]:
        emo = str(item.get("target_emotion", item.get("emotion", "Unknown"))).capitalize()
        
        # 1. Prompt'u bul
        prompt = item.get("prompt", item.get("input_text", item.get("input", "Prompt Bulunamadı")))
        
        # 2. Vanilla'yı bul
        vanilla = item.get("vanilla_text", item.get("vanilla_response", item.get("vanilla", "Vanilla Bulunamadı")))
        
        # 3. Steered'i zorla bul (Anahtar adı ne olursa olsun, içinde 'steered' geçen uzun metni al)
        steered = "N/A"
        for k, v in item.items():
            if "steered" in str(k).lower() and isinstance(v, str) and len(v) > 20:
                steered = v
                break
                
        # Eğer hala bulamadıysa, prompt ve vanilla dışındaki en uzun metni al (kesin çözüm)
        if steered == "N/A":
            other_strings = [v for k, v in item.items() if isinstance(v, str) and v != prompt and v != vanilla and len(v) > 20]
            if other_strings:
                steered = max(other_strings, key=len)

        # LaTeX formatını bozmamak için temizlik ve kısaltma (çok uzunsa tabloyu taşırır)
        def clean(text):
            t = str(text).replace('\n', ' ').replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('_', '\\_')
            return (t[:165] + "...") if len(t) > 165 else t

        print(f"\\textbf{{{emo}}} & \"{clean(prompt)}\" & \"{clean(vanilla)}\" & \"{clean(steered)}\" \\\\ \\midrule")

if __name__ == "__main__":
    extract_examples()