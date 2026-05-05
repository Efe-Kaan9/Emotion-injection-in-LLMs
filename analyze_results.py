import json
import pandas as pd
import ast

def analyze_linguistic(file_name):
    print(f"\n" + "="*50)
    print(f"--- {file_name} VAR 1 / VAR 2 DİLSEL (LINGUISTIC) SONUÇLARI ---")
    print("="*50)
    
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # eval_linguistic çıktısındaki listemiz "group_results" anahtarı altındadır
        records = data.get("group_results", [])
        
        if not records:
            print(f"Veri bulunamadı. Lütfen {file_name} dosyasını kontrol et.")
            return

        df = pd.DataFrame(records)
        
        # group_key şu formatta bir string: "('sadness', 1.0, 'all', 1)"
        # Bunu gerçek bir Python Tuple'ına çevirip 4. elemanını (index 3) alarak Variant'ı buluyoruz
        df['variant'] = df['group_key'].apply(lambda x: ast.literal_eval(x)[3])
        
        # Gruplama ve ortalama alma (Sütun isimleri dist_1, dist_2, self_bleu şeklinde)
        grouped = df.groupby(["variant"]).agg(
            dist1=("dist_1", "mean"),
            dist2=("dist_2", "mean"),
            self_bleu=("self_bleu", "mean")
        ).reset_index()
        
        print(grouped.to_string(index=False))
        
    except FileNotFoundError:
        print(f"HATA: {file_name} dosyası bulunamadı.")
    except Exception as e:
        print(f"Hata: {e}")

# Çalıştır
analyze_linguistic("eval_linguistic_phi4.json")
analyze_linguistic("eval_linguistic_qwen2.5.json")